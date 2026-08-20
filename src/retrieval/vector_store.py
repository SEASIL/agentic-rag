"""
PostgreSQL pgvector collection management using langchain-postgres.
"""
from __future__ import annotations

from langchain_postgres.vectorstores import PGVector
from configs.settings import settings
from src.ingestion.schema import Chunk
from src.retrieval.embeddings import get_dense_model

def get_vector_store() -> PGVector:
    if not settings.database_url:
        raise ValueError("DATABASE_URL environment variable is missing. Please set your Supabase connection string.")
        
    # Standardize the connection string for psycopg3
    connection_string = settings.database_url
    if connection_string.startswith("postgres://"):
        connection_string = connection_string.replace("postgres://", "postgresql+psycopg://")
    elif connection_string.startswith("postgresql://"):
        connection_string = connection_string.replace("postgresql://", "postgresql+psycopg://")

    return PGVector(
        embeddings=get_dense_model(),
        collection_name=settings.chroma_collection,
        connection=connection_string,
        use_jsonb=True
    )

def ensure_collection(client=None):
    # PGVector creates the collection automatically upon initialization/insertion
    return get_vector_store()

def upsert_chunks(chunks: list[Chunk], client=None, batch_size: int = 64) -> None:
    """Embeds and upserts chunks in batches using PGVector."""
    store = get_vector_store()

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        ids = [c.id for c in batch]

        metadatas = []
        for chunk in batch:
            metadatas.append({
                "source_type": str(chunk.source_type.value),
                "source_path": chunk.source_path,
                "page_number": chunk.page_number if chunk.page_number is not None else -1,
                "section_title": chunk.section_title or "",
            })

        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
