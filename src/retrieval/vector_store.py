"""
Chroma DB collection management.
"""
from __future__ import annotations

import chromadb

from configs.settings import settings
from src.ingestion.schema import Chunk
from src.retrieval.embeddings import embed_dense


from chromadb.config import Settings

def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=settings.chroma_db_dir,
        settings=Settings(anonymized_telemetry=False)
    )


def ensure_collection(client: chromadb.ClientAPI | None = None):
    client = client or get_client()
    return client.get_or_create_collection(name=settings.chroma_collection)


def upsert_chunks(chunks: list[Chunk], client: chromadb.ClientAPI | None = None, batch_size: int = 64) -> None:
    """Embeds and upserts chunks in batches using Chroma DB."""
    client = client or get_client()
    collection = ensure_collection(client)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        ids = [c.id for c in batch]

        dense_vecs = embed_dense(texts)

        metadatas = []
        for chunk in batch:
            # Chroma metadatas must be str, int, float, or bool
            metadatas.append({
                "source_type": str(chunk.source_type.value),
                "source_path": chunk.source_path,
                "page_number": chunk.page_number if chunk.page_number is not None else -1,
                "section_title": chunk.section_title or "",
            })

        collection.upsert(
            ids=ids,
            embeddings=dense_vecs,
            documents=texts,
            metadatas=metadatas
        )
