"""
Dense semantic retrieval using PostgreSQL pgvector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from configs.settings import settings
from src.retrieval.vector_store import get_vector_store

@dataclass
class RetrievedChunk:
    id: str
    text: str
    source_path: str
    page_number: int | None
    metadata: dict = field(default_factory=dict)
    dense_rank: int | None = None
    rerank_score: float | None = None


def hybrid_search(
    query: str,
    top_k: int | None = None,
    client=None,
) -> list[RetrievedChunk]:
    """Runs dense search against Postgres pgvector."""
    store = get_vector_store()
    top_k = top_k or settings.dense_top_k
    
    # Run similarity search
    results = store.similarity_search_with_score(query, k=top_k)

    retrieved = []
    
    for rank, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata or {}
        page_num = metadata.get("page_number", -1)
        
        # PGVector doesn't easily expose the raw ID from add_texts natively in the result Document 
        # without custom queries, but for standard RAG, we don't strictly need the UUID here.
        # We will fallback to a default if not present.
        doc_id = metadata.get("id", f"doc_{rank}")
        
        rc = RetrievedChunk(
            id=doc_id,
            text=doc.page_content,
            source_path=metadata.get("source_path", ""),
            page_number=page_num if page_num != -1 else None,
            metadata=metadata,
        )
        rc.dense_rank = rank
        retrieved.append(rc)

    return retrieved
