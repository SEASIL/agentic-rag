"""
Dense semantic retrieval using Chroma DB.
(Formerly hybrid search, refactored to dense-only for Chroma compatibility).
"""
from __future__ import annotations

import chromadb
from dataclasses import dataclass, field

from configs.settings import settings
from src.retrieval.embeddings import embed_dense_query
from src.retrieval.vector_store import ensure_collection, get_client


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source_path: str
    page_number: int | None
    metadata: dict = field(default_factory=dict)
    dense_rank: int | None = None
    rerank_score: float | None = None  # populated later by reranker.py


def hybrid_search(
    query: str,
    top_k: int | None = None,
    client: chromadb.ClientAPI | None = None,
) -> list[RetrievedChunk]:
    """Runs dense search against Chroma DB.
    (Kept function name 'hybrid_search' for backwards compatibility with retriever.py)"""
    client = client or get_client()
    collection = ensure_collection(client)
    top_k = top_k or settings.dense_top_k

    vec = embed_dense_query(query)
    
    results = collection.query(
        query_embeddings=[vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    
    if not results["ids"] or not results["ids"][0]:
        return retrieved
        
    for rank, (doc_id, text, metadata) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["metadatas"][0]), 
        start=1
    ):
        page_num = metadata.get("page_number", -1)
        
        rc = RetrievedChunk(
            id=doc_id,
            text=text,
            source_path=metadata.get("source_path", ""),
            page_number=page_num if page_num != -1 else None,
            metadata=metadata,
        )
        rc.dense_rank = rank
        retrieved.append(rc)

    return retrieved
