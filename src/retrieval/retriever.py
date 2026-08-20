"""
Single entry point the orchestration graph calls: query in, ranked
chunks out, plus a flag telling the planner whether to trigger web fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.hybrid_search import hybrid_search, RetrievedChunk


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    should_fallback_to_web: bool


def retrieve(query: str) -> RetrievalResult:
    candidates = hybrid_search(query)
    
    # Bypass reranker for free hosting
    top_chunks = candidates[:8]
    
    return RetrievalResult(
        chunks=top_chunks,
        should_fallback_to_web=len(top_chunks) == 0,
    )
