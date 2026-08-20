"""
Single entry point the orchestration graph calls: query in, ranked+reranked
chunks out, plus a flag telling the planner whether to trigger web fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.hybrid_search import hybrid_search, RetrievedChunk
from src.retrieval.reranker import rerank, needs_web_fallback


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    should_fallback_to_web: bool


def retrieve(query: str) -> RetrievalResult:
    candidates = hybrid_search(query)
    top_chunks = rerank(query, candidates)
    return RetrievalResult(
        chunks=top_chunks,
        should_fallback_to_web=needs_web_fallback(top_chunks),
    )
