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


# Keywords that strongly indicate the user needs live/real-time web data
_WEB_SIGNALS = [
    "today", "latest", "current", "news", "now", "recent", "live",
    "price", "weather", "stock", "score", "update", "trending",
    "2024", "2025", "2026", "this week", "this month", "right now",
]


def _needs_web(query: str) -> bool:
    """
    Lightweight heuristic: if the query contains real-time signals,
    web search is likely needed regardless of local doc hits.
    """
    q = query.lower()
    return any(signal in q for signal in _WEB_SIGNALS)


def retrieve(query: str, search_mode: str = "auto") -> RetrievalResult:
    candidates = hybrid_search(query)

    # Bypass reranker for free hosting
    top_chunks = candidates[:8]

    if search_mode == "local":
        # User explicitly wants local docs only — never go to web
        should_web = False
    elif search_mode == "web":
        # User explicitly wants web — always go to web
        should_web = True
    else:
        # Auto mode: use web if query has real-time signals OR local DB has no results
        should_web = _needs_web(query) or len(top_chunks) == 0

    return RetrievalResult(
        chunks=top_chunks,
        should_fallback_to_web=should_web,
    )
