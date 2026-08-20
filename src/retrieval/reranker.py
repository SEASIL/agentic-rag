"""
Cross-encoder reranking. Hybrid search (dense+sparse+RRF) is cheap and casts
a wide net, but bi-encoder similarity is a weak relevance signal on its own —
a cross-encoder that jointly attends over (query, chunk) pairs is far more
accurate at judging true relevance, at the cost of being too slow to run over
the whole corpus. So: hybrid search narrows the corpus to `hybrid_top_k`
candidates, then the reranker does the expensive precise pass over just those.

Also computes whether the top result is confident enough to skip the web
search fallback (see settings.min_rerank_score).
"""
from __future__ import annotations

from functools import lru_cache
from sentence_transformers import CrossEncoder

from configs.settings import settings
from src.retrieval.hybrid_search import RetrievedChunk


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model_name)


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
    """Scores every (query, candidate) pair with the cross-encoder and
    returns the top_k highest-scoring chunks, sorted descending."""
    if not candidates:
        return []

    top_k = top_k or settings.rerank_top_k
    model = get_reranker()

    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = float(score)

    ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
    return ranked[:top_k]


def needs_web_fallback(reranked: list[RetrievedChunk]) -> bool:
    """Confidence gate: if even the best reranked chunk scores below
    min_rerank_score, local retrieval likely doesn't cover the question and
    the orchestration graph should route to web search instead."""
    if not reranked:
        return True
    return reranked[0].rerank_score < settings.min_rerank_score
