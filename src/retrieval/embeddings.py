"""
Wraps dense embedding models behind one interface so ingestion and
query-time retrieval always produce vectors the same way.

Dense:  sentence-transformers (BGE) -> semantic similarity
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from configs.settings import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_dense_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.dense_model_name)


def embed_dense(texts: list[str]) -> list[list[float]]:
    model = get_dense_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_dense_query(query: str) -> list[float]:
    return embed_dense([query])[0]
