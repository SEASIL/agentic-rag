"""
Wraps dense embedding models behind one interface so ingestion and
query-time retrieval always produce vectors the same way.

Dense:  Google Gemini API -> semantic similarity
"""
from __future__ import annotations

from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from configs.settings import settings

@lru_cache(maxsize=1)
def get_dense_model() -> GoogleGenerativeAIEmbeddings:
    """
    Returns a Google Generative AI embeddings model.
    """
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2", 
        google_api_key=settings.gemini_api_key
    )

def embed_dense(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_dense_model()
    return model.embed_documents(texts)

def embed_dense_query(query: str) -> list[float]:
    model = get_dense_model()
    return model.embed_query(query)
