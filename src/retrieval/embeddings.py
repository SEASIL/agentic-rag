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
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
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
