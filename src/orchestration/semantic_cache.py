"""
Semantic Caching layer.
Intercepts incoming queries and returns cached answers if a semantically 
identical question was recently asked. Bypasses the entire LLM pipeline.
"""
from __future__ import annotations

import hashlib
import chromadb

from configs.settings import settings
from src.retrieval.embeddings import embed_dense_query

_cache_collection_name = "semantic_cache"

def _get_cache_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_db_dir)
    return client.get_or_create_collection(name=_cache_collection_name)

def check_cache(query: str, threshold: float = 0.15) -> dict | None:
    """
    Checks if a semantically similar query was asked.
    Returns the cached final_answer if found, else None.
    """
    collection = _get_cache_collection()
    if collection.count() == 0:
        return None
        
    vec = embed_dense_query(query)
    results = collection.query(
        query_embeddings=[vec],
        n_results=1,
        include=["metadatas", "distances"]
    )
    
    if not results["distances"] or not results["distances"][0]:
        return None
        
    distance = results["distances"][0][0]
    
    # Chroma uses L2 distance by default. Smaller distance = more similar.
    # A threshold of 0.15 is very strict, ensuring we don't return false positives.
    if distance < threshold:
        metadata = results["metadatas"][0][0]
        return {
            "final_answer": metadata.get("final_answer", ""),
            "citations": [], # Omitted for cached fast-path
            "is_cached": True
        }
    return None

def store_in_cache(query: str, final_answer: str) -> None:
    """Saves a successfully generated answer into the cache."""
    collection = _get_cache_collection()
    vec = embed_dense_query(query)
    
    query_id = hashlib.md5(query.encode()).hexdigest()
    
    collection.upsert(
        ids=[query_id],
        embeddings=[vec],
        documents=[query],
        metadatas=[{"final_answer": final_answer}]
    )
