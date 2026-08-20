"""
Centralized configuration for the Hybrid Agentic RAG system.
All tunables live here so retrieval, orchestration, and eval stay in sync.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database (Supabase / Postgres) ---
    database_url: str = ""
    chroma_collection: str = "enterprise_docs"

    # --- Embeddings ---
    dense_model_name: str = "BAAI/bge-m3"   # dense embedder
    reranker_model_name: str = "BAAI/bge-reranker-base"

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    table_chunk_strategy: str = "structured_json"  # options: structured_json | markdown | summarize

    # --- Retrieval ---
    dense_top_k: int = 40           # candidates pulled from dense search
    rerank_top_k: int = 8           # final chunks handed to the synthesizer
    min_rerank_score: float = 0.15  # below this -> trigger web search fallback

    # --- LLM Gateway (Smart Routing) ---
    ollama_base_url: str = "http://localhost:11434"
    # --- LLM Providers ---
    gemini_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    
    # Models to use if using Gemini
    fast_llm_model: str = "gemini-3.5-flash"       
    advanced_llm_model: str = "gemini-3.5-flash"   
    
    # Model to use if using OpenRouter
    openrouter_model: str = "openrouter/free"
    
    llm_temperature: float = 0.0

    # --- Web search fallback (Tavily) ---
    tavily_api_key: str = ""
    web_search_max_results: int = 5

    # --- Multi-hop orchestration ---
    max_planner_hops: int = 4

    # --- Eval ---
    ragas_thresholds: dict = {
        "context_recall": 0.75,
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.70,
    }


settings = Settings()
