"""
LLM Gateway for Smart Routing and Fallbacks.
This module acts as the central brain for deciding which model handles which task,
and provides a safety net if a model crashes.
"""
import os
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from configs.settings import settings

def get_llm(require_advanced: bool = False, temperature: float | None = None) -> BaseChatModel:
    """
    Returns the appropriate LLM based on available keys.
    Prefers OpenRouter (for Hermes) if available, then Gemini, then OpenAI.
    """
    temp = temperature if temperature is not None else settings.llm_temperature
    models = []
    
    gemini_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        model_name = settings.advanced_llm_model if require_advanced else settings.fast_llm_model
        models.append(ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temp,
            google_api_key=gemini_key,
            max_retries=1
        ))

    openrouter_key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if openrouter_key:
        models.append(ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            model=settings.openrouter_model,
            temperature=temp,
            max_retries=1
        ))

    openai_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        fallback_model = "gpt-4o" if require_advanced else "gpt-4o-mini"
        models.append(ChatOpenAI(
            model=fallback_model,
            temperature=temp,
            api_key=openai_key,
            max_retries=1
        ))

    if not models:
        raise ValueError("No API keys found! Please set OPENROUTER_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY in your .env file.")

    primary = models[0]
    if len(models) > 1:
        return primary.with_fallbacks(models[1:])
        
    return primary

