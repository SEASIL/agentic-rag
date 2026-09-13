"""
LLM Gateway for Smart Routing and Fallbacks.
This module acts as the central brain for deciding which model handles which task,
and provides a safety net if a model crashes.
"""
import os
from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models import ChatOllama

from configs.settings import settings

def get_llm(require_advanced: bool = False, temperature: float | None = None) -> BaseChatModel:
    temp = temperature if temperature is not None else settings.llm_temperature
    
    # Always use local Ollama for the benchmark to avoid API rate limits
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model="phi3",
        temperature=temp
    )

