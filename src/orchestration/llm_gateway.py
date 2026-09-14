"""
LLM Gateway for Smart Routing and Fallbacks.
This module acts as the central brain for deciding which model handles which task,
and provides a safety net if a model crashes.
"""
import os
from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from configs.settings import settings


def get_available_llms(require_advanced: bool = False, temperature: float | None = None) -> List[BaseChatModel]:
    temp = temperature if temperature is not None else settings.llm_temperature
    llms = []

    # 1. Groq (Ultra-fast, Llama 3 / open models)
    groq_key = os.getenv("GROQ_API_KEY") or settings.groq_api_key
    if groq_key and not groq_key.startswith("gsk_your"):
        try:
            model = "openai/gpt-oss-120b" if require_advanced else "openai/gpt-oss-20b"
            llms.append(ChatGroq(
                groq_api_key=groq_key,
                model_name=model,
                temperature=temp,
                max_retries=1,
            ))
        except Exception:
            pass

    # 2. Google Gemini (Massive context window & strong reasoning)
    gemini_key = os.getenv("GEMINI_API_KEY") or settings.gemini_api_key
    if gemini_key and not gemini_key.startswith("your-gemini"):
        try:
            model = settings.advanced_llm_model if require_advanced else settings.fast_llm_model
            llms.append(ChatGoogleGenerativeAI(
                google_api_key=gemini_key,
                model=model,
                temperature=temp,
                max_retries=1,
            ))
        except Exception:
            pass

    # 3. OpenRouter (Access to open-source models)
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or settings.openrouter_api_key
    if openrouter_key and not openrouter_key.startswith("sk-or-v1-your"):
        try:
            llms.append(ChatOpenAI(
                openai_api_key=openrouter_key,
                openai_api_base="https://openrouter.ai/api/v1",
                model_name=settings.openrouter_model,
                temperature=temp,
                max_retries=1,
            ))
        except Exception:
            pass

    # 4. OpenAI (GPT-4o / GPT-4o-mini)
    openai_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
    if openai_key and not openai_key.startswith("sk-your"):
        try:
            model = "gpt-4o" if require_advanced else "gpt-4o-mini"
            llms.append(ChatOpenAI(
                openai_api_key=openai_key,
                model_name=model,
                temperature=temp,
                max_retries=1,
            ))
        except Exception:
            pass

    return llms


def get_llm(require_advanced: bool = False, temperature: float | None = None) -> BaseChatModel:
    """
    Returns a primary LLM configured with automated runtime fallbacks.
    If the primary provider hits a 429 rate limit, quota error, or service outage,
    LangChain automatically retries using the next provider in the chain.
    """
    llms = get_available_llms(require_advanced=require_advanced, temperature=temperature)

    if not llms:
        # Ultimate fallback: local Ollama if no API keys are provided
        return ChatOpenAI(
            openai_api_key="ollama",
            openai_api_base=settings.ollama_base_url + "/v1",
            model_name="llama3",
            temperature=temperature if temperature is not None else settings.llm_temperature,
        )

    primary = llms[0]
    if len(llms) > 1:
        return primary.with_fallbacks(llms[1:])
    return primary


