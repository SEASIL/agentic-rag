"""
Web search fallback. Only invoked when local hybrid retrieval's reranker
confidence falls below settings.min_rerank_score (see reranker.needs_web_fallback)
— i.e. the corpus likely doesn't cover the question.

Uses Tavily Search — a search engine built specifically for AI agents.
It bypasses bot blocks and returns clean, extracted text from webpages
which is perfect for Agentic RAG. Requires a TAVILY_API_KEY in your .env file.
"""
from __future__ import annotations

from tavily import TavilyClient

from configs.settings import settings
from src.orchestration.state import GraphState

from src.orchestration.llm_gateway import get_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

def web_search_node(state: GraphState) -> dict:
    query = state.get("rewritten_query") or state["current_query"]

    # Contextualize query if there is chat history
    if state.get("chat_history"):
        from langchain_core.messages import HumanMessage, AIMessage
        history = []
        for msg in state.get("chat_history", [])[-4:]: # Only use last 4 turns to save tokens
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, formulate a standalone search query that can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is. Output ONLY the query."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        llm = get_llm(require_advanced=False)
        query = (contextualize_q_prompt | llm | StrOutputParser()).invoke({
            "chat_history": history,
            "question": query
        }).strip()

    raw_results = []
    
    if not settings.tavily_api_key:
        print("\n[Web Search Notice] TAVILY_API_KEY not found in .env file. Skipping web search.")
        return {"web_results": [], "citations": []}

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query,
            search_depth="advanced",
            max_results=settings.web_search_max_results
        )
        raw_results = response.get("results", [])
    except Exception as e:
        print(f"\n[Web Search Notice] Fallback search bypassed or failed: {e}")
        raw_results = []

    results = [
        {"url": r.get("url", ""), "title": r.get("title", ""), "content": r.get("content", "")}
        for r in raw_results
    ]

    citations = [{"source_path": r["url"], "page_number": None} for r in results]

    return {"web_results": results, "citations": citations}
