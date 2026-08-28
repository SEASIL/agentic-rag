"""
Thin LangGraph node wrapping src/retrieval/retriever.py. Kept separate from
the retrieval package itself so retrieval logic stays framework-agnostic and
testable outside of LangGraph.
"""
from __future__ import annotations

from src.orchestration.state import GraphState
from src.retrieval.retriever import retrieve


def retrieve_node(state: GraphState) -> dict:
    query = state.get("rewritten_query") or state["current_query"]
    search_mode = state.get("search_mode", "auto")
    result = retrieve(query, search_mode=search_mode)

    return {
        "retrieved_chunks": result.chunks,  # appended via operator.add reducer
        "should_use_web": result.should_fallback_to_web,
        "citations": [
            {"source_path": c.source_path, "page_number": c.page_number} for c in result.chunks
        ],
    }


def route_after_retrieval(state: GraphState) -> str:
    """Conditional edge: low-confidence retrieval -> web search fallback,
    otherwise go straight to deciding on the next hop."""
    return "web_search" if state["should_use_web"] else "advance_hop"
