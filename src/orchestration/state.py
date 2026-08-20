"""
Shared state passed between nodes in the LangGraph orchestration graph.
Every agent reads/writes this single object — this is how a multi-hop
question accumulates evidence across planner loop-backs.
"""
from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict
import operator

from src.retrieval.hybrid_search import RetrievedChunk


class SubQuestion(TypedDict):
    question: str
    answered: bool
    answer: Optional[str]


class GraphState(TypedDict):
    # Original user input, never mutated
    original_query: str
    
    # Search mode override: "auto", "local", or "web"
    search_mode: str
    
    # Previous messages in the conversation
    chat_history: list[dict]

    # Planner's decomposition of the query into sub-questions (multi-hop)
    sub_questions: list[SubQuestion]

    # The question currently being worked on this hop
    current_query: str

    # Rewritten version of current_query, optimized for retrieval
    rewritten_query: Optional[str]

    # Accumulates retrieved chunks across ALL hops (operator.add = append-only reducer)
    retrieved_chunks: Annotated[list[RetrievedChunk], operator.add]

    # Accumulates web search results across hops, same append-only pattern
    web_results: Annotated[list[dict], operator.add]

    # Whether the last retrieval pass was confident enough (see reranker.needs_web_fallback)
    should_use_web: bool

    # Hop counter — the planner stops looping once max_planner_hops is hit
    hop_count: int

    # Final synthesized answer, populated only at the end
    final_answer: Optional[str]

    # Guardrail status: Did this query pass security checks?
    is_safe: bool

    # Citations collected for the final answer: [{source_path, page_number}, ...]
    citations: Annotated[list[dict], operator.add]
