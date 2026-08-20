"""
Assembles the full agentic RAG graph:

    plan -> rewrite_query -> retrieve -> [web_search?] -> advance_hop --\\
                ^                                                        |
                |________________ loop until sub-questions done ________|
                                                                          v
                                                                    synthesize -> END

- plan: decomposes the query into sub-questions once, at the start.
- rewrite_query -> retrieve: standard per-hop retrieval.
- retrieve routes conditionally to web_search when reranker confidence is
  low (hard rule, see reranker.needs_web_fallback), otherwise straight to
  advance_hop.
- advance_hop marks the current sub-question done and either loops back to
  rewrite_query (more sub-questions left) or exits to synthesize, via the
  route_next_hop conditional edge.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from src.orchestration.state import GraphState
from src.orchestration.agents.web_search import web_search_node
from src.orchestration.agents.synthesizer import synthesize_node


from src.orchestration.agents.guardrails import input_guardrail_node, output_guardrail_node

from src.orchestration.agents.retrieval_node import retrieve_node, route_after_retrieval

def route_after_input_guardrail(state: GraphState) -> str:
    if not state.get("is_safe", True):
        return END
    if state.get("search_mode") == "web":
        return "web_search"
    return "retrieve"

def route_after_retrieve(state: GraphState) -> str:
    if state.get("search_mode") == "local":
        return "synthesize"
    return "web_search" if state.get("should_use_web") else "synthesize"

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("output_guardrail", output_guardrail_node)

    graph.set_entry_point("input_guardrail")

    graph.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {"retrieve": "retrieve", "web_search": "web_search", END: END}
    )
    
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"web_search": "web_search", "synthesize": "synthesize"}
    )
    
    graph.add_edge("web_search", "synthesize")
    graph.add_edge("synthesize", "output_guardrail")
    graph.add_edge("output_guardrail", END)

    return graph.compile()


def run_query(query: str, chat_history: list[dict] = None, search_mode: str = "auto") -> dict:
    """Convenience entry point: runs the full graph for a single question
    and returns the final state (answer + citations + full trace)."""
        
    app = build_graph()
    initial_state: GraphState = {
        "original_query": query,
        "search_mode": search_mode,
        "chat_history": chat_history or [],
        "sub_questions": [],
        "current_query": query,
        "rewritten_query": None,
        "retrieved_chunks": [],
        "web_results": [],
        "should_use_web": False,
        "hop_count": 0,
        "final_answer": None,
        "citations": [],
        "is_safe": True,
    }
    
    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What were the key findings in the latest report?"
    result = run_query(question)
    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])
    print("\n=== CITATIONS ===")
    for c in result["citations"]:
        print(c)
