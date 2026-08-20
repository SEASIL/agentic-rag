"""
Planner agent. Two jobs:

1. On the first hop: decompose the original query into sub-questions if it
   requires multi-hop reasoning (e.g. "compare X's Q3 revenue to Y's Q3
   revenue" -> two sub-questions). Simple questions get a single sub-question
   equal to the original query.

2. On every hop after retrieval: look at what's been answered so far and
   decide whether to (a) move to the next unanswered sub-question, or
   (b) stop and hand off to the synthesizer.

Design note: routing to web search is a HARD RULE based on reranker
confidence (see reranker.needs_web_fallback), not an LLM decision — this
keeps that failure mode debuggable and testable rather than depending on
the LLM's judgment call every time.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from src.orchestration.llm_gateway import get_llm
from langchain_core.output_parsers import JsonOutputParser

from configs.settings import settings
from src.orchestration.state import GraphState, SubQuestion

DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query planning agent for a document QA system. "
            "Decide if the user's question requires multiple retrieval steps "
            "(multi-hop) to answer fully — e.g. comparisons, questions with "
            "multiple named entities, or questions requiring facts to be "
            "combined. If it's a single-fact lookup, return one sub-question "
            "identical to the original.\n\n"
            'Respond ONLY with JSON: {{"sub_questions": ["...", "..."]}}',
        ),
        ("human", "{query}"),
    ]
)


from pydantic import BaseModel, Field

class PlanOutput(BaseModel):
    sub_questions: list[str] = Field(description="List of sub-questions required to fully answer the original query.")

def plan_node(state: GraphState) -> dict:
    """First-hop entry: decompose the query into sub-questions."""
    llm = get_llm(require_advanced=True)
    chain = DECOMPOSITION_PROMPT | llm.with_structured_output(PlanOutput)
    
    # with_structured_output returns the Pydantic object directly
    result = chain.invoke({"query": state["original_query"]})

    sub_questions: list[SubQuestion] = [
        {"question": q, "answered": False, "answer": None} for q in result.sub_questions
    ]

    first_question = sub_questions[0]["question"]

    return {
        "sub_questions": sub_questions,
        "current_query": first_question,
        "hop_count": 0,
    }


def route_next_hop(state: GraphState) -> str:
    """Conditional edge: decides whether to keep hopping or move to synthesis.
    Returns the name of the next node."""
    unanswered = [sq for sq in state["sub_questions"] if not sq["answered"]]

    if not unanswered or state["hop_count"] >= settings.max_planner_hops:
        return "synthesize"

    return "rewrite_query"


def advance_hop_node(state: GraphState) -> dict:
    """Marks the current sub-question answered and points current_query at
    the next unanswered one, if any."""
    updated = list(state["sub_questions"])
    for sq in updated:
        if sq["question"] == state["current_query"]:
            sq["answered"] = True
            # naive answer capture — in production, pull this from the
            # synthesizer's per-hop output rather than raw chunk text
            sq["answer"] = "; ".join(c.text[:200] for c in state["retrieved_chunks"][-3:])

    remaining = [sq for sq in updated if not sq["answered"]]
    next_query = remaining[0]["question"] if remaining else state["current_query"]

    return {
        "sub_questions": updated,
        "current_query": next_query,
        "hop_count": state["hop_count"] + 1,
    }
