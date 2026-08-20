"""
Query rewriter. Sub-questions from the planner are often phrased naturally
("what did they say about it") and lose antecedents/context that matter for
retrieval. This node rewrites current_query into a self-contained, retrieval-
optimized query before hitting the retriever — e.g. resolving pronouns using
the original query and prior answered sub-questions.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.orchestration.state import GraphState
from src.orchestration.llm_gateway import get_llm

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the given sub-question into a standalone search query "
            "optimized for semantic retrieval. Resolve any "
            "pronouns or references using the original question and prior "
            "answers as context. Keep it concise — a search query, not a "
            "sentence. Output ONLY the rewritten query, nothing else.",
        ),
        (
            "human",
            "Original question: {original_query}\n"
            "Prior answered sub-questions: {prior_answers}\n"
            "Current sub-question to rewrite: {current_query}",
        ),
    ]
)

def rewrite_query_node(state: GraphState) -> dict:
    prior_answers = [
        f"Q: {sq['question']} A: {sq['answer']}" for sq in state["sub_questions"] if sq["answered"]
    ]

    llm = get_llm(require_advanced=False)
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    rewritten = chain.invoke(
        {
            "original_query": state["original_query"],
            "prior_answers": "\n".join(prior_answers) or "none yet",
            "current_query": state["current_query"],
        }
    )

    return {"rewritten_query": rewritten.strip()}
