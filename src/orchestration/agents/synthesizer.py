"""
Synthesizer. Runs once, at the end of the graph, after all hops are done.
Combines every retrieved chunk + web result gathered across the whole
multi-hop trace into one grounded final answer. Explicitly instructed to
only use provided context (this is what Ragas' Faithfulness metric checks).
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.orchestration.llm_gateway import get_llm
from langchain_core.output_parsers import StrOutputParser

from configs.settings import settings
from src.orchestration.state import GraphState

def get_synthesis_prompt(search_mode: str) -> ChatPromptTemplate:
    if search_mode == "local":
        system_instructions = (
            "You are an expert assistant answering the user's question. "
            "Answer the question using ONLY the provided Document Context. "
            "If the document context does not contain the answer, explicitly state 'No relevant information found in the local documents.'\n\n"
            "FORMATTING RULES (you MUST follow these strictly):\n"
            "- Structure your ENTIRE answer as a bullet point list using markdown (- or *). "
            "Every sentence must be a separate bullet point. Never write paragraphs.\n"
            "- Do NOT include any citations, references, source tags, numbered references like [1] or [2], URLs, or links anywhere in the answer.\n"
            "- Do not include any structural headers. Just provide the bullet points directly."
        )
    elif search_mode in ["web", "web_search_only"]:
        system_instructions = (
            "You are an expert assistant answering the user's question. "
            "Answer the question using ONLY the provided Web Search Context. "
            "If no Web Search Context was provided, explicitly state 'No web search was performed for this query.'\n\n"
            "FORMATTING RULES (you MUST follow these strictly):\n"
            "- Structure your ENTIRE answer as a bullet point list using markdown (- or *). "
            "Every sentence must be a separate bullet point. Never write paragraphs.\n"
            "- Do NOT include any citations, references, source tags, numbered references like [1] or [2], URLs, or links anywhere in the answer.\n"
            "- Do not include any structural headers. Just provide the bullet points directly."
        )
    else: # auto
        system_instructions = (
            "You are an expert assistant answering the user's question. "
            "Answer the question by synthesizing information from the provided Document Context and Web Search Context. "
            "If the information is not found in either, explicitly state that.\n\n"
            "FORMATTING RULES (you MUST follow these strictly):\n"
            "- Structure your ENTIRE answer as a bullet point list using markdown (- or *). "
            "Every sentence must be a separate bullet point. Never write paragraphs.\n"
            "- Do NOT include any citations, references, source tags, numbered references like [1] or [2], URLs, or links anywhere in the answer.\n"
            "- Do not include any structural headers. Just provide the bullet points directly."
        )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_instructions),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Question: {question}\n\n"
                "Document Context:\n{doc_context}\n\n"
                "Web Search Context:\n{web_context}\n\n"
                "Answer:",
            ),
        ]
    )


def _format_doc_context(state: GraphState, max_chars: int = 20000) -> str:
    # 1. Deduplicate chunks (multiple hops might retrieve the same chunk)
    seen_ids = set()
    unique_chunks = []
    for c in state["retrieved_chunks"]:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            unique_chunks.append(c)
            
    # 2. Sort by rerank score (highest first) so most relevant context is kept if truncated
    unique_chunks.sort(key=lambda x: x.rerank_score or -1.0, reverse=True)
    
    # 3. Token Optimization: truncate if it exceeds the maximum context window
    lines = []
    current_chars = 0
    for c in unique_chunks:
        loc = f"{c.source_path}" + (f" p.{c.page_number}" if c.page_number else "")
        line_text = f"[{loc}] {c.text}"
        
        if current_chars + len(line_text) > max_chars:
            lines.append(f"...[TRUNCATED to preserve token limit]")
            break
            
        lines.append(line_text)
        current_chars += len(line_text)
        
    return "\n\n".join(lines) or "(no document context retrieved)"


def _format_web_context(state: GraphState) -> str:
    lines = [f"[{r['url']}] {r['content']}" for r in state["web_results"]]
    return "\n\n".join(lines) or "(no web search performed)"


def synthesize_node(state: GraphState) -> dict:
    import re

    llm = get_llm(require_advanced=True)
    prompt = get_synthesis_prompt(state.get("search_mode", "auto"))
    chain = prompt | llm | StrOutputParser()

    from langchain_core.messages import HumanMessage, AIMessage
    history = []
    for msg in state.get("chat_history", []):
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    answer = chain.invoke(
        {
            "question": state["original_query"],
            "doc_context": _format_doc_context(state),
            "web_context": _format_web_context(state),
            "chat_history": history,
        }
    )

    # Strip all citation patterns the LLM might produce:
    # 1. Markdown links in brackets: [[url](url), [url](url)]
    # 2. Simple bracketed URLs: [https://example.com]
    # 3. Simple source tags: [source]
    # 4. Numbered references: [1], [2], [1][3]
    answer = re.sub(r'\s*\[?\[https?://[^\]]*\]\([^\)]*\)[,\s]*\]?', '', answer)
    answer = re.sub(r'\s*\[https?://[^\]]+\]', '', answer)
    answer = re.sub(r'\s*\[[^\[\]]*source[^\[\]]*\]\.?', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\s*(\[\d+\])+', '', answer)
    # Clean up any leftover empty brackets or double spaces
    answer = re.sub(r'\s*\[\s*[,\s]*\]', '', answer)
    answer = re.sub(r'  +', ' ', answer).strip()

    return {"final_answer": answer}
