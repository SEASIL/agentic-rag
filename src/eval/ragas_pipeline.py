"""
Runs the full agentic RAG graph over a test set and scores it with Ragas on:
  - context_recall:    did retrieval surface the info needed to answer?
  - faithfulness:      is the answer actually grounded in retrieved context
                        (not hallucinated)?
  - answer_relevancy:  does the answer actually address the question asked?
  - context_precision: how much of what was retrieved was actually relevant
                        (catches a reranker that lets noise through even when
                        recall looks fine).

Used both for ad-hoc evaluation and as the CI gate (see scripts/run_eval_ci.py).
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

from configs.settings import settings
from src.orchestration.graph import run_query

# Ragas' metrics (faithfulness, answer_relevancy, etc.) use an LLM internally
# as a judge, and default to OpenAI if not told otherwise. Point it at the
# same local Ollama model, and use a local HuggingFace embedding model for
# any metrics that need embeddings — keeps the entire eval loop free/local.


def _get_ragas_llm() -> LangchainLLMWrapper:
    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0.0)
    return LangchainLLMWrapper(llm)


def _get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    embeddings = HuggingFaceEmbeddings(model_name=settings.dense_model_name)
    return LangchainEmbeddingsWrapper(embeddings)


def _load_testset(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_ragas_dataset(testset_path: str) -> Dataset:
    """Runs the full graph for every test question and assembles the
    question/answer/contexts/ground_truth format Ragas expects."""
    testset = _load_testset(testset_path)

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for record in testset:
        result = run_query(record["question"])

        contexts = [c.text for c in result["retrieved_chunks"]]
        contexts += [r["content"] for r in result["web_results"]]

        rows["question"].append(record["question"])
        rows["answer"].append(result["final_answer"] or "")
        rows["contexts"].append(contexts or [""])
        rows["ground_truth"].append(record["ground_truth"])

    return Dataset.from_dict(rows)


def run_evaluation(testset_path: str) -> dict:
    dataset = build_ragas_dataset(testset_path)

    result = evaluate(
        dataset,
        metrics=[context_recall, faithfulness, answer_relevancy, context_precision],
        llm=_get_ragas_llm(),
        embeddings=_get_ragas_embeddings(),
    )

    return result.to_pandas().mean(numeric_only=True).to_dict()


if __name__ == "__main__":
    import sys

    testset_path = sys.argv[1] if len(sys.argv) > 1 else "src/eval/golden_testset.jsonl"
    scores = run_evaluation(testset_path)

    print("\n=== Ragas Scores ===")
    for metric, score in scores.items():
        print(f"{metric}: {score:.3f}")
