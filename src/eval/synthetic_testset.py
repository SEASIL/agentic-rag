"""
Generates a synthetic QA test set from the ingested corpus for Ragas
evaluation. NOTE: synthetic questions tend to be phrased close to the source
text and are systematically easier than real user queries — treat this as a
regression-detection baseline, not a substitute for a human-curated eval set
(see eval/golden_testset.jsonl, which you should populate with real examples).
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from configs.settings import settings
from src.retrieval.vector_store import get_client
from qdrant_client import models

QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a passage from a document, write one question that can be "
            "answered using ONLY this passage, plus the ground-truth answer. "
            "Vary question style (factual, comparative, multi-hop-style) "
            'across calls. Respond ONLY with JSON: {{"question": "...", "ground_truth": "..."}}',
        ),
        ("human", "Passage:\n{passage}"),
    ]
)


def _sample_chunks(n: int) -> list[dict]:
    client = get_client()
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=n,
        with_payload=True,
        with_vectors=False,
    )
    return [p.payload for p in points]


def generate_testset(n: int = 20, output_path: str = "src/eval/synthetic_testset.jsonl") -> str:
    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
        format="json",
    )
    chain = QUESTION_GEN_PROMPT | llm | JsonOutputParser()

    chunks = _sample_chunks(n)
    records = []
    for chunk in chunks:
        result = chain.invoke({"passage": chunk["text"]})
        records.append(
            {
                "question": result["question"],
                "ground_truth": result["ground_truth"],
                "source_passage": chunk["text"],
                "source_path": chunk["source_path"],
            }
        )

    out = Path(output_path)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return str(out)


if __name__ == "__main__":
    path = generate_testset()
    print(f"Wrote synthetic test set to {path}")
