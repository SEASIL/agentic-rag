# Hybrid Agentic RAG System

An enterprise document assistant for multi-hop reasoning across heterogeneous
documents (PDFs, tables, live web). Hybrid dense+lexical retrieval, an
agentic LangGraph orchestration layer, and a Ragas-based CI evaluation gate.

**100% free and open-source stack — no API keys, no paid services required:**

| Component | Tool | Cost |
|---|---|---|
| Vector DB | Qdrant (self-hosted) | Free |
| Dense embeddings | BGE (sentence-transformers) | Free, local |
| Sparse/BM25 embeddings | fastembed | Free, local |
| Reranker | BGE cross-encoder | Free, local |
| LLM (planner/rewriter/synthesizer) | **Ollama** (Llama 3.1, local) | Free, local |
| Web search fallback | **DuckDuckGo search** | Free, no key |
| Eval judge (Ragas) | Same local Ollama model | Free, local |

Everything runs on your own machine (or CI runner) — nothing calls out to a
paid API.

## Architecture

```
                    ┌─────────────┐
   query ──────────▶│   planner   │  decomposes query into sub-questions
                    └──────┬──────┘  (multi-hop decomposition)
                           │
                    ┌──────▼──────┐
              ┌────▶│query_rewriter│ resolves references, optimizes for search
              │     └──────┬──────┘
              │            │
              │     ┌──────▼──────┐
              │     │  retrieve   │  hybrid search (dense+BM25, RRF fusion)
              │     │             │  + cross-encoder rerank
              │     └──────┬──────┘
              │            │
              │      low confidence?
              │       ┌────┴────┐
              │       │yes      │no
              │  ┌────▼───┐     │
              │  │web_search│    │
              │  └────┬───┘     │
              │       └────┬────┘
              │       ┌────▼─────┐
              └───────┤advance_hop│  more sub-questions? loop back : exit
                       └────┬─────┘
                            │
                     ┌──────▼──────┐
                     │ synthesize  │  grounded final answer + citations
                     └─────────────┘
```

### Retrieval (`src/retrieval/`)
- **`embeddings.py`** — dense (BGE via sentence-transformers) + sparse (BM25 via fastembed)
- **`vector_store.py`** — Qdrant collection with named dense+sparse vectors on each point
- **`hybrid_search.py`** — runs both searches, fuses with **Reciprocal Rank Fusion**
  (chosen over score-weighting because dense/BM25 scores aren't on comparable scales)
- **`reranker.py`** — BGE cross-encoder re-scores the fused candidates; also gates
  the web-search fallback via a **hard confidence threshold**, not an LLM decision,
  so that failure mode stays debuggable

### Ingestion (`src/ingestion/`)
- Format-aware: PDFs get page-tracked prose chunking; tables become **one
  chunk per table** (or per N rows for large tables) using a pluggable
  strategy (`structured_json` / `markdown` / `summarize`) — see
  `table_loader.py` docstring for tradeoffs.

### Orchestration (`src/orchestration/`)
LangGraph state machine: `plan → rewrite_query → retrieve → (web_search?) →
advance_hop → (loop | synthesize)`. Multi-hop is modeled as **loop-backs to
rewrite_query**, one sub-question per hop, capped by `max_planner_hops`.

### Evaluation (`src/eval/`)
- `synthetic_testset.py` — LLM-generated QA pairs from the corpus (fast, but
  systematically easier than real queries — treat as regression detection)
- `golden_testset.jsonl` — **populate this with real, human-verified
  question/answer pairs** before trusting CI gates
- `ragas_pipeline.py` — runs the full graph end-to-end per test question,
  scores with Ragas: `context_recall`, `faithfulness`, `answer_relevancy`,
  `context_precision`
- `scripts/run_eval_ci.py` — CI gate; supports both absolute thresholds
  (`configs/settings.py: ragas_thresholds`) and regression-only mode
  (`--baseline baseline_scores.json`)

## Setup

```bash
cp .env.example .env               # defaults already work — nothing to fill in
pip install -r requirements.txt
docker compose up -d               # starts Qdrant (6333) + Ollama (11434)

# Pull the local LLM (one-time, ~4.7GB for llama3.1:8b)
docker compose exec ollama ollama pull llama3.1
```

**Hardware note:** Llama 3.1 8B runs fine on CPU but is noticeably faster with
a GPU. For lighter hardware, swap `LLM_MODEL` in `.env` to a smaller model
like `llama3.2` (3B) or `qwen2.5:3b` — pull it the same way, then update
`configs/settings.py: llm_model` or the env var to match.

## Usage

```bash
# 1. Ingest documents
python scripts/ingest.py --path data/raw

# 2. Query the full agentic pipeline
python -m src.orchestration.graph "How did revenue change between the two reports?"

# 3. Generate a synthetic eval set (optional, for quick iteration)
python -m src.eval.synthetic_testset

# 4. Run the Ragas evaluation gate
python scripts/run_eval_ci.py --testset src/eval/golden_testset.jsonl
```

## Tests

```bash
pytest tests/ -v
```

Pure-logic tests (RRF fusion, chunking, table strategies) run without any
external services. Integration tests that touch Qdrant/LLMs require the
stack above running and are separated so CI can gate on the fast tests
independently of the slower Ragas eval job.

## Status / what's stubbed vs. functional

| Component | Status |
|---|---|
| Ingestion (PDF, table, chunking) | Functional |
| Hybrid search + RRF fusion | Functional |
| Cross-encoder reranking + web-fallback gate | Functional |
| LangGraph orchestration (planner/rewriter/synthesizer) | Wired end-to-end, uses local `llama3.1` via Ollama by default |
| Web search fallback | Functional, DuckDuckGo — no key needed |
| Ragas eval pipeline + CI gate | Functional, needs a populated `golden_testset.jsonl` |

## Next steps
- Populate `src/eval/golden_testset.jsonl` with 20-50 real, hard examples
  (multi-hop, table lookups, ambiguous phrasing) before trusting CI thresholds.
- Swap the naive word-based chunker (`chunker.py: _split_text`) for a
  token-aware splitter if context-window budgeting gets tight.
- Consider adding a `text-to-SQL`-style path for very large/structured tables
  instead of row-chunking, if `structured_json` chunks start blowing context.
