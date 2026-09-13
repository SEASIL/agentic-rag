# 🤖 Agentic RAG Document Assistant

A full-stack, AI-powered document assistant that uses **Agentic Retrieval-Augmented Generation (RAG)** to intelligently answer questions by searching through your local documents or browsing the live web.

**[👉 Click here to view the Live Demo on Render!](https://agentic-rag-jtvr.onrender.com/)**

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white)
<br/>
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white) ![TailwindCSS](https://img.shields.io/badge/TailwindCSS-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black) ![Marked.js](https://img.shields.io/badge/Marked.js-000000?style=flat-square&logo=markdown&logoColor=white)
<br/>
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?style=flat-square) ![LangGraph](https://img.shields.io/badge/LangGraph-blue?style=flat-square) ![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat-square) ![Tavily Search](https://img.shields.io/badge/Tavily_Search-000000?style=flat-square)
<br/>
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat-square&logo=supabase&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL_15-4169E1?style=flat-square&logo=postgresql&logoColor=white) ![ChromaDB](https://img.shields.io/badge/ChromaDB-FC5200?style=flat-square) ![PDF](https://img.shields.io/badge/PDF-EC1C24?style=flat-square&logo=adobeacrobatreader&logoColor=white) ![CSV / Excel](https://img.shields.io/badge/CSV_/_Excel-1D6F42?style=flat-square&logo=microsoftexcel&logoColor=white)
<br/>
![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=flat-square&logo=google&logoColor=white) ![Groq](https://img.shields.io/badge/Groq-F37021?style=flat-square) ![OpenRouter](https://img.shields.io/badge/OpenRouter-000000?style=flat-square) ![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square) ![Cohere](https://img.shields.io/badge/Cohere-3B71CA?style=flat-square)
<br/>
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white) ![Ragas](https://img.shields.io/badge/Ragas-FF4B4B?style=flat-square)
<br/>
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) ![Render](https://img.shields.io/badge/Render-000000?style=flat-square&logo=render&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)

---

## ✨ Key Features

1. **User-Guided Search:** Choose whether to search your local vectorized documents or search the live internet (via Tavily).
2. **Contextual Answers:** Uses either local private data or real-time web data to provide grounded, accurate answers.
3. **Beautiful UI:** A custom-built, responsive chat interface featuring Markdown rendering, auto-scrolling, and inline source citations (pills).
4. **Source Citations:** Every answer includes exact references to the document (and page number) or the website it pulled the information from, completely eliminating hallucinations.
5. **Multi-Provider Fallback:** Intelligent LLM routing across OpenRouter, Gemini, Groq, and local Ollama to ensure zero downtime during rate limits or outages.
6. **Input Guardrails:** Robust adversarial input checking to prevent prompt injection and unauthorized usage.
7. **Comprehensive Benchmarking:** Automated scripts for measuring Ragas evaluation scores, end-to-end latency, and failover success rates.

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **LLM Gateway** | Google Gemini, Groq (`llama-3.1-8b-instant`), OpenRouter, Ollama (local) |
| **Embedding Model** | `BAAI/bge-m3` (dense, 1024-dim) |
| **Reranker** | `BAAI/bge-reranker-base` (cross-encoder) |
| **Vector Database** | ChromaDB (HNSW index) |
| **Web Search** | Tavily Advanced Search API |
| **Orchestration** | LangGraph & LangChain |
| **Backend** | FastAPI + Uvicorn, Docker, Render |
| **Frontend** | HTML5, JavaScript, TailwindCSS, Marked.js |
| **Evaluation** | Ragas (faithfulness, recall, precision, relevancy) + Pytest |

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/SEASIL/agentic-rag.git
cd agentic-rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

```env
# Required
GEMINI_API_KEY=your_google_gemini_key_here
TAVILY_API_KEY=your_tavily_search_key_here

# Optional — fallback LLM providers (recommended for reliability)
OPENROUTER_API_KEY=your_openrouter_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional — advanced reranking
COHERE_API_KEY=your_cohere_api_key_here

# Optional — fully local/offline LLM (no key needed, requires Ollama installed)
OLLAMA_BASE_URL=http://localhost:11434
```

> Free keys for all providers: [Gemini](https://aistudio.google.com/) · [Tavily](https://tavily.com/) · [Groq](https://console.groq.com/keys) · [OpenRouter](https://openrouter.ai/keys) · [Cohere](https://dashboard.cohere.com/api-keys)

### 4. Ingest Your Documents
Place any PDF, CSV, or XLSX files inside `data/raw/`, then run:
```bash
set PYTHONPATH=. && python scripts/ingest.py
```

### 5. Start the Server
```bash
uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000` in your browser.

---

## 🏗️ Architecture

The backend operates on a state machine powered by **LangGraph**. When a user submits a query, the application state (`GraphState`) flows through the following nodes:

1. **Input Guardrail:** Checks if the query is safe from prompt injections.
2. **Query Rewriter:** Optimizes the user's raw query into an effective search string.
3. **Retrieval Node:** Embeds the query using `BAAI/bge-m3` and performs semantic similarity search against ChromaDB, then reranks with `BAAI/bge-reranker-base`.
4. **Web Search Node:** Hits the Tavily API to gather live internet context (if requested or retrieval score falls below threshold).
5. **Synthesizer:** Takes the gathered context and synthesizes a final, formatted Markdown response with citations.

---

## 📊 Benchmarking & Evaluation

This project includes a full evaluation suite. Run any script from the project root:

```bash
# Ragas scores: faithfulness, context recall, precision, answer relevancy
python -m src.eval.ragas_pipeline

# End-to-end p50/p95/p99 query latencies + ChromaDB index stats
python -m scripts.benchmark_latency

# LLM failover reliability and routing switch times
python -m scripts.benchmark_failover

# Guardrail block rate vs. false positive rate
python -m scripts.benchmark_guardrails
```

> **Note on the LLM backend:** Benchmarks use whichever LLM is configured in `llm_gateway.py`.
> For fastest results, use Groq (`llama-3.1-8b-instant`) — free tier handles the full suite in ~10 minutes.
> For fully offline evaluation, start Ollama locally (`ollama run phi3`) and set `OLLAMA_BASE_URL` in your `.env`.

---
