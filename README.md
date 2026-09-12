# 🤖 Agentic RAG Document Assistant

A full-stack, AI-powered document assistant that uses **Agentic Retrieval-Augmented Generation (RAG)** to intelligently answer questions by searching through your local documents or browsing the live web.

**[👉 Click here to view the Live Demo on Render!](https://agentic-rag-jtvr.onrender.com/)**

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-blue?style=flat-square) ![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat-square)
<br/>
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white) ![TailwindCSS](https://img.shields.io/badge/TailwindCSS-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
<br/>
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat-square&logo=supabase&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL_15-4169E1?style=flat-square&logo=postgresql&logoColor=white) ![ChromaDB](https://img.shields.io/badge/ChromaDB-FC5200?style=flat-square)
<br/>
![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=flat-square&logo=google&logoColor=white) ![OpenRouter](https://img.shields.io/badge/OpenRouter-000000?style=flat-square)
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
5. **Multi-Provider Fallback:** Intelligent LLM routing powered by OpenRouter, Gemini, and OpenAI to ensure zero downtime during rate limits.
6. **Input Guardrails:** Robust adversarial input checking to prevent prompt injection and unauthorized usage.
7. **Comprehensive Benchmarking:** Automated scripts for measuring Ragas evaluation scores, end-to-end latency, and failover success rates.

## 🛠️ Tech Stack

* **Large Language Model (LLM):** Google Gemini, OpenRouter, and OpenAI Fallbacks
* **Embedding Model:** Fast Embeddings
* **Vector Database:** ChromaDB
* **Web Search Engine:** Tavily Advanced Search API
* **Orchestration Framework:** LangGraph & LangChain
* **Backend & Hosting:** FastAPI (Python), Docker, Render
* **Frontend:** HTML5, JavaScript, TailwindCSS, Marked.js
* **Evaluation Framework:** Ragas & Pytest

---

## 🚀 How to Run Locally

If you want to run this project on your own machine, follow these steps:

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
Create a `.env` file in the root directory and add your free API keys:
```env
GEMINI_API_KEY=your_google_gemini_key_here
TAVILY_API_KEY=your_tavily_search_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. Ingest Sample Data
Place any PDF, CSV, or XLSX files you want the AI to read inside the `data/raw/` folder, then run the ingestion script to build the ChromaDB vector database:
```bash
set PYTHONPATH=. && python scripts/ingest.py
```

### 5. Start the Server
```bash
uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```
Then, open your browser and go to `http://127.0.0.1:8000` to interact with the agent!

---

## 🏗️ Architecture

The backend operates on a state machine powered by **LangGraph**. When a user submits a query, the application state (`GraphState`) flows through the following nodes:

1. **Input Guardrail:** Checks if the query is safe from prompt injections.
2. **Query Rewriter:** Optimizes the user's raw query into an optimized search string for the vector database and search engine.
3. **Retrieval Node:** Embeds the query and performs a semantic similarity search against the ChromaDB database.
4. **Web Search Node:** Hits the Tavily API to gather live internet context (if requested).
5. **Synthesizer:** Takes the gathered context and synthesizes a final, formatted Markdown response with citations.

---

## 📊 Benchmarking & Evaluation

This project includes a robust evaluation suite:
- `scripts/benchmark_latency.py`: Measures end-to-end p50/p95/p99 query latencies.
- `scripts/benchmark_failover.py`: Verifies LLM failover reliability and routing speeds.
- `scripts/benchmark_guardrails.py`: Tests the guardrail agent against a mix of legitimate and adversarial queries.
- `src/eval/ragas_pipeline.py`: Uses the **Ragas** framework to measure Context Precision, Answer Relevancy, and Faithfulness.

Run any of these directly to reproduce performance claims!

---

