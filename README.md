# 🤖 Agentic RAG Document Assistant

A full-stack, AI-powered document assistant that uses **Agentic Retrieval-Augmented Generation (RAG)** to intelligently answer questions by searching through your local documents or browsing the live web.

**[👉 Click here to view the Live Demo on Render!](https://your-app-name.onrender.com)**

![Clean UI with Markdown and Citations](https://img.shields.io/badge/UI-TailwindCSS-38B2AC?style=flat-square&logo=tailwind-css)
![FastAPI Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)
![LangGraph Orchestration](https://img.shields.io/badge/Orchestration-LangGraph-blue?style=flat-square)

---

## ✨ Key Features

1. **Smart Agentic Routing:** The system uses an LLM-driven decision engine (LangGraph) to decide whether to search your local vectorized documents, search the live internet (via Tavily), or synthesize both.
2. **Hybrid Context:** Combines local private data with real-time web data to provide grounded, accurate answers.
3. **Beautiful UI:** A custom-built, responsive chat interface featuring Markdown rendering, auto-scrolling, and inline source citations (pills).
4. **Source Citations:** Every answer includes exact references to the document (and page number) or the website it pulled the information from, completely eliminating hallucinations.

## 🛠️ Tech Stack

* **Large Language Model (LLM):** Google Gemini (gemini-3.5-flash)
* **Embedding Model:** BAAI/bge-m3 (1024 dimensions)
* **Vector Database:** ChromaDB (Local SQLite-based persistent storage)
* **Web Search Engine:** Tavily Advanced Search API
* **Orchestration Framework:** LangGraph & LangChain
* **Backend:** FastAPI (Python)
* **Frontend:** HTML5, JavaScript, TailwindCSS, Marked.js

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
```

### 4. Ingest Sample Data
Place any PDF, CSV, or XLSX files you want the AI to read inside the `data/raw/` folder, then run the ingestion script to build the vector database:
```bash
python scripts/ingest.py
```

### 5. Start the Server
```bash
uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```
Then, open your browser and go to `http://127.0.0.1:8000` to interact with the agent!

---

## 🏗️ Architecture

The backend operates on a state machine powered by **LangGraph**. When a user submits a query, the application state (`GraphState`) flows through the following nodes:

1. **Input Guardrail:** Checks if the query is safe and determines the user's selected search mode (Auto, Local, or Web).
2. **Query Rewriter:** Optimizes the user's raw query into an optimized search string for the vector database and search engine.
3. **Retrieval Node:** Embeds the query and performs a similarity search against the local ChromaDB.
4. **Web Search Node:** Hits the Tavily API to gather live internet context (if requested).
5. **Synthesizer:** Takes the gathered context (local + web) and synthesizes a final, formatted Markdown response with citations.

---
*Developed for resume demonstration purposes.*
