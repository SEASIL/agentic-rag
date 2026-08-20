from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import the existing run_query function from your orchestrator
from src.orchestration.graph import run_query

app = FastAPI(title="Agentic RAG API")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Allow CORS so external clients can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the raw index.html website
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = []
    search_mode: Optional[str] = "auto"

class Citation(BaseModel):
    source_path: str
    page_number: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]

import re

def is_greeting(text: str) -> bool:
    # Remove punctuation and check if the entire message is just a greeting
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text).strip().lower()
    greetings = {"hello", "hi", "hey", "hyy", "hy", "hola", "greetings", "good morning", "good evening", "good afternoon", "sup", "yo"}
    return cleaned in greetings

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        # Fast-path for simple greetings to save time and API costs
        if is_greeting(request.query):
            return ChatResponse(
                answer="Hello! I am your intelligent Web Agent. What can I help you research today?",
                citations=[]
            )
            
        # Convert chat_history to dicts
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
            
        # If it's not a simple greeting, run the full Agentic RAG pipeline
        result = run_query(request.query, chat_history=history_dicts, search_mode=request.search_mode)
        
        # Parse the citations to ensure they match our Pydantic model
        parsed_citations = []
        for c in result.get("citations", []):
            parsed_citations.append(Citation(
                source_path=str(c.get("source_path", "")),
                page_number=c.get("page_number")
            ))
            
        return ChatResponse(
            answer=str(result.get("final_answer", "Sorry, I could not generate an answer.")),
            citations=parsed_citations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
