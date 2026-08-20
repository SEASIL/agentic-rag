import uvicorn

if __name__ == "__main__":
    print("Starting Agentic RAG API Server...")
    # Run the FastAPI app defined in src/api/server.py
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)
