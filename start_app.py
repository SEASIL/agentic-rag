import subprocess
import sys
import os
import time

def main():
    print("========================================")
    print("Starting Agentic RAG Web Interface...")
    print("========================================")
    
    # 1. Start the FastAPI Backend
    print("-> Starting FastAPI Backend on http://127.0.0.1:8000")
    backend = subprocess.Popen([sys.executable, "main.py"])
    
    print("\n[INFO] Server is starting up! Press Ctrl+C in this terminal to stop.")
    
    try:
        # Keep the main process alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server...")
        backend.terminate()
        backend.wait()
        print("[INFO] Server stopped safely.")

if __name__ == "__main__":
    main()
