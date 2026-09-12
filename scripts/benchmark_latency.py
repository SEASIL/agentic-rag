import os
import time
import json
import statistics
from chromadb import HttpClient
from configs.settings import settings
from src.orchestration.graph import run_query

def get_chroma_stats():
    print("=== ChromaDB Stats ===")
    try:
        # Based on settings, it's typically local or hitting a service.
        # Here we'll try initializing a simple persistent client if we are using local,
        # or just try HttpClient assuming it runs on localhost:8000 (typical for chromadb server).
        # Wait, the project doesn't have an explicit chromadb config URL in settings, it just uses a local path probably.
        import chromadb
        client = chromadb.PersistentClient(path="data/chroma_db")
        collection = client.get_collection(settings.chroma_collection)
        count = collection.count()
        # To get dimensionality we can get one embedding
        peek = collection.peek(1)
        dim = len(peek['embeddings'][0]) if peek['embeddings'] else "N/A"
        print(f"Chunks indexed: {count}")
        print(f"Embedding dimensionality: {dim}")
        print(f"Index type: HNSW (default Chroma)")
    except Exception as e:
        print(f"Could not connect to ChromaDB: {e}")

def load_queries():
    queries = []
    with open("src/eval/golden_testset.jsonl", "r") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line)["question"])
    return queries

def benchmark_latency(queries, mode):
    print(f"\n=== End-to-End Latency Benchmark: {mode.upper()} ===")
    latencies = []
    for i, q in enumerate(queries[:10]):  # Run 10 queries for time
        start = time.time()
        try:
            run_query(q, search_mode=mode)
            elapsed = time.time() - start
            latencies.append(elapsed)
            print(f"Query {i+1}: {elapsed:.2f}s")
        except Exception as e:
            print(f"Query {i+1}: FAILED ({e})")
            
    if not latencies:
        return
        
    avg = statistics.mean(latencies)
    
    # Calculate p50, p95, p99 if we have enough
    latencies.sort()
    
    def percentile(data, p):
        k = (len(data) - 1) * p
        f = int(k)
        c = int(k) + 1 if (k % 1) != 0 else f
        if f == c:
            return data[int(k)]
        d0 = data[f]
        d1 = data[c]
        return d0 + (d1 - d0) * (k - f)

    p50 = percentile(latencies, 0.5)
    p95 = percentile(latencies, 0.95)
    p99 = percentile(latencies, 0.99)
    
    print(f"Average: {avg:.2f}s")
    print(f"p50: {p50:.2f}s")
    print(f"p95: {p95:.2f}s")
    print(f"p99: {p99:.2f}s")

if __name__ == "__main__":
    get_chroma_stats()
    queries = load_queries()
    benchmark_latency(queries, mode="local")
    benchmark_latency(queries, mode="web")
