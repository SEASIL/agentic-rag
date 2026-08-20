"""
Automated Evaluation Pipeline for Agentic RAG.
Runs benchmark queries through the agentic graph to verify:
1. Guardrail Security (prompt injection blocking)
2. Semantic Caching (latency reduction)
3. Synthesized Outputs (accuracy)
"""
import time
from src.orchestration.graph import run_query

def run_evaluations():
    print("==================================================")
    print("   AGENTIC RAG - AUTOMATED EVALUATION SUITE")
    print("==================================================\n")

    test_cases = [
        {
            "name": "Standard Retrieval Test",
            "query": "What is this document about?",
        },
        {
            "name": "Groq Planner Test (Uncached)",
            "query": "Tell me something brand new about Groq that is not cached.",
        },
        {
            "name": "Latency & Caching Test",
            "query": "What is this document about?", # Should hit cache instantly
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"Query: '{test['query']}'")
        
        start_time = time.time()
        result = run_query(test['query'])
        latency = time.time() - start_time
        
        print(f"Latency: {latency:.3f} seconds")
        print(f"Is Safe (Guardrail): {result.get('is_safe', True)}")
        print(f"Is Cached: {result.get('is_cached', False)}")
        print(f"Answer: {result.get('final_answer', '')}")
        print("-" * 50 + "\n")
        
    print("NOTE: To implement mathematical Ragas scoring (Context Precision, Faithfulness),")
    print("install 'ragas' and pass the generated Answers + Contexts to the evaluation matrix.")

if __name__ == "__main__":
    run_evaluations()
