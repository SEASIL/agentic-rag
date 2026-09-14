import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from src.orchestration.llm_gateway import get_available_llms, get_llm

load_dotenv()


def benchmark_providers():
    print("=== Per-Provider Latency Benchmark ===")
    available = get_available_llms()

    if not available:
        print("No online API keys found in .env. Using local Ollama fallback.")
        return {}

    results = {}
    for llm in available:
        provider_name = llm.__class__.__name__
        model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
        label = f"{provider_name} ({model_name})"
        try:
            start_time = time.time()
            res = llm.invoke([HumanMessage(content="Reply with exactly one word: 'Hello'.")])
            latency = time.time() - start_time
            results[label] = latency
            print(f"[OK] {label}: {latency:.2f}s")
        except Exception as e:
            print(f"[FAIL] {label}: FAILED ({e})")

    return results


def test_gateway_chain():
    print("\n=== Multi-Provider Gateway Chain Benchmark ===")
    gateway_llm = get_llm()
    print(f"Active Gateway Configuration: {gateway_llm}")

    start_time = time.time()
    try:
        res = gateway_llm.invoke([HumanMessage(content="Reply with 'Gateway active'.")])
        latency = time.time() - start_time
        print(f"[OK] Gateway invocation succeeded in {latency:.2f}s | Response: {res.content.strip()}")
    except Exception as e:
        print(f"[FAIL] Gateway invocation failed: {e}")


if __name__ == "__main__":
    benchmark_providers()
    test_gateway_chain()

