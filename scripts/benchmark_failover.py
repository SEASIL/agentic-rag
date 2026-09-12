import os
import time
from langchain_core.messages import HumanMessage
from src.orchestration.llm_gateway import get_llm
from configs.settings import settings

def benchmark_providers():
    print("=== Per-Provider Latency Benchmark ===")
    llms_to_test = []
    
    # Instantiate them individually based on available keys
    if settings.groq_api_key:
        from langchain_groq import ChatGroq
        llms_to_test.append(("Groq", ChatGroq(api_key=settings.groq_api_key, model="llama-3.1-8b-instant", temperature=0.0)))
        
    if settings.gemini_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llms_to_test.append(("Gemini", ChatGoogleGenerativeAI(google_api_key=settings.gemini_api_key, model=settings.fast_llm_model, temperature=0.0)))
        
    if settings.openrouter_api_key:
        from langchain_openai import ChatOpenAI
        llms_to_test.append(("OpenRouter", ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key, model=settings.openrouter_model, temperature=0.0)))
        
    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI
        llms_to_test.append(("OpenAI", ChatOpenAI(api_key=settings.openai_api_key, model="gpt-4o-mini", temperature=0.0)))
        
    results = {}
    for name, llm in llms_to_test:
        try:
            start_time = time.time()
            llm.invoke([HumanMessage(content="Reply with exactly one word: 'Hello'.")])
            latency = time.time() - start_time
            results[name] = latency
            print(f"{name}: {latency:.2f}s")
        except Exception as e:
            print(f"{name}: FAILED ({e})")
            
    return results

def test_failover_mechanism():
    print("\n=== Failover Mechanism Benchmark ===")
    
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    
    # We will use an invalid API key for the primary to force a fallback
    primary_failing_llm = ChatGroq(model="llama-3.1-8b-instant", api_key="invalid_key", max_retries=0)
    fallback_llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key, model=settings.openrouter_model, max_retries=1)
    
    llm_with_fallbacks = primary_failing_llm.with_fallbacks([fallback_llm])
    
    success_count = 0
    total_time = 0
    num_tests = 5
    
    for i in range(num_tests):
        start_time = time.time()
        try:
            res = llm_with_fallbacks.invoke([HumanMessage(content="Say 'Failover success'")])
            success_count += 1
        except Exception as e:
            print(f"Test {i+1} failed completely: {e}")
        finally:
            total_time += (time.time() - start_time)
            
    success_rate = (success_count / num_tests) * 100
    avg_latency = total_time / num_tests
    print(f"Handled {num_tests} simulated provider outages with {success_rate}% failover success, avg failover switch time of {avg_latency:.2f}s")

if __name__ == "__main__":
    benchmark_providers()
    test_failover_mechanism()
