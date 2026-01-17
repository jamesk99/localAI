"""
Benchmark script for testing RAG system performance and model capabilities.
Tests query latency, throughput, and RAG quality metrics.
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
from config import (
    LLM_MODEL, EMBED_MODEL, OLLAMA_BASE_URL,
    USE_ROCM, USE_NPU, GPU_LAYERS, LLM_CONTEXT_WINDOW
)
from query import initialize_rag_system, create_query_engine, format_response


class Benchmark:
    """Benchmark suite for RAG system performance."""
    
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "llm_model": LLM_MODEL,
                "embed_model": EMBED_MODEL,
                "ollama_url": OLLAMA_BASE_URL,
                "use_rocm": USE_ROCM,
                "use_npu": USE_NPU,
                "gpu_layers": GPU_LAYERS,
                "context_window": LLM_CONTEXT_WINDOW,
            },
            "tests": []
        }
    
    def run_query_latency_test(self, query_engine, test_queries: List[str]) -> Dict:
        """Test query response latency."""
        print("\n" + "=" * 60)
        print("QUERY LATENCY TEST")
        print("=" * 60)
        
        latencies = []
        token_counts = []
        
        for idx, query in enumerate(test_queries, 1):
            print(f"\n[{idx}/{len(test_queries)}] Testing query: {query[:60]}...")
            
            start_time = time.time()
            try:
                response = query_engine.query(query)
                end_time = time.time()
                
                latency = end_time - start_time
                latencies.append(latency)
                
                answer = str(response)
                token_count = len(answer.split())
                token_counts.append(token_count)
                
                tokens_per_sec = token_count / latency if latency > 0 else 0
                
                print(f"   ✓ Latency: {latency:.2f}s")
                print(f"   ✓ Tokens: {token_count}")
                print(f"   ✓ Throughput: {tokens_per_sec:.2f} tokens/sec")
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
                continue
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        avg_throughput = avg_tokens / avg_latency if avg_latency > 0 else 0
        
        results = {
            "test": "query_latency",
            "num_queries": len(test_queries),
            "avg_latency_sec": round(avg_latency, 2),
            "avg_tokens": round(avg_tokens, 1),
            "avg_throughput_tokens_per_sec": round(avg_throughput, 2),
            "min_latency_sec": round(min(latencies), 2) if latencies else 0,
            "max_latency_sec": round(max(latencies), 2) if latencies else 0,
        }
        
        print("\n" + "-" * 60)
        print("RESULTS:")
        print(f"  Average latency: {results['avg_latency_sec']}s")
        print(f"  Average throughput: {results['avg_throughput_tokens_per_sec']} tokens/sec")
        print(f"  Latency range: {results['min_latency_sec']}s - {results['max_latency_sec']}s")
        
        return results
    
    def run_retrieval_quality_test(self, query_engine, test_cases: List[Dict]) -> Dict:
        """Test retrieval quality with known good answers."""
        print("\n" + "=" * 60)
        print("RETRIEVAL QUALITY TEST")
        print("=" * 60)
        
        scores = []
        
        for idx, case in enumerate(test_cases, 1):
            query = case["query"]
            expected_keywords = case.get("expected_keywords", [])
            
            print(f"\n[{idx}/{len(test_cases)}] Query: {query[:60]}...")
            
            try:
                response = query_engine.query(query)
                result = format_response(response)
                answer = result["answer"].lower()
                
                keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in answer)
                keyword_score = keyword_hits / len(expected_keywords) if expected_keywords else 0
                
                retrieval_score = 0
                if result["sources"]:
                    top_score = result["sources"][0].get("score", 0)
                    retrieval_score = top_score
                
                combined_score = (keyword_score + retrieval_score) / 2
                scores.append(combined_score)
                
                print(f"   ✓ Keyword match: {keyword_hits}/{len(expected_keywords)} ({keyword_score:.2%})")
                print(f"   ✓ Top retrieval score: {retrieval_score:.3f}")
                print(f"   ✓ Combined score: {combined_score:.3f}")
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
                continue
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        results = {
            "test": "retrieval_quality",
            "num_test_cases": len(test_cases),
            "avg_quality_score": round(avg_score, 3),
            "min_score": round(min(scores), 3) if scores else 0,
            "max_score": round(max(scores), 3) if scores else 0,
        }
        
        print("\n" + "-" * 60)
        print("RESULTS:")
        print(f"  Average quality score: {results['avg_quality_score']:.3f}")
        print(f"  Score range: {results['min_score']:.3f} - {results['max_score']:.3f}")
        
        return results
    
    def run_context_window_test(self, query_engine, long_context_query: str) -> Dict:
        """Test handling of large context windows."""
        print("\n" + "=" * 60)
        print("CONTEXT WINDOW TEST")
        print("=" * 60)
        
        print(f"\nTesting with query requiring large context...")
        print(f"Configured context window: {LLM_CONTEXT_WINDOW} tokens")
        
        start_time = time.time()
        try:
            response = query_engine.query(long_context_query)
            end_time = time.time()
            
            result = format_response(response)
            latency = end_time - start_time
            num_sources = len(result["sources"])
            
            print(f"   ✓ Query completed successfully")
            print(f"   ✓ Latency: {latency:.2f}s")
            print(f"   ✓ Sources retrieved: {num_sources}")
            
            results = {
                "test": "context_window",
                "context_window_size": LLM_CONTEXT_WINDOW,
                "latency_sec": round(latency, 2),
                "sources_retrieved": num_sources,
                "success": True
            }
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            results = {
                "test": "context_window",
                "context_window_size": LLM_CONTEXT_WINDOW,
                "success": False,
                "error": str(e)
            }
        
        return results
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
        return filepath


def get_default_test_queries() -> List[str]:
    """Get default test queries for latency testing."""
    return [
        "What is the main topic of the documents?",
        "Summarize the key points from the ingested documents.",
        "What are the most important concepts discussed?",
        "Explain the methodology described in the documents.",
        "What conclusions can be drawn from the information provided?",
    ]


def get_default_quality_tests() -> List[Dict]:
    """Get default test cases for quality testing."""
    return [
        {
            "query": "What is this document about?",
            "expected_keywords": ["document", "information", "content"]
        },
        {
            "query": "Summarize the main points.",
            "expected_keywords": ["main", "key", "important"]
        },
    ]


def main():
    """Run benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark RAG system performance")
    parser.add_argument("--latency", action="store_true", help="Run latency test")
    parser.add_argument("--quality", action="store_true", help="Run quality test")
    parser.add_argument("--context", action="store_true", help="Run context window test")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--output", default="benchmarks", help="Output directory for results")
    
    args = parser.parse_args()
    
    if not any([args.latency, args.quality, args.context, args.all]):
        parser.print_help()
        return
    
    print("=" * 60)
    print("RAG SYSTEM BENCHMARK")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  LLM Model: {LLM_MODEL}")
    print(f"  Embed Model: {EMBED_MODEL}")
    print(f"  ROCm Enabled: {USE_ROCM}")
    print(f"  NPU Enabled: {USE_NPU}")
    print(f"  Context Window: {LLM_CONTEXT_WINDOW}")
    
    benchmark = Benchmark(output_dir=args.output)
    
    print("\n🔧 Initializing RAG system...")
    index = initialize_rag_system()
    query_engine = create_query_engine(index)
    print("✓ RAG system ready")
    
    if args.all or args.latency:
        test_queries = get_default_test_queries()
        result = benchmark.run_query_latency_test(query_engine, test_queries)
        benchmark.results["tests"].append(result)
    
    if args.all or args.quality:
        test_cases = get_default_quality_tests()
        result = benchmark.run_retrieval_quality_test(query_engine, test_cases)
        benchmark.results["tests"].append(result)
    
    if args.all or args.context:
        long_query = "Provide a comprehensive summary of all the key information across all documents, including main themes, important details, and conclusions."
        result = benchmark.run_context_window_test(query_engine, long_query)
        benchmark.results["tests"].append(result)
    
    benchmark.save_results()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
