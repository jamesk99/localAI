"""
Enhanced RAG Benchmark Suite v2
Tiered implementation for comprehensive RAG system evaluation.

Tiers:
  1. Infrastructure Metrics - Latency, throughput, hardware monitoring
  2. RAGAS Quality Metrics - Faithfulness, relevancy, precision, recall
  3. Retrieval Effectiveness - Precision@K, Recall@K, MRR, NDCG
  4. Multi-User Load Testing - Concurrent query handling
  5. Large Context & Scale Testing - Context window limits, corpus scaling

Usage:
  python src/benchmarkv2.py --tier1
  python src/benchmarkv2.py --tier2
  python src/benchmarkv2.py --all
"""

import os
import sys
import time
import json
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# System monitoring
import psutil

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    LLM_MODEL, EMBED_MODEL, OLLAMA_BASE_URL, TOP_K,
    CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_THRESHOLD
)
from query import initialize_rag_system, create_query_engine, format_response

# Import LlamaIndex components for detailed timing
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import get_response_synthesizer, PromptTemplate


# Optional GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVIDIA_GPU = True
except Exception:
    HAS_NVIDIA_GPU = False

# Optional RAGAS import (Tier 2)
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    from langchain_community.llms import Ollama as LangchainOllama
    from langchain_community.embeddings import OllamaEmbeddings
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False

# Progress indicator
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def progress_bar(iterable, desc="Processing", total=None):
    """Wrapper for optional tqdm progress bar."""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total)
    else:
        print(f"  {desc}...")
        return iterable


class HardwareMonitor:
    """Monitor system resources during benchmark execution."""
    
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
        self._monitoring = False
        self._thread = None
    
    def start(self):
        """Start background resource monitoring."""
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> Dict:
        """Stop monitoring and return statistics."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        return self._compute_stats()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            self.cpu_samples.append(psutil.cpu_percent())
            mem = psutil.virtual_memory()
            self.memory_samples.append(mem.used / (1024 ** 3))  # GB
            
            if HAS_NVIDIA_GPU:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_samples.append(util.gpu)
                except Exception:
                    pass
            
            time.sleep(0.5)
    
    def _compute_stats(self) -> Dict:
        """Compute resource usage statistics."""
        stats = {
            "cpu": {
                "avg_percent": round(statistics.mean(self.cpu_samples), 1) if self.cpu_samples else 0,
                "peak_percent": round(max(self.cpu_samples), 1) if self.cpu_samples else 0,
            },
            "memory": {
                "avg_gb": round(statistics.mean(self.memory_samples), 2) if self.memory_samples else 0,
                "peak_gb": round(max(self.memory_samples), 2) if self.memory_samples else 0,
            },
            "gpu_utilized": HAS_NVIDIA_GPU
        }
        
        if HAS_NVIDIA_GPU and self.gpu_samples:
            stats["gpu"] = {
                "avg_percent": round(statistics.mean(self.gpu_samples), 1),
                "peak_percent": round(max(self.gpu_samples), 1),
            }
        
        return stats


class BenchmarkV2:
    """Enhanced RAG Benchmark Suite with tiered evaluation."""
    
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.query_engine = None
        self.index = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of RAG system."""
        if not self._initialized:
            print("\n Initializing RAG system...")
            self.index = initialize_rag_system()
            self.query_engine = create_query_engine(self.index)
            
            # Create retriever matching production config
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=TOP_K,
            )
            
            # Create node postprocessors matching production
            self.node_postprocessors = [
                SimilarityPostprocessor(similarity_cutoff=SIMILARITY_THRESHOLD)
            ]
            
            # Create response synthesizer with SAME prompt as production (from query.py)
            qa_prompt_template = (
                "You are an AI assistant answering questions based on provided context documents.\n"
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information above, answer the following question in a clear, comprehensive, and well-structured manner.\n"
                "If the context doesn't contain enough information to fully answer the question, say so explicitly.\n"
                "Provide specific details and examples from the context when possible.\n"
                "Format your response with:\n"
                "1. A direct answer to the question\n"
                "2. Supporting details from the context\n"
                "3. Any relevant implications or considerations\n\n"
                "Question: {query_str}\n"
                "Answer: "
            )
            self.qa_prompt = PromptTemplate(qa_prompt_template)
            
            self.response_synthesizer = get_response_synthesizer(
                text_qa_template=self.qa_prompt,
                response_mode="compact"
            )
            
            self._initialized = True
            print("✓ RAG system ready\n")
    
    def _get_full_context_from_response(self, response) -> List[str]:
        """Extract full context text from response source nodes (not truncated)."""
        contexts = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                full_text = node.node.text
                contexts.append(full_text)
        return contexts
    
    def _execute_query_with_timing(self, query_text: str) -> Dict:
        """
        Execute a single query with component timing.
        This runs the query ONCE and measures each component SEPARATELY.
        
        Uses the pre-computed query embedding for vector search to avoid
        double-embedding and get accurate timing measurements.
        
        Returns dict with:
            - response: The actual response object
            - embedding_sec: Time to embed the query
            - retrieval_sec: Time for vector search only (using pre-computed embedding)
            - generation_sec: Time for LLM generation
            - total_sec: Total end-to-end time
            - nodes: Retrieved nodes
        """
        from llama_index.core import Settings as LlamaSettings
        from llama_index.core.schema import NodeWithScore, QueryBundle
        
        result = {
            "response": None,
            "embedding_sec": 0,
            "retrieval_sec": 0,
            "generation_sec": 0,
            "total_sec": 0,
            "nodes": [],
            "answer": ""
        }
        
        total_start = time.time()
        
        # Step 1: Embedding - measure by calling embed model directly
        embed_model = LlamaSettings.embed_model
        embedding_start = time.time()
        query_embedding = embed_model.get_query_embedding(query_text)
        embedding_end = time.time()
        result["embedding_sec"] = embedding_end - embedding_start
        
        # Step 2: Vector search - use pre-computed embedding to avoid re-embedding
        # Create QueryBundle with the embedding we already computed
        query_bundle = QueryBundle(query_str=query_text, embedding=query_embedding)
        
        retrieval_start = time.time()
        # Use _retrieve which accepts QueryBundle with pre-computed embedding
        nodes = self.retriever._retrieve(query_bundle)
        retrieval_end = time.time()
        result["retrieval_sec"] = retrieval_end - retrieval_start
        result["nodes"] = nodes
        
        # Apply postprocessors (same as production)
        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_str=query_text)
        
        # Step 3: Generation using production response synthesizer
        generation_start = time.time()
        response = self.response_synthesizer.synthesize(query_text, nodes)
        generation_end = time.time()
        result["generation_sec"] = generation_end - generation_start
        
        total_end = time.time()
        result["total_sec"] = total_end - total_start
        result["response"] = response
        result["answer"] = str(response)
        
        return result
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the LLM's tokenizer if available,
        otherwise fall back to approximation.
        """
        try:
            from llama_index.core import Settings as LlamaSettings
            llm = LlamaSettings.llm
            # Try to use the model's tokenizer if available
            if hasattr(llm, 'tokenizer') and llm.tokenizer is not None:
                return len(llm.tokenizer.encode(text))
            # Ollama models may have a different interface
            if hasattr(llm, '_tokenizer') and llm._tokenizer is not None:
                return len(llm._tokenizer.encode(text))
        except Exception:
            pass
        
        # Fallback: approximate tokens as words * 1.3 (typical for English text)
        # This is more accurate than just word count
        word_count = len(text.split())
        return int(word_count * 1.3)
    
    def _load_queries(self, query_file: str, key: str) -> List[Dict]:
        """Load test queries from JSON file."""
        filepath = Path(query_file)
        if not filepath.exists():
            raise FileNotFoundError(f"Query file not found: {query_file}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if key not in data:
            raise KeyError(f"Expected key '{key}' not found in {query_file}")
        
        return data[key]
    
    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentile statistics."""
        if not values:
            return {"p50": 0, "p95": 0, "p99": 0}
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        return {
            "p50": round(sorted_vals[int(n * 0.50)], 3),
            "p95": round(sorted_vals[int(min(n * 0.95, n - 1))], 3),
            "p99": round(sorted_vals[int(min(n * 0.99, n - 1))], 3),
        }
    
    def save_results(self, tier: str, results: Dict):
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tier}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to: {filepath}")
        return filepath

    # =========================================================================
    # TIER 1: Enhanced Infrastructure Metrics
    # =========================================================================
    
    def run_tier1_infrastructure(self, query_file: str = "tests/queries/infrastructure_test_set.json",
                                  iterations: int = 3) -> Dict:
        """
        Tier 1: Enhanced infrastructure metrics with hardware monitoring.
        
        Executes each query ONCE per iteration with integrated component timing.
        No double query execution - timing is measured during actual pipeline execution.
        
        Measures:
        - Query latency (mean, std, p50, p95, p99)
        - Component breakdown (retrieval = embed+search, generation)
        - Token throughput (tokens / generation_time)
        - Hardware utilization (CPU, memory, GPU)
        
        Args:
            query_file: Path to JSON file containing test queries
            iterations: Number of times to run each query (for statistical accuracy)
        """
        print("\n" + "=" * 70)
        print("TIER 1: ENHANCED INFRASTRUCTURE METRICS")
        print("=" * 70)
        
        self._ensure_initialized()
        
        # Load queries
        try:
            queries = self._load_queries(query_file, "infrastructure_tests")
        except (FileNotFoundError, KeyError) as e:
            print(f"✗ Error loading queries: {e}")
            return {"error": str(e)}
        
        print(f"\nLoaded {len(queries)} test queries")
        print(f"Running {iterations} iterations per query\n")
        
        # Start hardware monitoring
        hw_monitor = HardwareMonitor()
        hw_monitor.start()
        
        # Results storage
        all_latencies = []
        latencies_by_type = {"factual": [], "summary": [], "multi_hop": [], "analytical": []}
        query_results = []
        
        # Aggregate component timing accumulators
        all_embedding_times = []
        all_retrieval_times = []
        all_generation_times = []
        
        for query_data in progress_bar(queries, desc="Testing queries"):
            query_id = query_data["id"]
            query_text = query_data["query"]
            query_type = query_data.get("type", "unknown")
            
            print(f"\n[{query_id}] {query_text[:50]}...")
            
            # Per-query accumulators
            iteration_latencies = []
            iteration_tokens = []
            iteration_embedding = []
            iteration_retrieval = []
            iteration_generation = []
            iteration_throughputs = []
            
            for i in range(iterations):
                try:
                    # Execute query ONCE with component timing (no double execution)
                    timed_result = self._execute_query_with_timing(query_text)
                    
                    answer = timed_result["answer"]
                    total_latency = timed_result["total_sec"]
                    embedding_time = timed_result["embedding_sec"]
                    retrieval_time = timed_result["retrieval_sec"]
                    generation_time = timed_result["generation_sec"]
                    
                    # Count tokens properly
                    token_count = self._count_tokens(answer)
                    
                    # Throughput = tokens / generation_time (not total time)
                    # This is the actual token generation rate
                    throughput = token_count / generation_time if generation_time > 0 else 0
                    
                    iteration_latencies.append(total_latency)
                    iteration_tokens.append(token_count)
                    iteration_embedding.append(embedding_time)
                    iteration_retrieval.append(retrieval_time)
                    iteration_generation.append(generation_time)
                    iteration_throughputs.append(throughput)
                    
                except Exception as e:
                    print(f"   ✗ Iteration {i+1} failed: {e}")
                    continue
            
            if iteration_latencies:
                avg_latency = statistics.mean(iteration_latencies)
                std_latency = statistics.stdev(iteration_latencies) if len(iteration_latencies) > 1 else 0
                avg_tokens = statistics.mean(iteration_tokens)
                avg_embedding = statistics.mean(iteration_embedding)
                avg_retrieval = statistics.mean(iteration_retrieval)
                avg_generation = statistics.mean(iteration_generation)
                avg_throughput = statistics.mean(iteration_throughputs)
                
                all_latencies.extend(iteration_latencies)
                all_embedding_times.extend(iteration_embedding)
                all_retrieval_times.extend(iteration_retrieval)
                all_generation_times.extend(iteration_generation)
                
                if query_type in latencies_by_type:
                    latencies_by_type[query_type].extend(iteration_latencies)
                
                query_result = {
                    "id": query_id,
                    "query": query_text,
                    "type": query_type,
                    "iterations_completed": len(iteration_latencies),
                    "latency": {
                        "total_mean_sec": round(avg_latency, 3),
                        "total_std_sec": round(std_latency, 3),
                        "embedding_mean_sec": round(avg_embedding, 3),
                        "embedding_std_sec": round(statistics.stdev(iteration_embedding), 3) if len(iteration_embedding) > 1 else 0,
                        "retrieval_mean_sec": round(avg_retrieval, 3),
                        "retrieval_std_sec": round(statistics.stdev(iteration_retrieval), 3) if len(iteration_retrieval) > 1 else 0,
                        "generation_mean_sec": round(avg_generation, 3),
                        "generation_std_sec": round(statistics.stdev(iteration_generation), 3) if len(iteration_generation) > 1 else 0,
                    },
                    "tokens": round(avg_tokens, 1),
                    "throughput_tokens_per_sec": round(avg_throughput, 2),
                }
                
                query_results.append(query_result)
                
                print(f"   ✓ Total: {avg_latency:.2f}s (embed: {avg_embedding:.3f}s, search: {avg_retrieval:.3f}s, gen: {avg_generation:.2f}s)")
                print(f"   ✓ Throughput: {avg_throughput:.1f} tokens/sec (generation only)")
        
        # Stop hardware monitoring
        hw_stats = hw_monitor.stop()
        
        # Calculate aggregate statistics
        percentiles = self._calculate_percentiles(all_latencies)
        
        latency_by_type_stats = {}
        for qtype, lats in latencies_by_type.items():
            if lats:
                latency_by_type_stats[qtype] = round(statistics.mean(lats), 3)
        
        # Calculate aggregate throughput from query results
        all_throughputs = [q["throughput_tokens_per_sec"] for q in query_results]
        avg_throughput = statistics.mean(all_throughputs) if all_throughputs else 0
        
        # Build results
        results = {
            "timestamp": datetime.now().isoformat(),
            "tier": "tier1_infrastructure",
            "config": {
                "llm_model": LLM_MODEL,
                "embed_model": EMBED_MODEL,
                "top_k": TOP_K,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "num_queries": len(queries),
                "iterations_per_query": iterations
            },
            "aggregate_results": {
                "total_latency": {
                    "mean_sec": round(statistics.mean(all_latencies), 3) if all_latencies else 0,
                    "std_sec": round(statistics.stdev(all_latencies), 3) if len(all_latencies) > 1 else 0,
                    "p50_sec": percentiles["p50"],
                    "p95_sec": percentiles["p95"],
                    "p99_sec": percentiles["p99"],
                    "min_sec": round(min(all_latencies), 3) if all_latencies else 0,
                    "max_sec": round(max(all_latencies), 3) if all_latencies else 0,
                },
                "component_breakdown": {
                    "embedding_mean_sec": round(statistics.mean(all_embedding_times), 3) if all_embedding_times else 0,
                    "embedding_std_sec": round(statistics.stdev(all_embedding_times), 3) if len(all_embedding_times) > 1 else 0,
                    "retrieval_mean_sec": round(statistics.mean(all_retrieval_times), 3) if all_retrieval_times else 0,
                    "retrieval_std_sec": round(statistics.stdev(all_retrieval_times), 3) if len(all_retrieval_times) > 1 else 0,
                    "generation_mean_sec": round(statistics.mean(all_generation_times), 3) if all_generation_times else 0,
                    "generation_std_sec": round(statistics.stdev(all_generation_times), 3) if len(all_generation_times) > 1 else 0,
                },
                "throughput_tokens_per_sec": round(avg_throughput, 2),
                "latency_by_query_type": latency_by_type_stats,
            },
            "hardware": hw_stats,
            "per_query_results": query_results
        }
        
        # Calculate component percentages
        total_time = results["aggregate_results"]["total_latency"]["mean_sec"]
        if total_time > 0:
            embed_pct = (results["aggregate_results"]["component_breakdown"]["embedding_mean_sec"] / total_time) * 100
            retrieval_pct = (results["aggregate_results"]["component_breakdown"]["retrieval_mean_sec"] / total_time) * 100
            generation_pct = (results["aggregate_results"]["component_breakdown"]["generation_mean_sec"] / total_time) * 100
            results["aggregate_results"]["component_breakdown"]["embedding_percent"] = round(embed_pct, 1)
            results["aggregate_results"]["component_breakdown"]["retrieval_percent"] = round(retrieval_pct, 1)
            results["aggregate_results"]["component_breakdown"]["generation_percent"] = round(generation_pct, 1)
        
        # Print summary
        print("\n" + "-" * 70)
        print("TIER 1 SUMMARY")
        print("-" * 70)
        print(f"  Queries tested: {len(query_results)}")
        print(f"  Total iterations: {len(all_latencies)}")
        print(f"  Average total latency: {results['aggregate_results']['total_latency']['mean_sec']}s")
        print(f"    - Embedding: {results['aggregate_results']['component_breakdown']['embedding_mean_sec']}s")
        print(f"    - Vector search: {results['aggregate_results']['component_breakdown']['retrieval_mean_sec']}s")
        print(f"    - Generation: {results['aggregate_results']['component_breakdown']['generation_mean_sec']}s")
        print(f"  P95 latency: {results['aggregate_results']['total_latency']['p95_sec']}s")
        print(f"  Throughput: {results['aggregate_results']['throughput_tokens_per_sec']} tokens/sec (generation phase)")
        print(f"  Peak CPU: {hw_stats['cpu']['peak_percent']}%")
        print(f"  Peak Memory: {hw_stats['memory']['peak_gb']} GB")
        
        # Save results
        self.save_results("tier1_infrastructure", results)
        
        return results

    # =========================================================================
    # TIER 2: RAGAS Quality Metrics
    # =========================================================================
    
    def run_tier2_ragas(self, query_file: str = "tests/queries/ragas_test_set.json") -> Dict:
        """
        Tier 2: RAGAS quality metrics evaluation.
        
        Measures:
        - Faithfulness (answer grounded in context)
        - Answer Relevancy (answer addresses question)
        - Context Precision (retrieved chunks used)
        - Context Recall (all relevant info retrieved)
        """
        print("\n" + "=" * 70)
        print("TIER 2: RAGAS QUALITY METRICS")
        print("=" * 70)
        
        if not HAS_RAGAS:
            print("\n✗ RAGAS not installed. Install with:")
            print("  pip install ragas datasets")
            return {"error": "RAGAS not installed"}
        
        self._ensure_initialized()
        
        # Load queries
        try:
            queries = self._load_queries(query_file, "ragas_tests")
        except (FileNotFoundError, KeyError) as e:
            print(f"✗ Error loading queries: {e}")
            return {"error": str(e)}
        
        print(f"\nLoaded {len(queries)} RAGAS test queries\n")
        
        # Collect data for RAGAS evaluation
        questions = []
        contexts_list = []
        answers = []
        ground_truths = []
        per_query_results = []
        
        for query_data in progress_bar(queries, desc="Collecting responses"):
            query_id = query_data["id"]
            question = query_data["question"]
            ground_truth = query_data["ground_truth"]
            
            print(f"\n[{query_id}] {question[:50]}...")
            
            try:
                response = self.query_engine.query(question)
                answer = str(response)
                
                # Extract FULL context texts from source nodes (not truncated preview)
                # This is critical for accurate RAGAS evaluation
                contexts = self._get_full_context_from_response(response)
                
                # Fallback if no contexts found via source_nodes
                if not contexts:
                    result = format_response(response)
                    contexts = [source["text"] for source in result["sources"]]
                    print(f"      (Warning: Using truncated context, RAGAS accuracy may be reduced)")
                
                questions.append(question)
                contexts_list.append(contexts)
                answers.append(answer)
                ground_truths.append(ground_truth)
                
                # Calculate total context length for debugging
                total_context_chars = sum(len(c) for c in contexts)
                
                per_query_results.append({
                    "id": query_id,
                    "question": question,
                    "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                    "num_contexts": len(contexts),
                    "total_context_chars": total_context_chars
                })
                
                print(f"   ✓ Retrieved {len(contexts)} contexts ({total_context_chars} chars total)")
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
                continue
        
        if not questions:
            return {"error": "No successful queries to evaluate"}
        
        # Create RAGAS dataset
        print("\n Evaluating with RAGAS metrics...")
        
        try:
            dataset = Dataset.from_dict({
                "question": questions,
                "contexts": contexts_list,
                "answer": answers,
                "ground_truth": ground_truths
            })
            
            # Configure RAGAS to use Ollama (same as our pipeline) instead of OpenAI
            ragas_llm = LangchainOllama(
                model=LLM_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
            )
            ragas_embeddings = OllamaEmbeddings(
                model=EMBED_MODEL,
                base_url=OLLAMA_BASE_URL,
            )
            
            # Run RAGAS evaluation with our local Ollama models
            scores = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )
            
            # Convert to dict
            scores_dict = scores.to_pandas().to_dict()
            
            # Calculate aggregate scores
            agg_scores = {}
            score_distributions = {}
            
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric in scores_dict:
                    values = list(scores_dict[metric].values())
                    agg_scores[metric] = round(statistics.mean(values), 3) if values else 0
                    
                    # Calculate distribution
                    score_distributions[metric] = {
                        "above_0.9": round(sum(1 for v in values if v >= 0.9) / len(values) * 100, 1),
                        "above_0.8": round(sum(1 for v in values if v >= 0.8) / len(values) * 100, 1),
                        "above_0.7": round(sum(1 for v in values if v >= 0.7) / len(values) * 100, 1),
                        "above_0.6": round(sum(1 for v in values if v >= 0.6) / len(values) * 100, 1),
                    }
                    
                    # Add per-query scores
                    for i, pq in enumerate(per_query_results):
                        if i in scores_dict[metric]:
                            pq[metric] = round(scores_dict[metric][i], 3)
            
        except Exception as e:
            print(f"✗ RAGAS evaluation failed: {e}")
            return {"error": str(e)}
        
        # Build results
        results = {
            "timestamp": datetime.now().isoformat(),
            "tier": "tier2_ragas",
            "config": {
                "llm_model": LLM_MODEL,
                "embed_model": EMBED_MODEL,
                "num_queries": len(questions),
                "ragas_available": True
            },
            "aggregate_scores": agg_scores,
            "score_distribution": score_distributions,
            "per_query_results": per_query_results
        }
        
        # Print summary
        print("\n" + "-" * 70)
        print("TIER 2 SUMMARY")
        print("-" * 70)
        print(f"  Queries evaluated: {len(questions)}")
        for metric, score in agg_scores.items():
            print(f"  {metric}: {score:.3f}")
        
        # Save results
        self.save_results("tier2_ragas", results)
        
        return results

    # =========================================================================
    # TIER 3: Retrieval Effectiveness Analysis
    # =========================================================================
    
    def run_tier3_retrieval(self, query_file: str = "tests/queries/retrieval_test_set.json") -> Dict:
        """
        Tier 3: Retrieval effectiveness analysis.
        
        Measures:
        - Precision@K (K=3,5,10)
        - Recall@K
        - MRR (Mean Reciprocal Rank)
        - NDCG@K
        - Similarity score distribution
        """
        print("\n" + "=" * 70)
        print("TIER 3: RETRIEVAL EFFECTIVENESS ANALYSIS")
        print("=" * 70)
        
        self._ensure_initialized()
        
        # Load queries
        try:
            queries = self._load_queries(query_file, "retrieval_tests")
        except (FileNotFoundError, KeyError) as e:
            print(f"✗ Error loading queries: {e}")
            return {"error": str(e)}
        
        print(f"\nLoaded {len(queries)} retrieval test queries\n")
        
        # Metrics accumulators
        precision_at_k = {3: [], 5: [], 10: []}
        recall_at_k = {3: [], 5: [], 10: []}
        mrr_scores = []
        ndcg_at_k = {3: [], 5: [], 10: []}
        all_similarity_scores = []
        per_query_results = []
        
        for query_data in progress_bar(queries, desc="Analyzing retrieval"):
            query_id = query_data["id"]
            query_text = query_data["query"]
            relevant_keywords = query_data.get("relevant_chunk_keywords", [])
            
            print(f"\n[{query_id}] {query_text[:50]}...")
            
            try:
                response = self.query_engine.query(query_text)
                
                # Get full text from source nodes for better relevance detection
                full_texts = self._get_full_context_from_response(response)
                result = format_response(response)
                sources = result["sources"]
                
                # Collect similarity scores
                scores = [s.get("score", 0) for s in sources]
                all_similarity_scores.extend(scores)
                # Previous was to determine relevance based on keyword matching (I think) but was replaced witht the below mehtod.
                # Improved relevance detection using multiple signals:
                # 1. Similarity score (high score = likely relevant)
                # 2. Keyword matching with full text (not truncated)
                # 3. Weighted scoring for better accuracy
                relevance_labels = []
                relevance_details = []
                
                for idx, source in enumerate(sources):
                    # Use full text if available, otherwise use truncated
                    text_to_check = full_texts[idx] if idx < len(full_texts) else source["text"]
                    text_lower = text_to_check.lower()
                    
                    # Signal 1: Similarity score (normalized 0-1)
                    sim_score = source.get("score", 0)
                    sim_signal = 1.0 if sim_score >= 0.7 else (0.5 if sim_score >= 0.5 else 0.0)
                    
                    # Signal 2: Keyword matching with partial word matching
                    keyword_matches = 0
                    for kw in relevant_keywords:
                        kw_lower = kw.lower()
                        # Exact match
                        if kw_lower in text_lower:
                            keyword_matches += 1
                        # Partial match (for compound words like "back-propagation" vs "backpropagation")
                        elif len(kw_lower) > 4 and any(kw_lower in word or word in kw_lower 
                                                        for word in text_lower.split()):
                            keyword_matches += 0.5
                    
                    # Normalize keyword score
                    keyword_signal = min(1.0, keyword_matches / max(1, len(relevant_keywords) * 0.4))
                    
                    # Combined relevance score (weighted)
                    # Similarity score: 40%, Keyword matching: 60%
                    combined_score = (sim_signal * 0.4) + (keyword_signal * 0.6)
                    
                    # Threshold for relevance: 0.4 (allows for partial matches)
                    is_relevant = combined_score >= 0.4
                    relevance_labels.append(is_relevant)
                    relevance_details.append({
                        "sim_score": round(sim_score, 3),
                        "keyword_matches": keyword_matches,
                        "combined_score": round(combined_score, 3),
                        "is_relevant": is_relevant
                    })
                
                # Calculate Precision@K
                for k in [3, 5, 10]:
                    top_k = relevance_labels[:k]
                    if top_k:
                        precision = sum(top_k) / len(top_k)
                        precision_at_k[k].append(precision)
                
                # Calculate Recall@K (assuming all keywords represent distinct relevant items)
                total_relevant = max(1, len([kw for kw in relevant_keywords if any(kw.lower() in s["text"].lower() for s in sources)]))
                for k in [3, 5, 10]:
                    top_k = relevance_labels[:k]
                    retrieved_relevant = sum(top_k)
                    recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0
                    recall_at_k[k].append(recall)
                
                # Calculate MRR
                mrr = 0
                for rank, is_rel in enumerate(relevance_labels, 1):
                    if is_rel:
                        mrr = 1.0 / rank
                        break
                mrr_scores.append(mrr)
                
                # Calculate NDCG@K
                for k in [3, 5, 10]:
                    top_k = relevance_labels[:k]
                    dcg = sum((1 if rel else 0) / (i + 2) for i, rel in enumerate(top_k))
                    ideal_rels = sorted(relevance_labels, reverse=True)[:k]
                    idcg = sum((1 if rel else 0) / (i + 2) for i, rel in enumerate(ideal_rels))
                    ndcg = dcg / idcg if idcg > 0 else 0
                    ndcg_at_k[k].append(ndcg)
                
                per_query_results.append({
                    "id": query_id,
                    "query": query_text,
                    "num_retrieved": len(sources),
                    "num_relevant": sum(relevance_labels),
                    "precision_at_5": round(precision_at_k[5][-1], 3) if precision_at_k[5] else 0,
                    "mrr": round(mrr, 3),
                    "top_similarity_score": round(scores[0], 3) if scores else 0
                })
                
                print(f"   ✓ Retrieved: {len(sources)}, Relevant: {sum(relevance_labels)}")
                print(f"   ✓ P@5: {precision_at_k[5][-1]:.3f}, MRR: {mrr:.3f}")
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
                continue
        
        # Calculate aggregate metrics
        def safe_mean(lst):
            return round(statistics.mean(lst), 3) if lst else 0
        
        # Similarity score histogram
        score_histogram = {
            "0.9-1.0": sum(1 for s in all_similarity_scores if 0.9 <= s <= 1.0),
            "0.8-0.9": sum(1 for s in all_similarity_scores if 0.8 <= s < 0.9),
            "0.7-0.8": sum(1 for s in all_similarity_scores if 0.7 <= s < 0.8),
            "0.6-0.7": sum(1 for s in all_similarity_scores if 0.6 <= s < 0.7),
            "0.5-0.6": sum(1 for s in all_similarity_scores if 0.5 <= s < 0.6),
            "below_0.5": sum(1 for s in all_similarity_scores if s < 0.5),
        }
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "tier": "tier3_retrieval",
            "config": {
                "llm_model": LLM_MODEL,
                "embed_model": EMBED_MODEL,
                "top_k": TOP_K,
                "num_queries": len(queries)
            },
            "aggregate_metrics": {
                "precision_at_3": safe_mean(precision_at_k[3]),
                "precision_at_5": safe_mean(precision_at_k[5]),
                "precision_at_10": safe_mean(precision_at_k[10]),
                "recall_at_3": safe_mean(recall_at_k[3]),
                "recall_at_5": safe_mean(recall_at_k[5]),
                "recall_at_10": safe_mean(recall_at_k[10]),
                "mrr": safe_mean(mrr_scores),
                "ndcg_at_3": safe_mean(ndcg_at_k[3]),
                "ndcg_at_5": safe_mean(ndcg_at_k[5]),
                "ndcg_at_10": safe_mean(ndcg_at_k[10]),
            },
            "similarity_score_distribution": score_histogram,
            "insights": [],
            "per_query_results": per_query_results
        }
        
        # Generate insights
        p5 = results["aggregate_metrics"]["precision_at_5"]
        mrr = results["aggregate_metrics"]["mrr"]
        
        if p5 < 0.6:
            results["insights"].append("Low P@5 suggests reranking or improved embedding model may help")
        if mrr < 0.5:
            results["insights"].append("Low MRR indicates first-retrieved chunks often not relevant")
        if score_histogram.get("below_0.5", 0) > len(all_similarity_scores) * 0.3:
            results["insights"].append("Many low-similarity retrievals; consider increasing similarity threshold")
        
        # Print summary
        print("\n" + "-" * 70)
        print("TIER 3 SUMMARY")
        print("-" * 70)
        print(f"  Queries analyzed: {len(per_query_results)}")
        print(f"  Precision@5: {results['aggregate_metrics']['precision_at_5']:.3f}")
        print(f"  Recall@5: {results['aggregate_metrics']['recall_at_5']:.3f}")
        print(f"  MRR: {results['aggregate_metrics']['mrr']:.3f}")
        print(f"  NDCG@5: {results['aggregate_metrics']['ndcg_at_5']:.3f}")
        if results["insights"]:
            print("\n  Insights:")
            for insight in results["insights"]:
                print(f"    - {insight}")
        
        self.save_results("tier3_retrieval", results)
        
        return results

    # =========================================================================
    # TIER 4: Multi-User Load Testing
    # =========================================================================
    
    def run_tier4_multiuser(self, query_file: str = "tests/queries/infrastructure_test_set.json") -> Dict:
        """
        Tier 4: Multi-user concurrent load testing.
        
        Tests:
        - 1, 3, 5, 10 concurrent users
        - Latency degradation measurement
        - Error/timeout tracking
        - Resource contention analysis
        """
        print("\n" + "=" * 70)
        print("TIER 4: MULTI-USER LOAD TESTING")
        print("=" * 70)
        
        self._ensure_initialized()
        
        # Load queries
        try:
            queries = self._load_queries(query_file, "infrastructure_tests")
        except (FileNotFoundError, KeyError) as e:
            print(f"✗ Error loading queries: {e}")
            return {"error": str(e)}
        
        # Use first 5 queries for load testing (to limit duration)
        test_queries = [q["query"] for q in queries[:5]]
        print(f"\nUsing {len(test_queries)} queries for load testing\n")
        
        user_levels = [1, 3, 5, 10]
        scenario_results = []
        baseline_latency = None
        
        for num_users in user_levels:
            print(f"\n{'='*50}")
            print(f"Testing with {num_users} concurrent user(s)...")
            print(f"{'='*50}")
            
            # Start hardware monitoring
            hw_monitor = HardwareMonitor()
            hw_monitor.start()
            
            latencies = []
            errors = []
            
            # NOTE (Issue 2 - Component Breakdown): This uses simple query_engine.query() which
            # only measures total latency. To identify whether degradation is due to embedding
            # contention, vector search contention, or LLM contention, would need to use
            # _execute_query_with_timing() instead. However, that method creates its own
            # retriever/synthesizer which may not be thread-safe. Future fix: make component
            # timing thread-safe or add separate contention tests for each component.
            
            def execute_query(query_text: str) -> Tuple[float, Optional[str]]:
                """Execute a single query and return latency and error."""
                try:
                    start = time.time()
                    response = self.query_engine.query(query_text)
                    end = time.time()
                    return end - start, None
                except Exception as e:
                    return 0, str(e)
            
            # Run concurrent queries
            # Each user gets one query from the pool (round-robin assignment)
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = []
                for user_idx in range(num_users):
                    # Assign each user a query (cycling through available queries)
                    query = test_queries[user_idx % len(test_queries)]
                    futures.append(executor.submit(execute_query, query))
                
                # Collect results
                for future in progress_bar(as_completed(futures), 
                                          desc=f"{num_users} users", 
                                          total=len(futures)):
                    lat, err = future.result()
                    if err:
                        errors.append(err)
                    else:
                        latencies.append(lat)
            
            # Stop hardware monitoring
            hw_stats = hw_monitor.stop()
            
            # Calculate metrics
            avg_latency = statistics.mean(latencies) if latencies else 0
            
            if num_users == 1:
                baseline_latency = avg_latency
            
            degradation = ((avg_latency - baseline_latency) / baseline_latency * 100) if baseline_latency else 0
            
            scenario = {
                "num_users": num_users,
                "total_queries": len(futures),
                "successful_queries": len(latencies),
                "failed_queries": len(errors),
                "avg_latency_sec": round(avg_latency, 3),
                "min_latency_sec": round(min(latencies), 3) if latencies else 0,
                "max_latency_sec": round(max(latencies), 3) if latencies else 0,
                "std_latency_sec": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0,
                "degradation_percent": round(degradation, 1),
                "hardware": hw_stats,
                "errors": errors[:5] if errors else []  # Limit error logs
            }
            scenario_results.append(scenario)
            
            print(f"\n  ✓ Avg latency: {avg_latency:.2f}s")
            print(f"  ✓ Degradation: {degradation:.1f}%")
            print(f"  ✓ Failed: {len(errors)}/{len(futures)}")
            print(f"  ✓ Peak CPU: {hw_stats['cpu']['peak_percent']}%")
            print(f"  ✓ Peak Memory: {hw_stats['memory']['peak_gb']} GB")
        
        # Build results
        results = {
            "timestamp": datetime.now().isoformat(),
            "tier": "tier4_multiuser",
            "config": {
                "llm_model": LLM_MODEL,
                "embed_model": EMBED_MODEL,
                "queries_per_scenario": len(test_queries),
                "user_levels_tested": user_levels
            },
            "baseline_latency_sec": round(baseline_latency, 3) if baseline_latency else 0,
            "scenarios": scenario_results,
            "insights": []
        }
        
        # Generate insights
        for scenario in scenario_results:
            if scenario["degradation_percent"] > 100:
                results["insights"].append(
                    f"{scenario['num_users']} users: >100% degradation suggests significant bottleneck"
                )
            if scenario["failed_queries"] > 0:
                results["insights"].append(
                    f"{scenario['num_users']} users: {scenario['failed_queries']} failures detected"
                )
        
        # Print summary
        print("\n" + "-" * 70)
        print("TIER 4 SUMMARY")
        print("-" * 70)
        print(f"  Baseline latency (1 user): {baseline_latency:.2f}s")
        for s in scenario_results:
            print(f"  {s['num_users']} users: {s['avg_latency_sec']}s ({s['degradation_percent']:+.1f}%)")
        
        self.save_results("tier4_multiuser", results)
        
        return results

    # =========================================================================
    # TIER 5: Large Context & Scale Testing
    # =========================================================================
    
    def run_tier5_scale(self) -> Dict:
        """
        Tier 5: Large context and scale testing.
        
        Tests:
        - Variable chunk counts (5, 10, 15, 20)
        - Memory scaling with context size
        - Latency vs chunk count correlation
        """
        print("\n" + "=" * 70)
        print("TIER 5: LARGE CONTEXT & SCALE TESTING")
        print("=" * 70)
        
        # Ensure RAG system is initialized with proper error handling
        try:
            self._ensure_initialized()
        except Exception as init_error:
            print(f"\n✗ Failed to initialize RAG system: {init_error}")
            return {
                "timestamp": datetime.now().isoformat(),
                "tier": "tier5_scale",
                "error": f"Initialization failed: {str(init_error)[:200]}",
                "context_window_tests": [],
                "insights": ["RAG system initialization failed - check vector DB and Ollama status"]
            }
        
        # Verify index is accessible
        if self.index is None:
            print("\n✗ Index not available")
            return {
                "timestamp": datetime.now().isoformat(),
                "tier": "tier5_scale",
                "error": "Index is None - vector database may not be initialized",
                "context_window_tests": [],
                "insights": ["Run ingest.py first to create the vector database"]
            }
        
        # Test query requiring comprehensive answer
        test_query = "Provide a comprehensive summary of all key concepts, methodologies, and conclusions from the documents."
        
        chunk_levels = [5, 10, 15, 20]
        context_results = []
        consecutive_failures = 0
        max_consecutive_failures = 2  # Stop after 2 consecutive failures
        
        print("\n--- Large Context Window Tests ---")
        
        for target_chunks in chunk_levels:
            print(f"\nTesting with TOP_K = {target_chunks}...")
            
            hw_monitor = None
            try:
                # Verify index is still accessible before each test
                if not hasattr(self.index, '_vector_store'):
                    # Try to access the index to verify it's valid
                    pass  # Basic check - index object exists
                
                # Create retriever with modified TOP_K
                retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=target_chunks,
                )
                
                node_postprocessors = [
                    SimilarityPostprocessor(similarity_cutoff=SIMILARITY_THRESHOLD)
                ]
                
                # Use SAME prompt as production (query.py) for accurate comparison
                qa_prompt_template = (
                    "You are an AI assistant answering questions based on provided context documents.\n"
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above, answer the following question in a clear, comprehensive, and well-structured manner.\n"
                    "If the context doesn't contain enough information to fully answer the question, say so explicitly.\n"
                    "Provide specific details and examples from the context when possible.\n"
                    "Format your response with:\n"
                    "1. A direct answer to the question\n"
                    "2. Supporting details from the context\n"
                    "3. Any relevant implications or considerations\n\n"
                    "Question: {query_str}\n"
                    "Answer: "
                )
                qa_prompt = PromptTemplate(qa_prompt_template)
                
                response_synthesizer = get_response_synthesizer(
                    text_qa_template=qa_prompt,
                    response_mode="compact"
                )
                
                temp_engine = RetrieverQueryEngine(
                    retriever=retriever,
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=node_postprocessors,
                )
                
                # Start monitoring
                hw_monitor = HardwareMonitor()
                hw_monitor.start()
                
                start_time = time.time()
                response = temp_engine.query(test_query)
                end_time = time.time()
                
                hw_stats = hw_monitor.stop()
                hw_monitor = None  # Mark as stopped
                
                result = format_response(response)
                latency = end_time - start_time
                # Use same token estimation as Tier 1 (Ollama doesn't expose tokenizer)
                # Approximation: words * 1.3 is typical for English text with subword tokenization
                token_count = self._count_tokens(result["answer"])
                actual_chunks = len(result["sources"])
                
                context_results.append({
                    "target_chunks": target_chunks,
                    "actual_chunks": actual_chunks,
                    "latency_sec": round(latency, 3),
                    "tokens_generated": token_count,
                    "throughput_tokens_per_sec": round(token_count / latency, 2) if latency > 0 else 0,
                    "peak_memory_gb": hw_stats["memory"]["peak_gb"],
                    "peak_cpu_percent": hw_stats["cpu"]["peak_percent"],
                    "success": True
                })
                
                consecutive_failures = 0  # Reset on success
                print(f"   ✓ Latency: {latency:.2f}s")
                print(f"   ✓ Chunks retrieved: {actual_chunks}")
                print(f"   ✓ Peak memory: {hw_stats['memory']['peak_gb']} GB")
                
            except Exception as e:
                # Ensure hardware monitor is stopped on error
                if hw_monitor is not None:
                    try:
                        hw_monitor.stop()
                    except:
                        pass
                
                error_msg = str(e)
                error_lower = error_msg.lower()
                
                # Categorize error types
                is_oom = "memory" in error_lower or "oom" in error_lower or "out of memory" in error_lower
                is_timeout = "timeout" in error_lower or "timed out" in error_lower
                is_connection = "connection" in error_lower or "refused" in error_lower
                is_index_error = "index" in error_lower or "collection" in error_lower or "vector" in error_lower
                
                error_type = "unknown"
                if is_oom:
                    error_type = "out_of_memory"
                elif is_timeout:
                    error_type = "timeout"
                elif is_connection:
                    error_type = "connection_error"
                elif is_index_error:
                    error_type = "index_error"
                
                context_results.append({
                    "target_chunks": target_chunks,
                    "success": False,
                    "error": error_msg[:200],
                    "error_type": error_type,
                    "is_oom": is_oom
                })
                
                consecutive_failures += 1
                print(f"   ✗ Failed ({error_type}): {error_msg[:100]}")
                
                # Stop conditions
                if is_oom:
                    print("   ✗ OOM detected, stopping scale tests")
                    break
                if is_connection or is_index_error:
                    print("   ✗ Critical error detected, stopping scale tests")
                    break
                if consecutive_failures >= max_consecutive_failures:
                    print(f"   ✗ {max_consecutive_failures} consecutive failures, stopping scale tests")
                    break
        
        # Analyze scaling
        successful = [r for r in context_results if r.get("success")]
        
        latency_scaling = {}
        memory_scaling = {}
        
        # NOTE (Issue 3 - Scaling Baseline): The baseline (5 chunks) matches production TOP_K=5,
        # so the 5-chunk result IS your production baseline. Scaling multipliers are relative to
        # production config. If TOP_K changes in config.py, this baseline changes too.
        # Consider parameterizing baseline_chunks or always using a fixed reference point.
        if len(successful) >= 2:
            base = successful[0]
            for r in successful[1:]:
                latency_scaling[r["target_chunks"]] = round(r["latency_sec"] / base["latency_sec"], 2)
                memory_scaling[r["target_chunks"]] = round(r["peak_memory_gb"] / base["peak_memory_gb"], 2)
        
        # Build results
        results = {
            "timestamp": datetime.now().isoformat(),
            "tier": "tier5_scale",
            "config": {
                "llm_model": LLM_MODEL,
                "embed_model": EMBED_MODEL,
                "chunk_levels_tested": chunk_levels
            },
            "context_window_tests": context_results,
            "scaling_analysis": {
                "latency_multiplier_by_chunks": latency_scaling,
                "memory_multiplier_by_chunks": memory_scaling,
            },
            "max_successful_chunks": max([r.get("actual_chunks", 0) for r in successful]) if successful else 0,
            "insights": []
        }
        
        # Generate insights
        if successful:
            max_chunks = results["max_successful_chunks"]
            results["insights"].append(f"Successfully processed up to {max_chunks} chunks")
            
            if latency_scaling:
                max_scale = max(latency_scaling.values())
                if max_scale > 2:
                    results["insights"].append(f"Latency scales {max_scale}x with increased context")
        
        oom_tests = [r for r in context_results if r.get("is_oom")]
        if oom_tests:
            results["insights"].append(f"OOM at {oom_tests[0]['target_chunks']} chunks - consider reducing context")
        
        # Print summary
        print("\n" + "-" * 70)
        print("TIER 5 SUMMARY")
        print("-" * 70)
        print(f"  Max successful chunks: {results['max_successful_chunks']}")
        print(f"  Latency scaling: {latency_scaling}")
        print(f"  Memory scaling: {memory_scaling}")
        if results["insights"]:
            print("\n  Insights:")
            for insight in results["insights"]:
                print(f"    - {insight}")
        
        self.save_results("tier5_scale", results)
        
        return results


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced RAG Benchmark Suite v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/benchmarkv2.py --tier1
  python src/benchmarkv2.py --tier2
  python src/benchmarkv2.py --all
  python src/benchmarkv2.py --tier1 --queries tests/queries/custom.json
        """
    )
    
    parser.add_argument('--tier1', action='store_true', 
                        help='Run Tier 1: Enhanced Infrastructure Metrics')
    parser.add_argument('--tier2', action='store_true', 
                        help='Run Tier 2: RAGAS Quality Metrics')
    parser.add_argument('--tier3', action='store_true', 
                        help='Run Tier 3: Retrieval Effectiveness Analysis')
    parser.add_argument('--tier4', action='store_true', 
                        help='Run Tier 4: Multi-User Load Testing')
    parser.add_argument('--tier5', action='store_true', 
                        help='Run Tier 5: Large Context & Scale Testing')
    parser.add_argument('--all', action='store_true', 
                        help='Run all tiers sequentially')
    parser.add_argument('--queries', type=str, default=None,
                        help='Custom query file path (overrides default)')
    parser.add_argument('--output', type=str, default='benchmarks',
                        help='Output directory for results (default: benchmarks)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Iterations per query for Tier 1 (default: 3)')
    
    args = parser.parse_args()
    
    # Check if any tier selected
    if not any([args.tier1, args.tier2, args.tier3, args.tier4, args.tier5, args.all]):
        parser.print_help()
        print("\n✗ Please specify at least one tier to run (e.g., --tier1 or --all)")
        return
    
    print("=" * 70)
    print("ENHANCED RAG BENCHMARK SUITE v2")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  LLM Model: {LLM_MODEL}")
    print(f"  Embed Model: {EMBED_MODEL}")
    print(f"  Output Directory: {args.output}")
    print(f"  RAGAS Available: {HAS_RAGAS}")
    print(f"  GPU Monitoring: {HAS_NVIDIA_GPU}")
    
    # Initialize benchmark
    benchmark = BenchmarkV2(output_dir=args.output)
    
    # Run selected tiers
    if args.all or args.tier1:
        query_file = args.queries or "tests/queries/infrastructure_test_set.json"
        benchmark.run_tier1_infrastructure(
            query_file, 
            iterations=args.iterations
        )
    
    if args.all or args.tier2:
        query_file = args.queries or "tests/queries/ragas_test_set.json"
        benchmark.run_tier2_ragas(query_file)
    
    if args.all or args.tier3:
        query_file = args.queries or "tests/queries/retrieval_test_set.json"
        benchmark.run_tier3_retrieval(query_file)
    
    if args.all or args.tier4:
        query_file = args.queries or "tests/queries/infrastructure_test_set.json"
        benchmark.run_tier4_multiuser(query_file)
    
    if args.all or args.tier5:
        benchmark.run_tier5_scale()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
