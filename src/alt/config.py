import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# HARDWARE CONFIGURATION
# FUTURE: Hardware acceleration flags (not yet implemented in Ollama/LlamaIndex)
# These are placeholders for future ROCm/NPU support
USE_ROCM = os.getenv("USE_ROCM", "false").lower() == "true"  # Reserved for future ROCm GPU support
USE_NPU = os.getenv("USE_NPU", "false").lower() == "true"    # Reserved for future NPU support
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "999"))  # Offload ALL layers to iGPU (unified memory). the number of layers to offload to GPU (0 = auto). 999 means all layers.
NUM_GPU = int(os.getenv("NUM_GPU", "1"))  # Single iGPU. the number of GPUs to use. 1 means single GPU.

# Parallel Processing (leverage 16 Zen 5 cores)
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))  # Used in ingest.py for parallel document loading. parallel workers for embedding/ingesting documents. old was 4.
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Reserved for future batch processing features. batch size for embedding/ingesting documents. old was 16.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))  # Used in query.py and ingest.py for OllamaEmbedding. batch size for embedding/ingesting documents. old was 32.


# OLLAMA CONFIGURATION
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-arx") # Primary LLM (requires more RAM) - old was: LLM_MODEL=qwen3:30b 
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "llama3:latest") # Fallback LLM 
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3") # Embedding model

# Model-specific settings for large models (70B-120B)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "131072"))  # 131K for qwen3:30b (256K max). old default was as low as 8K (8192), but can go up to 128K (131072) since our machine can handle qwen3:30b and thus can handle the context window of 131K.
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "300.0"))  # 5min for large models. old default was 180.0.
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "4096"))  # Max tokens to generate. old default was as low as 512. then it was 2048. then it was 4096. Longer responses now with 131K context window.

# Ollama performance tuning (passed via additional_kwargs)
OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "16"))  # Match Zen 5 core count. old was 8. this is the number of threads to use for processing.
OLLAMA_NUM_BATCH = int(os.getenv("OLLAMA_NUM_BATCH", "1024"))  # this is the number of batches to process in parallel. # Larger batch for 128GB RAM. old was 256. then it was 512.
OLLAMA_MAIN_GPU = int(os.getenv("OLLAMA_MAIN_GPU", "0"))  # Reserved for multi-GPU setups. the gpu index to use. 
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "999"))  # Reserved for multi-GPU setups. the number of GPUs to use. 1 means single GPU.
OLLAMA_GPU_LAYERS = int(os.getenv("OLLAMA_GPU_LAYERS", "999"))  # Reserved for multi-GPU setups. the number of layers to use for processing. 999 means all layers.


# RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3072"))  # Larger chunks for 131K context, old default was 2048 for chunk_size. the number of tokens in each chunk. increasing it gives better context but also more chunks which affect performance.
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "384"))  # More overlap for continuity, old default was 256 for chunk_overlap. the number of tokens to overlap between chunks. increased overlap for continuity.
TOP_K = int(os.getenv("TOP_K", "25"))  # More candidates with 131K context window, old default was 15 for top_k. the number of chunks to retrieve.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))  # Slightly higher threshold, old default was 0.3 for similarity_threshold. the minimum similarity score.

# Advanced RAG settings (for large context models)
MAX_CHUNKS_IN_CONTEXT = int(os.getenv("MAX_CHUNKS_IN_CONTEXT", "40"))  # Maybe lower to 20 for better performance as it is a tradeoff between context and performance. More context in 131K window . the max chunks to include in prompt.
USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() == "true"  # Enable reranking
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "8"))  # Re-rank top 8 of 15 retrieved results.

# ChromaDB optimization for large corpus
CHROMA_BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", "5000"))  # Batch insert size
CHROMA_PERSIST_INTERVAL = int(os.getenv("CHROMA_PERSIST_INTERVAL", "1000"))  # Persist every N docs

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DOCS_DIR = os.path.join(DATA_DIR, "raw")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vectordb")
COLLECTION_NAME = "phase0_docs"
TRACKING_DB_PATH = os.path.join(DATA_DIR, "tracking.db")

# Ensure directories exist
os.makedirs(RAW_DOCS_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
