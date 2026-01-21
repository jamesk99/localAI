# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# HARDWARE CONFIGURATION
# ============================================================================
# HARDWARE CONFIGURATION (AMD Ryzen AI Max+ 395 / 128GB RAM)
# ============================================================================
USE_ROCM = os.getenv("USE_ROCM", "false").lower() == "true"
USE_NPU = os.getenv("USE_NPU", "false").lower() == "true"
# GPU_LAYERS = int(os.getenv("GPU_LAYERS", "0"))  # Number of layers to offload to GPU (0 = auto)
# NUM_GPU = int(os.getenv("NUM_GPU", "1"))  # Number of GPUs to use
# OLD VALUE (for lighter machines) above: GPU_LAYERS = int(os.getenv("GPU_LAYERS", "0"))
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "999"))  # Offload ALL layers to iGPU (unified memory)
NUM_GPU = int(os.getenv("NUM_GPU", "1"))  # Single iGPU

# Parallel Processing (leverage 16 Zen 5 cores)
# OLD VALUES (for lighter machines - these settings did not exist before):
# NUM_WORKERS = 4
# BATCH_SIZE = 16
# EMBED_BATCH_SIZE = 32
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))  # Parallel workers for embedding/ingestion
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Batch size for embedding generation
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))  # Larger batches with 128GB RAM

# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# OLD VALUES (for lighter machines):
# LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:32b-instruct")        # Primary LLM (requires more RAM) 
# LLM_FALLBACK = os.getenv("LLM_FALLBACK", "llama3:latest")  # Fallback LLM 
# EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:8b")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:30b")
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "llama3:latest")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# Model-specific settings for large models (70B-120B)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
# OLD VALUES (for lighter machines):
# LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))    # Default 8K, can go up to 128K
# LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "180.0"))
# LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "512"))           # Max tokens to generate
# OLD: LLM_CONTEXT_WINDOW default was 32768
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "131072"))  # 131K for qwen3:30b (256K max)
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "300.0"))  # 5min for large models
# OLD: LLM_NUM_PREDICT default was 2048
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "4096"))  # Longer responses with 131K context

# Ollama performance tuning (passed via additional_kwargs)
# OLD VALUES (for lighter machines - these settings did not exist before):
# OLLAMA_NUM_THREAD = 8
# OLLAMA_NUM_BATCH = 256
OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "16"))  # Match Zen 5 core count
# OLD: OLLAMA_NUM_BATCH default was 512
OLLAMA_NUM_BATCH = int(os.getenv("OLLAMA_NUM_BATCH", "1024"))  # Larger batch for 128GB RAM
OLLAMA_MAIN_GPU = int(os.getenv("OLLAMA_MAIN_GPU", "0"))  # Primary GPU index

# RAG Configuration
# NOTE: Changed from hardcoded values (e.g., CHUNK_SIZE = 1024) to environment variable pattern
# (e.g., CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))) in Phase 1.
# 
# Why this change:
# - Hardcoded: Fixed values in code, requires code edit to change settings
# - Environment variable: Read from .env file or system environment, allows runtime configuration
# 
# Benefits:
# - Different settings for different hardware (laptop vs. EVO-X2) without code changes
# - Easy experimentation during benchmarking (change .env, no code restart needed)
# - Deployment flexibility (dev/staging/prod configs via environment, not code)
# - Same codebase works across all environments
#
# Example: On old laptop use CHUNK_SIZE=1024, on new hardware use CHUNK_SIZE=2048
# Just edit .env file, don't touch config.py
# OLD VALUES (for lighter machines):
# CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))   # Increased for better context
# CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))   # Increased overlap for continuity
# TOP_K = int(os.getenv("TOP_K", "5"))    # Number of chunks to retrieve
# SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))    # Minimum similarity score
# OLD: CHUNK_SIZE default was 2048
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3072"))  # Larger chunks for 131K context
# OLD: CHUNK_OVERLAP default was 256
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "384"))  # More overlap for continuity
# OLD: TOP_K default was 15
TOP_K = int(os.getenv("TOP_K", "25"))  # More candidates with 131K context window
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))  # Slightly higher threshold

# Advanced RAG settings (for large context models)
# OLD VALUES (for lighter machines):
# MAX_CHUNKS_IN_CONTEXT = int(os.getenv("MAX_CHUNKS_IN_CONTEXT", "10"))   # Max chunks to include in prompt
# USE_RERANKING = os.getenv("USE_RERANKING", "false").lower() == "true"
# RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))   # Re-rank top N results
# OLD: MAX_CHUNKS_IN_CONTEXT default was 20
MAX_CHUNKS_IN_CONTEXT = int(os.getenv("MAX_CHUNKS_IN_CONTEXT", "40"))  # More context in 131K window
USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() == "true"  # Enable reranking
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "8"))  # Re-rank top 8 of 15 retrieved

# ChromaDB optimization for large corpus
# OLD VALUES (for lighter machines - these settings did not exist before):
# CHROMA_BATCH_SIZE = 1000
# CHROMA_PERSIST_INTERVAL = 500
CHROMA_BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", "5000"))  # Batch insert size
CHROMA_PERSIST_INTERVAL = int(os.getenv("CHROMA_PERSIST_INTERVAL", "1000"))  # Persist every N docs

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DOCS_DIR = os.path.join(DATA_DIR, "raw")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vectordb")
COLLECTION_NAME = "phase0_docs"
TRACKING_DB_PATH = os.path.join(DATA_DIR, "tracking.db")

# Supported document formats (via document_loaders.py):
# .txt, .md, .pdf, .docx, .csv, .json, .html, .htm, .xlsx, .xls

# Ensure directories exist
os.makedirs(RAW_DOCS_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Tracking database for conversation history (commented out) - old method
# TRACKING_DB_PATH = "data/tracking.db"

# Old Configurations
# LLM_MODEL = "llama3.1:8b"

# Old RAG Configuration
#CHUNK_SIZE = 512
#CHUNK_OVERLAP = 50
#TOP_K = 5  

# New Model Ideas
# For Ryzen AI with good VRAM: qwen2.5:32b-instruct -> qwen2.5:14b-instruct
# For CPU-only or limited RAM: llama3.2:3b-instruct -> phi3:3.8b-instruct
# Other Model 3: gemma2:2b-instruct
# Other Model 4: qwen2.5:7b-instruct (good balance of size and quality for fallback)
# Other Model 5: ???
# Other Model 6: ???
# Other Model 7: ???
# Other Model 8: ???
# Other Model 9: ???
# Other Model 10: 

# Embedding Model Ideas
# qwen3-embedding:8b
# embeddinggemma:300m
# bge-m3:567m (or :latest)
# nomic-embed-text


# TODO: ollama pull qwen3-embedding:8b
# TODO: ollama pull qwen2.5:32b-instruct