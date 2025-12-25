# config.py
import os

# Hardware Configuration
USE_ROCM = os.getenv("USE_ROCM", "false").lower() == "true"
USE_NPU = os.getenv("USE_NPU", "false").lower() == "true"
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "0"))  # Number of layers to offload to GPU (0 = auto)
NUM_GPU = int(os.getenv("NUM_GPU", "1"))  # Number of GPUs to use

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")        # Primary LLM
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "deepseek-r1:latest")  # Fallback LLM (requires more RAM)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Model-specific settings for large models (70B-120B)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))  # Default 8K, can go up to 128K
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "180.0"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "512"))  # Max tokens to generate

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
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))  # Increased for better context
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))  # Increased overlap for continuity
TOP_K = int(os.getenv("TOP_K", "5"))  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))  # Minimum similarity score

# Advanced RAG settings (for large context models)
MAX_CHUNKS_IN_CONTEXT = int(os.getenv("MAX_CHUNKS_IN_CONTEXT", "10"))  # Max chunks to include in prompt
USE_RERANKING = os.getenv("USE_RERANKING", "false").lower() == "true"
RERRANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))  # Re-rank top N results

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