import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# =============================================================================
# INFERENCE BACKEND SELECTION
# =============================================================================
# Choose which backend to use for LLM inference: "ollama" or "lmstudio"
# Embeddings ALWAYS use Ollama (LM Studio can't serve both LLM + embeddings)
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()  # "ollama" or "lmstudio"

# =============================================================================
# OLLAMA CONFIGURATION (Embeddings + optional LLM)
# =============================================================================
# Ollama runs on port 11434 by default
# Used for: Embeddings (always), LLM inference (when LLM_BACKEND="ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")  # Embedding model (always via Ollama)

# Ollama-specific LLM settings (used when LLM_BACKEND="ollama")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3-arx")
OLLAMA_LLM_FALLBACK = os.getenv("OLLAMA_LLM_FALLBACK", "llama3:latest")

# =============================================================================
# LM STUDIO CONFIGURATION (LLM inference only)
# =============================================================================
# LM Studio runs on port 1234 by default (OpenAI-compatible API)
# Used for: LLM inference (when LLM_BACKEND="lmstudio")
# NOTE: You must load the model in LM Studio GUI and start the server first!
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")

# LM Studio model identifier (as shown in LM Studio after loading)
# Common Qwen3-30B variants in LM Studio:
#   - "qwen3-30b-a3b-2507" (general purpose, non-thinking)
#   - "qwen3-30b-a3b-thinking-2507" (reasoning/thinking mode)
#   - "qwen3-coder-30b" (coding specialized)
LMSTUDIO_LLM_MODEL = os.getenv("LMSTUDIO_LLM_MODEL", "qwen3-30b-a3b-2507")

# =============================================================================
# UNIFIED LLM SETTINGS (applies to whichever backend is selected)
# =============================================================================
# These are used regardless of which LLM backend is active
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "131072"))  # 131K context
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "300.0"))  # 5min for large models
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "4096"))  # Max tokens to generate

# =============================================================================
# HARDWARE CONFIGURATION
# =============================================================================
USE_ROCM = os.getenv("USE_ROCM", "false").lower() == "true"
USE_NPU = os.getenv("USE_NPU", "false").lower() == "true"
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "999"))  # Offload all layers to GPU
NUM_GPU = int(os.getenv("NUM_GPU", "1"))

# Parallel Processing (leverage 16 Zen 5 cores)
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

# Ollama performance tuning (only applies when LLM_BACKEND="ollama")
OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "16"))
OLLAMA_NUM_BATCH = int(os.getenv("OLLAMA_NUM_BATCH", "1024"))
OLLAMA_MAIN_GPU = int(os.getenv("OLLAMA_MAIN_GPU", "0"))
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "999"))
OLLAMA_GPU_LAYERS = int(os.getenv("OLLAMA_GPU_LAYERS", "999"))

# =============================================================================
# RAG CONFIGURATION
# =============================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3072"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "384"))
TOP_K = int(os.getenv("TOP_K", "25"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# Advanced RAG settings
MAX_CHUNKS_IN_CONTEXT = int(os.getenv("MAX_CHUNKS_IN_CONTEXT", "40"))
USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() == "true"
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "8"))

# ChromaDB optimization
CHROMA_BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", "5000"))
CHROMA_PERSIST_INTERVAL = int(os.getenv("CHROMA_PERSIST_INTERVAL", "1000"))

# =============================================================================
# PATHS
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DOCS_DIR = os.path.join(DATA_DIR, "raw")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vectordb")
COLLECTION_NAME = "phase0_docs"
TRACKING_DB_PATH = os.path.join(DATA_DIR, "tracking.db")

# Ensure directories exist
os.makedirs(RAW_DOCS_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================
# These maintain compatibility with existing code that imports LLM_MODEL, LLM_FALLBACK
# They resolve to the correct model based on which backend is selected
if LLM_BACKEND == "lmstudio":
    LLM_MODEL = LMSTUDIO_LLM_MODEL
    LLM_FALLBACK = OLLAMA_LLM_FALLBACK  # Fallback still uses Ollama
else:
    LLM_MODEL = OLLAMA_LLM_MODEL
    LLM_FALLBACK = OLLAMA_LLM_FALLBACK
