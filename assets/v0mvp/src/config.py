# config.py
import os

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")        # Primary LLM (needs to be bigger than the fall back)
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "deepseek-r1:latest")  # Fallback LLM (requires more RAM)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

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
# Other Model 10: ???

# RAG Configuration
CHUNK_SIZE = 1024              # Increased for better context
CHUNK_OVERLAP = 128            # Increased overlap for continuity (consider 150-200)
TOP_K = 5                      # Number of chunks to retrieve - change to 8 or 10 on new hardware
SIMILARITY_THRESHOLD = 0.3     # Minimum similarity score (decreased from 0.5, increase again if irrelevant sources become prevalent)

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