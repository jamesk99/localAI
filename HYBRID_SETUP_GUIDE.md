# Hybrid Setup: Ollama + LM Studio for RAG

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        YOUR RAG PIPELINE                             │
│                                                                      │
│   Document Ingestion          Query Processing                       │
│         │                           │                                │
│         ▼                           ▼                                │
│   ┌──────────┐               ┌──────────────┐                       │
│   │ ChromaDB │◄──────────────│   Retriever  │                       │
│   │ (vectors)│               └──────┬───────┘                       │
│   └────┬─────┘                      │                               │
│        │                            ▼                               │
│        │                    ┌──────────────┐                        │
│        │                    │   LLM Gen    │                        │
│        │                    └──────┬───────┘                        │
│        │                           │                                │
│        ▼                           ▼                                │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                     INFERENCE LAYER                          │  │
│   ├─────────────────────────┬───────────────────────────────────┤  │
│   │     OLLAMA (:11434)     │       LM STUDIO (:1234)           │  │
│   │                         │                                    │  │
│   │  ✓ Embeddings (bge-m3)  │  ✓ LLM Inference (qwen3-30b)      │  │
│   │  ✓ Fallback LLM         │  ✓ Better GPU management          │  │
│   │                         │  ✓ Visual model loading           │  │
│   └─────────────────────────┴───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Why This Split?

LM Studio has a critical limitation: **it can only serve ONE type of model at a time** (either LLM OR embedding model, not both). Ollama doesn't have this limitation.

**Solution**: Use both simultaneously on different ports:
- **Ollama (port 11434)**: Handles embeddings (required) + fallback LLM
- **LM Studio (port 1234)**: Handles primary LLM inference

## Setup Steps

### 1. Install Dependencies

```bash
pip install llama-index-llms-openai-like
```

### 2. Start Ollama (Embeddings Server)

```bash
# Terminal 1: Start Ollama
ollama serve

# Pull embedding model (if not already done)
ollama pull bge-m3

# Optional: Pull fallback LLM
ollama pull llama3:latest
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

### 3. Start LM Studio (LLM Server)

1. **Open LM Studio**
2. **Download model**: Go to "Discover" tab → Search "qwen3-30b-a3b-2507" → Download
3. **Load model**: Go to "Developer" tab → Select the downloaded model → Click "Load"
4. **Start server**: In "Developer" tab → "Local Server" section → Click "Start Server"

Verify LM Studio is running:
```bash
curl http://localhost:1234/v1/models
```

Expected response:
```json
{
  "data": [
    {
      "id": "qwen3-30b-a3b-2507",
      "object": "model",
      ...
    }
  ]
}
```

### 4. Configure Your RAG System

Copy the example config:
```bash
cp .env.example .env
```

Edit `.env`:
```bash
# Use LM Studio for LLM
LLM_BACKEND=lmstudio
LMSTUDIO_LLM_MODEL=qwen3-30b-a3b-2507

# Ollama for embeddings (always)
EMBED_MODEL=bge-m3
```

### 5. Test the Setup

```bash
# Ingest documents (uses Ollama for embeddings)
python src/ingest.py

# Query (uses LM Studio for LLM, Ollama for query embeddings)
python src/query.py "What documents do you have?"
```

## Switching Between Backends

To switch back to Ollama-only mode:

```bash
# In .env
LLM_BACKEND=ollama
OLLAMA_LLM_MODEL=qwen3-arx
```

No code changes needed—just change the config.

## Troubleshooting

### "Connection refused" on port 1234
- LM Studio server not started
- Fix: Open LM Studio → Developer tab → Start Server

### "Connection refused" on port 11434
- Ollama not running
- Fix: Run `ollama serve` in terminal

### "Model not found" in LM Studio
- Model not loaded in LM Studio GUI
- Fix: Developer tab → Select model → Load

### Slow responses from LM Studio
- Model might be running on CPU
- Fix: Check LM Studio settings → Enable GPU acceleration

## Model Equivalents

| Ollama Model | LM Studio Equivalent | Notes |
|--------------|---------------------|-------|
| `bge-m3` | `lm-kit/bge-m3-gguf` | Keep on Ollama (embedding) |
| `qwen3:30b` | `qwen3-30b-a3b-2507` | General purpose |
| `qwen3:30b` | `qwen3-30b-a3b-thinking-2507` | Reasoning mode |
| `qwen3:30b` | `qwen3-coder-30b` | Code specialized |

## Port Reference

| Service | Default Port | Purpose |
|---------|-------------|---------|
| Ollama | 11434 | Embeddings + Fallback LLM |
| LM Studio | 1234 | Primary LLM inference |
| Flask App | 8000 | Web UI |
