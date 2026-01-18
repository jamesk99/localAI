# Tech Specs

## AMD Ryzen AI Max+ 395 (128GB RAM) Optimization Guide

Hardware Profile: AMD Ryzen AI Max+ 395
CPU: 16 Zen 5 cores (32 threads)
iGPU: RDNA 3.5 (40 CUs, ~100GB/s shared memory bandwidth)
NPU: XDNA 2 (50 TOPS)
RAM: 128GB unified memory (GPU can access all of it)
Key Advantage: Massive unified memory allows running 70B+ models that would fail on discrete GPUs with 24GB VRAM

## Setup Commands for EVO-X2

Run these commands on your new machine **in order**:

### Step 1: Pull Required Models

```powershell
# Primary 72B model (~42GB download)
ollama pull qwen2.5:72b-instruct-q4_K_M

# Fallback 32B model (~20GB download)
ollama pull qwen2.5:32b-instruct-q4_K_M

# High-quality 8B embedding model (~5GB download)
ollama pull qwen3-embedding:8b
```

### Step 2: Create Optimized Custom Model

```powershell
cd C:\Users\kelle\Projects\GitHub_Repository\localAI
ollama create qwen72b-rag -f Modelfile.qwen72b-optimized
```

Then update [.env] to use it:

```text
LLM_MODEL=qwen72b-rag
```

### Step 3: Re-ingest Documents (Required)

Since chunk size changed from 1024→2048, you **must** re-ingest:

```powershell
# Delete existing vector DB
Remove-Item -Recurse -Force .\data\vectordb
Remove-Item -Force .\data\tracking.db

# Re-ingest with new chunk settings
python src\ingest.py
```

---

## Key Optimizations Explained

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `LLM_MODEL` | llama3:latest | qwen2.5:72b | 72B fits in 128GB, dramatically better quality |
| `GPU_LAYERS` | 0 | 999 | Offload ALL layers to iGPU (unified memory) |
| `LLM_CONTEXT_WINDOW` | 8192 | 32768 | 4x more context for better RAG |
| `CHUNK_SIZE` | 1024 | 2048 | Larger chunks with bigger context window |
| `TOP_K` | 5 | 15 | Retrieve 3x more candidates |
| `MAX_CHUNKS_IN_CONTEXT` | 10 | 20 | Fit more context in 32K window |
| `EMBED_BATCH_SIZE` | (none) | 64 | Parallel embedding generation |
| `OLLAMA_NUM_THREAD` | (none) | 16 | Match Zen 5 core count |
| `OLLAMA_NUM_BATCH` | (none) | 512 | Large prompt batch processing |

---

## Expected Performance Improvement

| Metric | Before (llama3:8b, 8K ctx) | After (qwen72b, 32K ctx) |
|--------|----------------------------|--------------------------|
| **Throughput** | ~5 tokens/sec | ~15-25 tokens/sec (iGPU accel) |
| **Answer Quality** | Baseline | ~3-4x better (72B vs 8B) |
| **Context Utilization** | 10 chunks max | 20 chunks max |
| **First Token Latency** | ~5s | ~8-12s (larger model) |
| **Embedding Speed** | Serial | 8x faster (batch=64) |

---

### Optional: Environment Variables to Add to [.env]

You can add these for fine-tuning:

```env
# Threading (add to .env if needed)
OLLAMA_NUM_THREAD=16
OLLAMA_NUM_BATCH=512
NUM_WORKERS=8
EMBED_BATCH_SIZE=64
CHROMA_BATCH_SIZE=5000
```

---

## Verification

After setup, run the benchmark to verify improvements:

```powershell
python src\benchmarkv2.py --tier1
```

### About ROCm

**You do NOT need to install ROCm separately on Windows.**

Ollama on Windows uses **DirectML** (not ROCm) for AMD GPU acceleration. It's built into the Ollama installer—no extra setup required. The `USE_ROCM=true` in [.env] is a config flag for our code, not an OS dependency.

ROCm is only needed for:

- Linux systems with AMD GPUs
- Custom PyTorch/TensorFlow builds

### Quick Setup Checklist for EVO-X2

```powershell
# 1. Install Ollama (download from ollama.com)

# 2. Clone/copy repo and install Python deps
cd C:\Users\kelle\Projects\GitHub_Repository\localAI
pip install -r requirements.txt

# 3. Pull models (from specs.md)
ollama pull qwen2.5:72b-instruct-q4_K_M
ollama pull qwen2.5:32b-instruct-q4_K_M  
ollama pull qwen3-embedding:8b

# 4. Verify Ollama sees GPU
ollama run qwen2.5:72b-instruct-q4_K_M "Hello"
# Should show GPU acceleration in Ollama logs
```

### Optional (NPU Acceleration)

If you want to experiment with the XDNA 2 NPU later, you'd need the **AMD NPU Driver** from AMD's website, but that's not required for the current setup—Ollama doesn't use the NPU anyway.
