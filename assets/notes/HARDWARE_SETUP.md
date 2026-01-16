# Hardware Setup Guide - Phase 1

This guide covers setting up the GMKtec EVO-X2 AI Mini PC for optimal RAG system performance.

## Prerequisites

- GMKtec EVO-X2 AI Mini PC (AMD Ryzen AI Max+ 395)
- Windows 11
- Python 3.13.1
- Ollama installed

## Phase 1: Hardware Optimization Setup

### 1. ROCm Installation on Windows

ROCm (Radeon Open Compute) enables GPU acceleration for AMD hardware.

**Installation Steps:**

1. **Check Windows Version:**
   - Ensure you're running Windows 11 (ROCm support added late 2025)
   - Open PowerShell and run: `winver`

2. **Install AMD ROCm for Windows:**
   ```powershell
   # Download ROCm installer from AMD
   # https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
   
   # Install ROCm SDK
   # Follow AMD's installation wizard
   ```

3. **Verify ROCm Installation:**
   ```powershell
   # Check ROCm version
   rocm-smi
   
   # Should show Radeon 8060S iGPU details
   ```

4. **Configure Ollama for ROCm:**
   ```powershell
   # Set environment variable for ROCm
   $env:OLLAMA_GPU_DRIVER="rocm"
   
   # Restart Ollama service
   Stop-Service Ollama
   Start-Service Ollama
   ```

### 2. Ollama Configuration for Large Models

**Pull Large Models (70B-120B):**

```powershell
# Recommended models for testing
ollama pull qwen2.5:72b-instruct-q4_K_M    # Qwen 2.5 72B (Q4 quantization)
ollama pull llama3.1:70b-instruct-q4_K_M   # Llama 3.1 70B (Q4 quantization)
ollama pull deepseek-coder-v2:236b-q2_K    # DeepSeek V2 (Q2 for testing)

# Alternative embedding models
ollama pull bge-large                       # Better embeddings than nomic
ollama pull mxbai-embed-large              # Alternative embedding model
```

**Test Model Performance:**

```powershell
# Test inference speed
ollama run qwen2.5:72b-instruct-q4_K_M "Explain quantum computing in 3 sentences"

# Monitor GPU usage during inference
rocm-smi --showuse
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
# Hardware Settings
USE_ROCM=true
USE_NPU=false  # NPU support coming in future phase
GPU_LAYERS=0   # 0 = auto-detect, or specify number of layers
NUM_GPU=1

# Model Configuration
LLM_MODEL=qwen2.5:72b-instruct-q4_K_M
LLM_FALLBACK=llama3.1:70b-instruct-q4_K_M
EMBED_MODEL=bge-large

# Context Window (test with increasing values)
LLM_CONTEXT_WINDOW=32768  # Start with 32K, can go up to 128K
LLM_NUM_PREDICT=1024      # Max tokens to generate
LLM_TEMPERATURE=0.1
LLM_REQUEST_TIMEOUT=300.0

# Advanced RAG
TOP_K=10
MAX_CHUNKS_IN_CONTEXT=15
SIMILARITY_THRESHOLD=0.3
```

### 4. Performance Mode

The GMKtec EVO-X2 has a physical performance mode button:

1. **Locate the performance button** on the device
2. **Press to enable sustained high TDP mode** before benchmarking
3. **Monitor temperatures** during extended use

### 5. Memory Optimization

With 64-128GB unified memory:

**Ollama Memory Settings:**

```powershell
# Set Ollama to use more memory for context
$env:OLLAMA_MAX_LOADED_MODELS=2
$env:OLLAMA_NUM_PARALLEL=3  # Support 3-5 concurrent users

# Restart Ollama
Restart-Service Ollama
```

### 6. Benchmark Your Setup

Run the benchmark script to establish baseline performance:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run full benchmark suite
python src/benchmark.py --all

# Results saved to benchmarks/ directory
```

**Expected Performance Targets:**

- **Inference Speed:** 40-80+ tokens/sec (vs. 10-20 on old hardware)
- **Context Window:** 32K-128K tokens (vs. 4K-8K)
- **Concurrent Users:** 3-5 simultaneous queries
- **Model Size:** 70B-120B quantized (vs. 7B-13B)

## Troubleshooting

### ROCm Not Detecting GPU

```powershell
# Check GPU visibility
rocm-smi

# If not visible, reinstall AMD drivers
# Download latest from AMD website
```

### Ollama Out of Memory

```powershell
# Reduce context window
$env:LLM_CONTEXT_WINDOW=16384

# Or use smaller quantization
ollama pull qwen2.5:72b-instruct-q2_K  # Q2 instead of Q4
```

### Slow Inference Despite ROCm

```powershell
# Verify ROCm is being used
$env:OLLAMA_DEBUG=1
ollama run qwen2.5:72b-instruct-q4_K_M "test"

# Check logs for GPU offload confirmation
```

## Next Steps

After Phase 1 setup:

1. **Run benchmarks** and compare to old hardware
2. **Test with real documents** in your domain
3. **Tune context window** for optimal performance
4. **Move to Phase 2:** Advanced RAG techniques (hybrid search, reranking)

## Performance Monitoring

Monitor system during operation:

```powershell
# GPU usage
rocm-smi --showuse

# Memory usage
Get-Process ollama | Select-Object WorkingSet64

# CPU usage
Get-Counter '\Processor(_Total)\% Processor Time'
```

## References

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [Ollama GPU Configuration](https://github.com/ollama/ollama/blob/main/docs/gpu.md)
- [LlamaIndex Performance Tuning](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
