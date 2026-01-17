# RAG Platform Deployment Checklist
**Status**: ✅ READY FOR PRODUCTION (after fixes applied)

## Pre-Deployment Verification

### 1. Environment Setup
```bash
# Ensure Ollama is running
ollama serve

# Verify models are available
ollama list | grep qwen2.5:32b-instruct
ollama list | grep llama3:latest
ollama list | grep qwen3-embedding:8b

# If missing, pull models
ollama pull qwen2.5:32b-instruct
ollama pull llama3:latest
ollama pull qwen3-embedding:8b
```

### 2. Directory Structure
```bash
# Verify data directories exist (auto-created by config.py)
ls -la data/raw/
ls -la data/vectordb/
ls -la data/tracking.db  # Created on first ingest
```

### 3. Test Ingestion Pipeline
```bash
# Place a test document in data/raw/
echo "This is a test document for the RAG system." > data/raw/test.txt

# Run ingestion
python src/ingest.py

# Expected output:
# - "Loading: test.txt"
# - "Created X chunks from 1 documents"
# - "✓ test.txt: X chunks"
```

### 4. Test Query Pipeline
```bash
# Interactive mode
python src/query.py

# At prompt, enter:
# "What is in the test document?"

# Expected:
# - Retrieval filtering stats displayed
# - Answer generated
# - Sources shown with scores

# Single query mode
python src/query.py "What is in the test document?"
```

### 5. Test Database Manager
```bash
# View statistics
python src/db_manager.py --stats

# Expected:
# - Total documents: 1
# - Total chunks: X
# - First/last ingestion timestamps

# List documents
python src/db_manager.py --list

# Expected:
# - test.txt entry with chunk count
```

### 6. Test Benchmark
```bash
# Run Tier 1 benchmark
python src/benchmarkv2.py --tier1

# Expected:
# - Query latency metrics
# - Component timing breakdown
# - Hardware utilization stats
```

### 7. Test Web API (if using app.py)
```bash
# Start server
python src/app.py

# In another terminal, test endpoints
curl http://localhost:5000/

# Test query endpoint
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is in the test document?"}'

# Expected: JSON response with answer and sources
```

## Production Deployment Steps

### 1. Configuration
```bash
# Create .env file with production settings
cat > .env << EOF
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:32b-instruct
LLM_FALLBACK=llama3:latest
EMBED_MODEL=qwen3-embedding:8b

# LLM Settings
LLM_TEMPERATURE=0.1
LLM_CONTEXT_WINDOW=8192
LLM_REQUEST_TIMEOUT=180.0
LLM_NUM_PREDICT=512

# Hardware (adjust for your system)
USE_ROCM=false
USE_NPU=false
GPU_LAYERS=0
NUM_GPU=1

# RAG Configuration
CHUNK_SIZE=1024
CHUNK_OVERLAP=128
TOP_K=5
SIMILARITY_THRESHOLD=0.3
MAX_CHUNKS_IN_CONTEXT=10

# Logging (for web API)
RAG_LOG_LEVEL=INFO
RAG_LOG_QUESTIONS=true
EOF
```

### 2. Security Hardening
```bash
# Restrict permissions on data directory
chmod 700 data/
chmod 600 data/tracking.db

# Restrict permissions on logs (if using web API)
chmod 700 logs/
chmod 600 logs/*.log
```

### 3. Initial Document Ingestion
```bash
# Place production documents in data/raw/
# Run full ingestion
python src/ingest.py

# Verify with database manager
python src/db_manager.py --stats
python src/db_manager.py --list
```

### 4. Validation Queries
```bash
# Test with representative queries
python src/query.py "Your test question 1"
python src/query.py "Your test question 2"
python src/query.py "Your test question 3"

# Verify:
# - Answers are relevant
# - Sources are accurate
# - Response times acceptable
```

### 5. Benchmark Production Performance
```bash
# Run full benchmark suite
python src/benchmarkv2.py --all

# Review results in benchmarks/ directory
# Verify performance meets requirements
```

## Monitoring & Maintenance

### Daily Checks
- Monitor log files for errors (if using web API)
- Check disk space in data/vectordb/
- Verify Ollama service is running

### Weekly Checks
- Review query logs for common patterns
- Check for new documents to ingest
- Verify tracking database integrity: `python src/db_manager.py --verify`

### Monthly Tasks
- Backup data/ directory (tracking.db + vectordb/)
- Review and archive old logs
- Update models if new versions available
- Run benchmark suite to track performance trends

### Database Backup
```bash
# Create backup directory
mkdir -p backups/$(date +%Y-%m-%d)

# Backup tracking database
cp data/tracking.db backups/$(date +%Y-%m-%d)/

# Backup vector database
cp -r data/vectordb/ backups/$(date +%Y-%m-%d)/

# Create tarball
tar -czf backups/backup-$(date +%Y-%m-%d).tar.gz \
  backups/$(date +%Y-%m-%d)/
```

### Database Restore
```bash
# Stop any running processes
# Extract backup
tar -xzf backups/backup-YYYY-MM-DD.tar.gz

# Restore tracking database
cp backups/YYYY-MM-DD/tracking.db data/

# Restore vector database
rm -rf data/vectordb/
cp -r backups/YYYY-MM-DD/vectordb/ data/
```

## Troubleshooting

### Issue: Ollama Connection Failed
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama
ollama serve

# Verify connectivity
curl http://localhost:11434/api/tags
```

### Issue: Model Not Found
```bash
# List available models
ollama list

# Pull missing model
ollama pull qwen2.5:32b-instruct
```

### Issue: Out of Memory During Query
**Symptoms**: Status code 500, "system memory" error

**Solution**: System will automatically fall back to smaller model (LLM_FALLBACK)

**Prevention**: 
- Reduce LLM_CONTEXT_WINDOW
- Reduce TOP_K (fewer chunks retrieved)
- Use smaller model as primary

### Issue: Slow Query Performance
**Diagnosis**:
```bash
# Run benchmark to identify bottleneck
python src/benchmarkv2.py --tier1
```

**Common causes**:
- Embedding generation slow → GPU offloading: set GPU_LAYERS > 0
- Retrieval slow → Too many documents: check SIMILARITY_THRESHOLD
- Generation slow → Model too large: use smaller model or increase GPU_LAYERS

### Issue: Irrelevant Answers
**Diagnosis**: Review source chunks with scores

**Solutions**:
- Increase SIMILARITY_THRESHOLD (0.3 → 0.5)
- Reduce TOP_K (retrieve fewer chunks)
- Check document quality in data/raw/
- Review chunking strategy (CHUNK_SIZE, CHUNK_OVERLAP)

### Issue: Missing Context in Answers
**Solutions**:
- Decrease SIMILARITY_THRESHOLD (retrieve more chunks)
- Increase TOP_K (retrieve more chunks)
- Increase CHUNK_SIZE (larger chunks with more context)
- Increase CHUNK_OVERLAP (better continuity between chunks)

## Performance Tuning

### For Speed (Lower Latency)
```env
LLM_MODEL=llama3:latest  # Use smaller, faster model
TOP_K=3                   # Retrieve fewer chunks
LLM_CONTEXT_WINDOW=4096  # Smaller context window
GPU_LAYERS=32            # Offload more to GPU
```

### For Quality (Better Answers)
```env
LLM_MODEL=qwen2.5:32b-instruct  # Use larger, smarter model
TOP_K=8                          # Retrieve more context
LLM_CONTEXT_WINDOW=8192         # Larger context window
CHUNK_SIZE=1536                  # Larger chunks
CHUNK_OVERLAP=256                # More overlap
SIMILARITY_THRESHOLD=0.4         # More selective
```

### For Memory Constrained Systems
```env
LLM_MODEL=llama3.2:3b-instruct  # Small model
EMBED_MODEL=nomic-embed-text     # Smaller embedding model
TOP_K=3                          # Fewer chunks
LLM_CONTEXT_WINDOW=2048          # Minimal context
CHUNK_SIZE=512                   # Smaller chunks
```

## Production Readiness Checklist

- [x] Config typo fixed (RERANK_TOP_N)
- [x] Critical bug fixed (chunk counting)
- [x] All imports validated
- [x] Error handling reviewed
- [x] Custom modules integrated correctly
- [x] Benchmarks match production pipeline
- [ ] Test documents ingested successfully
- [ ] Test queries return correct results
- [ ] Database manager shows accurate stats
- [ ] Benchmark performance acceptable
- [ ] Backup strategy implemented
- [ ] Monitoring plan in place
- [ ] Documentation reviewed by team
- [ ] Security permissions set

**When all boxes checked**: ✅ APPROVED FOR PRODUCTION DEPLOYMENT
