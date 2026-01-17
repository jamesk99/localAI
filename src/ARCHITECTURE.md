# LocalAI RAG Platform Architecture

## Overview
Professional-grade, transparent RAG (Retrieval-Augmented Generation) platform designed for local AI deployment with emphasis on **data sovereignty**, **security**, and **complete customizability**.

## Design Philosophy

### Why Custom Implementation Over Standard LlamaIndex?

This codebase intentionally **replaces LlamaIndex abstractions** with transparent, pure Python implementations. This is **not over-engineering** - it's a deliberate architecture choice for:

1. **Data Sovereignty & Security**
   - All processing happens locally - no external API calls
   - Full visibility into every step of data handling
   - No hidden telemetry or data leakage
   - Complete control over sensitive document processing

2. **Transparency & Auditability**
   - Every line of code is visible and understandable
   - No black-box components that hide implementation details
   - Easy to audit for security compliance
   - Clear data flow from documents → chunks → embeddings → retrieval

3. **Customization & Optimization**
   - Fine-grained control over chunking strategy
   - Adjustable similarity filtering logic
   - Hardware-specific optimizations (ROCm, NPU, GPU layers)
   - Tunable for specific use cases without framework constraints

4. **Performance & Benchmarking**
   - Accurate performance measurement of each component
   - No hidden overhead from framework abstractions
   - Isolatable bottlenecks for optimization
   - Reproducible benchmarks that match production exactly

5. **Enterprise Requirements**
   - Extensible architecture for custom requirements
   - No vendor lock-in to framework decisions
   - Maintainable by teams without deep framework knowledge
   - Professional-grade error handling and logging

## Architecture Components

### Core Custom Modules

#### 1. `text_chunker.py` (235 lines)
**Purpose**: Replace LlamaIndex's `SentenceSplitter` with transparent, controllable chunking.

**Key Features**:
- Pure Python string manipulation (no NLP library dependencies)
- Two strategies: simple character-based, sentence-aware
- Configurable chunk size and overlap
- Edge case handling (empty docs, small files)
- Statistics generation for optimization

**Why Custom**:
- LlamaIndex's SentenceSplitter uses hidden tokenization
- Cannot inspect or modify chunking logic
- Hard to debug when chunks are suboptimal
- Custom implementation allows:
  - Code-aware chunking (preserve code blocks)
  - Domain-specific splitting (technical docs, legal text)
  - A/B testing different strategies
  - Chunk quality validation

#### 2. `similarity_filter.py` (166 lines)
**Purpose**: Replace LlamaIndex's `SimilarityPostprocessor` with transparent filtering.

**Key Features**:
- Simple threshold-based filtering
- Top-K selection with scoring
- Combined threshold + top-K filtering
- Detailed statistics generation

**Why Custom**:
- LlamaIndex's postprocessor filtering is opaque
- Cannot customize filtering logic (e.g., adaptive thresholds)
- Hard to A/B test different cutoff strategies
- Custom implementation enables:
  - Context-aware threshold adjustment
  - Domain-specific relevance scoring
  - Hybrid filtering strategies
  - Detailed filtering analytics for optimization

#### 3. `custom_retriever.py` (138 lines)
**Purpose**: Apply similarity filtering at **retrieval time** instead of postprocessing.

**Critical RAG Quality Improvement**:
```
Standard LlamaIndex Flow:
  Query → Retrieve 10 chunks → Filter to 5 → LLM sees 5 chunks
  Problem: LLM context already polluted during postprocessing

Custom FilteredRetriever Flow:
  Query → Retrieve 10 chunks → Filter to 5 BEFORE LLM → LLM sees only 5 chunks
  Benefit: LLM never sees low-quality chunks, better answer quality
```

**Why This Matters**:
- Prevents irrelevant context from confusing the LLM
- Reduces hallucinations from weak matches
- Optimizes limited context window usage
- Enables verbose filtering statistics for debugging

### Configuration System

#### `config.py` - Environment-Driven Configuration

**Hardware Abstraction**:
```python
USE_ROCM = os.getenv("USE_ROCM", "false").lower() == "true"
USE_NPU = os.getenv("USE_NPU", "false").lower() == "true"
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "0"))
NUM_GPU = int(os.getenv("NUM_GPU", "1"))
```

**LLM Control**:
```python
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "180.0"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "512"))
```

**RAG Tuning**:
```python
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
MAX_CHUNKS_IN_CONTEXT = int(os.getenv("MAX_CHUNKS_IN_CONTEXT", "10"))
```

**Why Environment Variables**:
- Different configs for different hardware (laptop vs server)
- Easy A/B testing (change .env, no code restart)
- Deployment flexibility (dev/staging/prod via environment)
- Same codebase across all environments
- Docker/container-friendly
- Secrets management (API keys in environment, not code)

### Document Processing Pipeline

#### `ingest.py` - Custom Document Processing

**Uses Custom Components**:
1. `SimpleDocument` dataclass (replaces LlamaIndex Document)
2. `text_chunker.chunk_text()` (replaces SentenceSplitter)
3. Manual `TextNode` creation with full metadata control

**Flow**:
```
Raw Documents → Load → Custom Chunk → Create Nodes → Embed → Store
                ↓
        DocumentTracker (SQLite)
         - Prevents re-ingestion
         - Tracks chunk counts
         - Audit trail
```

**Professional Features**:
- Incremental ingestion (skip already processed)
- Accurate chunk tracking via ChromaDB query
- Multiple document format support
- Graceful error handling per document
- Detailed progress logging

#### `query.py` - Production Query Engine

**Uses Custom Components**:
1. `FilteredRetriever` with retrieval-time filtering
2. Custom prompt template for structured responses
3. LLM fallback on OOM errors
4. Detailed source attribution

**Query Flow**:
```
Question
  ↓
Embed Query
  ↓
FilteredRetriever
  ├─ Vector Search (top-K)
  └─ Similarity Filter (threshold)
  ↓
Response Synthesizer (custom prompt)
  ↓
Answer + Sources
```

## Benchmarking System

### `benchmarkv2.py` - Professional Performance Testing

**Critical Fix Applied**: Now uses **SAME** custom components as production:
- `FilteredRetriever` (not standard VectorIndexRetriever)
- Empty postprocessors (filtering done at retrieval)
- Same prompt template as production
- **Benchmarks now accurately test production pipeline**

**5-Tier Benchmark Suite**:

**Tier 1: Infrastructure Metrics**
- Component-level timing (embedding, retrieval, generation)
- Hardware monitoring (CPU, RAM, GPU utilization)
- Token counting and throughput
- Accurate because components match production exactly

**Tier 2: RAGAS Quality Metrics**
- Faithfulness (answer accuracy to context)
- Answer relevancy (answer relevance to question)
- Context precision (retrieved chunks quality)
- Context recall (retrieved chunks completeness)

**Tier 3: Retrieval Effectiveness**
- Precision@K, Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Validates custom retriever performance

**Tier 4: Multi-User Load Testing**
- Concurrent query handling
- Throughput under load
- Resource contention analysis

**Tier 5: Scale Testing**
- Large context window stress tests
- Corpus scaling analysis
- Memory usage profiling

## Professional-Grade Features

### 1. Document Tracking (`document_tracker.py`)
- SQLite-based tracking database
- Prevents duplicate ingestion
- Audit trail for compliance
- Chunk count accuracy

### 2. Database Management (`db_manager.py`)
- CLI tool for database inspection
- Integrity verification
- New document detection
- Document removal (with safeguards)

### 3. Document Loaders (`document_loaders.py`)
- Extensible loader architecture
- Supports: PDF, DOCX, CSV, JSON, HTML, XLSX, TXT, MD
- Graceful degradation on errors
- Easy to add new formats

### 4. Error Handling
- LLM fallback on OOM errors
- Per-document error isolation
- Graceful degradation
- Detailed error logging

### 5. Security & Privacy
- All processing 100% local
- No external API dependencies (uses Ollama)
- No telemetry or tracking
- Complete data sovereignty

## Testing & Validation

### Why This Architecture Enables Better Testing

1. **Component Isolation**: Each custom module can be unit tested independently
2. **No Black Boxes**: Every operation is visible and testable
3. **Reproducible Results**: No framework version dependencies causing variance
4. **Performance Attribution**: Know exactly which component is slow
5. **A/B Testing**: Easy to swap implementations and compare

### Benchmark Accuracy

**Before Fix** (INVALID):
```python
# benchmarkv2.py was using:
retriever = VectorIndexRetriever(...)
postprocessors = [SimilarityPostprocessor(...)]
# This tested a DIFFERENT pipeline than production!
```

**After Fix** (VALID):
```python
# benchmarkv2.py now uses:
retriever = create_filtered_retriever(...)
postprocessors = []  # Filtering at retrieval time
# This tests the EXACT SAME pipeline as production!
```

## Deployment Considerations

### Hardware Flexibility
- ROCm support for AMD GPUs
- NPU support for AI accelerators
- Multi-GPU support
- CPU fallback for constrained environments

### Model Selection
- Primary: `qwen2.5:32b-instruct` (high quality)
- Fallback: `llama3:latest` (memory constrained)
- Embedding: `qwen3-embedding:8b` (high dimensional)
- All configurable via environment

### Scalability
- ChromaDB for vector storage (production-ready)
- Incremental ingestion (process new docs only)
- Efficient similarity search
- Configurable context window (8K-128K tokens)

## Comparison: Custom vs Standard LlamaIndex

| Aspect | Standard LlamaIndex | This Implementation |
|--------|-------------------|---------------------|
| **Transparency** | Black-box abstractions | Fully visible code |
| **Customization** | Limited to framework options | Complete control |
| **Debugging** | Hard to trace issues | Easy to debug |
| **Testing** | Framework-dependent | Independent |
| **Performance** | Hidden overhead | Measurable, optimizable |
| **Security Audit** | Framework complexity | Simple, auditable |
| **Vendor Lock-in** | High | None |
| **Learning Curve** | Steep (framework docs) | Simple (Python code) |
| **Enterprise Ready** | Depends on version | Stable, maintainable |

## Use Cases Requiring This Architecture

1. **Healthcare/Medical**: HIPAA compliance, audit trails, data sovereignty
2. **Legal**: Document confidentiality, audit requirements
3. **Financial**: Regulatory compliance, security audits
4. **Government**: Air-gapped deployment, security clearance
5. **Research**: Reproducible results, custom algorithms
6. **Enterprise**: No vendor lock-in, long-term maintainability

## Conclusion

This is **not over-engineering** - it's a **professional-grade architecture** for users who need:
- Complete control over their data
- Transparent, auditable AI systems
- Customizable RAG pipelines
- Accurate performance benchmarking
- Long-term maintainability

The custom components (539 lines total) provide **essential capabilities** that abstracted frameworks cannot offer. For local AI platforms focused on **data sovereignty and security**, this architecture is the **correct approach**.
