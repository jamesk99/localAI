# RAG Platform Functional Audit
**Date**: January 17, 2026  
**Auditor**: Chief Software Engineer Review  
**Status**: CRITICAL ISSUES FOUND

## Executive Summary

Platform has **1 CRITICAL BUG** and several warnings that need attention before production deployment.

---

## CRITICAL ISSUES

### ❌ CRITICAL #1: Chunk Count Logic Bug in ingest.py

**Location**: `src/ingest.py:254-264`

**Problem**: Chunk counting queries ALL documents in ChromaDB, not just newly ingested ones.

**Current Code**:
```python
# Lines 254-259
all_items = chroma_collection.get(include=['metadatas'])
if all_items and all_items['metadatas']:
    for metadata in all_items['metadatas']:
        source_file = metadata.get('file_path')
        if source_file:
            doc_chunk_counts[source_file] = doc_chunk_counts.get(source_file, 0) + 1
```

**Bug**: This counts ALL chunks for ALL documents ever ingested, not just the NEW documents being processed in this run.

**Impact**: 
- If you re-run ingestion with new documents, the tracking database will show inflated chunk counts
- Example: 
  - First run: ingest doc1.pdf → 10 chunks → tracked as 10 ✓
  - Second run: ingest doc2.pdf → 5 chunks
  - Bug: doc2.pdf tracked as **15 chunks** (10 from doc1 + 5 from doc2) ❌

**Severity**: HIGH - Data integrity issue in tracking database

**Fix Required**: YES - See fixes section below

---

## CONFIGURATION ISSUES

### ✅ FIXED: Typo in config.py
- **Issue**: `RERRANK_TOP_N` → `RERANK_TOP_N`
- **Status**: FIXED
- **Impact**: Variable was defined but never used (USE_RERANKING feature not implemented)

---

## WARNINGS

### ⚠️ WARNING #1: Unused Configuration Variables

**Location**: `src/config.py:45-46`

```python
USE_RERANKING = os.getenv("USE_RERANKING", "false").lower() == "true"
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))
```

**Issue**: These variables are defined but never used in the codebase.

**Recommendation**: 
- Document that reranking is planned but not implemented
- OR remove these variables to avoid confusion

---

### ⚠️ WARNING #2: GPU Configuration Not Validated

**Location**: `src/config.py:5-8`

```python
USE_ROCM = os.getenv("USE_ROCM", "false").lower() == "true"
USE_NPU = os.getenv("USE_NPU", "false").lower() == "true"
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "0"))
NUM_GPU = int(os.getenv("NUM_GPU", "1"))
```

**Issue**: Variables are defined but:
- No validation that hardware actually exists
- No fallback if GPU unavailable
- Not clear how these affect Ollama model loading

**Recommendation**: 
- Add hardware detection and validation
- Document how these settings interact with Ollama

---

### ⚠️ WARNING #3: No Validation for Ollama Connection

**Location**: `src/ingest.py:167-177`, `src/query.py:30-110`

**Issue**: Both modules assume Ollama is running and accessible at `OLLAMA_BASE_URL`

**Current Behavior**:
- `ingest.py`: Tries to connect, prints error, returns early ✓ (acceptable)
- `query.py`: Tries primary model, falls back to secondary, then raises ✓ (acceptable)

**Recommendation**: 
- Current error handling is adequate
- Consider adding a health check endpoint

---

## INTEGRATION VALIDATION

### ✅ Module Dependencies - ALL CORRECT

**Custom Modules**:
- `custom_retriever.py` → imported by `query.py`, `benchmarkv2.py` ✓
- `similarity_filter.py` → imported by `custom_retriever.py` ✓  
- `text_chunker.py` → imported by `ingest.py` ✓

**Standard Dependencies**:
- `document_tracker.py` → imported by `ingest.py`, `db_manager.py` ✓
- `document_loaders.py` → imported by `ingest.py`, `db_manager.py` ✓
- `config.py` → imported by all modules ✓

**Conclusion**: All imports are valid, no missing modules.

---

### ✅ Query Engine Integration - CORRECT

**Flow**: `query.py` → `create_query_engine()` → `FilteredRetriever` → `ResponseSynthesizer`

**Validation**:
1. ✅ Uses `VectorIndexRetriever` as base
2. ✅ Wraps with `create_filtered_retriever()`
3. ✅ Applies custom prompt template
4. ✅ Empty postprocessors (filtering done at retrieval)
5. ✅ LLM fallback on OOM errors

**Conclusion**: Query engine correctly uses custom components.

---

### ✅ Benchmark Integration - FIXED

**Status**: benchmarkv2.py now matches production pipeline

**Validation**:
1. ✅ Imports `create_filtered_retriever`
2. ✅ Uses same retriever as `query.py`
3. ✅ Empty postprocessors (matches production)
4. ✅ Same prompt template as production

**Conclusion**: Benchmarks will accurately test production system.

---

## ERROR HANDLING REVIEW

### ✅ Document Loading - ADEQUATE

**Location**: `src/ingest.py:73-118`

```python
for file_path in files:
    if tracker.is_document_ingested(file_path):
        print(f"     Skipping (already ingested): {file_path.name}")
        skipped_count += 1
        continue
    
    try:
        # ... load document ...
    except Exception as e:
        print(f"    Error loading {file_path.name}: {e}")
```

**Validation**:
- ✅ Per-document error isolation (one bad file doesn't stop ingestion)
- ✅ Skip already-ingested files
- ✅ Reports errors clearly

---

### ✅ LLM Initialization - ADEQUATE

**Location**: `src/query.py:42-75`

**Validation**:
- ✅ Try primary model first
- ✅ Fall back to secondary model
- ✅ Raise if both fail (appropriate)
- ✅ Clear error messages

---

### ✅ Query Execution - ADEQUATE

**Location**: `src/query.py:282-309`

**Validation**:
- ✅ Try query with primary model
- ✅ Detect OOM errors
- ✅ Fall back to smaller model
- ✅ Re-raise other exceptions (appropriate)

---

## FIXES REQUIRED

### FIX #1: Chunk Count Bug in ingest.py

**Current problematic code** (`lines 250-264`):
```python
doc_chunk_counts = {}
try:
    # Query ChromaDB for all items and count by source file
    all_items = chroma_collection.get(include=['metadatas'])
    if all_items and all_items['metadatas']:
        for metadata in all_items['metadatas']:
            source_file = metadata.get('file_path')
            if source_file:
                doc_chunk_counts[source_file] = doc_chunk_counts.get(source_file, 0) + 1
except Exception as e:
    print(f"     Could not get chunk counts from ChromaDB: {e}")
    # Fallback to estimation if ChromaDB query fails
    for doc in documents:
        doc_chunk_counts[str(Path(doc.metadata['file_path']))] = len(doc.text) // CHUNK_SIZE
```

**CORRECT FIX** - Option 1 (Simpler, Recommended):
```python
# Don't query ChromaDB at all - use the nodes we just created
doc_chunk_counts = {}
for doc in documents:
    file_path = str(Path(doc.metadata['file_path']))
    # Count nodes we created for this document
    doc_chunk_count = sum(1 for node in all_nodes 
                          if node.metadata.get('file_path') == file_path)
    doc_chunk_counts[file_path] = doc_chunk_count
```

**CORRECT FIX** - Option 2 (Query only new documents):
```python
doc_chunk_counts = {}
for doc in documents:
    file_path_str = str(Path(doc.metadata['file_path']))
    try:
        # Query ChromaDB for THIS SPECIFIC document only
        result = chroma_collection.get(
            where={"file_path": file_path_str},
            include=['metadatas']
        )
        if result and result['metadatas']:
            doc_chunk_counts[file_path_str] = len(result['metadatas'])
    except Exception as e:
        print(f"     Could not get chunk count for {Path(file_path_str).name}: {e}")
        # Fallback: count from all_nodes
        doc_chunk_counts[file_path_str] = sum(
            1 for node in all_nodes 
            if node.metadata.get('file_path') == file_path_str
        )
```

**Recommendation**: Use Option 1 (simpler, more reliable, no database query needed).

---

## FUNCTIONALITY CHECKLIST

### Core Ingestion Pipeline
- ✅ Document loading (multiple formats)
- ✅ Document tracking (prevents duplicates)
- ✅ Custom chunking (text_chunker.py)
- ✅ Embedding generation (Ollama)
- ✅ Vector storage (ChromaDB)
- ❌ **Chunk counting (HAS BUG - see Fix #1)**
- ✅ Error handling (per-document isolation)

### Core Query Pipeline
- ✅ Index loading from ChromaDB
- ✅ Query embedding generation
- ✅ Vector retrieval (base retriever)
- ✅ Similarity filtering (FilteredRetriever)
- ✅ LLM generation (with fallback)
- ✅ Response formatting with sources
- ✅ Error handling (OOM detection)

### Custom Components
- ✅ FilteredRetriever (retrieval-time filtering)
- ✅ SimilarityFilter (pure Python filtering)
- ✅ TextChunker (sentence-aware chunking)
- ✅ SimpleDocument (transparent document class)
- ✅ DocumentTracker (SQLite tracking)

### Benchmarking
- ✅ benchmarkv2.py uses production components
- ✅ Component-level timing
- ✅ Hardware monitoring support
- ✅ 5-tier benchmark suite

### Web API
- ✅ Flask app integration
- ✅ Query endpoint
- ✅ Logging configuration
- ✅ CORS support (if enabled)

---

## DEPLOYMENT READINESS

### Must Fix Before Production
1. ❌ **Fix chunk counting bug in ingest.py** (CRITICAL)

### Should Address Before Production
1. ⚠️ Document or remove unused reranking config
2. ⚠️ Add hardware validation for GPU settings
3. ⚠️ Create .env.example file with all config options

### Nice to Have
1. Add unit tests for custom modules
2. Add integration test suite
3. Add health check endpoint for web API
4. Add Ollama connectivity check at startup

---

## SMOKE TEST PROCEDURE

Before deploying, run this manual smoke test:

### 1. Test Ingestion
```bash
# Place a test document in data/raw/
python src/ingest.py
# Expected: Document ingested successfully, chunk count displayed
```

### 2. Test Query
```bash
python src/query.py "What is this document about?"
# Expected: Answer generated with sources displayed
```

### 3. Test Database Manager
```bash
python src/db_manager.py --stats
# Expected: Statistics displayed correctly
```

### 4. Test Benchmark
```bash
python src/benchmarkv2.py --tier1
# Expected: Benchmark runs, metrics displayed
```

### 5. Test Web API (if using app.py)
```bash
python src/app.py
# In another terminal:
curl http://localhost:5000/health
curl -X POST http://localhost:5000/query -H "Content-Type: application/json" -d '{"question":"test"}'
# Expected: Endpoints respond correctly
```

---

## FINAL VERDICT

**Status**: ❌ **NOT READY FOR PRODUCTION**

**Reason**: Critical bug in chunk counting logic

**Required Actions**:
1. Apply Fix #1 (chunk counting)
2. Run smoke tests
3. Verify chunk counts are accurate
4. Then: ✅ READY FOR PRODUCTION

**Architecture Quality**: ⭐⭐⭐⭐⭐ (5/5)
- Excellent custom implementation
- Professional error handling  
- Good separation of concerns
- Transparent, auditable code

**Code Quality**: ⭐⭐⭐⭐ (4/5)
- Well-documented
- Clean structure
- One critical bug (chunk counting)
- Some unused config variables

**Production Readiness**: ⭐⭐⭐ (3/5 - pending critical fix)
- All core functionality works
- Error handling adequate
- Needs chunk count fix before deployment
- Missing some nice-to-have features (tests, health checks)

---

## RECOMMENDATIONS

### Immediate (Before Any Production Use)
1. **Fix chunk counting bug** (see Fix #1)
2. Test with multiple ingestion runs
3. Verify tracking database accuracy

### Short Term (Within 1 Week)
1. Create .env.example with all configuration options
2. Add Ollama connectivity validation at startup
3. Document GPU configuration behavior
4. Remove or implement reranking features

### Long Term (Within 1 Month)
1. Add unit tests for custom modules
2. Add integration test suite
3. Add monitoring/alerting for production
4. Consider adding health check endpoint

---

**Audit Completed**: Platform is well-architected but has one critical bug that must be fixed before production deployment.
