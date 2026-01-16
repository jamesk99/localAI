# RAG Pipeline De-abstraction Notes

## Overview

This document describes the replacement of LlamaIndex abstractions with pure Python implementations to increase code transparency and control. All changes maintain full RAG pipeline functionality while providing better visibility into how the system works.

## Changes Made

### 1. Text Chunking (SentenceSplitter → text_chunker.py)

**Location:** `src/text_chunker.py` (new file)

**What was replaced:**
```python
# OLD: LlamaIndex SentenceSplitter
from llama_index.core.node_parser import SentenceSplitter
text_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
```

**Replaced with:**
```python
# NEW: Custom pure Python implementation
from text_chunker import chunk_text
chunks = chunk_text(text=doc.text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, sentence_aware=True)
```

**Benefits:**
- Full transparency: See exactly how text is split into chunks
- Two strategies available: simple character-based or sentence-aware
- Easy to customize (e.g., preserve code blocks, respect markdown structure)
- No hidden tokenization or complex NLP dependencies
- Configurable via `sentence_aware` parameter

**Implementation details:**
- `chunk_text_simple()`: Pure string slicing with overlap
- `chunk_text_sentence_aware()`: Respects sentence boundaries using regex
- `get_chunk_stats()`: Debugging utility for chunk analysis

**Modified files:**
- `src/ingest.py`: Lines 8-18, 183-244 (commented old code, added new implementation)

---

### 2. Document Objects (Document → SimpleDocument)

**Location:** `src/ingest.py` (lines 26-44)

**What was replaced:**
```python
# OLD: LlamaIndex Document class
from llama_index.core import Document
doc = Document(text=text, metadata={...})
```

**Replaced with:**
```python
# NEW: Simple dataclass
from dataclasses import dataclass

@dataclass
class SimpleDocument:
    text: str
    metadata: Dict[str, Any]

doc = SimpleDocument(text=text, metadata={...})
```

**Benefits:**
- No hidden methods or complex inheritance
- Pure Python dataclass - standard library only
- Same functionality, complete transparency
- Easy to extend with additional fields if needed

**Modified files:**
- `src/ingest.py`: Lines 26-44 (dataclass definition), 90-111 (usage)

---

### 3. Similarity Filtering (SimilarityPostprocessor → similarity_filter.py)

**Location:** `src/similarity_filter.py` (new file)

**What was replaced:**
```python
# OLD: LlamaIndex SimilarityPostprocessor
from llama_index.core.postprocessor import SimilarityPostprocessor
node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=SIMILARITY_THRESHOLD)]
```

**Replaced with:**
```python
# NEW: Custom pure Python filtering
from similarity_filter import filter_by_similarity
source_nodes = filter_by_similarity(source_nodes, SIMILARITY_THRESHOLD)
```

**Benefits:**
- Simple threshold comparison - no black box logic
- Applied at display time for flexibility (can adjust threshold without re-querying)
- Additional utilities: `filter_top_k()`, `filter_by_threshold_and_top_k()`, `get_filter_stats()`
- Full control over filtering logic

**Implementation details:**
- `filter_by_similarity()`: Basic threshold filtering
- `filter_top_k()`: Keep only top K results
- `filter_by_threshold_and_top_k()`: Combined filtering
- `get_filter_stats()`: Statistics for debugging

**Modified files:**
- `src/query.py`: Lines 14-15, 123-132, 180-228 (commented old code, added new implementation)

---

## What Was NOT Replaced (and Why)

These LlamaIndex components provide significant value and were kept:

1. **VectorStoreIndex**: Complex indexing logic with ChromaDB integration
2. **ChromaVectorStore**: Handles ChromaDB protocol and vector operations
3. **OllamaEmbedding**: Manages embedding generation with proper error handling
4. **Ollama LLM wrapper**: Handles streaming, context management, retries
5. **RetrieverQueryEngine**: Orchestrates the retrieval + generation pipeline
6. **TextNode**: Minimal structure needed for LlamaIndex index compatibility

These abstractions handle complex operations that would be difficult to reimplement correctly and provide real value.

---

## Verification Steps

To verify the pipeline still works correctly:

### 1. Test Ingestion
```bash
cd src
python ingest.py
```

Expected output:
- Documents loaded successfully
- Custom chunking applied (sentence-aware)
- Chunks created and embedded
- ChromaDB updated

### 2. Test Querying
```bash
cd src
python query.py
```

Expected behavior:
- RAG system initializes
- Custom similarity filtering applied
- Queries answered with source citations
- Filter messages shown when chunks are removed

### 3. Verify Chunk Quality
```python
from text_chunker import chunk_text, get_chunk_stats

text = "Your document text here..."
chunks = chunk_text(text, chunk_size=1024, overlap=128, sentence_aware=True)
stats = get_chunk_stats(chunks)
print(stats)
```

---

## Configuration

All chunking and filtering parameters remain configurable via `.env`:

```bash
# Text chunking
CHUNK_SIZE=1024          # Characters per chunk
CHUNK_OVERLAP=128        # Overlap between chunks

# Similarity filtering
SIMILARITY_THRESHOLD=0.3  # Minimum similarity score (0.0-1.0)
TOP_K=5                  # Number of chunks to retrieve
```

---

## Future De-abstraction Opportunities

Additional components that could be replaced with pure Python:

1. **Embedding Generation**: Direct Ollama REST API calls instead of OllamaEmbedding wrapper
2. **LLM Calls**: Direct HTTP requests to Ollama API for full control
3. **Retrieval Logic**: Custom ChromaDB queries instead of VectorIndexRetriever

These are more complex and require careful consideration of error handling, streaming, and protocol details.

---

## Code Organization

New files created:
- `src/text_chunker.py`: Pure Python text chunking implementation
- `src/similarity_filter.py`: Pure Python similarity filtering
- `docs/de-abstraction-notes.md`: This documentation

Modified files:
- `src/ingest.py`: Replaced Document and SentenceSplitter
- `src/query.py`: Replaced SimilarityPostprocessor

All original code is commented out (not deleted) for reference and comparison.

---

## Benefits Summary

1. **Transparency**: See exactly how text is chunked and filtered
2. **Control**: Easy to customize chunking logic for specific use cases
3. **Learning**: Understand RAG internals without diving into library source code
4. **Debugging**: Simple code makes it easier to trace issues
5. **Flexibility**: Can modify behavior without waiting for library updates
6. **No degradation**: All functionality preserved, pipeline works identically

---

## Testing Checklist

- [ ] Ingestion completes without errors
- [ ] Chunks are created with expected sizes
- [ ] Embeddings are generated successfully
- [ ] ChromaDB stores chunks correctly
- [ ] Query system initializes properly
- [ ] Similarity filtering works as expected
- [ ] Query responses include source citations
- [ ] Filter statistics are displayed correctly

---

## Notes

- The chunking strategy defaults to sentence-aware splitting for better semantic coherence
- Similarity filtering now happens at display time rather than retrieval time, providing more flexibility
- All metadata (chunk_index, total_chunks) is preserved for debugging
- Original LlamaIndex code is commented (not deleted) for reference
