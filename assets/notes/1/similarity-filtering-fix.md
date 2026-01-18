# Similarity Filtering Fix - Critical RAG Quality Improvement

## The Problem

The initial de-abstraction implementation had a critical flaw in how similarity filtering was applied.

### Original (Incorrect) Implementation

```
Query Flow:
1. Retriever gets top-K chunks (e.g., 5 chunks with scores: 0.9, 0.7, 0.5, 0.2, 0.1)
2. ALL 5 chunks sent to LLM (including low-quality 0.2 and 0.1)
3. LLM generates answer based on ALL chunks (good + bad)
4. Filter applied at display time (cosmetic only)
5. User sees filtered sources, but answer already contaminated
```

**Why this was wrong:**
- LLM saw irrelevant chunks (scores 0.2, 0.1) when generating the answer
- Low-similarity chunks confused the LLM and diluted signal
- Increased risk of hallucinations from weak matches
- Wasted context window tokens on junk
- Filtering at display time was cosmetic - didn't improve answer quality

### Corrected Implementation

```
Query Flow:
1. Retriever gets top-K chunks (e.g., 5 chunks with scores: 0.9, 0.7, 0.5, 0.2, 0.1)
2. FilteredRetriever applies threshold (0.3) BEFORE LLM
3. Only high-quality chunks sent to LLM (scores: 0.9, 0.7, 0.5)
4. LLM generates answer based on FILTERED chunks only
5. User sees sources that actually influenced the answer
```

**Why this is correct:**
- LLM only sees relevant, high-quality chunks
- No irrelevant context to confuse the model
- Reduced hallucinations from weak matches
- Efficient use of context window
- Answer quality improved at the source

## Implementation Details

### New File: `custom_retriever.py`

Created a custom retriever wrapper that applies similarity filtering at retrieval time:

```python
class FilteredRetriever(BaseRetriever):
    """
    Wraps a base retriever and filters results BEFORE they reach the LLM.
    
    Critical for RAG quality:
    - Filters out low-similarity chunks before LLM sees them
    - Prevents irrelevant context from confusing the LLM
    - Reduces hallucinations caused by weak matches
    """
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Get initial results
        nodes = self._base_retriever.retrieve(query_bundle)
        
        # Apply similarity filtering BEFORE returning to LLM
        filtered_nodes = filter_by_similarity(nodes, self._similarity_threshold)
        
        return filtered_nodes  # Only high-quality chunks reach LLM
```

### Modified: `query.py`

Updated query engine creation to use filtered retriever:

```python
# Create base retriever
base_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=effective_top_k,
)

# Wrap with filtering at retrieval time
retriever = create_filtered_retriever(
    base_retriever=base_retriever,
    similarity_threshold=SIMILARITY_THRESHOLD,
    verbose=True  # Show filtering stats
)

# Create query engine with filtered retriever
query_engine = RetrieverQueryEngine(
    retriever=retriever,  # Uses filtered retriever
    response_synthesizer=response_synthesizer,
)
```

## RAG Quality Improvements

### Before Fix (Display-Time Filtering)

**Example scenario:**
- Query: "What is the capital of France?"
- Retrieved chunks:
  - Chunk 1 (score 0.9): "Paris is the capital of France..."
  - Chunk 2 (score 0.7): "France is a country in Europe..."
  - Chunk 3 (score 0.2): "The weather in London is rainy..." (irrelevant)

**What happened:**
- LLM saw all 3 chunks including irrelevant London weather
- Answer might mention London or get confused
- Filtering at display only hid the bad chunk from user, didn't help LLM

### After Fix (Retrieval-Time Filtering)

**Same scenario with threshold 0.3:**
- Query: "What is the capital of France?"
- Retrieved chunks:
  - Chunk 1 (score 0.9): "Paris is the capital of France..."
  - Chunk 2 (score 0.7): "France is a country in Europe..."
  - Chunk 3 (score 0.2): FILTERED OUT before LLM

**What happens:**
- LLM only sees relevant chunks about France
- Answer is focused and accurate
- No confusion from irrelevant context

## Transparency Features

The FilteredRetriever includes verbose logging:

```
[Retrieval Filter] Retrieved 5 chunks, filtered to 3 (removed 2 below threshold 0.3)
[Retrieval Filter] Score range: 0.500 - 0.900
```

This shows:
- How many chunks were retrieved initially
- How many passed the filter
- Score distribution of filtered chunks

## Configuration

Similarity threshold is configurable via `.env`:

```bash
SIMILARITY_THRESHOLD=0.3  # Adjust based on your needs
```

**Tuning guidance:**
- Lower threshold (0.2): More permissive, retrieves more chunks (risk: noise)
- Higher threshold (0.5): More strict, only high-quality chunks (risk: miss relevant info)
- Recommended: 0.3 (good balance)

## Why This Matters for RAG Systems

This fix addresses a fundamental principle of RAG:

**Garbage In, Garbage Out**

If you feed the LLM irrelevant context:
- It will try to use that context (even if weak)
- It may hallucinate connections between unrelated chunks
- It wastes precious context window tokens
- Answer quality degrades

**The fix ensures:**
- Only high-quality context reaches the LLM
- LLM focuses on relevant information
- Better answers with fewer hallucinations
- Efficient use of context window

## Comparison to LlamaIndex SimilarityPostprocessor

LlamaIndex's `SimilarityPostprocessor` actually does apply filtering at the right time (retrieval time), but:

1. It's a black box - you don't see exactly how it works
2. Our custom implementation is transparent
3. We have full control over the filtering logic
4. We can add custom logging and statistics
5. Easier to extend with additional filtering criteria

Both approaches filter at retrieval time, but our implementation gives us transparency and control.

## Summary

**Critical fix:** Moved similarity filtering from display time to retrieval time.

**Impact:** LLM now only sees high-quality, relevant chunks, significantly improving answer quality.

**Implementation:** Custom `FilteredRetriever` wrapper that filters before chunks reach LLM.

**Transparency:** Verbose logging shows filtering statistics for debugging.

This is a fundamental improvement to RAG quality - ensuring the LLM works with clean, relevant context rather than noisy, irrelevant chunks.
