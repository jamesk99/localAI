# Benchmark Test Query Sets

This directory contains test query JSON files for the enhanced RAG benchmark suite (`src/benchmarkv2.py`).

## File Structure

| File | Tier | Description |
|------|------|-------------|
| `infrastructure_test_set.json` | Tier 1 | Infrastructure metrics testing (latency, throughput, hardware) |
| `ragas_test_set.json` | Tier 2 | RAGAS quality evaluation with ground truth answers |
| `retrieval_test_set.json` | Tier 3 | Retrieval effectiveness with relevance labels |

---

## Query File Formats

### Infrastructure Test Set (Tier 1)

```json
{
  "metadata": {
    "version": "1.0",
    "description": "Description of the test set"
  },
  "infrastructure_tests": [
    {
      "id": "inf_001",
      "query": "Your test question here",
      "type": "factual|summary|multi_hop|analytical",
      "expected_tokens": 150,
      "complexity": "low|medium|high"
    }
  ]
}
```

**Query Types:**
- **factual**: Direct fact-based questions with concise answers
- **summary**: Questions requiring aggregation across documents
- **multi_hop**: Questions requiring reasoning across multiple concepts
- **analytical**: Questions requiring analysis and inference

---

### RAGAS Test Set (Tier 2)

```json
{
  "metadata": {
    "version": "1.0",
    "description": "RAGAS quality evaluation queries"
  },
  "ragas_tests": [
    {
      "id": "ragas_001",
      "question": "Your test question here",
      "ground_truth": "The expected correct answer based on your corpus",
      "type": "factual|conceptual|analytical"
    }
  ]
}
```

**Important:** Ground truth answers must reflect the actual content in your document corpus for accurate RAGAS scoring.

---

### Retrieval Test Set (Tier 3)

```json
{
  "metadata": {
    "version": "1.0",
    "description": "Retrieval effectiveness queries"
  },
  "retrieval_tests": [
    {
      "id": "ret_001",
      "query": "Your test question here",
      "relevant_doc_ids": ["doc_123", "doc_456"],
      "relevant_chunk_keywords": ["keyword1", "keyword2", "keyword3"]
    }
  ]
}
```

**Relevance Labeling:**
- `relevant_doc_ids`: Document IDs that should be retrieved (optional, update based on your corpus)
- `relevant_chunk_keywords`: Keywords that indicate a chunk is relevant (at least 2 matches = relevant)

---

## Adding New Queries

### Best Practices

1. **Diverse Query Types**: Include a mix of factual, summary, and analytical questions
2. **Corpus-Specific**: Tailor questions to your actual document content
3. **Ground Truth Accuracy**: For RAGAS tests, ensure ground truth matches your corpus
4. **Keyword Coverage**: For retrieval tests, use distinctive keywords that appear in relevant chunks

### Creating Custom Test Sets

1. Copy an existing template file
2. Update the `metadata` section
3. Add your queries following the format
4. Run the benchmark with `--queries` flag:

```bash
python src/benchmarkv2.py --tier1 --queries tests/queries/my_custom_tests.json
```

---

## Updating for Your Corpus

The default queries are generic AI/ML questions. To optimize for your specific document corpus:

1. **Review your documents** to identify key topics and concepts
2. **Update ground truth answers** in `ragas_test_set.json` to match your corpus
3. **Add relevant keywords** to `retrieval_test_set.json` based on your documents
4. **Create domain-specific queries** that test your actual use cases

---

## Query Count Recommendations

| Tier | Minimum | Recommended | Purpose |
|------|---------|-------------|---------|
| Tier 1 | 10 | 20-50 | Latency/throughput variance |
| Tier 2 | 20 | 50-100 | Statistically significant RAGAS |
| Tier 3 | 20 | 50+ | Retrieval precision analysis |

---

## Validation

Before running benchmarks, validate your query files:

```python
import json

with open("tests/queries/infrastructure_test_set.json") as f:
    data = json.load(f)
    print(f"Loaded {len(data['infrastructure_tests'])} infrastructure queries")

with open("tests/queries/ragas_test_set.json") as f:
    data = json.load(f)
    print(f"Loaded {len(data['ragas_tests'])} RAGAS queries")

with open("tests/queries/retrieval_test_set.json") as f:
    data = json.load(f)
    print(f"Loaded {len(data['retrieval_tests'])} retrieval queries")
```
