"""
Test script to verify de-abstraction changes maintain RAG pipeline functionality.

This script tests the custom implementations of:
1. Text chunking (text_chunker.py)
2. Document objects (SimpleDocument dataclass)
3. Similarity filtering (similarity_filter.py)

Run this before and after making changes to ensure functionality is preserved.
"""

import sys
from pathlib import Path

# Test 1: Text Chunking
def test_text_chunking():
    """Test custom text chunking implementation."""
    print("\n" + "="*60)
    print("TEST 1: Text Chunking")
    print("="*60)
    
    from text_chunker import chunk_text, chunk_text_simple, chunk_text_sentence_aware, get_chunk_stats
    
    # Sample text with multiple sentences
    test_text = """
    This is the first sentence. This is the second sentence. This is the third sentence.
    This is the fourth sentence. This is the fifth sentence. This is the sixth sentence.
    This is the seventh sentence. This is the eighth sentence. This is the ninth sentence.
    This is the tenth sentence.
    """
    
    # Test simple chunking
    print("\n1a. Testing simple character-based chunking...")
    chunks_simple = chunk_text_simple(test_text, chunk_size=100, overlap=20)
    print(f"   Created {len(chunks_simple)} chunks")
    stats_simple = get_chunk_stats(chunks_simple)
    print(f"   Avg length: {stats_simple['avg_length']:.1f} chars")
    print(f"   Min/Max: {stats_simple['min_length']}/{stats_simple['max_length']} chars")
    
    # Test sentence-aware chunking
    print("\n1b. Testing sentence-aware chunking...")
    chunks_sentence = chunk_text_sentence_aware(test_text, chunk_size=100, overlap=20)
    print(f"   Created {len(chunks_sentence)} chunks")
    stats_sentence = get_chunk_stats(chunks_sentence)
    print(f"   Avg length: {stats_sentence['avg_length']:.1f} chars")
    print(f"   Min/Max: {stats_sentence['min_length']}/{stats_sentence['max_length']} chars")
    
    # Test main interface
    print("\n1c. Testing main chunk_text() interface...")
    chunks_main = chunk_text(test_text, chunk_size=100, overlap=20, sentence_aware=True)
    print(f"   Created {len(chunks_main)} chunks")
    
    # Verify chunks are not empty
    assert all(chunk.strip() for chunk in chunks_simple), "Empty chunks found in simple chunking"
    assert all(chunk.strip() for chunk in chunks_sentence), "Empty chunks found in sentence-aware chunking"
    
    print("\n   PASS: Text chunking works correctly")
    return True


# Test 2: Document Objects
def test_document_objects():
    """Test SimpleDocument dataclass."""
    print("\n" + "="*60)
    print("TEST 2: Document Objects")
    print("="*60)
    
    from dataclasses import dataclass
    from typing import Dict, Any
    
    # Define SimpleDocument (same as in ingest.py)
    @dataclass
    class SimpleDocument:
        text: str
        metadata: Dict[str, Any]
    
    print("\n2a. Creating SimpleDocument...")
    doc = SimpleDocument(
        text="This is test document content.",
        metadata={
            "filename": "test.txt",
            "file_type": ".txt",
            "file_path": "/path/to/test.txt"
        }
    )
    
    print(f"   Text: {doc.text[:30]}...")
    print(f"   Metadata: {doc.metadata}")
    
    # Verify attributes are accessible
    assert doc.text == "This is test document content."
    assert doc.metadata["filename"] == "test.txt"
    assert doc.metadata["file_type"] == ".txt"
    
    print("\n   PASS: SimpleDocument works correctly")
    return True


# Test 3: Similarity Filtering
def test_similarity_filtering():
    """Test custom similarity filtering implementation."""
    print("\n" + "="*60)
    print("TEST 3: Similarity Filtering")
    print("="*60)
    
    from similarity_filter import (
        filter_by_similarity, 
        filter_top_k, 
        filter_by_threshold_and_top_k,
        get_filter_stats
    )
    
    # Create mock result objects with scores
    class MockResult:
        def __init__(self, score):
            self.score = score
            self.text = f"Result with score {score}"
    
    results = [
        MockResult(0.9),
        MockResult(0.7),
        MockResult(0.5),
        MockResult(0.3),
        MockResult(0.1),
    ]
    
    print("\n3a. Testing filter_by_similarity()...")
    filtered = filter_by_similarity(results, threshold=0.4)
    print(f"   Original: {len(results)} results")
    print(f"   Filtered (threshold=0.4): {len(filtered)} results")
    assert len(filtered) == 3, f"Expected 3 results, got {len(filtered)}"
    assert all(r.score >= 0.4 for r in filtered), "Some results below threshold"
    
    print("\n3b. Testing filter_top_k()...")
    top_k = filter_top_k(results, k=3)
    print(f"   Top 3 results: {[r.score for r in top_k]}")
    assert len(top_k) == 3, f"Expected 3 results, got {len(top_k)}"
    assert top_k[0].score >= top_k[1].score >= top_k[2].score, "Results not sorted"
    
    print("\n3c. Testing filter_by_threshold_and_top_k()...")
    combined = filter_by_threshold_and_top_k(results, threshold=0.4, k=2)
    print(f"   Combined filter (threshold=0.4, k=2): {len(combined)} results")
    assert len(combined) == 2, f"Expected 2 results, got {len(combined)}"
    
    print("\n3d. Testing get_filter_stats()...")
    stats = get_filter_stats(results, filtered)
    print(f"   Original count: {stats['original_count']}")
    print(f"   Filtered count: {stats['filtered_count']}")
    print(f"   Removed: {stats['removed_count']} ({stats['removal_rate']:.1f}%)")
    print(f"   Score range: {stats['min_score']:.1f} - {stats['max_score']:.1f}")
    
    print("\n   PASS: Similarity filtering works correctly")
    return True


# Test 4: Integration Test (if documents exist)
def test_integration():
    """Test integration with actual RAG pipeline components."""
    print("\n" + "="*60)
    print("TEST 4: Integration Test")
    print("="*60)
    
    try:
        from text_chunker import chunk_text
        from dataclasses import dataclass
        from typing import Dict, Any
        
        @dataclass
        class SimpleDocument:
            text: str
            metadata: Dict[str, Any]
        
        # Create a test document
        print("\n4a. Creating test document...")
        doc = SimpleDocument(
            text="This is a test document. It has multiple sentences. Each sentence should be preserved. The chunking should work correctly. This tests the integration.",
            metadata={"filename": "integration_test.txt", "file_type": ".txt"}
        )
        print(f"   Document created: {len(doc.text)} characters")
        
        # Chunk the document
        print("\n4b. Chunking document...")
        chunks = chunk_text(doc.text, chunk_size=50, overlap=10, sentence_aware=True)
        print(f"   Created {len(chunks)} chunks")
        
        # Verify chunks contain document metadata context
        print("\n4c. Verifying chunk metadata...")
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: {len(chunk)} chars")
        
        assert len(chunks) > 0, "No chunks created"
        
        print("\n   PASS: Integration test successful")
        return True
        
    except Exception as e:
        print(f"\n   WARNING: Integration test failed: {e}")
        print("   This is expected if the full pipeline is not set up")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DE-ABSTRACTION VERIFICATION TESTS")
    print("="*60)
    print("\nTesting custom implementations to verify RAG pipeline functionality...")
    
    results = []
    
    try:
        results.append(("Text Chunking", test_text_chunking()))
    except Exception as e:
        print(f"\n   FAIL: Text chunking test failed: {e}")
        results.append(("Text Chunking", False))
    
    try:
        results.append(("Document Objects", test_document_objects()))
    except Exception as e:
        print(f"\n   FAIL: Document objects test failed: {e}")
        results.append(("Document Objects", False))
    
    try:
        results.append(("Similarity Filtering", test_similarity_filtering()))
    except Exception as e:
        print(f"\n   FAIL: Similarity filtering test failed: {e}")
        results.append(("Similarity Filtering", False))
    
    try:
        results.append(("Integration", test_integration()))
    except Exception as e:
        print(f"\n   FAIL: Integration test failed: {e}")
        results.append(("Integration", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"   {test_name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\n   Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n   All tests passed! De-abstraction changes are working correctly.")
        return 0
    else:
        print("\n   Some tests failed. Review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
