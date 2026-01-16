"""
Pure Python similarity filtering implementation.

This module provides transparent similarity-based filtering to replace
LlamaIndex's SimilarityPostprocessor abstraction. The implementation gives
direct control over how retrieved chunks are filtered based on similarity scores.

Key features:
- Simple threshold-based filtering
- No hidden abstractions - all logic is visible
- Compatible with existing RAG pipeline
- Easy to extend with custom filtering logic

Design decisions:
- Uses simple comparison operations instead of complex postprocessor classes
- Preserves all metadata from original results
- Handles edge cases (empty results, missing scores, etc.)
"""

from typing import List, Dict, Any, Optional


def filter_by_similarity(results: List[Any], threshold: float) -> List[Any]:
    """
    Filter retrieval results by similarity score threshold.
    
    This is a transparent replacement for LlamaIndex's SimilarityPostprocessor.
    It simply filters out any results with a similarity score below the threshold.
    
    Args:
        results: List of result objects with 'score' attribute
        threshold: Minimum similarity score (0.0 to 1.0)
        
    Returns:
        Filtered list of results that meet the threshold
        
    Example:
        results = retriever.retrieve(query)
        filtered = filter_by_similarity(results, threshold=0.3)
        # Only results with score >= 0.3 are kept
    """
    if not results:
        return []
    
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")
    
    filtered = []
    for result in results:
        # Check if result has a score attribute
        if hasattr(result, 'score'):
            if result.score is not None and result.score >= threshold:
                filtered.append(result)
        else:
            # If no score attribute, include the result (fail-safe behavior)
            filtered.append(result)
    
    return filtered


def filter_top_k(results: List[Any], k: int) -> List[Any]:
    """
    Keep only the top K results by similarity score.
    
    Args:
        results: List of result objects with 'score' attribute
        k: Number of top results to keep
        
    Returns:
        Top K results sorted by score (descending)
    """
    if not results:
        return []
    
    if k <= 0:
        return []
    
    # Sort by score in descending order (highest scores first)
    sorted_results = sorted(
        results,
        key=lambda x: x.score if hasattr(x, 'score') and x.score is not None else 0.0,
        reverse=True
    )
    
    return sorted_results[:k]


def filter_by_threshold_and_top_k(results: List[Any], threshold: float, k: int) -> List[Any]:
    """
    Combined filtering: apply similarity threshold AND keep only top K.
    
    This is useful when you want to ensure a minimum quality threshold
    while also limiting the number of results.
    
    Args:
        results: List of result objects with 'score' attribute
        threshold: Minimum similarity score
        k: Maximum number of results to keep
        
    Returns:
        Filtered and limited results
        
    Example:
        # Get top 5 results that have at least 0.3 similarity
        filtered = filter_by_threshold_and_top_k(results, threshold=0.3, k=5)
    """
    # First apply threshold filter
    filtered = filter_by_similarity(results, threshold)
    
    # Then keep only top K
    return filter_top_k(filtered, k)


def get_filter_stats(original_results: List[Any], filtered_results: List[Any]) -> Dict[str, Any]:
    """
    Calculate statistics about filtering operation.
    
    Useful for debugging and understanding how filtering affects results.
    
    Args:
        original_results: Results before filtering
        filtered_results: Results after filtering
        
    Returns:
        Dictionary with statistics:
        - original_count: Number of results before filtering
        - filtered_count: Number of results after filtering
        - removed_count: Number of results removed
        - removal_rate: Percentage of results removed
        - min_score: Minimum score in filtered results
        - max_score: Maximum score in filtered results
        - avg_score: Average score in filtered results
    """
    original_count = len(original_results)
    filtered_count = len(filtered_results)
    removed_count = original_count - filtered_count
    
    stats = {
        'original_count': original_count,
        'filtered_count': filtered_count,
        'removed_count': removed_count,
        'removal_rate': (removed_count / original_count * 100) if original_count > 0 else 0.0
    }
    
    # Calculate score statistics for filtered results
    if filtered_results:
        scores = [
            r.score for r in filtered_results 
            if hasattr(r, 'score') and r.score is not None
        ]
        
        if scores:
            stats['min_score'] = min(scores)
            stats['max_score'] = max(scores)
            stats['avg_score'] = sum(scores) / len(scores)
        else:
            stats['min_score'] = None
            stats['max_score'] = None
            stats['avg_score'] = None
    else:
        stats['min_score'] = None
        stats['max_score'] = None
        stats['avg_score'] = None
    
    return stats
