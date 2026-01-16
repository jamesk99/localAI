"""
Custom retriever with pure Python similarity filtering at retrieval time.

This module implements a retriever wrapper that applies similarity filtering
BEFORE chunks are sent to the LLM, ensuring only high-quality, relevant
context is used for answer generation.

Critical for RAG quality:
- Filters out low-similarity chunks before LLM sees them
- Prevents irrelevant context from confusing the LLM
- Reduces hallucinations caused by weak matches
- Optimizes context window usage
"""

from typing import List, Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from similarity_filter import filter_by_similarity


class FilteredRetriever(BaseRetriever):
    """
    Custom retriever that applies similarity filtering at retrieval time.
    
    This wrapper takes a base retriever and applies similarity threshold filtering
    to the results BEFORE they are passed to the LLM. This is critical for RAG
    quality because:
    
    1. LLM only sees high-quality, relevant chunks
    2. Low-similarity chunks don't pollute the context
    3. Reduces hallucinations from weak matches
    4. Better use of limited context window
    
    The filtering happens at retrieval time (not display time), which means:
    - Filtered chunks never reach the LLM
    - Answer quality is improved at the source
    - More transparent than postprocessors
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        similarity_threshold: float = 0.3,
        verbose: bool = False
    ):
        """
        Initialize the filtered retriever.
        
        Args:
            base_retriever: The underlying retriever (e.g., VectorIndexRetriever)
            similarity_threshold: Minimum similarity score (0.0-1.0)
            verbose: If True, print filtering statistics
        """
        self._base_retriever = base_retriever
        self._similarity_threshold = similarity_threshold
        self._verbose = verbose
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes and apply similarity filtering.
        
        This is the core method that:
        1. Calls the base retriever to get initial results
        2. Applies similarity threshold filtering
        3. Returns only high-quality chunks to the LLM
        
        Args:
            query_bundle: The query to retrieve for
            
        Returns:
            Filtered list of NodeWithScore objects
        """
        # Get initial retrieval results from base retriever
        nodes = self._base_retriever.retrieve(query_bundle)
        
        if not nodes:
            return []
        
        # Apply similarity filtering using our pure Python implementation
        original_count = len(nodes)
        filtered_nodes = filter_by_similarity(nodes, self._similarity_threshold)
        filtered_count = len(filtered_nodes)
        
        # Log filtering statistics if verbose
        if self._verbose and filtered_count < original_count:
            removed_count = original_count - filtered_count
            print(f"   [Retrieval Filter] Retrieved {original_count} chunks, "
                  f"filtered to {filtered_count} (removed {removed_count} below threshold {self._similarity_threshold})")
            
            # Show score distribution for transparency
            if filtered_nodes:
                scores = [n.score for n in filtered_nodes if n.score is not None]
                if scores:
                    print(f"   [Retrieval Filter] Score range: {min(scores):.3f} - {max(scores):.3f}")
        
        return filtered_nodes
    
    @property
    def base_retriever(self) -> BaseRetriever:
        """Access to the underlying base retriever."""
        return self._base_retriever


def create_filtered_retriever(
    base_retriever: BaseRetriever,
    similarity_threshold: float = 0.3,
    verbose: bool = True
) -> FilteredRetriever:
    """
    Factory function to create a filtered retriever.
    
    This is a convenience function that wraps any base retriever with
    similarity filtering at retrieval time.
    
    Args:
        base_retriever: The retriever to wrap
        similarity_threshold: Minimum similarity score
        verbose: Whether to print filtering statistics
        
    Returns:
        FilteredRetriever instance
        
    Example:
        from llama_index.core.retrievers import VectorIndexRetriever
        
        base = VectorIndexRetriever(index=index, similarity_top_k=10)
        filtered = create_filtered_retriever(base, similarity_threshold=0.3)
        
        # Now filtered retriever will only return chunks with score >= 0.3
        # BEFORE they reach the LLM
    """
    return FilteredRetriever(
        base_retriever=base_retriever,
        similarity_threshold=similarity_threshold,
        verbose=verbose
    )
