"""
Pure Python text chunking implementation.

This module provides transparent, low-level text chunking functionality
to replace LlamaIndex's SentenceSplitter abstraction. The implementation
gives direct control over how text is split into chunks for embedding.

Key features:
- Character-based chunking with configurable size and overlap
- Sentence-aware splitting (respects sentence boundaries when possible)
- No hidden abstractions - all logic is visible and modifiable
- Compatible with existing RAG pipeline

Design decisions:
- Uses simple string manipulation instead of complex NLP libraries
- Preserves context through overlapping chunks
- Handles edge cases (empty text, small documents, etc.)
"""

import re
from typing import List


def chunk_text_simple(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks using simple character-based splitting.
    
    This is the most transparent chunking method - pure string slicing with overlap.
    Use this when you want predictable, consistent chunk sizes.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
        
    Example:
        text = "This is a test. This is only a test."
        chunks = chunk_text_simple(text, chunk_size=20, overlap=5)
        # chunks[0]: "This is a test. This"
        # chunks[1]: "This is only a test."
    """
    if not text or not text.strip():
        return []
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate end position for this chunk
        end = start + chunk_size
        
        # Extract chunk
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)
        
        # Move start position forward (accounting for overlap)
        start += chunk_size - overlap
        
        # Prevent infinite loop if we're at the end
        if start >= text_length:
            break
    
    return chunks


def chunk_text_sentence_aware(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks with sentence boundary awareness.
    
    This method tries to break at sentence boundaries when possible, making
    chunks more semantically coherent. Falls back to character-based splitting
    if sentences are too long.
    
    Algorithm:
    1. Split text into sentences using basic punctuation rules
    2. Build chunks by adding complete sentences until near chunk_size
    3. Add overlap by including sentences from previous chunk
    
    Args:
        text: The text to chunk
        chunk_size: Target characters per chunk (approximate)
        overlap: Number of characters to overlap between chunks (approximate)
        
    Returns:
        List of text chunks
        
    Note:
        Actual chunk sizes may vary slightly to preserve sentence boundaries.
        If a single sentence exceeds chunk_size, it will be kept intact.
    """
    if not text or not text.strip():
        return []
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    
    # Split into sentences using basic punctuation rules
    # This regex looks for sentence-ending punctuation followed by whitespace
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed chunk_size, finalize current chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            # Join sentences in current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            # Include sentences from end of previous chunk to create overlap
            overlap_sentences = []
            overlap_length = 0
            
            # Work backwards through current_chunk to build overlap
            for prev_sentence in reversed(current_chunk):
                if overlap_length + len(prev_sentence) <= overlap:
                    overlap_sentences.insert(0, prev_sentence)
                    overlap_length += len(prev_sentence)
                else:
                    break
            
            # Reset current chunk with overlap sentences
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        # Add current sentence to chunk
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add final chunk if it has content
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int, 
               sentence_aware: bool = True) -> List[str]:
    """
    Main chunking function with configurable strategy.
    
    This is the primary interface for text chunking. It provides a choice
    between simple character-based chunking and sentence-aware chunking.
    
    Args:
        text: The text to chunk
        chunk_size: Target characters per chunk
        overlap: Number of characters to overlap between chunks
        sentence_aware: If True, try to break at sentence boundaries (default: True)
        
    Returns:
        List of text chunks
        
    Usage:
        # Sentence-aware chunking (default, better for semantic coherence)
        chunks = chunk_text(document_text, chunk_size=1024, overlap=128)
        
        # Simple character-based chunking (predictable sizes)
        chunks = chunk_text(document_text, chunk_size=1024, overlap=128, 
                          sentence_aware=False)
    """
    if sentence_aware:
        return chunk_text_sentence_aware(text, chunk_size, overlap)
    else:
        return chunk_text_simple(text, chunk_size, overlap)


def get_chunk_stats(chunks: List[str]) -> dict:
    """
    Calculate statistics about a list of chunks.
    
    Useful for debugging and optimizing chunk parameters.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Dictionary with statistics:
        - num_chunks: Total number of chunks
        - avg_length: Average chunk length in characters
        - min_length: Shortest chunk length
        - max_length: Longest chunk length
        - total_chars: Total characters across all chunks
    """
    if not chunks:
        return {
            'num_chunks': 0,
            'avg_length': 0,
            'min_length': 0,
            'max_length': 0,
            'total_chars': 0
        }
    
    lengths = [len(chunk) for chunk in chunks]
    
    return {
        'num_chunks': len(chunks),
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'total_chars': sum(lengths)
    }
