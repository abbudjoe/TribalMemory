"""Shared embedding utilities for mock implementations.

Provides consistent, deterministic embedding generation for testing.
"""

import hashlib
import math
import random
import re


def hash_to_embedding(text: str, dimensions: int = 1536) -> list[float]:
    """Convert text to deterministic embedding that preserves semantic similarity.
    
    Uses word-level hashing so texts with shared words have similar embeddings.
    Suitable for basic mock testing scenarios.
    
    Args:
        text: Text to embed.
        dimensions: Output embedding dimensions.
        
    Returns:
        Normalized embedding vector.
    """
    embedding = [0.0] * dimensions
    
    def add_term(term: str, weight: float = 1.0):
        """Add a term's contribution to the embedding."""
        term_hash = hashlib.sha256(term.encode()).digest()
        random.seed(int.from_bytes(term_hash[:8], 'big'))
        for i in range(dimensions):
            embedding[i] += random.gauss(0, 1) * weight
    
    # Normalize text
    text_lower = text.lower()
    
    # Add contribution for each unique word (skip very short words)
    words = set(re.findall(r'\b\w+\b', text_lower))
    for word in words:
        if len(word) > 2:
            add_term(word, 1.0)
    
    # Add short text as a whole (helps exact match queries)
    if len(text) < 200:
        add_term(text_lower.strip(), 2.0)
    
    # Normalize to unit vector
    norm = math.sqrt(sum(x * x for x in embedding))
    if norm == 0:
        random.seed(42)
        embedding = [random.gauss(0, 1) for _ in range(dimensions)]
        norm = math.sqrt(sum(x * x for x in embedding))
    
    return [x / norm for x in embedding]


def hash_to_embedding_extended(text: str, dimensions: int = 1536) -> list[float]:
    """Convert text to deterministic embedding with sliding window support.
    
    Extended version that uses sliding windows to catch substring matches.
    Better for tests that need substring similarity detection.
    
    Args:
        text: Text to embed.
        dimensions: Output embedding dimensions.
        
    Returns:
        Normalized embedding vector.
    """
    embedding = [0.0] * dimensions
    
    def add_term(term: str, weight: float = 1.0):
        """Add a term's contribution to the embedding with given weight."""
        term_hash = hashlib.sha256(term.encode()).digest()
        random.seed(int.from_bytes(term_hash[:8], 'big'))
        for i in range(dimensions):
            embedding[i] += random.gauss(0, 1) * weight
    
    # Normalize text
    text_lower = text.lower()
    
    # Add contribution for each unique word
    words = set(re.findall(r'\b\w+\b', text_lower))
    for word in words:
        if len(word) > 2:  # Skip very short words
            add_term(word, 1.0)
    
    # Add sliding windows of characters (catches substrings)
    # Use windows of size 20, 40, 80 characters
    for window_size in [20, 40, 80]:
        seen = set()
        for i in range(0, len(text_lower) - window_size + 1, window_size // 2):
            chunk = text_lower[i:i + window_size].strip()
            if chunk and chunk not in seen:
                seen.add(chunk)
                add_term(chunk, 2.0)
    
    # For short texts (likely queries), add the whole text as a term
    # This ensures short exact matches get high similarity
    if len(text) < 200:
        add_term(text_lower.strip(), 5.0)
    
    # Normalize to unit vector
    norm = math.sqrt(sum(x * x for x in embedding))
    if norm == 0:
        random.seed(42)
        embedding = [random.gauss(0, 1) for _ in range(dimensions)]
        norm = math.sqrt(sum(x * x for x in embedding))
    
    return [x / norm for x in embedding]
