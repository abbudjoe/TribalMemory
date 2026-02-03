"""Shared utility functions for TribalMemory.

This module provides common functions used across multiple components
to prevent code duplication and ensure consistency.
"""

import math


def normalize_embedding(embedding: list[float]) -> list[float]:
    """Normalize embedding to unit length for consistent similarity math.
    
    Args:
        embedding: Vector of floats representing an embedding.
        
    Returns:
        Normalized embedding with unit length (L2 norm = 1).
        Returns the original embedding if it has zero magnitude.
    """
    norm = math.sqrt(sum(x * x for x in embedding))
    if norm == 0:
        return embedding
    return [x / norm for x in embedding]
