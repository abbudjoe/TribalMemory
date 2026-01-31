"""Deduplication provider."""

from typing import Optional

from .base import DeduplicationProvider, StorageProvider, EmbeddingProvider, ProviderHealth, ProviderStatus
from ..config.providers import DeduplicationConfig


class EmbeddingDeduplicationProvider(DeduplicationProvider[DeduplicationConfig]):
    """Deduplication using embedding similarity.
    
    Detects duplicates by comparing embeddings against stored memories.
    Uses configurable thresholds for exact and near-duplicate detection.
    """
    
    def __init__(
        self,
        config: DeduplicationConfig,
        storage_provider: StorageProvider,
        embedding_provider: EmbeddingProvider,
    ):
        """Initialize deduplication provider.
        
        Args:
            config: Deduplication configuration
            storage_provider: Initialized storage provider for recall queries
            embedding_provider: Initialized embedding provider for similarity calculations
        """
        super().__init__(config)
        self._storage = storage_provider
        self._embedding = embedding_provider
    
    async def initialize(self) -> None:
        """Initialize provider. Storage and embedding must already be initialized."""
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Shutdown provider."""
        self._initialized = False
    
    async def health_check(self) -> ProviderHealth:
        """Check provider health."""
        return ProviderHealth(
            status=ProviderStatus.HEALTHY,
            message="Deduplication ready"
        )
    
    async def is_duplicate(
        self,
        content: str,
        embedding: list[float],
    ) -> tuple[bool, Optional[str]]:
        """Check if content is a duplicate.
        
        Args:
            content: Text content to check
            embedding: Pre-computed embedding for the content
            
        Returns:
            Tuple of (is_duplicate, duplicate_id)
        """
        results = await self._storage.recall(
            embedding,
            limit=1,
            min_similarity=self.config.exact_threshold
        )
        
        if results and results[0].similarity_score >= self.config.exact_threshold:
            return True, results[0].memory.id
        
        return False, None
    
    async def find_similar(
        self,
        content: str,
        embedding: list[float],
        threshold: float = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar memories.
        
        Args:
            content: Text content to search for
            embedding: Pre-computed embedding for the content
            threshold: Minimum similarity (defaults to config.near_threshold)
            limit: Maximum results
            
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        threshold = threshold or self.config.near_threshold
        
        results = await self._storage.recall(
            embedding,
            limit=limit,
            min_similarity=threshold
        )
        
        return [(r.memory.id, r.similarity_score) for r in results]
