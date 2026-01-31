"""Mock providers for testing."""

from datetime import datetime
from typing import Optional

from .base import (
    EmbeddingProvider,
    StorageProvider,
    ProviderHealth,
    ProviderStatus,
)
from ..config.providers import EmbeddingConfig
from ...testing.embedding_utils import hash_to_embedding


class MockEmbeddingProvider(EmbeddingProvider[EmbeddingConfig]):
    """Mock embedding provider using deterministic hashing."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
    
    @property
    def dimensions(self) -> int:
        return self.config.dimensions
    
    @property
    def model_name(self) -> str:
        return "mock-embedding"
    
    async def initialize(self) -> None:
        self._initialized = True
    
    async def shutdown(self) -> None:
        self._initialized = False
    
    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            status=ProviderStatus.HEALTHY,
            latency_ms=0.1,
            message="Mock provider always healthy"
        )
    
    async def embed(self, text: str) -> list[float]:
        return self._hash_to_embedding(text)
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_embedding(t) for t in texts]
    
    def _hash_to_embedding(self, text: str) -> list[float]:
        """Convert text to deterministic embedding that preserves semantic similarity.
        
        Delegates to shared utility for consistent behavior across mock implementations.
        """
        return hash_to_embedding(text, self.dimensions)
