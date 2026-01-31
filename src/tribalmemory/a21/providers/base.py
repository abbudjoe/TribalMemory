"""Abstract base classes for all providers.

These define the contracts that provider implementations must satisfy.
Designed for extensibility and forward compatibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any, TypeVar, Generic
from enum import Enum

from ...interfaces import MemoryEntry, RecallResult, StoreResult, MemorySource


# Type variable for provider-specific configuration
TConfig = TypeVar('TConfig')


class ProviderStatus(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    INITIALIZING = "initializing"


@dataclass
class ProviderHealth:
    """Health check result for a provider."""
    status: ProviderStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    last_check: datetime = None
    
    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.utcnow()


class Provider(ABC, Generic[TConfig]):
    """Base class for all providers.
    
    Provides common functionality:
    - Configuration management
    - Health checking
    - Lifecycle management (init/shutdown)
    - Metrics collection hooks
    """
    
    def __init__(self, config: TConfig):
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider. Called once before first use."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the provider."""
        pass
    
    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Check provider health and connectivity."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    async def __aenter__(self):
        if not self._initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()


class EmbeddingProvider(Provider[TConfig]):
    """Abstract embedding provider.
    
    Responsible for converting text to vector embeddings.
    Implementations may use OpenAI, local models, or other services.
    """
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimension size."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently."""
        pass
    
    def similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Uses the formula: cos(θ) = (a · b) / (||a|| * ||b||)
        
        Args:
            a: First embedding vector
            b: Second embedding vector
            
        Returns:
            Cosine similarity score between -1.0 and 1.0
        """
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class StorageProvider(Provider[TConfig]):
    """Abstract storage provider.
    
    Responsible for persisting and retrieving memory entries.
    Implementations may use LanceDB, Pinecone, Postgres+pgvector, etc.
    """
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> StoreResult:
        """Store a memory entry."""
        pass
    
    @abstractmethod
    async def recall(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RecallResult]:
        """Recall memories similar to query embedding.
        
        Args:
            query_embedding: Vector to search for
            limit: Maximum results
            min_similarity: Minimum similarity threshold
            filters: Optional metadata filters (e.g., tags, source_instance)
        """
        pass
    
    @abstractmethod
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        pass
    
    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Soft delete a memory."""
        pass
    
    @abstractmethod
    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[MemoryEntry]:
        """List memories with pagination and filtering."""
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[dict[str, Any]] = None) -> int:
        """Count memories matching filters."""
        pass


class TimestampProvider(Provider[TConfig]):
    """Abstract timestamp provider.
    
    Responsible for generating cryptographic timestamps (RFC 3161).
    Used for provenance verification.
    """
    
    @abstractmethod
    async def timestamp(self, data: bytes) -> bytes:
        """Generate a timestamp token for data."""
        pass
    
    @abstractmethod
    async def verify(self, data: bytes, token: bytes) -> tuple[bool, Optional[datetime]]:
        """Verify a timestamp token."""
        pass


class DeduplicationProvider(Provider[TConfig]):
    """Abstract deduplication provider.
    
    Responsible for detecting duplicate or near-duplicate memories.
    May use embedding similarity, hashing, or hybrid approaches.
    """
    
    @abstractmethod
    async def is_duplicate(
        self,
        content: str,
        embedding: list[float],
    ) -> tuple[bool, Optional[str]]:
        """Check if content is duplicate.
        
        Returns:
            Tuple of (is_duplicate, duplicate_id).
            Use find_similar() or get_duplicate_report() if similarity score needed.
        """
        pass
    
    @abstractmethod
    async def find_similar(
        self,
        content: str,
        embedding: list[float],
        threshold: float = 0.85,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar memories.
        
        Returns:
            List of (memory_id, similarity_score)
        """
        pass
