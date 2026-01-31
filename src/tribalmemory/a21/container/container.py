"""Dependency injection container for A2.1."""

from typing import Optional, TypeVar, Type
import logging

from ..config import SystemConfig
from ..providers.base import (
    EmbeddingProvider,
    StorageProvider,
    TimestampProvider,
    DeduplicationProvider,
    ProviderHealth,
    ProviderStatus,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Container:
    """Dependency injection container.
    
    Manages the lifecycle of all providers and provides them to consumers.
    Supports lazy initialization and graceful shutdown.
    
    Usage:
        container = Container(config)
        await container.initialize()
        
        embedding = container.embedding
        storage = container.storage
        
        await container.shutdown()
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self._embedding: Optional[EmbeddingProvider] = None
        self._storage: Optional[StorageProvider] = None
        self._timestamp: Optional[TimestampProvider] = None
        self._deduplication: Optional[DeduplicationProvider] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all providers.
        
        Providers are created and initialized in dependency order:
        1. Embedding (no dependencies)
        2. Storage (depends on embedding dimensions)
        3. Deduplication (depends on both storage and embedding)
        4. Timestamp (no dependencies)
        """
        if self._initialized:
            return
        
        logger.info(f"Initializing container for instance: {self.config.instance_id}")
        
        # Create and initialize providers in dependency order
        self._embedding = self._create_embedding_provider()
        await self._embedding.initialize()
        
        self._storage = self._create_storage_provider()
        await self._storage.initialize()
        
        # Deduplication depends on both storage and embedding - create after they're initialized
        self._deduplication = self._create_deduplication_provider(
            storage=self._storage,
            embedding=self._embedding,
        )
        if self._deduplication:
            await self._deduplication.initialize()
        
        if self.config.timestamp.provider.value != "none":
            self._timestamp = self._create_timestamp_provider()
            await self._timestamp.initialize()
        
        self._initialized = True
        logger.info("Container initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown all providers gracefully."""
        if not self._initialized:
            return
        
        logger.info("Shutting down container")
        
        # Shutdown in reverse order
        if self._timestamp:
            await self._timestamp.shutdown()
        if self._deduplication:
            await self._deduplication.shutdown()
        await self._storage.shutdown()
        await self._embedding.shutdown()
        
        self._initialized = False
        logger.info("Container shutdown complete")
    
    async def health_check(self) -> dict[str, ProviderHealth]:
        """Check health of all providers."""
        results = {}
        
        if self._embedding:
            results["embedding"] = await self._embedding.health_check()
        if self._storage:
            results["storage"] = await self._storage.health_check()
        if self._timestamp:
            results["timestamp"] = await self._timestamp.health_check()
        if self._deduplication:
            results["deduplication"] = await self._deduplication.health_check()
        
        return results
    
    @property
    def embedding(self) -> EmbeddingProvider:
        """Get the embedding provider."""
        if not self._embedding:
            raise RuntimeError("Container not initialized. Call initialize() first.")
        return self._embedding
    
    @property
    def storage(self) -> StorageProvider:
        """Get the storage provider."""
        if not self._storage:
            raise RuntimeError("Container not initialized. Call initialize() first.")
        return self._storage
    
    @property
    def timestamp(self) -> Optional[TimestampProvider]:
        """Get the timestamp provider (may be None if disabled)."""
        return self._timestamp
    
    @property
    def deduplication(self) -> Optional[DeduplicationProvider]:
        """Get the deduplication provider."""
        return self._deduplication
    
    def _create_embedding_provider(self) -> EmbeddingProvider:
        """Create embedding provider based on config."""
        from ..config.providers import EmbeddingProviderType
        from ..providers.openai import OpenAIEmbeddingProvider
        from ..providers.mock import MockEmbeddingProvider
        
        cfg = self.config.embedding
        
        if cfg.provider == EmbeddingProviderType.OPENAI:
            return OpenAIEmbeddingProvider(cfg)
        elif cfg.provider == EmbeddingProviderType.MOCK:
            return MockEmbeddingProvider(cfg)
        else:
            raise ValueError(f"Unknown embedding provider: {cfg.provider}")
    
    def _create_storage_provider(self) -> StorageProvider:
        """Create storage provider based on config."""
        from ..config.providers import StorageProviderType
        from ..providers.lancedb import LanceDBStorageProvider
        from ..providers.memory import InMemoryStorageProvider
        
        cfg = self.config.storage
        
        if cfg.provider == StorageProviderType.LANCEDB:
            return LanceDBStorageProvider(cfg, self._embedding)
        elif cfg.provider == StorageProviderType.MEMORY:
            return InMemoryStorageProvider(cfg, self._embedding)
        else:
            raise ValueError(f"Unknown storage provider: {cfg.provider}")
    
    def _create_timestamp_provider(self) -> TimestampProvider:
        """Create timestamp provider based on config."""
        from ..config.providers import TimestampProviderType
        from ..providers.timestamp import RFC3161TimestampProvider, MockTimestampProvider
        
        cfg = self.config.timestamp
        
        if cfg.provider == TimestampProviderType.RFC3161:
            return RFC3161TimestampProvider(cfg)
        elif cfg.provider == TimestampProviderType.MOCK:
            return MockTimestampProvider(cfg)
        else:
            raise ValueError(f"Unknown timestamp provider: {cfg.provider}")
    
    def _create_deduplication_provider(
        self,
        storage: StorageProvider,
        embedding: EmbeddingProvider,
    ) -> Optional[DeduplicationProvider]:
        """Create deduplication provider based on config.
        
        Args:
            storage: Initialized storage provider
            embedding: Initialized embedding provider
            
        Returns:
            DeduplicationProvider or None if disabled
        """
        if not self.config.deduplication.enabled:
            return None
        
        from ..providers.deduplication import EmbeddingDeduplicationProvider
        
        return EmbeddingDeduplicationProvider(
            self.config.deduplication,
            storage_provider=storage,
            embedding_provider=embedding,
        )
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
