"""Tests for A2.1 dependency injection container."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tribalmemory.a21.container import Container
from tribalmemory.a21.config import SystemConfig
from tribalmemory.a21.config.providers import (
    EmbeddingProviderType,
    StorageProviderType,
    TimestampProviderType,
)
from tribalmemory.a21.providers.base import (
    EmbeddingProvider,
    StorageProvider,
    DeduplicationProvider,
    ProviderHealth,
    ProviderStatus,
)


class TestContainerInitialization:
    """Tests for container initialization."""
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration with mock providers."""
        return SystemConfig.for_testing(instance_id="container-test")
    
    async def test_initialize_creates_providers(self, test_config):
        """Test that initialize creates all required providers."""
        container = Container(test_config)
        
        assert container._embedding is None
        assert container._storage is None
        
        await container.initialize()
        
        assert container._embedding is not None
        assert container._storage is not None
        assert container._initialized
    
    async def test_initialize_is_idempotent(self, test_config):
        """Test that double initialization is safe."""
        container = Container(test_config)
        
        await container.initialize()
        embedding1 = container._embedding
        
        await container.initialize()  # Second call
        embedding2 = container._embedding
        
        assert embedding1 is embedding2  # Same instance
        await container.shutdown()
    
    async def test_context_manager(self, test_config):
        """Test async context manager."""
        async with Container(test_config) as container:
            assert container._initialized
            assert container.embedding is not None
            assert container.storage is not None
        
        assert not container._initialized
    
    async def test_provider_properties_require_init(self, test_config):
        """Test that accessing providers before init raises error."""
        container = Container(test_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = container.embedding
        
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = container.storage


class TestContainerProviderCreation:
    """Tests for provider factory methods."""
    
    async def test_creates_mock_embedding_provider(self):
        """Test creation of mock embedding provider."""
        config = SystemConfig.for_testing()
        container = Container(config)
        
        provider = container._create_embedding_provider()
        
        from tribalmemory.a21.providers.mock import MockEmbeddingProvider
        assert isinstance(provider, MockEmbeddingProvider)
    
    async def test_creates_openai_embedding_provider(self):
        """Test creation of OpenAI embedding provider."""
        config = SystemConfig.for_testing()
        config.embedding.provider = EmbeddingProviderType.OPENAI
        config.embedding.api_key = "sk-test"
        
        container = Container(config)
        provider = container._create_embedding_provider()
        
        from tribalmemory.a21.providers.openai import OpenAIEmbeddingProvider
        assert isinstance(provider, OpenAIEmbeddingProvider)
    
    async def test_creates_memory_storage_provider(self):
        """Test creation of in-memory storage provider."""
        config = SystemConfig.for_testing()
        container = Container(config)
        
        # Need embedding provider first
        container._embedding = container._create_embedding_provider()
        await container._embedding.initialize()
        
        provider = container._create_storage_provider()
        
        from tribalmemory.a21.providers.memory import InMemoryStorageProvider
        assert isinstance(provider, InMemoryStorageProvider)
        
        await container._embedding.shutdown()
    
    async def test_creates_deduplication_provider_when_enabled(self):
        """Test deduplication provider creation when enabled."""
        config = SystemConfig.for_testing()
        config.deduplication.enabled = True
        
        container = Container(config)
        
        # Initialize embedding and storage first
        container._embedding = container._create_embedding_provider()
        await container._embedding.initialize()
        container._storage = container._create_storage_provider()
        await container._storage.initialize()
        
        provider = container._create_deduplication_provider(
            storage=container._storage,
            embedding=container._embedding,
        )
        
        assert provider is not None
        from tribalmemory.a21.providers.deduplication import EmbeddingDeduplicationProvider
        assert isinstance(provider, EmbeddingDeduplicationProvider)
        
        await container._storage.shutdown()
        await container._embedding.shutdown()
    
    async def test_no_deduplication_when_disabled(self):
        """Test no deduplication provider when disabled."""
        config = SystemConfig.for_testing()
        config.deduplication.enabled = False
        
        container = Container(config)
        
        # Create mock providers for the call
        mock_storage = MagicMock()
        mock_embedding = MagicMock()
        
        provider = container._create_deduplication_provider(
            storage=mock_storage,
            embedding=mock_embedding,
        )
        
        assert provider is None
    
    async def test_unknown_embedding_provider_raises(self):
        """Test that unknown embedding provider raises error."""
        config = SystemConfig.for_testing()
        config.embedding.provider = MagicMock()  # Invalid provider type
        
        container = Container(config)
        
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            container._create_embedding_provider()


class TestContainerLifecycle:
    """Tests for container lifecycle management."""
    
    @pytest.fixture
    def test_config(self):
        return SystemConfig.for_testing()
    
    async def test_shutdown_order(self, test_config):
        """Test providers are shut down in reverse order."""
        container = Container(test_config)
        await container.initialize()
        
        # Track shutdown calls
        embedding_shutdown = AsyncMock()
        storage_shutdown = AsyncMock()
        
        container._embedding.shutdown = embedding_shutdown
        container._storage.shutdown = storage_shutdown
        
        shutdown_order = []
        embedding_shutdown.side_effect = lambda: shutdown_order.append("embedding")
        storage_shutdown.side_effect = lambda: shutdown_order.append("storage")
        
        await container.shutdown()
        
        # Storage should shut down before embedding (reverse of init order)
        assert shutdown_order == ["storage", "embedding"]
    
    async def test_shutdown_not_initialized(self, test_config):
        """Test shutdown without initialization is safe."""
        container = Container(test_config)
        await container.shutdown()  # Should not raise
    
    async def test_double_shutdown(self, test_config):
        """Test double shutdown is safe."""
        container = Container(test_config)
        await container.initialize()
        
        await container.shutdown()
        await container.shutdown()  # Should be safe


class TestContainerHealthCheck:
    """Tests for container health checking."""
    
    async def test_health_check_all_providers(self):
        """Test health check returns status for all providers."""
        config = SystemConfig.for_testing()
        config.deduplication.enabled = True
        
        async with Container(config) as container:
            health = await container.health_check()
            
            assert "embedding" in health
            assert "storage" in health
            assert "deduplication" in health
            
            assert all(isinstance(h, ProviderHealth) for h in health.values())
    
    async def test_health_check_excludes_none_providers(self):
        """Test health check skips disabled providers."""
        config = SystemConfig.for_testing()
        config.deduplication.enabled = False
        
        async with Container(config) as container:
            health = await container.health_check()
            
            assert "embedding" in health
            assert "storage" in health
            assert "deduplication" not in health


class TestContainerProviderAccess:
    """Tests for accessing providers from container."""
    
    async def test_embedding_property(self):
        """Test embedding property returns provider."""
        config = SystemConfig.for_testing()
        
        async with Container(config) as container:
            embedding = container.embedding
            assert isinstance(embedding, EmbeddingProvider)
    
    async def test_storage_property(self):
        """Test storage property returns provider."""
        config = SystemConfig.for_testing()
        
        async with Container(config) as container:
            storage = container.storage
            assert isinstance(storage, StorageProvider)
    
    async def test_deduplication_property(self):
        """Test deduplication property returns provider or None."""
        # With dedup enabled
        config = SystemConfig.for_testing()
        config.deduplication.enabled = True
        
        async with Container(config) as container:
            dedup = container.deduplication
            assert isinstance(dedup, DeduplicationProvider)
        
        # With dedup disabled
        config.deduplication.enabled = False
        
        async with Container(config) as container:
            dedup = container.deduplication
            assert dedup is None
    
    async def test_timestamp_property_when_disabled(self):
        """Test timestamp property returns None when disabled."""
        config = SystemConfig.for_testing()
        # Default is NONE provider
        
        async with Container(config) as container:
            timestamp = container.timestamp
            assert timestamp is None
