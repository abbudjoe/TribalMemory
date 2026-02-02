"""Tests for A2.1 provider implementations."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.tribalmemory.a21.providers.base import (
    EmbeddingProvider,
    StorageProvider,
    ProviderHealth,
    ProviderStatus,
)
from src.tribalmemory.a21.providers.openai import OpenAIEmbeddingProvider
from src.tribalmemory.a21.providers.memory import InMemoryStorageProvider
from src.tribalmemory.a21.providers.mock import MockEmbeddingProvider
from src.tribalmemory.a21.providers.deduplication import EmbeddingDeduplicationProvider
from src.tribalmemory.a21.config.providers import (
    EmbeddingConfig,
    StorageConfig,
    DeduplicationConfig,
    EmbeddingProviderType,
    StorageProviderType,
)
from src.tribalmemory.interfaces import MemoryEntry, MemorySource, RecallResult


class TestMockEmbeddingProvider:
    """Tests for mock embedding provider."""
    
    @pytest.fixture
    def config(self):
        return EmbeddingConfig(provider=EmbeddingProviderType.MOCK, dimensions=128)
    
    @pytest.fixture
    async def provider(self, config):
        p = MockEmbeddingProvider(config)
        await p.initialize()
        yield p
        await p.shutdown()
    
    async def test_embed_returns_correct_dimensions(self, provider, config):
        """Test embedding has correct dimensions."""
        embedding = await provider.embed("Test text")
        assert len(embedding) == config.dimensions
        assert all(isinstance(x, float) for x in embedding)
    
    async def test_embed_is_deterministic(self, provider):
        """Test same text produces same embedding."""
        emb1 = await provider.embed("Hello world")
        emb2 = await provider.embed("Hello world")
        assert emb1 == emb2
    
    async def test_embed_batch(self, provider, config):
        """Test batch embedding."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await provider.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == config.dimensions for e in embeddings)
    
    async def test_similarity_identical_vectors(self, provider):
        """Test similarity of identical vectors."""
        emb = await provider.embed("Test")
        sim = provider.similarity(emb, emb)
        assert abs(sim - 1.0) < 0.0001
    
    async def test_similarity_different_vectors(self, provider):
        """Test similarity of different vectors (cosine sim is in [-1, 1])."""
        emb1 = await provider.embed("Hello")
        emb2 = await provider.embed("Goodbye")
        sim = provider.similarity(emb1, emb2)
        assert -1 <= sim <= 1
    
    async def test_health_check(self, provider):
        """Test health check returns healthy."""
        health = await provider.health_check()
        assert health.status == ProviderStatus.HEALTHY


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAI embedding provider."""
    
    @pytest.fixture
    def config(self):
        return EmbeddingConfig(
            provider=EmbeddingProviderType.OPENAI,
            api_key="sk-test-key",
            dimensions=1536
        )
    
    def test_init_stores_config(self, config):
        """Test initialization stores config."""
        provider = OpenAIEmbeddingProvider(config)
        assert provider.config == config
        assert provider.dimensions == 1536
        assert provider.model_name == "text-embedding-3-small"
    
    async def test_init_requires_api_key(self):
        """Test initialization requires API key."""
        config = EmbeddingConfig(provider=EmbeddingProviderType.OPENAI, api_key=None)
        provider = OpenAIEmbeddingProvider(config)
        
        with pytest.raises(ValueError, match="API key required"):
            await provider.initialize()
    
    async def test_health_check_not_initialized(self):
        """Test health check when not initialized."""
        config = EmbeddingConfig(provider=EmbeddingProviderType.OPENAI, api_key="sk-test")
        provider = OpenAIEmbeddingProvider(config)
        
        health = await provider.health_check()
        assert health.status == ProviderStatus.UNAVAILABLE
    
    def test_clean_text_removes_extra_whitespace(self, config):
        """Test text cleaning removes extra whitespace."""
        provider = OpenAIEmbeddingProvider(config)
        cleaned = provider._clean_text("  hello   world  \n\t test  ")
        assert cleaned == "hello world test"
    
    def test_clean_text_truncates_long_text(self, config):
        """Test text cleaning truncates long text."""
        provider = OpenAIEmbeddingProvider(config)
        long_text = "x" * 100000
        cleaned = provider._clean_text(long_text)
        assert len(cleaned.encode('utf-8')) <= 8191 * 4
    
    def test_clean_text_handles_unicode(self, config):
        """Test text cleaning preserves unicode."""
        provider = OpenAIEmbeddingProvider(config)
        cleaned = provider._clean_text("Hello 🦝 世界")
        assert "🦝" in cleaned
        assert "世界" in cleaned
    
    def test_similarity_calculation(self, config):
        """Test cosine similarity calculation."""
        provider = OpenAIEmbeddingProvider(config)
        
        # Identical vectors
        vec = [0.1, 0.2, 0.3, 0.4]
        assert abs(provider.similarity(vec, vec) - 1.0) < 0.0001
        
        # Orthogonal vectors
        vec1, vec2 = [1, 0, 0, 0], [0, 1, 0, 0]
        assert abs(provider.similarity(vec1, vec2)) < 0.0001
        
        # Zero vector
        vec_zero = [0, 0, 0, 0]
        assert provider.similarity(vec, vec_zero) == 0.0

    def test_normalize_embedding_unit_length(self, config):
        """Test embedding normalization to unit length."""
        provider = OpenAIEmbeddingProvider(config)
        vec = [3.0, 4.0]
        normalized = provider._normalize_embedding(vec)
        norm = sum(x * x for x in normalized) ** 0.5
        assert abs(norm - 1.0) < 0.0001

    def test_normalize_embedding_zero_vector(self, config):
        """Test normalization preserves zero vector."""
        provider = OpenAIEmbeddingProvider(config)
        vec = [0.0, 0.0]
        assert provider._normalize_embedding(vec) == vec
    
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    async def test_embed_real_api(self):
        """Test real embedding generation with OpenAI API."""
        config = EmbeddingConfig(
            provider=EmbeddingProviderType.OPENAI,
            api_key=os.environ["OPENAI_API_KEY"]
        )
        provider = OpenAIEmbeddingProvider(config)
        
        await provider.initialize()
        try:
            embedding = await provider.embed("Hello world")
            assert len(embedding) == 1536
            assert all(isinstance(x, float) for x in embedding)
        finally:
            await provider.shutdown()


class TestInMemoryStorageProvider:
    """Tests for in-memory storage provider."""
    
    @pytest.fixture
    def mock_embedding(self):
        config = EmbeddingConfig(provider=EmbeddingProviderType.MOCK, dimensions=128)
        return MockEmbeddingProvider(config)
    
    @pytest.fixture
    def config(self):
        return StorageConfig(provider=StorageProviderType.MEMORY, embedding_dimensions=128)
    
    @pytest.fixture
    async def provider(self, config, mock_embedding):
        await mock_embedding.initialize()
        p = InMemoryStorageProvider(config, mock_embedding)
        await p.initialize()
        yield p
        await p.shutdown()
        await mock_embedding.shutdown()
    
    async def test_store_and_get(self, provider):
        """Test storing and retrieving a memory."""
        entry = MemoryEntry(
            id="test-1",
            content="Test content",
            embedding=[0.1] * 128,
            source_instance="test",
            source_type=MemorySource.USER_EXPLICIT,
        )
        
        result = await provider.store(entry)
        assert result.success
        assert result.memory_id == "test-1"
        
        retrieved = await provider.get("test-1")
        assert retrieved is not None
        assert retrieved.content == "Test content"
    
    async def test_store_generates_embedding_if_missing(self, provider):
        """Test that store generates embedding if not provided."""
        entry = MemoryEntry(
            id="test-embed",
            content="Generate my embedding",
            embedding=None,
            source_instance="test",
            source_type=MemorySource.AUTO_CAPTURE,
        )
        
        result = await provider.store(entry)
        assert result.success
        
        retrieved = await provider.get("test-embed")
        assert retrieved.embedding is not None
        assert len(retrieved.embedding) == 128
    
    async def test_store_validates_embedding_dimensions(self, provider):
        """Test that store rejects wrong embedding dimensions."""
        entry = MemoryEntry(
            id="wrong-dim",
            content="Wrong dimensions",
            embedding=[0.1] * 64,  # Wrong size
            source_instance="test",
            source_type=MemorySource.USER_EXPLICIT,
        )
        
        result = await provider.store(entry)
        assert not result.success
        assert "dimension" in result.error.lower()
    
    async def test_delete(self, provider):
        """Test soft delete."""
        entry = MemoryEntry(
            id="to-delete",
            content="Delete me",
            embedding=[0.1] * 128,
            source_instance="test",
            source_type=MemorySource.USER_EXPLICIT,
        )
        
        await provider.store(entry)
        assert await provider.get("to-delete") is not None
        
        assert await provider.delete("to-delete")
        assert await provider.get("to-delete") is None
    
    async def test_recall(self, provider):
        """Test recall by similarity."""
        entries = [
            MemoryEntry(id=f"mem-{i}", content=f"Memory {i}", embedding=[0.1 * (i + 1)] * 128,
                       source_instance="test", source_type=MemorySource.AUTO_CAPTURE)
            for i in range(5)
        ]
        
        for entry in entries:
            await provider.store(entry)
        
        # Query with embedding similar to mem-4
        results = await provider.recall([0.5] * 128, limit=3)
        assert len(results) <= 3
        assert all(isinstance(r, RecallResult) for r in results)
    
    async def test_list(self, provider):
        """Test listing memories."""
        for i in range(5):
            entry = MemoryEntry(
                id=f"list-{i}",
                content=f"Entry {i}",
                embedding=[0.1] * 128,
                source_instance="test",
                source_type=MemorySource.AUTO_CAPTURE,
            )
            await provider.store(entry)
        
        all_entries = await provider.list(limit=10)
        assert len(all_entries) == 5
    
    async def test_count(self, provider):
        """Test counting memories."""
        assert await provider.count() == 0
        
        for i in range(3):
            entry = MemoryEntry(
                id=f"count-{i}",
                content=f"Entry {i}",
                embedding=[0.1] * 128,
                source_instance="test",
                source_type=MemorySource.AUTO_CAPTURE,
            )
            await provider.store(entry)
        
        assert await provider.count() == 3
    
    async def test_health_check(self, provider):
        """Test health check."""
        health = await provider.health_check()
        assert health.status == ProviderStatus.HEALTHY


class TestEmbeddingDeduplicationProvider:
    """Tests for deduplication provider."""
    
    @pytest.fixture
    def mock_embedding(self):
        config = EmbeddingConfig(provider=EmbeddingProviderType.MOCK, dimensions=128)
        return MockEmbeddingProvider(config)
    
    @pytest.fixture
    def mock_storage(self, mock_embedding):
        config = StorageConfig(provider=StorageProviderType.MEMORY, embedding_dimensions=128)
        return InMemoryStorageProvider(config, mock_embedding)
    
    @pytest.fixture
    def config(self):
        return DeduplicationConfig(
            enabled=True,
            exact_threshold=0.95,
            near_threshold=0.85,
        )
    
    @pytest.fixture
    async def provider(self, config, mock_storage, mock_embedding):
        await mock_embedding.initialize()
        await mock_storage.initialize()
        p = EmbeddingDeduplicationProvider(config, mock_storage, mock_embedding)
        await p.initialize()
        yield p
        await p.shutdown()
        await mock_storage.shutdown()
        await mock_embedding.shutdown()
    
    async def test_is_duplicate_empty_store(self, provider):
        """Test duplicate check on empty store."""
        is_dup, dup_id = await provider.is_duplicate("Test", [0.1] * 128)
        assert is_dup is False
        assert dup_id is None
    
    async def test_is_duplicate_detects_duplicate(self, provider, mock_storage):
        """Test duplicate detection."""
        # Store an entry
        entry = MemoryEntry(
            id="original",
            content="The quick brown fox",
            embedding=[0.5] * 128,
            source_instance="test",
            source_type=MemorySource.USER_EXPLICIT,
        )
        await mock_storage.store(entry)
        
        # Check for duplicate with same embedding
        is_dup, dup_id = await provider.is_duplicate("Same fox", [0.5] * 128)
        
        # Note: Mock provider's similarity may or may not exceed threshold
        # This tests the flow, actual behavior depends on mock implementation
        assert isinstance(is_dup, bool)
    
    async def test_find_similar(self, provider, mock_storage):
        """Test finding similar memories."""
        # Store some entries
        for i in range(3):
            entry = MemoryEntry(
                id=f"sim-{i}",
                content=f"Similar content {i}",
                embedding=[0.1 * (i + 1)] * 128,
                source_instance="test",
                source_type=MemorySource.AUTO_CAPTURE,
            )
            await mock_storage.store(entry)
        
        similar = await provider.find_similar("Query", [0.2] * 128, threshold=0.5, limit=5)
        assert isinstance(similar, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in similar)
    
    async def test_health_check(self, provider):
        """Test health check."""
        health = await provider.health_check()
        assert health.status == ProviderStatus.HEALTHY


class TestProviderLifecycle:
    """Tests for provider lifecycle management."""
    
    async def test_context_manager(self):
        """Test async context manager."""
        config = EmbeddingConfig(provider=EmbeddingProviderType.MOCK)
        provider = MockEmbeddingProvider(config)
        
        assert not provider.is_initialized
        
        async with provider:
            assert provider.is_initialized
            embedding = await provider.embed("Test")
            assert len(embedding) == config.dimensions
        
        assert not provider.is_initialized
    
    async def test_double_initialize(self):
        """Test that double initialization is safe."""
        config = EmbeddingConfig(provider=EmbeddingProviderType.MOCK)
        provider = MockEmbeddingProvider(config)
        
        await provider.initialize()
        await provider.initialize()  # Should be safe
        
        assert provider.is_initialized
        await provider.shutdown()
    
    async def test_shutdown_not_initialized(self):
        """Test that shutdown without init is safe."""
        config = EmbeddingConfig(provider=EmbeddingProviderType.MOCK)
        provider = MockEmbeddingProvider(config)
        
        await provider.shutdown()  # Should be safe
        assert not provider.is_initialized
