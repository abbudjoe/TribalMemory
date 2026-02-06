"""Tests for production service implementations."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tribalmemory.interfaces import MemorySource, MemoryEntry, RecallResult
from tribalmemory.utils import normalize_embedding
from tribalmemory.services.vector_store import InMemoryVectorStore
from tribalmemory.services.deduplication import SemanticDeduplicationService
from tribalmemory.services.memory import TribalMemoryService, create_memory_service


class TestNormalizeEmbedding:
    """Tests for embedding normalization utilities."""

    def test_normalize_embedding_unit_length(self):
        """Test embedding normalization to unit length."""
        vec = [3.0, 4.0]
        normalized = normalize_embedding(vec)
        norm = sum(x * x for x in normalized) ** 0.5
        assert abs(norm - 1.0) < 0.0001

    def test_normalize_embedding_zero_vector(self):
        """Test normalization preserves zero vector."""
        vec = [0.0, 0.0]
        assert normalize_embedding(vec) == vec


class TestInMemoryVectorStore:
    """Tests for in-memory vector store."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        service = MagicMock()
        service.embed = AsyncMock(return_value=[0.1] * 384)
        service.similarity = MagicMock(return_value=0.95)
        return service
    
    @pytest.fixture
    def store(self, mock_embedding_service):
        return InMemoryVectorStore(mock_embedding_service)
    
    async def test_store_and_retrieve(self, store):
        entry = MemoryEntry(id="test-1", content="Test content", embedding=[0.1] * 384)
        
        result = await store.store(entry)
        assert result.success
        
        retrieved = await store.get("test-1")
        assert retrieved is not None
        assert retrieved.content == "Test content"
    
    async def test_delete(self, store):
        entry = MemoryEntry(id="test-del", content="To delete", embedding=[0.1] * 384)
        await store.store(entry)
        
        assert await store.delete("test-del")
        assert await store.get("test-del") is None


class TestSemanticDeduplicationService:
    """Tests for semantic deduplication."""
    
    async def test_detects_exact_duplicate(self):
        store = MagicMock()
        store.recall = AsyncMock(return_value=[
            RecallResult(
                memory=MemoryEntry(id="existing", content="Joe prefers TypeScript"),
                similarity_score=0.99,
                retrieval_time_ms=10
            )
        ])
        
        dedup = SemanticDeduplicationService(
            vector_store=store,
            embedding_service=MagicMock(),
            exact_threshold=0.98
        )
        
        is_dup, dup_id = await dedup.is_duplicate("Joe prefers TypeScript", [0.1] * 384)
        
        assert is_dup is True
        assert dup_id == "existing"
    
    async def test_allows_non_duplicates(self):
        store = MagicMock()
        store.recall = AsyncMock(return_value=[
            RecallResult(
                memory=MemoryEntry(id="other", content="Different"),
                similarity_score=0.7,
                retrieval_time_ms=10
            )
        ])
        
        dedup = SemanticDeduplicationService(
            vector_store=store,
            embedding_service=MagicMock(),
            exact_threshold=0.98
        )
        
        is_dup, dup_id = await dedup.is_duplicate("New content", [0.1] * 384)
        
        assert is_dup is False
        assert dup_id is None


class TestTribalMemoryService:
    """Tests for the main memory service."""
    
    @pytest.fixture
    def mock_components(self):
        embedding_service = MagicMock()
        embedding_service.embed = AsyncMock(return_value=[0.1] * 384)
        embedding_service.similarity = MagicMock(return_value=0.5)
        
        vector_store = MagicMock()
        vector_store.store = AsyncMock(return_value=MagicMock(success=True, memory_id="new-id"))
        vector_store.recall = AsyncMock(return_value=[])
        vector_store.get = AsyncMock(return_value=None)
        vector_store.delete = AsyncMock(return_value=True)
        
        return embedding_service, vector_store
    
    async def test_remember_basic(self, mock_components):
        embedding_service, vector_store = mock_components
        
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        result = await service.remember("Test memory")
        assert result.success or result.duplicate_of
    
    async def test_remember_rejects_empty(self, mock_components):
        embedding_service, vector_store = mock_components
        
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        result = await service.remember("")
        assert not result.success
        assert "Empty" in result.error
    
    async def test_correct_creates_chain(self, mock_components):
        embedding_service, vector_store = mock_components
        
        original = MemoryEntry(id="original", content="Original", tags=["tag1"])
        vector_store.get = AsyncMock(return_value=original)
        
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        await service.correct("original", "Corrected")
        
        stored = vector_store.store.call_args[0][0]
        assert stored.supersedes == "original"
        assert stored.source_type == MemorySource.CORRECTION

    async def test_recall_filters_superseded(self, mock_components):
        embedding_service, vector_store = mock_components

        original = MemoryEntry(id="orig", content="Old info")
        corrected = MemoryEntry(
            id="new",
            content="Corrected info",
            source_type=MemorySource.CORRECTION,
            supersedes="orig",
        )
        vector_store.recall = AsyncMock(return_value=[
            RecallResult(memory=original, similarity_score=0.8, retrieval_time_ms=1),
            RecallResult(memory=corrected, similarity_score=0.9, retrieval_time_ms=1),
        ])

        service = TribalMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=vector_store
        )

        results = await service.recall("Query")
        ids = [r.memory.id for r in results]
        assert "orig" not in ids
        assert "new" in ids


class TestCreateMemoryService:
    """Tests for the factory function."""
    
    def test_creates_with_defaults(self):
        """Test that create_memory_service creates a service with defaults."""
        service = create_memory_service(instance_id="test")
        assert service.instance_id == "test"


class TestErrorPaths:
    """Tests for error handling and edge cases."""
    
    async def test_embedding_failure_handling(self):
        """Test that embedding failures are handled gracefully."""
        embedding_service = MagicMock()
        embedding_service.embed = AsyncMock(side_effect=RuntimeError("API failure"))
        
        vector_store = MagicMock()
        vector_store.recall = AsyncMock(return_value=[])
        
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        result = await service.remember("Test content")
        assert not result.success
        assert "Embedding generation failed" in result.error
    
    async def test_recall_with_embedding_failure(self):
        """Test that recall returns empty on embedding failure."""
        embedding_service = MagicMock()
        embedding_service.embed = AsyncMock(side_effect=RuntimeError("API failure"))
        
        vector_store = MagicMock()
        
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        results = await service.recall("Test query")
        assert results == []
    
    async def test_correct_missing_original(self):
        """Test correction of non-existent memory."""
        embedding_service = MagicMock()
        embedding_service.embed = AsyncMock(return_value=[0.1] * 384)
        
        vector_store = MagicMock()
        vector_store.get = AsyncMock(return_value=None)
        
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        result = await service.correct("nonexistent-id", "Corrected content")
        assert not result.success
        assert "not found" in result.error
    
    async def test_dedup_at_exact_threshold(self):
        """Test deduplication behavior at exact threshold boundary."""
        store = MagicMock()
        store.recall = AsyncMock(return_value=[
            RecallResult(
                memory=MemoryEntry(id="boundary", content="Boundary case"),
                similarity_score=0.90,  # Exactly at threshold
                retrieval_time_ms=10
            )
        ])
        
        dedup = SemanticDeduplicationService(
            vector_store=store,
            embedding_service=MagicMock(),
            exact_threshold=0.90
        )
        
        is_dup, dup_id = await dedup.is_duplicate("Test", [0.1] * 384)
        assert is_dup is True
        assert dup_id == "boundary"
    
    async def test_dedup_just_below_threshold(self):
        """Test deduplication behavior just below threshold."""
        store = MagicMock()
        store.recall = AsyncMock(return_value=[
            RecallResult(
                memory=MemoryEntry(id="below", content="Below threshold"),
                similarity_score=0.899,  # Just below threshold
                retrieval_time_ms=10
            )
        ])
        
        dedup = SemanticDeduplicationService(
            vector_store=store,
            embedding_service=MagicMock(),
            exact_threshold=0.90
        )
        
        is_dup, dup_id = await dedup.is_duplicate("Test", [0.1] * 384)
        assert is_dup is False
    
    async def test_vector_store_invalid_id(self):
        """Test vector store rejects invalid memory IDs."""
        from tribalmemory.services.vector_store import LanceDBVectorStore
        
        embedding_service = MagicMock()
        embedding_service.dimensions = 384
        
        # Can't fully test without LanceDB, but can test the sanitization
        store = LanceDBVectorStore(
            embedding_service=embedding_service,
            db_path="/tmp/test_tribal"
        )
        
        # Test sanitization
        with pytest.raises(ValueError, match="Invalid memory_id"):
            store._sanitize_id("'; DROP TABLE memories;--")
        
        # Valid UUID should pass
        valid_id = store._sanitize_id("550e8400-e29b-41d4-a716-446655440000")
        assert valid_id == "550e8400-e29b-41d4-a716-446655440000"
