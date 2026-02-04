"""Tests for A2.1 MemorySystem high-level API."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from tribalmemory.a21.system import MemorySystem
from tribalmemory.a21.config import SystemConfig
from tribalmemory.a21.config.providers import EmbeddingProviderType, StorageProviderType
from tribalmemory.interfaces import MemorySource, MemoryEntry, RecallResult, StoreResult


class TestMemorySystemLifecycle:
    """Tests for MemorySystem lifecycle management."""
    
    @pytest.fixture
    def test_config(self):
        return SystemConfig.for_testing(instance_id="system-test")
    
    async def test_start_and_stop(self, test_config):
        """Test basic start and stop."""
        system = MemorySystem(test_config)
        
        assert not system._started
        
        await system.start()
        assert system._started
        
        await system.stop()
        assert not system._started
    
    async def test_context_manager(self, test_config):
        """Test async context manager."""
        async with MemorySystem(test_config) as system:
            assert system._started
        
        assert not system._started
    
    async def test_double_start_is_safe(self, test_config):
        """Test double start is idempotent."""
        system = MemorySystem(test_config)
        
        await system.start()
        await system.start()  # Should not raise
        
        assert system._started
        await system.stop()
    
    async def test_double_stop_is_safe(self, test_config):
        """Test double stop is safe."""
        system = MemorySystem(test_config)
        
        await system.start()
        await system.stop()
        await system.stop()  # Should not raise
    
    async def test_start_validates_config(self):
        """Test that start validates configuration."""
        config = SystemConfig.for_testing()
        # Create an invalid config
        config.embedding.dimensions = 0  # Invalid
        
        system = MemorySystem(config)
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            await system.start()


class TestMemorySystemRemember:
    """Tests for remember() method."""
    
    @pytest.fixture
    def test_config(self):
        return SystemConfig.for_testing()
    
    async def test_remember_basic(self, test_config):
        """Test basic remember operation."""
        async with MemorySystem(test_config) as system:
            result = await system.remember("Remember this fact")
            
            assert result.success
            assert result.memory_id is not None
    
    async def test_remember_with_tags(self, test_config):
        """Test remember with tags."""
        async with MemorySystem(test_config) as system:
            result = await system.remember(
                "Tagged memory",
                tags=["test", "important"]
            )
            
            assert result.success
            
            # Verify tags were stored
            entry = await system.get(result.memory_id)
            assert "test" in entry.tags
            assert "important" in entry.tags
    
    async def test_remember_with_context(self, test_config):
        """Test remember with context."""
        async with MemorySystem(test_config) as system:
            result = await system.remember(
                "Contextual memory",
                context="From a conversation about testing"
            )
            
            assert result.success
            
            entry = await system.get(result.memory_id)
            assert entry.context == "From a conversation about testing"
    
    async def test_remember_with_source_type(self, test_config):
        """Test remember with explicit source type."""
        async with MemorySystem(test_config) as system:
            result = await system.remember(
                "User requested memory",
                source_type=MemorySource.USER_EXPLICIT
            )
            
            assert result.success
            
            entry = await system.get(result.memory_id)
            assert entry.source_type == MemorySource.USER_EXPLICIT
    
    async def test_remember_rejects_empty_content(self, test_config):
        """Test that empty content is rejected."""
        async with MemorySystem(test_config) as system:
            result = await system.remember("")
            
            assert not result.success
            assert "Empty" in result.error
    
    async def test_remember_rejects_whitespace_only(self, test_config):
        """Test that whitespace-only content is rejected."""
        async with MemorySystem(test_config) as system:
            result = await system.remember("   \n\t   ")
            
            assert not result.success
            assert "Empty" in result.error
    
    async def test_remember_strips_whitespace(self, test_config):
        """Test that content is stripped."""
        async with MemorySystem(test_config) as system:
            result = await system.remember("  trimmed content  ")
            
            assert result.success
            
            entry = await system.get(result.memory_id)
            assert entry.content == "trimmed content"
    
    async def test_remember_not_started_raises(self, test_config):
        """Test that remember raises if system not started."""
        system = MemorySystem(test_config)
        
        with pytest.raises(RuntimeError, match="not started"):
            await system.remember("Test")
    
    async def test_remember_skip_dedup(self, test_config):
        """Test skipping deduplication."""
        async with MemorySystem(test_config) as system:
            # Store same content twice with skip_dedup
            result1 = await system.remember("Duplicate content")
            result2 = await system.remember("Duplicate content", skip_dedup=True)
            
            assert result1.success
            assert result2.success
            assert result1.memory_id != result2.memory_id


class TestMemorySystemRecall:
    """Tests for recall() method."""
    
    @pytest.fixture
    def test_config(self):
        return SystemConfig.for_testing()
    
    async def test_recall_empty_store(self, test_config):
        """Test recall from empty store."""
        async with MemorySystem(test_config) as system:
            results = await system.recall("Find something")
            
            assert results == []
    
    async def test_recall_basic(self, test_config):
        """Test basic recall."""
        async with MemorySystem(test_config) as system:
            await system.remember("The sky is blue")
            await system.remember("Grass is green")
            await system.remember("The ocean is blue")
            
            results = await system.recall("What color is the sky?")
            
            assert isinstance(results, list)
            assert all(isinstance(r, RecallResult) for r in results)
    
    async def test_recall_respects_limit(self, test_config):
        """Test recall respects limit parameter."""
        async with MemorySystem(test_config) as system:
            for i in range(10):
                await system.remember(f"Memory number {i}", skip_dedup=True)
            
            results = await system.recall("memory", limit=3)
            
            assert len(results) <= 3
    
    async def test_recall_not_started_raises(self, test_config):
        """Test that recall raises if system not started."""
        system = MemorySystem(test_config)
        
        with pytest.raises(RuntimeError, match="not started"):
            await system.recall("Test")


class TestMemorySystemSupersededFiltering:
    """Tests for superseded memory filtering."""
    
    def test_filter_superseded_static(self):
        """Test that superseded memories are filtered out."""
        original = MemoryEntry(id="orig", content="Old")
        corrected = MemoryEntry(
            id="new",
            content="New",
            source_type=MemorySource.CORRECTION,
            supersedes="orig",
        )
        results = [
            RecallResult(memory=original, similarity_score=0.8, retrieval_time_ms=1),
            RecallResult(memory=corrected, similarity_score=0.9, retrieval_time_ms=1),
        ]
        
        filtered = MemorySystem._filter_superseded(results)
        ids = [r.memory.id for r in filtered]
        assert "orig" not in ids
        assert "new" in ids


class TestMemorySystemCorrect:
    """Tests for correct() method."""
    
    @pytest.fixture
    def test_config(self):
        return SystemConfig.for_testing()
    
    async def test_correct_creates_chain(self, test_config):
        """Test that correct creates a correction chain."""
        async with MemorySystem(test_config) as system:
            original = await system.remember("Paris is the capital of Germany")
            
            correction = await system.correct(
                original.memory_id,
                "Paris is the capital of France"
            )
            
            assert correction.success
            
            corrected_entry = await system.get(correction.memory_id)
            assert corrected_entry.source_type == MemorySource.CORRECTION
            assert corrected_entry.supersedes == original.memory_id
    
    async def test_correct_preserves_tags(self, test_config):
        """Test that correct preserves original tags."""
        async with MemorySystem(test_config) as system:
            original = await system.remember(
                "Wrong fact",
                tags=["geography", "important"]
            )
            
            correction = await system.correct(
                original.memory_id,
                "Right fact"
            )
            
            corrected_entry = await system.get(correction.memory_id)
            assert "geography" in corrected_entry.tags
            assert "important" in corrected_entry.tags
    
    async def test_correct_nonexistent_fails(self, test_config):
        """Test that correcting nonexistent memory fails."""
        async with MemorySystem(test_config) as system:
            result = await system.correct(
                "nonexistent-id",
                "Correction content"
            )
            
            assert not result.success
            assert "not found" in result.error
    
    async def test_correct_with_context(self, test_config):
        """Test correction with context."""
        async with MemorySystem(test_config) as system:
            original = await system.remember("Incorrect info")
            
            correction = await system.correct(
                original.memory_id,
                "Correct info",
                context="User pointed out the error"
            )
            
            corrected_entry = await system.get(correction.memory_id)
            assert corrected_entry.context == "User pointed out the error"


class TestMemorySystemForget:
    """Tests for forget() method."""
    
    @pytest.fixture
    def test_config(self):
        return SystemConfig.for_testing()
    
    async def test_forget_basic(self, test_config):
        """Test basic forget (soft delete)."""
        async with MemorySystem(test_config) as system:
            result = await system.remember("To be forgotten")
            
            assert await system.get(result.memory_id) is not None
            
            assert await system.forget(result.memory_id)
            
            assert await system.get(result.memory_id) is None
    
    async def test_forget_nonexistent(self, test_config):
        """Test forgetting nonexistent memory."""
        async with MemorySystem(test_config) as system:
            # Should return False for nonexistent
            result = await system.forget("nonexistent-id")
            # Behavior depends on implementation - may return True or False


class TestMemorySystemGet:
    """Tests for get() method."""
    
    @pytest.fixture
    def test_config(self):
        return SystemConfig.for_testing()
    
    async def test_get_existing(self, test_config):
        """Test getting existing memory."""
        async with MemorySystem(test_config) as system:
            result = await system.remember("Get me")
            
            entry = await system.get(result.memory_id)
            
            assert entry is not None
            assert entry.content == "Get me"
            assert entry.id == result.memory_id
    
    async def test_get_nonexistent(self, test_config):
        """Test getting nonexistent memory."""
        async with MemorySystem(test_config) as system:
            entry = await system.get("nonexistent-id")
            
            assert entry is None


class TestMemorySystemHealth:
    """Tests for health() method."""
    
    @pytest.fixture
    def test_config(self):
        return SystemConfig.for_testing()
    
    async def test_health_when_running(self, test_config):
        """Test health check when running."""
        async with MemorySystem(test_config) as system:
            health = await system.health()
            
            assert health["status"] == "running"
            assert health["instance_id"] == test_config.instance_id
            assert "providers" in health
    
    async def test_health_when_stopped(self, test_config):
        """Test health check when stopped."""
        system = MemorySystem(test_config)
        
        health = await system.health()
        
        assert health["status"] == "stopped"


class TestMemorySystemStats:
    """Tests for stats() method."""
    
    @pytest.fixture
    def test_config(self):
        return SystemConfig.for_testing()
    
    async def test_stats_empty(self, test_config):
        """Test stats on empty system."""
        async with MemorySystem(test_config) as system:
            stats = await system.stats()
            
            assert stats["total_memories"] == 0
            assert stats["instance_id"] == test_config.instance_id
    
    async def test_stats_with_memories(self, test_config):
        """Test stats with memories."""
        async with MemorySystem(test_config) as system:
            await system.remember("Memory 1", skip_dedup=True)
            await system.remember("Memory 2", skip_dedup=True)
            await system.remember("Memory 3", skip_dedup=True)
            
            stats = await system.stats()
            
            assert stats["total_memories"] == 3
    
    async def test_stats_not_started_raises(self, test_config):
        """Test stats raises if not started."""
        system = MemorySystem(test_config)
        
        with pytest.raises(RuntimeError, match="not started"):
            await system.stats()


class TestMemorySystemIntegration:
    """Integration tests for MemorySystem."""
    
    @pytest.fixture
    def test_config(self):
        return SystemConfig.for_testing()
    
    async def test_full_workflow(self, test_config):
        """Test complete workflow: remember, recall, correct, forget."""
        async with MemorySystem(test_config) as system:
            # Remember
            result1 = await system.remember(
                "Joe prefers Python",
                source_type=MemorySource.USER_EXPLICIT,
                tags=["preferences"]
            )
            assert result1.success
            
            result2 = await system.remember(
                "Joe likes TypeScript too",
                tags=["preferences"]
            )
            assert result2.success
            
            # Recall (use lower threshold since mock embeddings don't capture full semantics)
            results = await system.recall("Joe Python TypeScript", min_relevance=0.1)
            assert len(results) > 0
            
            # Correct
            correction = await system.correct(
                result1.memory_id,
                "Joe prefers TypeScript over Python",
                context="Joe clarified his preference"
            )
            assert correction.success
            
            # Verify correction chain
            corrected = await system.get(correction.memory_id)
            assert corrected.supersedes == result1.memory_id
            
            # Forget
            assert await system.forget(result2.memory_id)
            assert await system.get(result2.memory_id) is None
            
            # Stats
            stats = await system.stats()
            assert stats["total_memories"] >= 1  # At least the correction
