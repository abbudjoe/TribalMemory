"""Tier 1: Functional Tests (Must Pass)

These tests verify core functionality. All must pass before deployment.
"""

import asyncio
import pytest
from datetime import datetime

from tribalmemory.interfaces import MemorySource, MemoryEntry


class TestWriteReadIntegrity:
    """T1.1: Write-Read Integrity Tests"""
    
    @pytest.mark.tier1
    async def test_t1_1_1_basic_write_read(self, memory_service):
        """T1.1.1: Basic write and read."""
        content = "Joe prefers concise responses"
        
        # Store
        result = await memory_service.remember(content)
        assert result.success, f"Failed to store: {result.error}"
        assert result.memory_id is not None
        
        # Recall with exact query
        results = await memory_service.recall(content, limit=1)
        assert len(results) >= 1, "No results returned"
        assert results[0].similarity_score > 0.95, f"Score too low: {results[0].similarity_score}"
        assert results[0].memory.content == content
    
    @pytest.mark.tier1
    @pytest.mark.edge_case
    async def test_t1_1_2_unicode_emoji_handling(self, memory_service):
        """T1.1.2: Unicode and emoji handling."""
        content = "Joe's favorite emoji is 🦝 and he likes カタカナ"
        
        result = await memory_service.remember(content)
        assert result.success, f"Failed to store unicode: {result.error}"
        
        results = await memory_service.recall("Joe's favorite emoji is 🦝")
        assert len(results) >= 1
        # Verify no encoding corruption
        assert "🦝" in results[0].memory.content
        assert "カタカナ" in results[0].memory.content
    
    @pytest.mark.tier1
    @pytest.mark.edge_case
    async def test_t1_1_3_long_memory_storage(self, memory_service):
        """T1.1.3: Long memory storage (>4k tokens)."""
        # Generate a long document (~5000 tokens worth)
        long_content = "Technical documentation section. " * 500
        long_content += "KEY_PHRASE_FOR_RETRIEVAL"
        long_content += " More technical content. " * 500
        
        result = await memory_service.remember(long_content)
        assert result.success, f"Failed to store long content: {result.error}"
        
        results = await memory_service.recall("KEY_PHRASE_FOR_RETRIEVAL", min_relevance=0.3)
        assert len(results) >= 1
        # Verify not truncated
        assert "KEY_PHRASE_FOR_RETRIEVAL" in results[0].memory.content
    
    @pytest.mark.tier1
    @pytest.mark.edge_case
    async def test_t1_1_4_empty_whitespace_rejection(self, memory_service):
        """T1.1.4: Empty and whitespace rejection."""
        # Empty string
        result = await memory_service.remember("")
        assert not result.success, "Empty string should be rejected"
        
        # Whitespace only
        result = await memory_service.remember("   ")
        assert not result.success, "Whitespace-only should be rejected"
        
        # Newlines only
        result = await memory_service.remember("\n\n")
        assert not result.success, "Newline-only should be rejected"
    
    @pytest.mark.tier1
    @pytest.mark.edge_case
    @pytest.mark.security
    async def test_t1_1_5_special_character_handling(self, memory_service):
        """T1.1.5: Special characters (injection patterns)."""
        injection_content = "'; DROP TABLE memories;--"
        
        # Should store safely
        result = await memory_service.remember(injection_content)
        assert result.success, "Should store injection pattern safely"
        
        # Should retrieve unchanged
        results = await memory_service.recall(injection_content)
        assert len(results) >= 1
        assert results[0].memory.content == injection_content
    
    @pytest.mark.tier1
    @pytest.mark.edge_case
    async def test_t1_1_6_concurrent_writes(self, memory_service):
        """T1.1.6: Concurrent write safety."""
        contents = [f"Concurrent memory {i}" for i in range(20)]
        
        # Write all concurrently
        tasks = [memory_service.remember(c) for c in contents]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        successful = [r for r in results if r.success]
        assert len(successful) == len(contents), f"Only {len(successful)}/{len(contents)} succeeded"
        
        # All should be retrievable
        for content in contents:
            recall_results = await memory_service.recall(content, limit=1)
            assert len(recall_results) >= 1, f"Failed to recall: {content}"


class TestCrossInstancePropagation:
    """T1.2: Cross-Instance Propagation Tests
    
    Note: Full cross-instance tests require multi-instance setup.
    These tests verify the interface contracts.
    """
    
    @pytest.mark.tier1
    async def test_t1_2_1_basic_propagation_interface(self, memory_service):
        """T1.2.1: Verify provenance tracking for cross-instance."""
        content = "Memory from test instance"
        
        result = await memory_service.remember(content)
        assert result.success
        
        # Verify source instance is tracked
        provenance = await memory_service.get(result.memory_id)
        assert provenance is not None
        assert provenance.source_instance == "test-instance"
    
    @pytest.mark.tier1
    @pytest.mark.edge_case
    async def test_t1_2_5_store_failure_graceful(self, capacity_limited_store, embedding_service):
        """T1.2.5: Graceful handling when store is unavailable."""
        from tribalmemory.testing import MockMemoryService
        
        memory_service = MockMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=capacity_limited_store
        )
        
        # Fill to capacity
        for i in range(10):
            await memory_service.remember(f"Filler memory {i}")
        
        # Next should fail gracefully
        result = await memory_service.remember("One more memory")
        assert not result.success
        assert result.error is not None


class TestDeduplication:
    """T1.3: Deduplication Tests"""
    
    @pytest.mark.tier1
    async def test_t1_3_1_exact_duplicate_detection(self, memory_service):
        """T1.3.1: Exact duplicate detection."""
        content = "This is a unique fact"
        
        # First store should succeed
        result1 = await memory_service.remember(content)
        assert result1.success
        
        # Second store of same content should be rejected as duplicate
        result2 = await memory_service.remember(content)
        assert not result2.success
        assert result2.duplicate_of == result1.memory_id
    
    @pytest.mark.tier1
    @pytest.mark.edge_case
    async def test_t1_3_4_intentional_duplicate_defined_behavior(self, memory_service):
        """T1.3.4: Intentional duplicates have defined behavior."""
        content = "Remember this important fact"
        
        result1 = await memory_service.remember(content, source_type=MemorySource.USER_EXPLICIT)
        assert result1.success
        
        # Second attempt - should have defined behavior (reject with reference)
        result2 = await memory_service.remember(content, source_type=MemorySource.USER_EXPLICIT)
        # Either rejected with duplicate_of, or allowed with flag
        if not result2.success:
            assert result2.duplicate_of is not None


class TestProvenanceTracking:
    """T1.4: Provenance Tracking Tests"""
    
    @pytest.mark.tier1
    async def test_t1_4_1_source_attribution(self, memory_service):
        """T1.4.1: Basic source attribution."""
        content = "Memory with provenance"
        
        result = await memory_service.remember(content)
        assert result.success
        
        provenance = await memory_service.get(result.memory_id)
        assert provenance is not None
        assert provenance.source_instance == "test-instance"
        assert provenance.source_type == MemorySource.AUTO_CAPTURE
        assert provenance.created_at is not None
    
    @pytest.mark.tier1
    @pytest.mark.edge_case
    async def test_t1_4_2_legacy_memory_handling(self, vector_store, embedding_service):
        """T1.4.2: Legacy memory without source field."""
        # Manually insert a legacy memory
        legacy_entry = MemoryEntry(
            id="legacy-123",
            content="Old memory before tribal system",
            embedding=await embedding_service.embed("Old memory before tribal system"),
            source_instance="unknown",
            source_type=MemorySource.LEGACY
        )
        
        await vector_store.store(legacy_entry)
        
        # Should be retrievable with legacy marker
        retrieved = await vector_store.get("legacy-123")
        assert retrieved is not None
        assert retrieved.source_type == MemorySource.LEGACY


class TestPerformance:
    """T1.5: Performance Tests"""
    
    @pytest.mark.tier1
    async def test_t1_5_1_baseline_latency(self, memory_service, latency_tracker):
        """T1.5.1: Baseline latency measurement."""
        content = "Performance test memory"
        await memory_service.remember(content)
        
        # Measure recall latency
        measurements = []
        for _ in range(10):
            key = latency_tracker.start("recall")
            await memory_service.recall("Performance test")
            measurements.append(latency_tracker.stop(key))
        
        stats = latency_tracker.get_stats("recall")
        # Mock should be fast - real implementation will have higher baseline
        assert stats["p95_ms"] < 1000, f"P95 latency too high: {stats['p95_ms']}ms"
    
    @pytest.mark.tier1
    @pytest.mark.edge_case
    @pytest.mark.slow
    async def test_t1_5_2_latency_under_load(self, memory_service, latency_tracker):
        """T1.5.2: Latency with 10,000 memories."""
        # Store many memories
        for i in range(100):  # Reduced for mock; real test uses 10k
            await memory_service.remember(f"Load test memory number {i} with unique content")
        
        # Measure recall latency under load
        key = latency_tracker.start("recall_under_load")
        await memory_service.recall("Load test memory")
        duration = latency_tracker.stop(key)
        
        # Should still be reasonable
        assert duration < 5000, f"Latency under load too high: {duration}ms"
    
    @pytest.mark.tier1
    @pytest.mark.edge_case
    @pytest.mark.failure_mode
    async def test_t1_5_3_embedding_timeout_graceful(self, timeout_embedding_service):
        """T1.5.3: Graceful handling of embedding API timeout."""
        from tribalmemory.testing import MockMemoryService
        
        memory_service = MockMemoryService(
            instance_id="test",
            embedding_service=timeout_embedding_service
        )
        
        # First few should work
        for i in range(3):
            result = await memory_service.remember(f"Memory {i}")
            assert result.success
        
        # After timeout_after_n, should timeout but not crash
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                memory_service.remember("This will timeout"),
                timeout=1.0
            )


class TestCorrections:
    """Additional Tier 1 tests for correction flow."""
    
    @pytest.mark.tier1
    async def test_correction_supersedes_original(self, memory_service):
        """Test that corrections properly supersede originals."""
        # Store original
        original = await memory_service.remember("Joe's timezone is Eastern")
        assert original.success
        
        # Store correction
        correction = await memory_service.correct(
            original.memory_id,
            "Joe's timezone is Mountain",
            context="User correction"
        )
        assert correction.success
        
        # Verify correction links to original
        corrected = await memory_service.get(correction.memory_id)
        assert corrected.supersedes == original.memory_id
        assert corrected.source_type == MemorySource.CORRECTION
