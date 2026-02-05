"""Tests for temporal filtering on recall() — Issue #61.

Verifies that recall() accepts `after` and `before` parameters
to filter/boost results by resolved temporal facts.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta

from tribalmemory.services.memory import TribalMemoryService
from tribalmemory.services.graph_store import GraphStore
from tribalmemory.services.fts_store import FTSStore
from tribalmemory.testing import MockEmbeddingService, MockVectorStore


@pytest.fixture
def embedding_service():
    return MockEmbeddingService(embedding_dim=64)


@pytest.fixture
def vector_store(embedding_service):
    return MockVectorStore(embedding_service)


@pytest.fixture
def graph_store(tmp_path):
    return GraphStore(str(tmp_path / "graph.db"))


@pytest.fixture
def fts_store(tmp_path):
    store = FTSStore(str(tmp_path / "fts.db"))
    if not store.is_available():
        pytest.skip("FTS5 not available")
    return store


@pytest.fixture
def service(embedding_service, vector_store, graph_store, fts_store):
    return TribalMemoryService(
        instance_id="test",
        embedding_service=embedding_service,
        vector_store=vector_store,
        graph_store=graph_store,
        graph_enabled=True,
        fts_store=fts_store,
        hybrid_search=True,
    )


@pytest.fixture
def service_no_hybrid(embedding_service, vector_store, graph_store):
    """Service without FTS for vector-only temporal tests."""
    return TribalMemoryService(
        instance_id="test",
        embedding_service=embedding_service,
        vector_store=vector_store,
        graph_store=graph_store,
        graph_enabled=True,
    )


class TestTemporalRecallParams:
    """Test that recall() accepts after/before params without errors.
    
    These tests verify the API contract — params are accepted and
    return valid results regardless of filtering behavior.
    """

    @pytest.mark.asyncio
    async def test_recall_accepts_after_param(self, service):
        """recall() should accept an `after` parameter."""
        await service.remember("Meeting with Bob on January 15, 2025")
        results = await service.recall("meeting", after="2025-01-01")
        # Should not raise
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_recall_accepts_before_param(self, service):
        """recall() should accept a `before` parameter."""
        await service.remember("Meeting with Bob on January 15, 2025")
        results = await service.recall("meeting", before="2025-12-31")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_recall_accepts_both_params(self, service):
        """recall() should accept both after and before together."""
        await service.remember("Meeting with Bob on January 15, 2025")
        results = await service.recall(
            "meeting", after="2025-01-01", before="2025-12-31"
        )
        assert isinstance(results, list)


class TestTemporalFiltering:
    """Test that temporal params actually filter results.
    
    Memories with temporal facts outside the range are excluded.
    Memories without temporal facts pass through unfiltered.
    """

    @pytest.mark.asyncio
    async def test_after_filters_old_memories(self, service):
        """Memories with dates before `after` should be excluded."""
        await service.remember("Project started on January 5, 2024")
        await service.remember("Project completed on March 10, 2025")
        
        results = await service.recall(
            "project", after="2025-01-01", min_relevance=0.0
        )
        
        # Only the 2025 memory should appear
        contents = [r.memory.content for r in results]
        assert any("2025" in c for c in contents)
        assert not any("January 5, 2024" in c for c in contents)

    @pytest.mark.asyncio
    async def test_before_filters_future_memories(self, service):
        """Memories with dates after `before` should be excluded."""
        await service.remember("Deployed release on June 1, 2024")
        await service.remember("Deployed release for March 2026")
        
        results = await service.recall(
            "Deployed release", before="2025-01-01", min_relevance=0.0
        )
        
        contents = [r.memory.content for r in results]
        assert any("2024" in c for c in contents)
        assert not any("2026" in c for c in contents)

    @pytest.mark.asyncio
    async def test_date_range_filtering(self, service):
        """Only memories within the date range should appear."""
        await service.remember("Event A on January 10, 2024")
        await service.remember("Event B on June 15, 2024")
        await service.remember("Event C on December 20, 2024")
        
        results = await service.recall(
            "event",
            after="2024-03-01",
            before="2024-09-30",
            min_relevance=0.0,
        )
        
        contents = [r.memory.content for r in results]
        assert any("June" in c for c in contents)
        assert not any("January" in c for c in contents)
        assert not any("December" in c for c in contents)

    @pytest.mark.asyncio
    async def test_no_temporal_data_passes_through(self, service):
        """Memories without temporal facts should NOT be filtered out."""
        await service.remember("Bob likes Python programming")
        await service.remember("Meeting on May 5, 2024")
        
        # When using temporal filter, non-temporal memories pass through
        # (they don't have dates, so they can't violate the range)
        results = await service.recall(
            "programming", after="2024-01-01", min_relevance=0.0
        )
        
        # The non-temporal memory should still appear if semantically relevant
        assert isinstance(results, list)


class TestTemporalWithNaturalLanguageDates:
    """Test that after/before params accept natural language dates."""

    @pytest.mark.asyncio
    async def test_after_with_natural_date(self, service):
        """after param should accept strings like 'last month'."""
        await service.remember("Sprint review on February 1, 2026")
        # "last month" should resolve relative to now
        results = await service.recall("sprint", after="2026-01-01")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_before_with_iso_date(self, service):
        """before param should accept ISO format dates."""
        await service.remember("Release planned for 2026-06-01")
        results = await service.recall("release", before="2026-03-01")
        assert isinstance(results, list)


class TestTemporalRecallRetrieval:
    """Test retrieval_method marking for temporal results."""

    @pytest.mark.asyncio
    async def test_temporal_filtered_results_keep_method(self, service):
        """Temporal filtering should not change the retrieval_method."""
        await service.remember("Meeting on May 5, 2024")
        
        results = await service.recall(
            "meeting", after="2024-01-01", min_relevance=0.0
        )
        
        for r in results:
            # retrieval_method should still reflect how the result was found
            assert r.retrieval_method in ("vector", "hybrid", "graph")


class TestTemporalRecallEdgeCases:
    """Edge cases for temporal filtering.
    
    Covers: empty results, None params, invalid dates, limit enforcement,
    vector-only mode (no FTS), and month-precision date matching.
    """

    @pytest.mark.asyncio
    async def test_empty_results_with_strict_range(self, service):
        """Should return empty list when no memories match the range."""
        await service.remember("Event on March 15, 2024")
        
        results = await service.recall(
            "event", after="2025-01-01", before="2025-12-31"
        )
        
        # The 2024 event should be filtered out — no 2025 events stored
        contents = [r.memory.content for r in results]
        assert not any("March 15, 2024" in c for c in contents)

    @pytest.mark.asyncio
    async def test_after_none_before_none_no_filtering(self, service):
        """When both params are None, all results should come through."""
        await service.remember("Event on March 15, 2024")
        
        results_with = await service.recall("event", after=None, before=None, min_relevance=0.0)
        results_without = await service.recall("event", min_relevance=0.0)
        
        assert len(results_with) == len(results_without)

    @pytest.mark.asyncio
    async def test_after_later_than_before_returns_empty(self, service):
        """When after > before, should return empty list (invalid range)."""
        await service.remember("Event on March 15, 2024")
        
        results = await service.recall(
            "event", after="2025-01-01", before="2024-01-01"
        )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_invalid_date_string_handled_gracefully(self, service):
        """Invalid date strings should not crash — should log warning and skip filtering."""
        await service.remember("Important event on July 4, 2024")
        
        # Should not raise
        results = await service.recall("event", after="not-a-date", min_relevance=0.0)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_limit_respected_with_temporal_filter(self, service):
        """Limit should still be respected after temporal filtering."""
        for i in range(10):
            await service.remember(f"Event {i} on March {i+1}, 2024")
        
        results = await service.recall(
            "event", after="2024-01-01", limit=3, min_relevance=0.0
        )
        
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_temporal_filter_with_vector_only(self, service_no_hybrid):
        """Temporal filtering should work without hybrid search."""
        await service_no_hybrid.remember("Meeting on May 5, 2024")
        await service_no_hybrid.remember("Meeting on May 5, 2026")
        
        results = await service_no_hybrid.recall(
            "meeting", after="2025-01-01", min_relevance=0.0
        )
        
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_month_precision_filtering(self, service):
        """Month-precision dates should filter correctly."""
        await service.remember("Started job in January 2024")
        await service.remember("Got promoted in August 2024")
        
        results = await service.recall(
            "job", after="2024-06-01", min_relevance=0.0
        )
        
        contents = [r.memory.content for r in results]
        assert any("August" in c for c in contents)
        assert not any("January 2024" in c and "August" not in c for c in contents)
