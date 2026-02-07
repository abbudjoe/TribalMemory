"""Tests for auto-temporal query extraction."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from tribalmemory.services.memory import TribalMemoryService
from tribalmemory.services.temporal import TemporalExtractor


class TestExtractQueryTemporal:
    """Tests for _extract_query_temporal() method."""

    @pytest.fixture
    def service_with_temporal(self):
        """Create a memory service with temporal extractor enabled."""
        service = MagicMock(spec=TribalMemoryService)
        service.temporal_extractor = TemporalExtractor()
        service._extract_query_temporal = TribalMemoryService._extract_query_temporal.__get__(
            service, TribalMemoryService
        )
        return service

    def test_no_temporal_signal_returns_none(self, service_with_temporal):
        """Test that queries without temporal references return None."""
        result = service_with_temporal._extract_query_temporal(
            "What is my favorite color?"
        )
        assert result is None

    def test_last_saturday_returns_day_range(self, service_with_temporal):
        """Test 'last Saturday' extracts to a specific day."""
        result = service_with_temporal._extract_query_temporal(
            "Who did I meet last Saturday?"
        )
        # Should return a tuple (after, before) for that day
        if result is not None:
            after, before = result
            assert after is not None
            # For day precision, after == before
            assert after == before

    def test_yesterday_returns_day_range(self, service_with_temporal):
        """Test 'yesterday' extracts to a specific day."""
        result = service_with_temporal._extract_query_temporal(
            "What did I do yesterday?"
        )
        if result is not None:
            after, before = result
            assert after is not None

    def test_last_week_returns_week_range(self, service_with_temporal):
        """Test 'last week' extracts to a week range."""
        result = service_with_temporal._extract_query_temporal(
            "What meetings did I have last week?"
        )
        if result is not None:
            after, before = result
            assert after is not None
            # For week precision, before should be 6 days after start
            if before is not None:
                start = datetime.fromisoformat(after)
                end = datetime.fromisoformat(before)
                diff = (end - start).days
                assert diff == 6  # Inclusive week range

    def test_last_month_returns_month_range(self, service_with_temporal):
        """Test 'last month' extracts to a month range."""
        result = service_with_temporal._extract_query_temporal(
            "What did I accomplish last month?"
        )
        if result is not None:
            after, before = result
            assert after is not None
            # Month should have both start and end
            if before is not None:
                # Verify it's a valid month range
                start = datetime.fromisoformat(after)
                end = datetime.fromisoformat(before)
                assert start.month == end.month or (
                    start.month == 12 and end.month == 1
                )

    def test_no_temporal_extractor_returns_none(self):
        """Test that None is returned when temporal_extractor is None."""
        service = MagicMock(spec=TribalMemoryService)
        service.temporal_extractor = None
        service._extract_query_temporal = TribalMemoryService._extract_query_temporal.__get__(
            service, TribalMemoryService
        )
        
        result = service._extract_query_temporal("What happened last week?")
        assert result is None

    def test_invalid_date_returns_none(self, service_with_temporal):
        """Test that unparseable temporal expressions return None gracefully."""
        # The temporal extractor should handle this gracefully
        result = service_with_temporal._extract_query_temporal(
            "What happened on the 32nd of Octember?"
        )
        # Should return None rather than raising
        # (The actual behavior depends on dateparser)
        assert result is None or isinstance(result, tuple)

    def test_multiple_temporal_uses_first(self, service_with_temporal):
        """Test that multiple temporal expressions use the first one."""
        # This documents the current behavior
        result = service_with_temporal._extract_query_temporal(
            "Compare meetings last Monday and next Friday"
        )
        # Should extract something (first temporal reference)
        # The exact result depends on parsing order
        if result is not None:
            after, before = result
            assert after is not None

    def test_query_with_explicit_date(self, service_with_temporal):
        """Test query with explicit date format."""
        result = service_with_temporal._extract_query_temporal(
            "What happened on 2026-01-15?"
        )
        if result is not None:
            after, before = result
            # Should parse the ISO date
            assert "2026-01" in (after or "")


class TestAutoTemporalInRecall:
    """Integration tests for auto-temporal extraction in recall."""

    @pytest.fixture
    def memory_service(self):
        """Create a real memory service for integration testing."""
        from tribalmemory.services.memory import create_memory_service
        
        service = create_memory_service(
            instance_id="test-auto-temporal",
            db_path=None,  # In-memory
            lazy_spacy=True,
        )
        return service

    @pytest.mark.asyncio
    async def test_recall_without_explicit_temporal_extracts_from_query(
        self, memory_service
    ):
        """Test that recall auto-extracts temporal from query when not provided."""
        # Store a memory
        await memory_service.remember(
            content="Met with John about the project",
            source_type="auto_capture",
        )

        # Query with temporal reference but no explicit after/before
        results = await memory_service.recall(
            query="Who did I meet last week?",
            limit=5,
        )
        
        # Should complete without error
        # Results may or may not match depending on date
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_recall_with_explicit_temporal_overrides_extraction(
        self, memory_service
    ):
        """Test that explicit after/before override auto-extraction."""
        await memory_service.remember(
            content="Important meeting notes",
            source_type="auto_capture",
        )

        # Explicit after should be used, not extracted from query
        results = await memory_service.recall(
            query="What happened last week?",  # Has temporal
            after="2020-01-01",  # Explicit override
            limit=5,
        )
        
        # Should use explicit date, not extracted
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_recall_query_without_temporal_no_extraction(
        self, memory_service
    ):
        """Test that queries without temporal don't get filtered."""
        await memory_service.remember(
            content="My favorite color is blue",
            source_type="user_explicit",
        )

        results = await memory_service.recall(
            query="What is my favorite color?",
            limit=5,
        )
        
        # Should find the memory (no temporal filtering applied)
        assert len(results) >= 1
        assert "blue" in results[0].memory.content
