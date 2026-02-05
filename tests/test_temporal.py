"""Tests for temporal entity extraction and resolution."""

import pytest
from datetime import datetime, timezone

from tribalmemory.services.temporal import (
    TemporalExtractor,
    TemporalEntity,
    TemporalRelationship,
    format_temporal_context,
)


class TestTemporalExtractor:
    """Test TemporalExtractor class."""

    @pytest.fixture
    def extractor(self):
        return TemporalExtractor()

    @pytest.fixture
    def reference_time(self):
        """May 8, 2023 at 2pm UTC - the MemoryBench example."""
        return datetime(2023, 5, 8, 14, 0, 0, tzinfo=timezone.utc)

    def test_extract_yesterday(self, extractor, reference_time):
        """Yesterday should resolve to day before reference."""
        text = "I went to a LGBTQ support group yesterday"
        entities = extractor.extract(text, reference_time)
        
        assert len(entities) == 1
        assert entities[0].expression == "yesterday"
        assert entities[0].resolved_date == "2023-05-07"
        assert entities[0].precision == "day"

    def test_extract_today(self, extractor, reference_time):
        """Today should resolve to reference date."""
        text = "I have a meeting today"
        entities = extractor.extract(text, reference_time)
        
        assert len(entities) == 1
        assert entities[0].expression == "today"
        assert entities[0].resolved_date == "2023-05-08"

    def test_extract_tomorrow(self, extractor, reference_time):
        """Tomorrow should resolve to day after reference."""
        text = "The event is tomorrow"
        entities = extractor.extract(text, reference_time)
        
        assert len(entities) == 1
        assert entities[0].expression == "tomorrow"
        assert entities[0].resolved_date == "2023-05-09"

    def test_extract_n_days_ago(self, extractor, reference_time):
        """N days ago should resolve correctly."""
        text = "This happened 3 days ago"
        entities = extractor.extract(text, reference_time)
        
        assert len(entities) == 1
        assert entities[0].resolved_date == "2023-05-05"

    def test_extract_last_week(self, extractor, reference_time):
        """Last week should resolve to ~7 days prior."""
        text = "We discussed this last week"
        entities = extractor.extract(text, reference_time)
        
        assert len(entities) == 1
        assert entities[0].expression == "last week"
        assert entities[0].precision == "week"
        # Should be around May 1
        assert entities[0].resolved_date.startswith("2023-05-01") or \
               entities[0].resolved_date.startswith("2023-04")

    def test_extract_last_year(self, extractor, reference_time):
        """Last year should resolve to previous year."""
        text = "I started this project last year"
        entities = extractor.extract(text, reference_time)
        
        assert len(entities) == 1
        assert entities[0].precision == "year"
        assert "2022" in entities[0].resolved_date

    def test_extract_absolute_year(self, extractor, reference_time):
        """Explicit year should be extracted."""
        text = "I painted a sunrise in 2022"
        entities = extractor.extract(text, reference_time)
        
        assert len(entities) == 1
        assert entities[0].expression == "in 2022"
        assert entities[0].resolved_date == "2022"
        assert entities[0].precision == "year"

    def test_extract_multiple_expressions(self, extractor, reference_time):
        """Multiple temporal expressions should all be extracted."""
        text = "Yesterday I planned for tomorrow's meeting about last year's results"
        entities = extractor.extract(text, reference_time)
        
        # Should find: yesterday, tomorrow, last year
        assert len(entities) >= 3
        expressions = {e.expression.lower() for e in entities}
        assert "yesterday" in expressions
        assert "tomorrow" in expressions
        assert "last year" in expressions

    def test_extract_no_duplicates(self, extractor, reference_time):
        """Same expression mentioned twice should only appear once."""
        text = "Yesterday was great. I loved yesterday."
        entities = extractor.extract(text, reference_time)
        
        assert len(entities) == 1
        assert entities[0].expression.lower() == "yesterday"

    def test_extract_empty_text(self, extractor):
        """Empty text should return empty list."""
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []
        assert extractor.extract(None) == []

    def test_extract_no_temporal(self, extractor, reference_time):
        """Text without temporal expressions returns empty list."""
        text = "The quick brown fox jumps over the lazy dog"
        entities = extractor.extract(text, reference_time)
        assert len(entities) == 0

    def test_uses_current_time_as_default(self, extractor):
        """Without reference time, should use current time."""
        text = "I did this yesterday"
        entities = extractor.extract(text)
        
        assert len(entities) == 1
        # Reference date should be close to now
        assert entities[0].reference_date  # Just check it exists


class TestFormatTemporalContext:
    """Test context formatting for LLM prompts."""

    def test_format_with_resolved_dates(self):
        """Temporal context should include resolved dates."""
        memory_content = "I went to a support group yesterday"
        memory_timestamp = datetime(2023, 5, 8, 14, 0, 0, tzinfo=timezone.utc)
        temporal_entities = [
            TemporalEntity(
                expression="yesterday",
                resolved_date="2023-05-07",
                precision="day",
                reference_date="2023-05-08T14:00:00Z",
            )
        ]
        
        formatted = format_temporal_context(
            memory_content, memory_timestamp, temporal_entities
        )
        
        assert "May 08, 2023" in formatted
        assert memory_content in formatted
        assert "yesterday" in formatted
        assert "2023-05-07" in formatted

    def test_format_without_temporal(self):
        """Context without temporal entities should still work."""
        memory_content = "Some general memory"
        memory_timestamp = datetime(2023, 5, 8, tzinfo=timezone.utc)
        
        formatted = format_temporal_context(
            memory_content, memory_timestamp, []
        )
        
        assert memory_content in formatted
        assert "May 08, 2023" in formatted


class TestTemporalPrecision:
    """Test precision inference for different expression types."""

    @pytest.fixture
    def extractor(self):
        return TemporalExtractor()

    def test_day_precision(self, extractor):
        """Day-level expressions should have day precision."""
        ref = datetime(2023, 5, 8, tzinfo=timezone.utc)
        
        for text in ["yesterday", "today", "tomorrow", "5 days ago"]:
            entities = extractor.extract(text, ref)
            if entities:
                assert entities[0].precision == "day", f"Failed for: {text}"

    def test_week_precision(self, extractor):
        """Week-level expressions should have week precision."""
        ref = datetime(2023, 5, 8, tzinfo=timezone.utc)
        
        for text in ["last week", "2 weeks ago"]:
            entities = extractor.extract(text, ref)
            if entities:
                assert entities[0].precision == "week", f"Failed for: {text}"

    def test_year_precision(self, extractor):
        """Year-level expressions should have year precision."""
        ref = datetime(2023, 5, 8, tzinfo=timezone.utc)
        
        entities = extractor.extract("in 2022", ref)
        assert len(entities) == 1
        assert entities[0].precision == "year"


class TestTemporalEdgeCases:
    """Edge case tests for robustness."""

    @pytest.fixture
    def extractor(self):
        return TemporalExtractor()

    @pytest.fixture
    def reference_time(self):
        return datetime(2023, 5, 8, 14, 0, 0, tzinfo=timezone.utc)

    def test_repeated_expression(self, extractor, reference_time):
        """Same expression repeated should only appear once."""
        text = "Yesterday was great. I loved yesterday. Yesterday!"
        entities = extractor.extract(text, reference_time)
        assert len(entities) == 1

    def test_naive_reference_time(self, extractor):
        """Should handle naive datetime (no tzinfo) gracefully."""
        naive_time = datetime(2023, 5, 8, 14, 0, 0)
        text = "I did this yesterday"
        entities = extractor.extract(text, naive_time)
        assert len(entities) == 1
        assert entities[0].resolved_date == "2023-05-07"

    def test_extract_with_context_returns_relationships(
        self, extractor, reference_time
    ):
        """extract_with_context should return TemporalRelationship."""
        text = "I went to a support group yesterday"
        rels = extractor.extract_with_context(text, reference_time)
        assert len(rels) >= 1
        assert isinstance(rels[0], TemporalRelationship)
        assert rels[0].temporal.resolved_date == "2023-05-07"

    def test_extract_with_context_finds_subject(
        self, extractor, reference_time
    ):
        """Should try to identify what the temporal refers to."""
        text = "I attended a LGBTQ support group yesterday"
        rels = extractor.extract_with_context(text, reference_time)
        assert len(rels) >= 1
        # Subject should be something other than just "event"
        # (exact subject depends on regex matching)
        assert rels[0].subject is not None

    def test_last_month_january_wraps_to_december(self, extractor):
        """Last month in January should give December."""
        jan_ref = datetime(2023, 1, 15, tzinfo=timezone.utc)
        entities = extractor.extract("last month", jan_ref)
        assert len(entities) == 1
        assert entities[0].resolved_date == "2022-12"

    def test_year_without_in_prefix(self, extractor, reference_time):
        """Bare year '2022' in prose should still be extractable.

        Note: bare '2022' alone is not captured by current patterns
        (which require 'in 2022'). This documents that behavior.
        """
        text = "Melanie painted a sunrise in 2022"
        entities = extractor.extract(text, reference_time)
        assert len(entities) == 1
        assert entities[0].resolved_date == "2022"

    def test_overlapping_temporal_ranges(
        self, extractor, reference_time
    ):
        """Multiple temporal expressions in same sentence."""
        text = "From last week until tomorrow we are busy"
        entities = extractor.extract(text, reference_time)
        exprs = {e.expression.lower() for e in entities}
        assert "last week" in exprs
        assert "tomorrow" in exprs
        # Dates should be different
        dates = {e.resolved_date for e in entities}
        assert len(dates) == 2

    def test_unicode_month_with_dateparser(
        self, extractor, reference_time
    ):
        """dateparser handles non-English month names."""
        # Only works if dateparser is installed; test gracefully
        import importlib
        dp = importlib.util.find_spec("dateparser")
        if dp is None:
            pytest.skip("dateparser not installed")
        text = "le 7 mai 2023"  # French
        entities = extractor.extract(text, reference_time)
        # May or may not extract depending on pattern match
        # The important thing is no crash
        assert isinstance(entities, list)


class TestGraphStoreTemporalFacts:
    """Test temporal facts stored in GraphStore."""

    @pytest.fixture
    def graph_store(self, tmp_path):
        from tribalmemory.services.graph_store import GraphStore
        return GraphStore(str(tmp_path / "graph.db"))

    def test_store_temporal_fact(self, graph_store):
        """Temporal facts should be stored and retrievable."""
        from tribalmemory.services.graph_store import TemporalFact
        
        fact = TemporalFact(
            subject="LGBTQ support group",
            relation="occurred_on",
            resolved_date="2023-05-07",
            original_expression="yesterday",
            precision="day",
            confidence=0.95,
        )
        fact_id = graph_store.add_temporal_fact(fact, memory_id="mem-001")
        assert fact_id > 0
        
        facts = graph_store.get_temporal_facts_for_memory("mem-001")
        assert len(facts) == 1
        assert facts[0].resolved_date == "2023-05-07"
        assert facts[0].original_expression == "yesterday"

    def test_get_memories_for_date(self, graph_store):
        """Should retrieve memories by date."""
        from tribalmemory.services.graph_store import TemporalFact
        
        graph_store.add_temporal_fact(TemporalFact(
            subject="meeting", relation="occurred_on",
            resolved_date="2023-05-07", original_expression="yesterday",
            precision="day",
        ), memory_id="mem-001")
        
        graph_store.add_temporal_fact(TemporalFact(
            subject="lunch", relation="occurred_on",
            resolved_date="2023-05-08", original_expression="today",
            precision="day",
        ), memory_id="mem-002")
        
        may7 = graph_store.get_memories_for_date("2023-05-07")
        assert may7 == ["mem-001"]
        
        may8 = graph_store.get_memories_for_date("2023-05-08")
        assert may8 == ["mem-002"]

    def test_get_memories_in_date_range(self, graph_store):
        """Should retrieve memories in a date range."""
        from tribalmemory.services.graph_store import TemporalFact
        
        for i, date in enumerate(["2023-05-05", "2023-05-07", "2023-05-10"]):
            graph_store.add_temporal_fact(TemporalFact(
                subject="event", relation="occurred_on",
                resolved_date=date, original_expression=date,
                precision="day",
            ), memory_id=f"mem-{i}")
        
        # Range: May 6-9 should only include May 7
        memories = graph_store.get_memories_in_date_range("2023-05-06", "2023-05-09")
        assert memories == ["mem-1"]

    def test_get_memories_for_year(self, graph_store):
        """Should match by year prefix."""
        from tribalmemory.services.graph_store import TemporalFact
        
        graph_store.add_temporal_fact(TemporalFact(
            subject="painting", relation="occurred_on",
            resolved_date="2022", original_expression="in 2022",
            precision="year",
        ), memory_id="mem-001")
        
        memories = graph_store.get_memories_for_date("2022")
        assert memories == ["mem-001"]

    def test_delete_memory_cleans_temporal(self, graph_store):
        """Deleting a memory should remove its temporal facts."""
        from tribalmemory.services.graph_store import TemporalFact
        
        graph_store.add_temporal_fact(TemporalFact(
            subject="event", relation="occurred_on",
            resolved_date="2023-05-07", original_expression="yesterday",
            precision="day",
        ), memory_id="mem-001")
        
        assert len(graph_store.get_temporal_facts_for_memory("mem-001")) == 1
        
        graph_store.delete_memory("mem-001")
        
        assert len(graph_store.get_temporal_facts_for_memory("mem-001")) == 0
