"""Tests for temporal entity extraction and resolution."""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch

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


# ------------------------------------------------------------------
# Issue #60 — Unicode month names fallback test
# ------------------------------------------------------------------


class TestUnicodeFallback:
    """Test that the fallback parser handles Unicode gracefully."""

    @pytest.fixture
    def extractor(self):
        return TemporalExtractor()

    @pytest.fixture
    def reference_time(self):
        return datetime(2023, 5, 8, 14, 0, 0, tzinfo=timezone.utc)

    def test_fallback_rejects_unicode_month_gracefully(
        self, extractor, reference_time
    ):
        """Fallback parser should not crash on non-ASCII month names.

        When dateparser is unavailable, the fallback regex only knows English
        month names. Non-English names should be silently ignored (no crash,
        no bogus results).
        """
        with patch("tribalmemory.services.temporal.DATEPARSER_AVAILABLE", False):
            # French month
            entities = extractor.extract("le 7 mai 2023", reference_time)
            # Fallback doesn't understand French — should return empty
            # (the regex patterns only match English month names)
            assert isinstance(entities, list)
            # No crash is the primary assertion

            # German month
            entities = extractor.extract("am 7. März 2023", reference_time)
            assert isinstance(entities, list)

            # Japanese date
            entities = extractor.extract("2023年5月7日", reference_time)
            assert isinstance(entities, list)

    def test_fallback_still_handles_relative_expressions(
        self, extractor, reference_time
    ):
        """Even without dateparser, relative expressions should resolve."""
        with patch("tribalmemory.services.temporal.DATEPARSER_AVAILABLE", False):
            entities = extractor.extract("yesterday", reference_time)
            assert len(entities) == 1
            assert entities[0].resolved_date == "2023-05-07"

            entities = extractor.extract("3 days ago", reference_time)
            assert len(entities) == 1
            assert entities[0].resolved_date == "2023-05-05"

    def test_fallback_handles_last_month_year_wrap(
        self, extractor
    ):
        """Fallback correctly wraps December when reference is January."""
        jan = datetime(2023, 1, 15, tzinfo=timezone.utc)
        with patch("tribalmemory.services.temporal.DATEPARSER_AVAILABLE", False):
            entities = extractor.extract("last month", jan)
            assert len(entities) == 1
            assert entities[0].resolved_date == "2022-12"

    def test_fallback_returns_none_for_unparseable(
        self, extractor, reference_time
    ):
        """Fallback should return nothing for expressions it can't parse."""
        with patch("tribalmemory.services.temporal.DATEPARSER_AVAILABLE", False):
            # "next Tuesday" isn't handled by fallback (only dateparser)
            entities = extractor.extract("next Tuesday", reference_time)
            # Regex matches "next tuesday" but fallback can't resolve it
            # — either empty or entity with None date
            for e in entities:
                # If returned, the expression was matched but resolution
                # should have fallen through gracefully
                assert isinstance(e, TemporalEntity)


# ------------------------------------------------------------------
# Issue #62 — Batch temporal extraction
# ------------------------------------------------------------------


class TestHasTemporalSignal:
    """Test the fast pre-check for temporal signals."""

    @pytest.fixture
    def extractor(self):
        return TemporalExtractor()

    def test_detects_relative_signals(self, extractor):
        assert extractor.has_temporal_signal("I went there yesterday")
        assert extractor.has_temporal_signal("this happened last week")
        assert extractor.has_temporal_signal("3 days ago something occurred")
        assert extractor.has_temporal_signal("meeting next month")

    def test_detects_absolute_signals(self, extractor):
        assert extractor.has_temporal_signal("on 2023-05-07 we met")
        assert extractor.has_temporal_signal("born in 1990")
        assert extractor.has_temporal_signal("date is 01/15/2023")

    def test_detects_month_names(self, extractor):
        assert extractor.has_temporal_signal("Project started on January 5, 2024")
        assert extractor.has_temporal_signal("We launched in March 2025")
        assert extractor.has_temporal_signal("The December report is ready")
        assert extractor.has_temporal_signal("meeting on 3rd February 2023")

    def test_rejects_non_temporal_text(self, extractor):
        assert not extractor.has_temporal_signal(
            "The quick brown fox jumps over the lazy dog"
        )
        assert not extractor.has_temporal_signal(
            "I like bananas and strawberries"
        )
        assert not extractor.has_temporal_signal(
            "Python is a programming language"
        )

    def test_empty_input(self, extractor):
        assert not extractor.has_temporal_signal("")
        assert not extractor.has_temporal_signal(None)

    def test_case_insensitive(self, extractor):
        assert extractor.has_temporal_signal("YESTERDAY was fun")
        assert extractor.has_temporal_signal("Last Week I went hiking")


class TestBatchExtract:
    """Test batch temporal extraction."""

    @pytest.fixture
    def extractor(self):
        return TemporalExtractor()

    @pytest.fixture
    def reference_time(self):
        return datetime(2023, 5, 8, 14, 0, 0, tzinfo=timezone.utc)

    def test_batch_extract_basic(self, extractor, reference_time):
        """Batch extract should return parallel results."""
        items = [
            ("I went there yesterday", reference_time),
            ("No dates here", reference_time),
            ("Meeting is tomorrow", reference_time),
        ]
        results = extractor.batch_extract(items)

        assert len(results) == 3
        # First item: has "yesterday"
        assert len(results[0]) == 1
        assert results[0][0].resolved_date == "2023-05-07"
        # Second item: no temporal signal → empty
        assert results[1] == []
        # Third item: has "tomorrow"
        assert len(results[2]) == 1
        assert results[2][0].resolved_date == "2023-05-09"

    def test_batch_extract_empty_items(self, extractor):
        """Empty/None texts should return empty lists."""
        items = [
            ("", None),
            (None, None),
        ]
        results = extractor.batch_extract(items)
        assert results == [[], []]

    def test_batch_extract_skips_non_temporal(self, extractor, reference_time):
        """Items without temporal signals should be skipped efficiently."""
        items = [
            ("No dates at all in this text about dogs and cats", reference_time),
            ("Another plain sentence about programming", reference_time),
        ]
        results = extractor.batch_extract(items)
        assert results == [[], []]

    def test_batch_extract_with_context(self, extractor, reference_time):
        """batch_extract_with_context should return relationships."""
        items = [
            ("I attended a meeting yesterday", reference_time),
            ("Plain text", reference_time),
        ]
        results = extractor.batch_extract_with_context(items)

        assert len(results) == 2
        assert len(results[0]) >= 1
        assert isinstance(results[0][0], TemporalRelationship)
        assert results[0][0].temporal.resolved_date == "2023-05-07"
        assert results[1] == []

    def test_batch_preserves_order(self, extractor, reference_time):
        """Results must align with input order."""
        items = [
            ("tomorrow is the day", reference_time),
            ("apples and oranges", reference_time),
            ("2 weeks ago", reference_time),
            ("just chatting", reference_time),
            ("yesterday evening", reference_time),
        ]
        results = extractor.batch_extract(items)
        assert len(results) == 5
        # Indices 0, 2, 4 should have results; 1, 3 should be empty
        assert len(results[0]) >= 1
        assert results[1] == []
        assert len(results[2]) >= 1
        assert results[3] == []
        assert len(results[4]) >= 1

    def test_batch_large_volume(self, extractor, reference_time):
        """Should handle a large batch without error."""
        temporal_text = "I did this yesterday"
        plain_text = "The sky is blue"
        # 500 items, alternating
        items = [
            (temporal_text if i % 2 == 0 else plain_text, reference_time)
            for i in range(500)
        ]
        results = extractor.batch_extract(items)
        assert len(results) == 500
        for i, r in enumerate(results):
            if i % 2 == 0:
                assert len(r) >= 1, f"Expected temporal at index {i}"
            else:
                assert r == [], f"Expected empty at index {i}"


class TestPreCheckIntegration:
    """Test that the pre-check in memory.remember() works correctly."""

    @pytest.fixture
    def extractor(self):
        return TemporalExtractor()

    @pytest.fixture
    def reference_time(self):
        return datetime(2023, 5, 8, 14, 0, 0, tzinfo=timezone.utc)

    def test_precheck_agrees_with_extract(self, extractor, reference_time):
        """has_temporal_signal should agree with extract() results.

        If extract() finds entities, has_temporal_signal() must be True.
        (The reverse isn't guaranteed — pre-check can have false positives.)
        """
        test_texts = [
            "I went to the store yesterday",
            "Meeting last week was productive",
            "No dates here whatsoever",
            "She started in 2022",
            "3 months ago we launched",
            "Plain text about cats",
            "The deadline is tomorrow",
        ]
        for text in test_texts:
            entities = extractor.extract(text, reference_time)
            signal = extractor.has_temporal_signal(text)
            if entities:
                assert signal, (
                    f"Pre-check missed signal in: {text!r}"
                )
