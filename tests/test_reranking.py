"""Tests for result reranking (issue #39).

TDD: RED → GREEN → REFACTOR
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from tribalmemory.interfaces import MemoryEntry, MemorySource, RecallResult
from tribalmemory.server.config import SearchConfig
from tribalmemory.services.reranker import (
    IReranker,
    NoopReranker,
    HeuristicReranker,
    CrossEncoderReranker,
    create_reranker,
)


class TestNoopReranker:
    """Tests for pass-through reranker."""

    def test_noop_returns_unchanged(self):
        """NoopReranker should return results unchanged."""
        reranker = NoopReranker()
        
        results = [
            self._create_result("mem-1", "content 1", 0.9),
            self._create_result("mem-2", "content 2", 0.8),
            self._create_result("mem-3", "content 3", 0.7),
        ]
        
        reranked = reranker.rerank("test query", results, top_k=2)
        
        assert len(reranked) == 2
        assert reranked[0].memory.id == "mem-1"
        assert reranked[1].memory.id == "mem-2"
        assert reranked[0].similarity_score == 0.9
        assert reranked[1].similarity_score == 0.8

    def test_noop_respects_top_k(self):
        """NoopReranker should respect top_k parameter."""
        reranker = NoopReranker()
        
        results = [
            self._create_result(f"mem-{i}", f"content {i}", 0.9 - i * 0.1)
            for i in range(10)
        ]
        
        reranked = reranker.rerank("query", results, top_k=3)
        assert len(reranked) == 3

    @staticmethod
    def _create_result(memory_id: str, content: str, score: float) -> RecallResult:
        """Helper to create RecallResult."""
        return RecallResult(
            memory=MemoryEntry(
                id=memory_id,
                content=content,
                source_instance="test",
                source_type=MemorySource.AUTO_CAPTURE,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            ),
            similarity_score=score,
            retrieval_time_ms=1.0,
        )


class TestHeuristicReranker:
    """Tests for heuristic reranking with recency, tags, and length signals."""

    def test_recency_boost(self):
        """Newer memories should get higher scores."""
        reranker = HeuristicReranker(
            recency_decay_days=30.0,
            tag_boost_weight=0.0,  # Disable tag boost for this test
        )
        
        now = datetime.utcnow()
        old_date = now - timedelta(days=60)
        recent_date = now - timedelta(days=5)
        
        results = [
            self._create_result("old", "old memory", 0.8, created_at=old_date),
            self._create_result("recent", "recent memory", 0.8, created_at=recent_date),
        ]
        
        reranked = reranker.rerank("query", results, top_k=2)
        
        # Recent should be first due to recency boost
        assert reranked[0].memory.id == "recent"
        assert reranked[0].similarity_score > 0.8  # Boosted
        assert reranked[1].similarity_score >= 0.8  # Old memory baseline preserved
        assert reranked[0].similarity_score > reranked[1].similarity_score  # Boost direction

    def test_tag_boost(self):
        """Memories with matching tags should score higher."""
        reranker = HeuristicReranker(
            recency_decay_days=1000.0,  # Disable recency for this test
            tag_boost_weight=0.2,
        )
        
        results = [
            self._create_result("no-tags", "content", 0.8, tags=[]),
            self._create_result("match", "content", 0.8, tags=["python", "coding"]),
        ]
        
        # Query contains "python"
        reranked = reranker.rerank("python programming", results, top_k=2)
        
        # Tagged result should rank higher
        assert reranked[0].memory.id == "match"
        assert reranked[0].similarity_score > 0.8

    def test_length_penalty_very_short(self):
        """Very short content should get slight penalty."""
        reranker = HeuristicReranker(
            recency_decay_days=1000.0,
            tag_boost_weight=0.0,
        )
        
        results = [
            self._create_result("short", "Hi", 0.8),
            self._create_result("normal", "This is a normal length memory with sufficient detail.", 0.8),
        ]
        
        reranked = reranker.rerank("query", results, top_k=2)
        
        # Normal length should rank higher
        assert reranked[0].memory.id == "normal"

    def test_length_penalty_very_long(self):
        """Very long content should get slight penalty."""
        reranker = HeuristicReranker(
            recency_decay_days=1000.0,
            tag_boost_weight=0.0,
        )
        
        long_content = "word " * 500  # Very long
        normal_content = "This is a normal length memory."
        
        results = [
            self._create_result("long", long_content, 0.8),
            self._create_result("normal", normal_content, 0.8),
        ]
        
        reranked = reranker.rerank("query", results, top_k=2)
        
        # Normal length should rank higher
        assert reranked[0].memory.id == "normal"

    def test_combined_scoring(self):
        """Test combined recency + tag + length scoring."""
        reranker = HeuristicReranker(
            recency_decay_days=30.0,
            tag_boost_weight=0.15,
        )
        
        now = datetime.utcnow()
        
        results = [
            # Old, no tags, normal length → baseline
            self._create_result(
                "baseline",
                "Some older memory content here",
                0.7,
                created_at=now - timedelta(days=90),
                tags=[],
            ),
            # Recent, matching tags, normal length → should win
            self._create_result(
                "winner",
                "Recent python development notes",
                0.7,
                created_at=now - timedelta(days=2),
                tags=["python", "dev"],
            ),
        ]
        
        reranked = reranker.rerank("python coding", results, top_k=2)
        
        assert reranked[0].memory.id == "winner"
        assert reranked[0].similarity_score > reranked[1].similarity_score

    def test_recency_boost_extreme_age(self):
        """Very old memories (1+ year) should still get valid scores without issues."""
        reranker = HeuristicReranker(
            recency_decay_days=30.0,
            tag_boost_weight=0.0,
        )
        
        now = datetime.utcnow()
        very_old = now - timedelta(days=365)
        ancient = now - timedelta(days=3650)  # 10 years
        
        results = [
            self._create_result("very-old", "old memory", 0.8, created_at=very_old),
            self._create_result("ancient", "ancient memory", 0.8, created_at=ancient),
            self._create_result("recent", "recent memory", 0.8, created_at=now - timedelta(days=1)),
        ]
        
        reranked = reranker.rerank("query", results, top_k=3)
        
        # Recent should rank first
        assert reranked[0].memory.id == "recent"
        # All scores should be positive (no negative scores from extreme decay)
        for r in reranked:
            assert r.similarity_score > 0, f"Score for {r.memory.id} should be positive"
        # Ancient shouldn't score higher than very old
        ancient_score = next(r.similarity_score for r in reranked if r.memory.id == "ancient")
        very_old_score = next(r.similarity_score for r in reranked if r.memory.id == "very-old")
        assert very_old_score >= ancient_score

    def test_preserves_original_order_when_tied(self):
        """When scores are equal, preserve original order."""
        reranker = HeuristicReranker()
        
        now = datetime.utcnow()
        results = [
            self._create_result("first", "content", 0.8, created_at=now),
            self._create_result("second", "content", 0.8, created_at=now),
        ]
        
        reranked = reranker.rerank("query", results, top_k=2)
        
        # Original order preserved
        assert reranked[0].memory.id == "first"
        assert reranked[1].memory.id == "second"

    @staticmethod
    def _create_result(
        memory_id: str,
        content: str,
        score: float,
        created_at: datetime | None = None,
        tags: list[str] | None = None,
    ) -> RecallResult:
        """Helper to create RecallResult with custom fields."""
        return RecallResult(
            memory=MemoryEntry(
                id=memory_id,
                content=content,
                source_instance="test",
                source_type=MemorySource.AUTO_CAPTURE,
                created_at=created_at or datetime.utcnow(),
                updated_at=datetime.utcnow(),
                tags=tags or [],
            ),
            similarity_score=score,
            retrieval_time_ms=1.0,
        )


class TestCrossEncoderReranker:
    """Tests for cross-encoder model-based reranking."""

    def test_cross_encoder_unavailable(self):
        """Should raise when cross-encoder requested but unavailable."""
        with patch("tribalmemory.services.reranker.CROSS_ENCODER_AVAILABLE", False):
            with pytest.raises(ImportError, match="sentence-transformers"):
                CrossEncoderReranker()

    def test_cross_encoder_reranks_by_model_score(self):
        """Should reorder results based on cross-encoder scores."""
        # Mock the model
        mock_model = MagicMock()
        # Return scores: query+mem2 is best, query+mem1 is worst
        mock_model.predict.return_value = [0.3, 0.9, 0.5]  # mem1, mem2, mem3
        
        with patch("tribalmemory.services.reranker.CROSS_ENCODER_AVAILABLE", True):
            with patch("tribalmemory.services.reranker.CrossEncoder", return_value=mock_model):
                reranker = CrossEncoderReranker()
                
                results = [
                    self._create_result("mem-1", "content one", 0.8),
                    self._create_result("mem-2", "content two", 0.7),
                    self._create_result("mem-3", "content three", 0.9),
                ]
                
                reranked = reranker.rerank("test query", results, top_k=3)
                
                # Should be reordered by cross-encoder score
                assert reranked[0].memory.id == "mem-2"  # Score 0.9
                assert reranked[1].memory.id == "mem-3"  # Score 0.5
                assert reranked[2].memory.id == "mem-1"  # Score 0.3

    def test_cross_encoder_respects_top_k(self):
        """Should return only top_k results."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        with patch("tribalmemory.services.reranker.CROSS_ENCODER_AVAILABLE", True):
            with patch("tribalmemory.services.reranker.CrossEncoder", return_value=mock_model):
                reranker = CrossEncoderReranker()
                
                results = [
                    self._create_result(f"mem-{i}", f"content {i}", 0.8)
                    for i in range(5)
                ]
                
                reranked = reranker.rerank("query", results, top_k=2)
                
                assert len(reranked) == 2

    def test_cross_encoder_updates_scores(self):
        """Should update similarity_score with cross-encoder score."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95]
        
        with patch("tribalmemory.services.reranker.CROSS_ENCODER_AVAILABLE", True):
            with patch("tribalmemory.services.reranker.CrossEncoder", return_value=mock_model):
                reranker = CrossEncoderReranker()
                
                results = [self._create_result("mem-1", "content", 0.7)]
                reranked = reranker.rerank("query", results, top_k=1)
                
                assert reranked[0].similarity_score == 0.95

    @staticmethod
    def _create_result(memory_id: str, content: str, score: float) -> RecallResult:
        """Helper to create RecallResult."""
        return RecallResult(
            memory=MemoryEntry(
                id=memory_id,
                content=content,
                source_instance="test",
                source_type=MemorySource.AUTO_CAPTURE,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            ),
            similarity_score=score,
            retrieval_time_ms=1.0,
        )


class TestRerankerFactory:
    """Tests for create_reranker factory function."""

    def test_factory_none(self):
        """Should create NoopReranker for 'none'."""
        config = SearchConfig(reranking="none")
        reranker = create_reranker(config)
        assert isinstance(reranker, NoopReranker)

    def test_factory_heuristic(self):
        """Should create HeuristicReranker for 'heuristic'."""
        config = SearchConfig(
            reranking="heuristic",
            recency_decay_days=20.0,
            tag_boost_weight=0.2,
        )
        reranker = create_reranker(config)
        assert isinstance(reranker, HeuristicReranker)
        assert reranker.recency_decay_days == 20.0
        assert reranker.tag_boost_weight == 0.2

    def test_factory_cross_encoder(self):
        """Should create CrossEncoderReranker for 'cross-encoder'."""
        with patch("tribalmemory.services.reranker.CROSS_ENCODER_AVAILABLE", True):
            with patch("tribalmemory.services.reranker.CrossEncoder"):
                config = SearchConfig(reranking="cross-encoder")
                reranker = create_reranker(config)
                assert isinstance(reranker, CrossEncoderReranker)

    def test_factory_cross_encoder_unavailable(self):
        """Should raise when cross-encoder requested but unavailable."""
        with patch("tribalmemory.services.reranker.CROSS_ENCODER_AVAILABLE", False):
            config = SearchConfig(reranking="cross-encoder")
            with pytest.raises(ImportError, match="sentence-transformers"):
                create_reranker(config)

    def test_factory_auto_with_cross_encoder(self):
        """Auto should use cross-encoder when available."""
        with patch("tribalmemory.services.reranker.CROSS_ENCODER_AVAILABLE", True):
            with patch("tribalmemory.services.reranker.CrossEncoder"):
                config = SearchConfig(reranking="auto")
                reranker = create_reranker(config)
                assert isinstance(reranker, CrossEncoderReranker)

    def test_factory_auto_fallback(self):
        """Auto should fall back to heuristic when cross-encoder unavailable."""
        with patch("tribalmemory.services.reranker.CROSS_ENCODER_AVAILABLE", False):
            config = SearchConfig(reranking="auto")
            reranker = create_reranker(config)
            assert isinstance(reranker, HeuristicReranker)

    def test_factory_invalid_mode(self):
        """Should raise on invalid reranking mode."""
        with pytest.raises(ValueError, match="Invalid reranking mode"):
            SearchConfig(reranking="invalid_mode")
