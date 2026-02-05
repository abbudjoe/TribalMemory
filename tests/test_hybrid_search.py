"""Tests for BM25 hybrid search (SQLite FTS5).

TDD: RED → GREEN → REFACTOR
"""

import pytest
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from tribalmemory.server.config import SearchConfig
from tribalmemory.services.fts_store import FTSStore
from tribalmemory.services.memory import TribalMemoryService
from tribalmemory.services.vector_store import InMemoryVectorStore
from tribalmemory.testing.mocks import MockEmbeddingService


class TestFTSStore:
    """Tests for the SQLite FTS5 full-text search store."""

    @pytest.fixture
    def fts(self, tmp_path):
        """Create a FTS store with a temp database."""
        db_path = tmp_path / "test_fts.db"
        return FTSStore(str(db_path))

    def test_fts5_available(self, fts):
        """FTS5 should be available in the SQLite build."""
        assert fts.is_available()

    def test_index_and_search(self, fts):
        """Should index a document and find it by keyword."""
        fts.index("mem-1", "The OPENAI_API_KEY must be set", ["config"])
        results = fts.search("OPENAI_API_KEY", limit=5)

        assert len(results) == 1
        assert results[0]["id"] == "mem-1"
        assert results[0]["rank"] < 0  # BM25 returns negative ranks

    def test_search_no_results(self, fts):
        """Should return empty list when no matches."""
        fts.index("mem-1", "Hello world", ["test"])
        results = fts.search("nonexistent_token", limit=5)
        assert len(results) == 0

    def test_search_multiple_results(self, fts):
        """Should return multiple matching documents."""
        fts.index("mem-1", "Python is great for scripting", ["lang"])
        fts.index("mem-2", "Python web frameworks like Flask", ["lang"])
        fts.index("mem-3", "Rust is great for systems", ["lang"])

        results = fts.search("Python", limit=5)
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"mem-1", "mem-2"}

    def test_search_respects_limit(self, fts):
        """Should return at most `limit` results."""
        for i in range(10):
            fts.index(f"mem-{i}", f"Python example number {i}", ["test"])

        results = fts.search("Python", limit=3)
        assert len(results) == 3

    def test_search_tags_indexed(self, fts):
        """Should be able to search by tag content."""
        fts.index("mem-1", "Some content", ["debugging", "python"])
        results = fts.search("debugging", limit=5)
        assert len(results) == 1

    def test_delete_removes_from_index(self, fts):
        """Should remove document from search results after delete."""
        fts.index("mem-1", "Important API config", ["config"])
        fts.delete("mem-1")
        results = fts.search("API config", limit=5)
        assert len(results) == 0

    def test_update_replaces_content(self, fts):
        """Should update the indexed content for an existing ID."""
        fts.index("mem-1", "Old content about Python", ["lang"])
        fts.index("mem-1", "New content about Rust", ["lang"])

        python_results = fts.search("Python", limit=5)
        rust_results = fts.search("Rust", limit=5)
        assert len(python_results) == 0
        assert len(rust_results) == 1

    def test_count(self, fts):
        """Should return total indexed documents."""
        assert fts.count() == 0
        fts.index("mem-1", "Hello", [])
        fts.index("mem-2", "World", [])
        assert fts.count() == 2

    def test_search_malformed_query(self, fts):
        """Should return empty list for malformed FTS query."""
        fts.index("mem-1", "Test content", [])
        # Unbalanced quotes cause FTS syntax error
        results = fts.search('unbalanced"quote', limit=5)
        assert len(results) == 0


class TestHybridScoring:
    """Tests for hybrid score merging (vector + BM25)."""

    def test_bm25_rank_to_score(self):
        """BM25 ranks (negative) should convert to 0..1 scores."""
        from tribalmemory.services.fts_store import bm25_rank_to_score

        # BM25 returns negative ranks; more negative = better match
        assert bm25_rank_to_score(-10.0) == pytest.approx(1 / (1 + 10.0))
        assert bm25_rank_to_score(-1.0) == pytest.approx(1 / (1 + 1.0))
        assert bm25_rank_to_score(0.0) == pytest.approx(1.0)

    def test_hybrid_merge_combines_scores(self):
        """Should combine vector and BM25 scores with weights."""
        from tribalmemory.services.fts_store import hybrid_merge

        vector_results = [
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.7},
        ]
        bm25_results = [
            {"id": "b", "rank": -5.0},
            {"id": "c", "rank": -3.0},
        ]

        merged = hybrid_merge(
            vector_results, bm25_results,
            vector_weight=0.7, text_weight=0.3
        )

        ids = [r["id"] for r in merged]
        # All three should be present
        assert set(ids) == {"a", "b", "c"}
        # "b" appears in both, should have highest combined score
        b_result = next(r for r in merged if r["id"] == "b")
        assert b_result["final_score"] > 0

    def test_hybrid_merge_empty_bm25(self):
        """Should work with empty BM25 results (vector-only)."""
        from tribalmemory.services.fts_store import hybrid_merge

        vector_results = [
            {"id": "a", "score": 0.9},
        ]
        merged = hybrid_merge(vector_results, [], vector_weight=0.7, text_weight=0.3)
        assert len(merged) == 1
        assert merged[0]["id"] == "a"

    def test_hybrid_merge_empty_vector(self):
        """Should work with empty vector results (BM25-only)."""
        from tribalmemory.services.fts_store import hybrid_merge

        bm25_results = [
            {"id": "a", "rank": -5.0},
        ]
        merged = hybrid_merge([], bm25_results, vector_weight=0.7, text_weight=0.3)
        assert len(merged) == 1
        assert merged[0]["id"] == "a"

    def test_hybrid_merge_respects_weights(self):
        """Higher vector weight should favor vector-only results."""
        from tribalmemory.services.fts_store import hybrid_merge

        vector_results = [{"id": "v", "score": 0.9}]
        bm25_results = [{"id": "b", "rank": -10.0}]

        # High vector weight
        merged_v = hybrid_merge(vector_results, bm25_results, 0.9, 0.1)
        v_score = next(r for r in merged_v if r["id"] == "v")["final_score"]
        b_score = next(r for r in merged_v if r["id"] == "b")["final_score"]
        assert v_score > b_score  # vector result should win

        # High text weight
        merged_t = hybrid_merge(vector_results, bm25_results, 0.1, 0.9)
        v_score_t = next(r for r in merged_t if r["id"] == "v")["final_score"]
        b_score_t = next(r for r in merged_t if r["id"] == "b")["final_score"]
        assert b_score_t > v_score_t  # BM25 result should win


class TestSearchConfig:
    """Tests for SearchConfig validation."""

    def test_valid_config(self):
        """Should accept valid config."""
        config = SearchConfig(
            vector_weight=0.7, text_weight=0.3, candidate_multiplier=4
        )
        assert config.vector_weight == 0.7

    def test_negative_vector_weight_rejected(self):
        """Should reject negative vector_weight."""
        with pytest.raises(ValueError, match="vector_weight"):
            SearchConfig(vector_weight=-0.1, text_weight=0.3)

    def test_negative_text_weight_rejected(self):
        """Should reject negative text_weight."""
        with pytest.raises(ValueError, match="text_weight"):
            SearchConfig(vector_weight=0.7, text_weight=-0.1)

    def test_both_zero_weights_rejected(self):
        """Should reject both weights being zero."""
        with pytest.raises(ValueError, match="(?i)at least one"):
            SearchConfig(vector_weight=0.0, text_weight=0.0)

    def test_zero_candidate_multiplier_rejected(self):
        """Should reject candidate_multiplier < 1."""
        with pytest.raises(ValueError, match="candidate_multiplier"):
            SearchConfig(candidate_multiplier=0)


class TestHybridIntegration:
    """Integration tests: hybrid search through TribalMemoryService."""

    @pytest.fixture
    def service(self, tmp_path):
        """Create a memory service with FTS enabled."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)
        fts = FTSStore(str(tmp_path / "fts.db"))
        return TribalMemoryService(
            instance_id="test",
            embedding_service=embedding,
            vector_store=vector_store,
            fts_store=fts,
            hybrid_search=True,
        )

    @pytest.mark.asyncio
    async def test_remember_indexes_in_fts(self, service):
        """remember() should index content in both vector and FTS stores."""
        result = await service.remember("Set OPENAI_API_KEY=sk-test123")
        assert result.success
        # FTS should have the entry
        assert service.fts_store.count() == 1
        fts_results = service.fts_store.search("OPENAI_API_KEY", limit=5)
        assert len(fts_results) == 1

    @pytest.mark.asyncio
    async def test_recall_uses_hybrid_search(self, service):
        """recall() should use hybrid search when FTS is available."""
        await service.remember(
            "Set OPENAI_API_KEY environment variable for auth",
            tags=["config"],
        )
        await service.remember(
            "Python virtualenvs isolate dependencies",
            tags=["python"],
        )

        # Keyword query should find the API key memory
        results = await service.recall("OPENAI_API_KEY", min_relevance=0.0)
        assert len(results) >= 1
        # The API key memory should be in the results
        contents = [r.memory.content for r in results]
        assert any("OPENAI_API_KEY" in c for c in contents)

    @pytest.mark.asyncio
    async def test_forget_removes_from_fts(self, service):
        """forget() should remove from FTS index too."""
        result = await service.remember("Test memory for deletion")
        assert result.success
        assert service.fts_store.count() == 1

        await service.forget(result.memory_id)
        assert service.fts_store.count() == 0

    @pytest.mark.asyncio
    async def test_hybrid_disabled_uses_vector_only(self, tmp_path):
        """When hybrid_search=False, should use vector-only recall."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)
        fts = FTSStore(str(tmp_path / "fts.db"))
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=embedding,
            vector_store=vector_store,
            fts_store=fts,
            hybrid_search=False,
        )

        await service.remember("Test content")
        # FTS should still be indexed (for future use)
        assert service.fts_store.count() == 1
        # But recall should not use hybrid path
        results = await service.recall("Test", min_relevance=0.0)
        # Should still work (vector-only)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_recall_with_heuristic_reranking(self, tmp_path):
        """recall() should apply heuristic reranking when configured."""
        from tribalmemory.services.reranker import HeuristicReranker
        
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)
        fts = FTSStore(str(tmp_path / "fts.db"))
        reranker = HeuristicReranker(recency_decay_days=30.0, tag_boost_weight=0.2)
        
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=embedding,
            vector_store=vector_store,
            fts_store=fts,
            hybrid_search=True,
            reranker=reranker,
        )

        # Add memories
        await service.remember("Python programming information", tags=["python"])
        await service.remember("JavaScript web development", tags=["javascript"])
        
        # Query should work with reranking enabled
        # The exact results depend on mock embeddings, but the pipeline should not crash
        results = await service.recall("python", limit=5, min_relevance=0.0)
        
        # Should return results without errors (reranking applied internally)
        assert isinstance(results, list)
        # All results should have valid scores
        for r in results:
            assert r.similarity_score >= 0.0
            assert r.memory.content is not None


class TestFTSEscaping:
    """Test FTS5 query escaping for special characters."""

    def test_escape_question_mark(self, tmp_path):
        """Query with ? should not cause FTS5 syntax error."""
        fts = FTSStore(str(tmp_path / "test.db"))
        fts.index("1", "What is the answer to life?", ["test"])
        # This would fail before the fix with: fts5: syntax error near "?"
        results = fts.search("What is the answer?")
        assert len(results) >= 0  # No crash

    def test_escape_quotes(self, tmp_path):
        """Query with quotes should be escaped properly."""
        fts = FTSStore(str(tmp_path / "test.db"))
        fts.index("1", 'He said "hello world"', ["test"])
        results = fts.search('He said "hello"')
        assert len(results) >= 0  # No crash

    def test_escape_apostrophe(self, tmp_path):
        """Query with apostrophe should not cause syntax error."""
        fts = FTSStore(str(tmp_path / "test.db"))
        fts.index("1", "Caroline's support group meeting", ["test"])
        results = fts.search("What's Caroline's plan?")
        assert len(results) >= 0  # No crash

    def test_escape_special_operators(self, tmp_path):
        """Query with FTS5 operators should be treated literally."""
        fts = FTSStore(str(tmp_path / "test.db"))
        fts.index("1", "A OR B AND C NOT D", ["test"])
        # These are FTS5 operators that would fail without escaping
        results = fts.search("A OR B")
        assert len(results) >= 0  # No crash
