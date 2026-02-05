"""Tests for session search pagination (#47)."""

import pytest
from datetime import datetime, timezone, timedelta

from tribalmemory.services.session_store import (
    InMemorySessionStore,
    SessionStore,
)
from tribalmemory.services.vector_store import InMemoryVectorStore
from tribalmemory.testing.mocks import MockEmbeddingService


@pytest.fixture
def embedding_service():
    return MockEmbeddingService(embedding_dim=64)


@pytest.fixture
def store(embedding_service):
    vector_store = InMemoryVectorStore(embedding_service)
    return InMemorySessionStore(
        instance_id="test",
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


async def _ingest_messages(store, session_id, count):
    """Helper: ingest N messages into a session."""
    from tribalmemory.services.session_store import SessionMessage

    messages = [
        SessionMessage(
            role="user" if i % 2 == 0 else "assistant",
            content=f"Message {i} about topic-{session_id}",
            timestamp=(
                datetime(2024, 1, 1, tzinfo=timezone.utc)
                + timedelta(minutes=i)
            ),
        )
        for i in range(count)
    ]
    await store.ingest(session_id, messages, instance_id="test")


class TestSearchPagination:
    """Test offset-based pagination on session search."""

    @pytest.mark.asyncio
    async def test_search_returns_total_count(self, store):
        """Search results should include total_count."""
        await _ingest_messages(store, "sess-1", 20)

        results = await store.search(
            "Message", limit=5, min_relevance=0.0
        )
        assert isinstance(results, dict)
        assert "items" in results
        assert "total_count" in results
        assert results["total_count"] >= len(results["items"])

    @pytest.mark.asyncio
    async def test_search_offset_zero_is_default(self, store):
        """offset=0 should behave like no offset."""
        await _ingest_messages(store, "sess-1", 20)

        r1 = await store.search(
            "Message", limit=3, min_relevance=0.0
        )
        r2 = await store.search(
            "Message", limit=3, offset=0, min_relevance=0.0
        )
        assert r1["items"] == r2["items"]

    @pytest.mark.asyncio
    async def test_search_offset_skips_results(self, store):
        """offset=N should skip the first N results."""
        await _ingest_messages(store, "sess-1", 20)

        page1 = await store.search(
            "Message", limit=3, offset=0, min_relevance=0.0
        )
        page2 = await store.search(
            "Message", limit=3, offset=3, min_relevance=0.0
        )

        # Pages should not overlap
        ids1 = {r["chunk_id"] for r in page1["items"]}
        ids2 = {r["chunk_id"] for r in page2["items"]}
        assert ids1.isdisjoint(ids2)

    @pytest.mark.asyncio
    async def test_search_offset_beyond_results(self, store):
        """offset beyond total should return empty items."""
        await _ingest_messages(store, "sess-1", 5)

        results = await store.search(
            "Message", limit=10, offset=1000, min_relevance=0.0
        )
        assert results["items"] == []
        assert results["total_count"] >= 0

    @pytest.mark.asyncio
    async def test_search_total_count_independent_of_offset(
        self, store
    ):
        """total_count should be the same regardless of offset."""
        await _ingest_messages(store, "sess-1", 20)

        r0 = await store.search(
            "Message", limit=5, offset=0, min_relevance=0.0
        )
        r5 = await store.search(
            "Message", limit=5, offset=5, min_relevance=0.0
        )
        assert r0["total_count"] == r5["total_count"]

    @pytest.mark.asyncio
    async def test_pagination_walks_all_results(self, store):
        """Walking pages should visit all results without dups."""
        await _ingest_messages(store, "sess-1", 20)

        all_ids = set()
        offset = 0
        page_size = 3

        while True:
            page = await store.search(
                "Message",
                limit=page_size,
                offset=offset,
                min_relevance=0.0,
            )
            if not page["items"]:
                break
            for item in page["items"]:
                assert item["chunk_id"] not in all_ids, (
                    f"Duplicate chunk: {item['chunk_id']}"
                )
                all_ids.add(item["chunk_id"])
            offset += page_size

        assert len(all_ids) == page["total_count"]

    @pytest.mark.asyncio
    async def test_offset_with_session_filter(self, store):
        """Pagination should work with session_id filter."""
        await _ingest_messages(store, "sess-1", 10)
        await _ingest_messages(store, "sess-2", 10)

        results = await store.search(
            "Message",
            session_id="sess-1",
            limit=2,
            offset=0,
            min_relevance=0.0,
        )
        # Only sess-1 chunks
        for item in results["items"]:
            assert item["session_id"] == "sess-1"

    @pytest.mark.asyncio
    async def test_negative_offset_treated_as_zero(self, store):
        """Negative offset should be clamped to 0."""
        await _ingest_messages(store, "sess-1", 10)

        r_neg = await store.search(
            "Message", limit=3, offset=-5, min_relevance=0.0
        )
        r_zero = await store.search(
            "Message", limit=3, offset=0, min_relevance=0.0
        )
        assert r_neg["items"] == r_zero["items"]

    @pytest.mark.asyncio
    async def test_has_more_flag(self, store):
        """has_more should be False when all results fit."""
        await _ingest_messages(store, "sess-1", 10)

        result = await store.search(
            "Message", limit=50, offset=0, min_relevance=0.0
        )
        # Small dataset — should fit within pool cap
        assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_very_large_offset(self, store):
        """Very large offset should return empty without error."""
        await _ingest_messages(store, "sess-1", 10)

        result = await store.search(
            "Message", limit=5, offset=999999, min_relevance=0.0
        )
        assert result["items"] == []
        assert result["total_count"] >= 0

    @pytest.mark.asyncio
    async def test_limit_clamped_to_valid_range(self, store):
        """Limit should be clamped between 1 and 50."""
        await _ingest_messages(store, "sess-1", 10)

        # limit=0 should clamp to 1
        r1 = await store.search(
            "Message", limit=0, min_relevance=0.0
        )
        assert len(r1["items"]) <= 1

        # limit=999 should clamp to 50
        r2 = await store.search(
            "Message", limit=999, min_relevance=0.0
        )
        assert len(r2["items"]) <= 50


class TestLanceDBPagination:
    """Test pagination with LanceDB store (if available)."""

    @pytest.fixture
    def lance_store(self, tmp_path, embedding_service):
        try:
            from tribalmemory.services.session_store import (
                LanceDBSessionStore,
            )
            vector_store = InMemoryVectorStore(embedding_service)
            return LanceDBSessionStore(
                instance_id="test",
                embedding_service=embedding_service,
                vector_store=vector_store,
                db_path=tmp_path / "lance_sessions",
            )
        except ImportError:
            pytest.skip("LanceDB not installed")

    @pytest.mark.asyncio
    async def test_lancedb_offset_pagination(self, lance_store):
        """LanceDB store should support offset pagination."""
        # Ingest enough messages to produce multiple chunks
        # (~400 tokens per chunk, so 200 messages should give >1)
        await _ingest_messages(lance_store, "sess-1", 200)

        page1 = await lance_store.search(
            "Message", limit=2, offset=0, min_relevance=0.0
        )
        page2 = await lance_store.search(
            "Message", limit=2, offset=2, min_relevance=0.0
        )

        assert isinstance(page1, dict)
        assert "items" in page1
        assert "total_count" in page1
        assert len(page1["items"]) >= 1

        if page2["items"]:
            ids1 = {r["chunk_id"] for r in page1["items"]}
            ids2 = {r["chunk_id"] for r in page2["items"]}
            assert ids1.isdisjoint(ids2)

    @pytest.mark.asyncio
    async def test_lancedb_total_count(self, lance_store):
        """LanceDB total_count should reflect filtered results."""
        await _ingest_messages(lance_store, "sess-1", 200)

        results = await lance_store.search(
            "Message", limit=5, offset=0, min_relevance=0.0
        )
        assert results["total_count"] >= 1
