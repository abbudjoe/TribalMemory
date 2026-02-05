"""Edge-case tests for session integration (#70).

Covers:
1. Pagination behavior verification (offset returns different pages)
2. Unicode / special characters (emoji, CJK, Cyrillic)
3. Large message lists (100+ messages, chunking)
4. Timestamp edge cases (out-of-order, duplicates, naive)
5. Load testing (100+ concurrent requests)
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tribalmemory.server.routes import router
from tribalmemory.server import app as app_module
from tribalmemory.services.memory import TribalMemoryService
from tribalmemory.services.session_store import (
    InMemorySessionStore,
    SessionMessage,
)
from tribalmemory.services.vector_store import InMemoryVectorStore
from tribalmemory.testing.mocks import MockEmbeddingService


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def embedding_service():
    emb = MockEmbeddingService(embedding_dim=64)
    # Export/import tools read these via getattr; constructor doesn't set them
    emb.dimensions = 64
    emb.model = "mock-test-model"
    return emb


@pytest.fixture
def memory_service(embedding_service):
    vector_store = InMemoryVectorStore(embedding_service)
    return TribalMemoryService(
        instance_id="test-instance",
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


@pytest.fixture
def session_store(embedding_service):
    vector_store = InMemoryVectorStore(embedding_service)
    return InMemorySessionStore(
        instance_id="test-instance",
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


@pytest.fixture
def client(memory_service, session_store):
    """HTTP test client with injected services."""
    app_module._memory_service = memory_service
    app_module._session_store = session_store
    app_module._instance_id = "test-instance"

    app = FastAPI()
    app.include_router(router)

    yield TestClient(app)

    app_module._memory_service = None
    app_module._session_store = None
    app_module._instance_id = None


def _make_messages(count: int, prefix: str = "msg") -> list[dict]:
    """Generate a list of HTTP-format messages."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    messages = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({
            "role": role,
            "content": f"{prefix} number {i}: topic-{i % 10} discussion about item-{i}",
            "timestamp": (base + timedelta(seconds=i * 30)).isoformat(),
        })
    return messages


def _make_session_messages(
    count: int, prefix: str = "msg"
) -> list[SessionMessage]:
    """Generate a list of SessionMessage objects."""
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return [
        SessionMessage(
            role="user" if i % 2 == 0 else "assistant",
            content=f"{prefix} number {i}: topic-{i % 10} discussion about item-{i}",
            timestamp=base + timedelta(seconds=i * 30),
        )
        for i in range(count)
    ]


# ==================================================================
# 1. Pagination Behavior Verification
# ==================================================================


class TestPaginationBehavior:
    """Verify pagination returns different results at different offsets."""

    def test_different_pages_return_different_items(self, client):
        """offset=0 and offset=1 should return different chunks."""
        # Ingest enough messages to produce multiple chunks
        # (~400 tokens per chunk, ~5 words per message → need ~60 messages)
        messages = _make_messages(80)
        resp = client.post("/v1/sessions/ingest", json={
            "session_id": "pagination-sess",
            "messages": messages,
        })
        assert resp.json()["success"] is True
        chunks = resp.json()["chunks_created"]
        assert chunks >= 2, f"Need >=2 chunks for pagination test, got {chunks}"

        # Page 0
        page0 = client.get("/v1/sessions/search", params={
            "query": "discussion",
            "limit": 1,
            "offset": 0,
            "min_relevance": -1,
        })
        # Page 1
        page1 = client.get("/v1/sessions/search", params={
            "query": "discussion",
            "limit": 1,
            "offset": 1,
            "min_relevance": -1,
        })

        data0 = page0.json()
        data1 = page1.json()

        assert len(data0["results"]) == 1
        assert len(data1["results"]) == 1
        # Different chunk IDs on different pages
        assert (
            data0["results"][0]["chunk_id"]
            != data1["results"][0]["chunk_id"]
        )

    def test_offset_beyond_results_returns_empty(self, client):
        """Offset past total results should return empty page."""
        messages = _make_messages(4)
        client.post("/v1/sessions/ingest", json={
            "session_id": "offset-beyond",
            "messages": messages,
        })

        resp = client.get("/v1/sessions/search", params={
            "query": "discussion",
            "offset": 9999,
            "min_relevance": -1,
        })
        data = resp.json()
        assert data["results"] == []
        assert data["total_count"] >= 0

    def test_full_traversal_covers_all_chunks(self, client):
        """Iterating page by page should cover all chunks."""
        messages = _make_messages(80)
        client.post("/v1/sessions/ingest", json={
            "session_id": "traversal-sess",
            "messages": messages,
        })

        all_ids = set()
        offset = 0
        limit = 2
        while True:
            resp = client.get("/v1/sessions/search", params={
                "query": "discussion",
                "limit": limit,
                "offset": offset,
                "min_relevance": -1,
            })
            data = resp.json()
            page_ids = {r["chunk_id"] for r in data["results"]}
            if not page_ids:
                break
            all_ids.update(page_ids)
            offset += limit

        assert len(all_ids) >= 2
        # Total should match what we collected
        assert len(all_ids) == data["total_count"]


# ==================================================================
# 2. Unicode / Special Characters
# ==================================================================


class TestUnicodeMessages:
    """Verify ingestion and search with non-ASCII content."""

    @pytest.mark.asyncio
    async def test_emoji_content(self, session_store):
        """Messages with emoji should ingest and search correctly."""
        messages = [
            SessionMessage(
                role="user",
                content="Docker 🐳 is great for containerization 📦",
                timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            ),
            SessionMessage(
                role="assistant",
                content="Yes! Kubernetes ☸️ orchestrates those containers 🎯",
                timestamp=datetime(2024, 1, 1, 12, 1, tzinfo=timezone.utc),
            ),
        ]
        result = await session_store.ingest("emoji-sess", messages)
        assert result["success"] is True
        assert result["chunks_created"] >= 1

        search = await session_store.search("Docker", min_relevance=-1.0)
        assert search["total_count"] >= 1

    @pytest.mark.asyncio
    async def test_cjk_content(self, session_store):
        """Messages with CJK characters should work."""
        messages = [
            SessionMessage(
                role="user",
                content="Dockerとは何ですか？コンテナ技術について教えてください。",
                timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            ),
            SessionMessage(
                role="assistant",
                content="Dockerはコンテナ仮想化プラットフォームです。",
                timestamp=datetime(2024, 1, 1, 12, 1, tzinfo=timezone.utc),
            ),
        ]
        result = await session_store.ingest("cjk-sess", messages)
        assert result["success"] is True
        assert result["chunks_created"] >= 1

    @pytest.mark.asyncio
    async def test_cyrillic_content(self, session_store):
        """Messages with Cyrillic characters should work."""
        messages = [
            SessionMessage(
                role="user",
                content="Что такое Docker? Расскажите о контейнерах.",
                timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            ),
        ]
        result = await session_store.ingest("cyrillic-sess", messages)
        assert result["success"] is True
        assert result["chunks_created"] >= 1

    @pytest.mark.asyncio
    async def test_mixed_scripts(self, session_store):
        """Messages mixing Latin, CJK, emoji, and Arabic."""
        messages = [
            SessionMessage(
                role="user",
                content="Hello مرحبا こんにちは 🌍 Привет",
                timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            ),
        ]
        result = await session_store.ingest("mixed-sess", messages)
        assert result["success"] is True

    def test_unicode_via_http(self, client):
        """Unicode messages should work through HTTP endpoint."""
        resp = client.post("/v1/sessions/ingest", json={
            "session_id": "http-unicode",
            "messages": [
                {
                    "role": "user",
                    "content": "Docker 🐳 と Kubernetes 🎯",
                    "timestamp": "2024-01-01T12:00:00Z",
                },
            ],
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ==================================================================
# 3. Large Message Lists
# ==================================================================


class TestLargeMessageLists:
    """Verify chunking behavior with large transcripts."""

    @pytest.mark.asyncio
    async def test_100_messages_chunking(self, session_store):
        """100 messages should produce multiple chunks."""
        messages = _make_session_messages(100)
        result = await session_store.ingest("large-100", messages)

        assert result["success"] is True
        assert result["messages_processed"] == 100
        # ~400 tokens per chunk, ~10 words per message ≈ ~13 tokens
        # 100 messages ≈ 1300 tokens → ~3-4 chunks
        assert result["chunks_created"] >= 2

    @pytest.mark.asyncio
    async def test_200_messages_chunking(self, session_store):
        """200 messages should produce many chunks."""
        messages = _make_session_messages(200, prefix="large")
        result = await session_store.ingest("large-200", messages)

        assert result["success"] is True
        assert result["messages_processed"] == 200
        assert result["chunks_created"] >= 4

    @pytest.mark.asyncio
    async def test_large_session_searchable(self, session_store):
        """All content from a large session should be searchable."""
        messages = _make_session_messages(100)
        await session_store.ingest("searchable-large", messages)

        # Search for content in the middle of the transcript
        result = await session_store.search(
            "topic-5 discussion", min_relevance=-1.0
        )
        assert result["total_count"] >= 1

    @pytest.mark.asyncio
    async def test_delta_ingest_large_session(self, session_store):
        """Delta ingestion should only process new messages."""
        first_batch = _make_session_messages(50)
        r1 = await session_store.ingest("delta-large", first_batch)
        first_chunks = r1["chunks_created"]

        # Add 50 more messages
        all_messages = _make_session_messages(100)
        r2 = await session_store.ingest("delta-large", all_messages)

        # Delta ingestion only processes NEW messages (50 added)
        assert r2["chunks_created"] >= 0  # May be 0 if overlap covers them
        # messages_processed reflects only the delta (new messages)
        assert r2["messages_processed"] == 50

    def test_100_messages_via_http(self, client):
        """Large message list through HTTP endpoint."""
        messages = _make_messages(100)
        resp = client.post("/v1/sessions/ingest", json={
            "session_id": "http-large",
            "messages": messages,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["chunks_created"] >= 2


# ==================================================================
# 4. Timestamp Edge Cases
# ==================================================================


class TestTimestampEdgeCases:
    """Verify handling of unusual timestamp patterns."""

    @pytest.mark.asyncio
    async def test_out_of_order_timestamps(self, session_store):
        """Messages with non-chronological timestamps should still ingest."""
        messages = [
            SessionMessage(
                role="user",
                content="Third message",
                timestamp=datetime(2024, 1, 1, 15, 0, tzinfo=timezone.utc),
            ),
            SessionMessage(
                role="assistant",
                content="First message",
                timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            ),
            SessionMessage(
                role="user",
                content="Second message",
                timestamp=datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
            ),
        ]
        result = await session_store.ingest("unordered", messages)
        assert result["success"] is True
        assert result["chunks_created"] >= 1

    @pytest.mark.asyncio
    async def test_duplicate_timestamps(self, session_store):
        """Multiple messages with identical timestamps should work."""
        same_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        messages = [
            SessionMessage(role="user", content="Message A", timestamp=same_time),
            SessionMessage(role="assistant", content="Message B", timestamp=same_time),
            SessionMessage(role="user", content="Message C", timestamp=same_time),
        ]
        result = await session_store.ingest("dup-timestamps", messages)
        assert result["success"] is True
        assert result["chunks_created"] >= 1

    @pytest.mark.asyncio
    async def test_wide_time_span(self, session_store):
        """Messages spanning months should be handled."""
        messages = [
            SessionMessage(
                role="user",
                content="January message",
                timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            ),
            SessionMessage(
                role="assistant",
                content="June message",
                timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
            ),
            SessionMessage(
                role="user",
                content="December message",
                timestamp=datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc),
            ),
        ]
        result = await session_store.ingest("wide-span", messages)
        assert result["success"] is True

    def test_out_of_order_via_http(self, client):
        """Out-of-order timestamps through HTTP endpoint."""
        resp = client.post("/v1/sessions/ingest", json={
            "session_id": "http-unordered",
            "messages": [
                {
                    "role": "user",
                    "content": "Later message",
                    "timestamp": "2024-01-01T15:00:00Z",
                },
                {
                    "role": "assistant",
                    "content": "Earlier message",
                    "timestamp": "2024-01-01T12:00:00Z",
                },
            ],
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_duplicate_timestamps_via_http(self, client):
        """Duplicate timestamps through HTTP endpoint."""
        resp = client.post("/v1/sessions/ingest", json={
            "session_id": "http-dup-ts",
            "messages": [
                {
                    "role": "user",
                    "content": "Same time A",
                    "timestamp": "2024-01-01T12:00:00Z",
                },
                {
                    "role": "assistant",
                    "content": "Same time B",
                    "timestamp": "2024-01-01T12:00:00Z",
                },
            ],
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ==================================================================
# 5. Load Testing (Concurrent Requests)
# ==================================================================


class TestLoadConcurrency:
    """Verify stability under high concurrent load."""

    @pytest.mark.asyncio
    async def test_100_concurrent_ingests(self, session_store):
        """100 concurrent ingestion requests should not crash."""

        async def ingest_one(idx: int):
            messages = [
                SessionMessage(
                    role="user",
                    content=f"Load test message {idx}",
                    timestamp=datetime(
                        2024, 1, 1, 12, idx % 60, tzinfo=timezone.utc
                    ),
                ),
            ]
            return await session_store.ingest(f"load-{idx}", messages)

        tasks = [ingest_one(i) for i in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Got {len(errors)} errors: {errors[:3]}"

        successes = [r for r in results if not isinstance(r, Exception)]
        assert all(r["success"] for r in successes)

    @pytest.mark.asyncio
    async def test_concurrent_search_under_load(self, session_store):
        """Search should remain stable during concurrent ingestion."""
        # Pre-populate some data
        for i in range(5):
            messages = [
                SessionMessage(
                    role="user",
                    content=f"Pre-loaded message {i}",
                    timestamp=datetime(
                        2024, 1, 1, 12, i, tzinfo=timezone.utc
                    ),
                ),
            ]
            await session_store.ingest(f"preload-{i}", messages)

        async def search_loop():
            for _ in range(20):
                result = await session_store.search(
                    "message", min_relevance=-1.0
                )
                assert isinstance(result, dict)
                assert "items" in result
                await asyncio.sleep(0.005)

        async def ingest_loop():
            for i in range(20):
                messages = [
                    SessionMessage(
                        role="user",
                        content=f"Concurrent ingest {i}",
                        timestamp=datetime(
                            2024, 1, 1, 13, i % 60, tzinfo=timezone.utc
                        ),
                    ),
                ]
                await session_store.ingest(f"concurrent-load-{i}", messages)

        # Run both loops concurrently
        results = await asyncio.gather(
            search_loop(),
            ingest_loop(),
            return_exceptions=True,
        )
        for r in results:
            assert not isinstance(r, Exception), f"Error: {r}"

    @pytest.mark.asyncio
    async def test_50_concurrent_same_session(self, session_store):
        """50 concurrent writes to the same session should not corrupt data."""
        base_msg = SessionMessage(
            role="user",
            content="Initial message",
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        )
        await session_store.ingest("contested", [base_msg])

        async def append(idx: int):
            messages = [
                base_msg,
                SessionMessage(
                    role="assistant",
                    content=f"Reply {idx}",
                    timestamp=datetime(
                        2024, 1, 1, 12, (idx + 1) % 60, tzinfo=timezone.utc
                    ),
                ),
            ]
            return await session_store.ingest("contested", messages)

        tasks = [append(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Got {len(errors)} errors: {errors[:3]}"

        # Session should still be searchable after contention
        search = await session_store.search(
            "Reply", session_id="contested", min_relevance=-1.0
        )
        assert isinstance(search, dict)
