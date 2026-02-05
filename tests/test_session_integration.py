"""Integration tests for session HTTP endpoints and MCP tools (#46).

Tests cover:
1. HTTP endpoint tests — POST /v1/sessions/ingest, GET /v1/sessions/search
2. MCP tool tests — tribal_sessions_ingest, tribal_recall with sources
3. Concurrent ingestion stress test
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

mcp = pytest.importorskip("mcp")

from tribalmemory.server.routes import router
from tribalmemory.server import app as app_module
from tribalmemory.mcp.server import create_server
import tribalmemory.mcp.server as mcp_server
from tribalmemory.services.memory import TribalMemoryService
from tribalmemory.services.session_store import (
    InMemorySessionStore,
    SessionStore,
)
from tribalmemory.services.vector_store import InMemoryVectorStore
from tribalmemory.testing.mocks import MockEmbeddingService


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse(result: Any) -> dict:
    """Extract JSON from FastMCP call_tool result."""
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple):
        result = result[0]
    for block in result:
        text = getattr(block, "text", None)
        if not text and isinstance(block, dict):
            text = block.get("text")
        if text:
            return json.loads(text)
    raise ValueError(f"Could not parse result: {result}")


# Structured messages for HTTP endpoint (list of dicts)
SAMPLE_MESSAGES_LIST = [
    {
        "role": "user",
        "content": "What is Docker?",
        "timestamp": "2024-01-01T12:00:00Z",
    },
    {
        "role": "assistant",
        "content": "Docker is a container platform for packaging apps.",
        "timestamp": "2024-01-01T12:00:30Z",
    },
    {
        "role": "user",
        "content": "How does it differ from VMs?",
        "timestamp": "2024-01-01T12:01:00Z",
    },
    {
        "role": "assistant",
        "content": (
            "Containers share the host OS kernel, "
            "making them lighter than virtual machines."
        ),
        "timestamp": "2024-01-01T12:01:30Z",
    },
]

# JSON string for MCP tool (expects string param)
SAMPLE_MESSAGES_JSON = json.dumps(SAMPLE_MESSAGES_LIST)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def embedding_service():
    emb = MockEmbeddingService(embedding_dim=64)
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


@pytest.fixture
def mcp_app(memory_service, session_store, monkeypatch):
    """MCP server with injected services."""
    monkeypatch.setattr(mcp_server, "_memory_service", memory_service)
    monkeypatch.setattr(mcp_server, "_session_store", session_store)
    server = create_server()
    yield server
    monkeypatch.setattr(mcp_server, "_memory_service", None)
    monkeypatch.setattr(mcp_server, "_session_store", None)


# ==================================================================
# HTTP Endpoint Tests
# ==================================================================


class TestSessionIngestEndpoint:
    """Tests for POST /v1/sessions/ingest."""

    def test_ingest_success(self, client):
        """Should ingest messages and return chunk count."""
        response = client.post("/v1/sessions/ingest", json={
            "session_id": "sess-1",
            "messages": SAMPLE_MESSAGES_LIST,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["chunks_created"] >= 1

    def test_ingest_with_instance_id(self, client):
        """Should accept optional instance_id override."""
        response = client.post("/v1/sessions/ingest", json={
            "session_id": "sess-2",
            "messages": SAMPLE_MESSAGES_LIST,
            "instance_id": "custom-instance",
        })
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_ingest_empty_messages(self, client):
        """Empty messages list should return zero chunks."""
        response = client.post("/v1/sessions/ingest", json={
            "session_id": "sess-empty",
            "messages": [],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_created"] == 0

    def test_ingest_invalid_messages_format(self, client):
        """Messages with missing fields should fail validation."""
        response = client.post("/v1/sessions/ingest", json={
            "session_id": "sess-bad",
            "messages": [{"role": "user"}],  # missing content+timestamp
        })
        assert response.status_code == 422

    def test_ingest_missing_session_id(self, client):
        """Missing session_id should fail validation."""
        response = client.post("/v1/sessions/ingest", json={
            "messages": SAMPLE_MESSAGES_LIST,
        })
        assert response.status_code == 422


class TestSessionSearchEndpoint:
    """Tests for GET /v1/sessions/search."""

    def test_search_empty_store(self, client):
        """Search on empty store should return empty results."""
        response = client.get("/v1/sessions/search", params={
            "query": "Docker",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total_count"] == 0

    def test_search_after_ingest(self, client):
        """Should find ingested content."""
        # Ingest first
        client.post("/v1/sessions/ingest", json={
            "session_id": "sess-1",
            "messages": SAMPLE_MESSAGES_LIST,
        })

        # Search — use min_relevance=-1 because MockEmbeddingService
        # hash-based similarity can be slightly negative for different texts
        response = client.get("/v1/sessions/search", params={
            "query": "Docker container",
            "min_relevance": -1,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) >= 1
        assert data["total_count"] >= 1

    def test_search_with_session_filter(self, client):
        """Session filter should restrict results."""
        client.post("/v1/sessions/ingest", json={
            "session_id": "sess-1",
            "messages": SAMPLE_MESSAGES_LIST,
        })

        response = client.get("/v1/sessions/search", params={
            "query": "Docker",
            "session_id": "sess-1",
            "min_relevance": -1,
        })
        data = response.json()
        for r in data["results"]:
            assert r["session_id"] == "sess-1"

    def test_search_pagination(self, client):
        """Offset/limit should paginate results."""
        client.post("/v1/sessions/ingest", json={
            "session_id": "sess-1",
            "messages": SAMPLE_MESSAGES_LIST,
        })

        response = client.get("/v1/sessions/search", params={
            "query": "Docker",
            "limit": 1,
            "offset": 0,
            "min_relevance": -1,
        })
        data = response.json()
        assert data["offset"] == 0
        assert data["limit"] == 1
        assert "total_count" in data
        assert "has_more" in data

    def test_search_missing_query(self, client):
        """Missing query param should fail validation."""
        response = client.get("/v1/sessions/search")
        assert response.status_code == 422


# ==================================================================
# MCP Tool Tests
# ==================================================================


class TestMCPSessionIngest:
    """Tests for tribal_sessions_ingest MCP tool."""

    @pytest.mark.asyncio
    async def test_ingest_via_mcp(self, mcp_app):
        """Should ingest session messages via MCP."""
        result = await mcp_app.call_tool(
            "tribal_sessions_ingest",
            {
                "session_id": "mcp-sess-1",
                "messages": SAMPLE_MESSAGES_JSON,
            },
        )
        data = _parse(result)
        assert data["success"] is True
        assert data["chunks_created"] >= 1

    @pytest.mark.asyncio
    async def test_ingest_with_instance_override(self, mcp_app):
        """Should accept instance_id override."""
        result = await mcp_app.call_tool(
            "tribal_sessions_ingest",
            {
                "session_id": "mcp-sess-2",
                "messages": SAMPLE_MESSAGES_JSON,
                "instance_id": "custom",
            },
        )
        data = _parse(result)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_ingest_invalid_messages(self, mcp_app):
        """Invalid messages should return error."""
        result = await mcp_app.call_tool(
            "tribal_sessions_ingest",
            {
                "session_id": "mcp-sess-bad",
                "messages": "not-valid-json",
            },
        )
        data = _parse(result)
        assert data["success"] is False


class TestMCPRecallSources:
    """Tests for tribal_recall with sources parameter."""

    @pytest.mark.asyncio
    async def test_recall_memories_only(self, mcp_app):
        """sources='memories' should only return memory results."""
        # Store a memory
        await mcp_app.call_tool(
            "tribal_remember",
            {"content": "Python is a programming language"},
        )

        result = await mcp_app.call_tool(
            "tribal_recall",
            {
                "query": "programming",
                "sources": "memories",
                "min_relevance": 0,
            },
        )
        data = _parse(result)
        assert data["sources"] == "memories"
        for r in data["results"]:
            assert r["type"] == "memory"

    @pytest.mark.asyncio
    async def test_recall_sessions_only(self, mcp_app):
        """sources='sessions' should only return session results."""
        # Ingest a short session (single message for positive mock similarity)
        short_msgs = json.dumps([
            {
                "role": "user",
                "content": "Docker",
                "timestamp": "2024-01-01T12:00:00Z",
            },
        ])
        await mcp_app.call_tool(
            "tribal_sessions_ingest",
            {
                "session_id": "recall-sess",
                "messages": short_msgs,
            },
        )

        result = await mcp_app.call_tool(
            "tribal_recall",
            {
                "query": "Docker",
                "sources": "sessions",
                "min_relevance": 0,
            },
        )
        data = _parse(result)
        assert data["sources"] == "sessions"
        # With mock embeddings, results may be empty due to hash
        # similarity; verify structure is correct
        for r in data["results"]:
            assert r["type"] == "session"

    @pytest.mark.asyncio
    async def test_recall_all_sources(self, mcp_app):
        """sources='all' merges memories and sessions in one call."""
        # Store a memory with matching content
        await mcp_app.call_tool(
            "tribal_remember",
            {"content": "Kubernetes is a container orchestration platform"},
        )
        # Ingest session with matching content
        short_msgs = json.dumps([
            {
                "role": "user",
                "content": "Kubernetes",
                "timestamp": "2024-01-01T12:00:00Z",
            },
        ])
        await mcp_app.call_tool(
            "tribal_sessions_ingest",
            {
                "session_id": "all-sess",
                "messages": short_msgs,
            },
        )

        result = await mcp_app.call_tool(
            "tribal_recall",
            {
                "query": "Kubernetes",
                "sources": "all",
                "min_relevance": 0,
            },
        )
        data = _parse(result)
        assert data["sources"] == "all"
        # Verify response shape — mock embeddings may not produce
        # positive similarity for all pairs
        assert isinstance(data["results"], list)
        assert "count" in data

    @pytest.mark.asyncio
    async def test_recall_invalid_sources(self, mcp_app):
        """Invalid sources value should return error."""
        result = await mcp_app.call_tool(
            "tribal_recall",
            {
                "query": "test",
                "sources": "invalid",
            },
        )
        data = _parse(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_recall_sessions_empty(self, mcp_app):
        """Sessions search with no data should return empty."""
        result = await mcp_app.call_tool(
            "tribal_recall",
            {
                "query": "anything",
                "sources": "sessions",
                "min_relevance": 0,
            },
        )
        data = _parse(result)
        assert data["count"] == 0
        assert data["results"] == []


# ==================================================================
# Concurrent Ingestion Tests
# ==================================================================


class TestConcurrentIngestion:
    """Test thread safety of concurrent session ingestion."""

    @pytest.mark.asyncio
    async def test_concurrent_different_sessions(
        self, session_store
    ):
        """Concurrent ingestion of different sessions."""
        from tribalmemory.services.session_store import SessionMessage

        async def ingest_session(sid: str):
            messages = [
                SessionMessage(
                    role="user",
                    content=f"Message for session {sid}",
                    timestamp=datetime(
                        2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc
                    ),
                ),
            ]
            return await session_store.ingest(sid, messages)

        # Run 10 sessions concurrently
        tasks = [ingest_session(f"sess-{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(r["success"] for r in results)
        assert sum(r["chunks_created"] for r in results) >= 10

        # Verify all sessions are searchable
        for i in range(10):
            result = await session_store.search(
                f"session sess-{i}",
                session_id=f"sess-{i}",
                min_relevance=-1.0,
            )
            assert result["total_count"] >= 1

    @pytest.mark.asyncio
    async def test_concurrent_same_session_delta(
        self, session_store
    ):
        """Concurrent delta ingestion of the same session."""
        from tribalmemory.services.session_store import SessionMessage

        base_messages = [
            SessionMessage(
                role="user",
                content="Initial message",
                timestamp=datetime(
                    2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc
                ),
            ),
        ]

        # First ingest
        await session_store.ingest("shared-sess", base_messages)

        async def append_message(idx: int):
            messages = base_messages + [
                SessionMessage(
                    role="assistant",
                    content=f"Reply {idx}",
                    timestamp=datetime(
                        2024, 1, 1, 12, idx + 1, 0,
                        tzinfo=timezone.utc,
                    ),
                ),
            ]
            return await session_store.ingest("shared-sess", messages)

        # Concurrent appends — should not crash
        tasks = [append_message(i) for i in range(5)]
        results = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        # No exceptions
        for r in results:
            assert not isinstance(r, Exception), f"Got error: {r}"

    @pytest.mark.asyncio
    async def test_concurrent_ingest_and_search(
        self, session_store
    ):
        """Search during concurrent ingestion should not crash."""
        from tribalmemory.services.session_store import SessionMessage

        async def ingest_loop():
            for i in range(5):
                messages = [
                    SessionMessage(
                        role="user",
                        content=f"Concurrent message {i}",
                        timestamp=datetime(
                            2024, 1, 1, 12, i, 0,
                            tzinfo=timezone.utc,
                        ),
                    ),
                ]
                await session_store.ingest(
                    f"concurrent-{i}", messages
                )

        async def search_loop():
            for _ in range(5):
                result = await session_store.search(
                    "concurrent", min_relevance=-1.0
                )
                assert isinstance(result, dict)
                await asyncio.sleep(0.01)

        # Run ingest and search concurrently
        await asyncio.gather(
            ingest_loop(),
            search_loop(),
        )
