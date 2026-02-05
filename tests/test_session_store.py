"""Tests for session transcript indexing.

TDD: RED → GREEN → REFACTOR
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Optional

from tribalmemory.services.session_store import (
    SessionStore,
    SessionMessage,
    SessionChunk,
)
from tribalmemory.services.vector_store import InMemoryVectorStore
from tribalmemory.testing.mocks import MockEmbeddingService


class TestSessionMessage:
    """Tests for SessionMessage data class."""

    def test_create_message(self):
        """Should create a session message with all fields."""
        msg = SessionMessage(
            role="user",
            content="What is Docker?",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )
        assert msg.role == "user"
        assert msg.content == "What is Docker?"
        assert msg.timestamp == datetime(2024, 1, 1, 12, 0, 0)

    def test_message_without_timestamp(self):
        """Should create message with current timestamp if not provided."""
        msg = SessionMessage(role="assistant", content="Docker is a container platform")
        assert msg.role == "assistant"
        assert isinstance(msg.timestamp, datetime)


class TestSessionChunk:
    """Tests for SessionChunk data class."""

    def test_create_chunk(self):
        """Should create a session chunk with all fields."""
        chunk = SessionChunk(
            chunk_id="chunk-1",
            session_id="session-123",
            instance_id="clawdio-1",
            content="What is Docker? Docker is a container platform.",
            embedding=[0.1] * 64,
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 12, 1, 0),
            chunk_index=0,
        )
        assert chunk.chunk_id == "chunk-1"
        assert chunk.session_id == "session-123"
        assert chunk.instance_id == "clawdio-1"
        assert len(chunk.embedding) == 64
        assert chunk.chunk_index == 0


class TestSessionStoreChunking:
    """Tests for the chunking algorithm."""

    @pytest.fixture
    def store(self) -> SessionStore:
        """Create a SessionStore with mock embedding service."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)
        return SessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
        )

    @pytest.mark.asyncio
    async def test_chunk_short_conversation(self, store):
        """Should create single chunk for short conversation."""
        messages = [
            SessionMessage("user", "What is Docker?", datetime(2024, 1, 1, 12, 0, 0)),
            SessionMessage("assistant", "Docker is a container platform", datetime(2024, 1, 1, 12, 0, 30)),
        ]

        chunks = await store._chunk_messages(messages, "session-1", "test-instance")
        assert len(chunks) == 1
        assert chunks[0].session_id == "session-1"
        assert "What is Docker?" in chunks[0].content
        assert "Docker is a container platform" in chunks[0].content

    @pytest.mark.asyncio
    async def test_chunk_long_conversation(self, store):
        """Should split long conversation into multiple chunks."""
        # Create a conversation with ~1000 tokens (about 750 words)
        messages = []
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(50):
            # Each message ~15 words = ~20 tokens
            content = " ".join([f"word{j}" for j in range(15)])
            messages.append(SessionMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=content,
                timestamp=base_time + timedelta(seconds=i * 10),
            ))

        chunks = await store._chunk_messages(messages, "session-1", "test-instance")
        
        # Should have multiple chunks (approximate)
        assert len(chunks) >= 2
        assert len(chunks) <= 5
        
        # Each chunk should have incremental index
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    @pytest.mark.asyncio
    async def test_chunk_overlap(self, store):
        """Should include overlap between chunks for context continuity."""
        # Create enough messages to force multiple chunks
        messages = []
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(40):
            content = " ".join([f"message{i}_word{j}" for j in range(15)])
            messages.append(SessionMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=content,
                timestamp=base_time + timedelta(seconds=i * 10),
            ))

        chunks = await store._chunk_messages(messages, "session-1", "test-instance")
        
        if len(chunks) > 1:
            # Check that there's some overlap in content between consecutive chunks
            # (last part of chunk N should appear in beginning of chunk N+1)
            chunk0_words = set(chunks[0].content.split())
            chunk1_words = set(chunks[1].content.split())
            overlap = chunk0_words & chunk1_words
            assert len(overlap) > 0, "Chunks should have some overlapping words"


class TestSessionStoreIngest:
    """Tests for session ingestion."""

    @pytest.fixture
    def store(self) -> SessionStore:
        """Create a SessionStore with mock embedding service."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)
        return SessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
        )

    @pytest.mark.asyncio
    async def test_ingest_new_session(self, store):
        """Should ingest a new session and create chunks."""
        messages = [
            SessionMessage("user", "What is Docker?", datetime(2024, 1, 1, 12, 0, 0)),
            SessionMessage("assistant", "Docker is a container platform", datetime(2024, 1, 1, 12, 0, 30)),
        ]

        result = await store.ingest("session-1", messages)
        
        assert result["success"] is True
        assert result["chunks_created"] == 1
        assert result["messages_processed"] == 2

    @pytest.mark.asyncio
    async def test_ingest_empty_messages(self, store):
        """Should handle empty message list gracefully."""
        result = await store.ingest("session-1", [])
        
        assert result["success"] is True
        assert result["chunks_created"] == 0
        assert result["messages_processed"] == 0

    @pytest.mark.asyncio
    async def test_delta_ingestion(self, store):
        """Should only process new messages on subsequent ingestion."""
        messages_batch1 = [
            SessionMessage("user", "What is Docker?", datetime(2024, 1, 1, 12, 0, 0)),
            SessionMessage("assistant", "Docker is a container platform", datetime(2024, 1, 1, 12, 0, 30)),
        ]

        # First ingest
        result1 = await store.ingest("session-1", messages_batch1)
        assert result1["messages_processed"] == 2

        # Second ingest with same messages (should skip)
        result2 = await store.ingest("session-1", messages_batch1)
        assert result2["messages_processed"] == 0
        assert result2["chunks_created"] == 0

        # Third ingest with additional messages
        messages_batch2 = messages_batch1 + [
            SessionMessage("user", "How do I install it?", datetime(2024, 1, 1, 12, 1, 0)),
            SessionMessage("assistant", "Run: apt install docker.io", datetime(2024, 1, 1, 12, 1, 30)),
        ]
        result3 = await store.ingest("session-1", messages_batch2)
        assert result3["messages_processed"] == 2  # Only the new ones

    @pytest.mark.asyncio
    async def test_ingest_with_instance_id(self, store):
        """Should use provided instance_id for chunks."""
        messages = [
            SessionMessage("user", "Hello", datetime(2024, 1, 1, 12, 0, 0)),
        ]

        await store.ingest("session-1", messages, instance_id="custom-instance")
        
        # Search to verify
        results = (await store.search("Hello", session_id="session-1"))["items"]
        assert len(results) > 0
        assert results[0]["instance_id"] == "custom-instance"


class TestSessionStoreSearch:
    """Tests for session search."""

    @pytest.fixture
    def store(self) -> SessionStore:
        """Create a SessionStore with mock embedding service."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)
        return SessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
        )

    @pytest.mark.asyncio
    async def test_search_across_all_sessions(self, store):
        """Should search across all sessions when session_id not specified."""
        # Ingest multiple sessions
        await store.ingest("session-1", [
            SessionMessage("user", "What is Docker?", datetime(2024, 1, 1, 12, 0, 0)),
            SessionMessage("assistant", "Docker is a container platform", datetime(2024, 1, 1, 12, 0, 30)),
        ])
        await store.ingest("session-2", [
            SessionMessage("user", "What is Kubernetes?", datetime(2024, 1, 2, 12, 0, 0)),
            SessionMessage("assistant", "Kubernetes orchestrates containers", datetime(2024, 1, 2, 12, 0, 30)),
        ])

        results = (await store.search("container", limit=10))["items"]
        
        # Should find results from both sessions
        assert len(results) >= 2
        session_ids = {r["session_id"] for r in results}
        assert "session-1" in session_ids
        assert "session-2" in session_ids

    @pytest.mark.asyncio
    async def test_search_specific_session(self, store):
        """Should filter results to specific session when session_id provided."""
        # Ingest multiple sessions
        await store.ingest("session-1", [
            SessionMessage("user", "What is Docker?", datetime(2024, 1, 1, 12, 0, 0)),
            SessionMessage("assistant", "Docker is a container platform", datetime(2024, 1, 1, 12, 0, 30)),
        ])
        await store.ingest("session-2", [
            SessionMessage("user", "What is Kubernetes?", datetime(2024, 1, 2, 12, 0, 0)),
            SessionMessage("assistant", "Kubernetes orchestrates containers", datetime(2024, 1, 2, 12, 0, 30)),
        ])

        results = (await store.search("container", session_id="session-1"))["items"]
        
        # Should only find results from session-1
        assert len(results) >= 1
        for result in results:
            assert result["session_id"] == "session-1"

    @pytest.mark.asyncio
    async def test_search_min_relevance(self, store):
        """Should filter results by minimum relevance score."""
        await store.ingest("session-1", [
            SessionMessage("user", "Docker setup", datetime(2024, 1, 1, 12, 0, 0)),
        ])

        # High relevance threshold should return fewer results
        results_strict = (await store.search("Docker", min_relevance=0.9))["items"]
        results_loose = (await store.search("Docker", min_relevance=0.0))["items"]
        
        assert len(results_loose) >= len(results_strict)

    @pytest.mark.asyncio
    async def test_search_limit(self, store):
        """Should limit number of results returned."""
        # Ingest multiple sessions
        for i in range(10):
            await store.ingest(f"session-{i}", [
                SessionMessage("user", "What is Docker?", datetime(2024, 1, i + 1, 12, 0, 0)),
            ])

        results = (await store.search("Docker", limit=3))["items"]
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_returns_metadata(self, store):
        """Should return chunk metadata with search results."""
        await store.ingest("session-1", [
            SessionMessage("user", "What is Docker?", datetime(2024, 1, 1, 12, 0, 0)),
            SessionMessage("assistant", "Docker is a container platform", datetime(2024, 1, 1, 12, 0, 30)),
        ])

        results = (await store.search("Docker"))["items"]
        
        assert len(results) > 0
        result = results[0]
        assert "chunk_id" in result
        assert "session_id" in result
        assert "instance_id" in result
        assert "content" in result
        assert "similarity_score" in result
        assert "start_time" in result
        assert "end_time" in result
        assert "chunk_index" in result


class TestSessionStoreCleanup:
    """Tests for session retention and cleanup."""

    @pytest.fixture
    def store(self) -> SessionStore:
        """Create a SessionStore with mock embedding service."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)
        return SessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
        )

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, store):
        """Should delete chunks older than retention period."""
        # Ingest old session (35 days ago)
        old_time = datetime.now(timezone.utc) - timedelta(days=35)
        await store.ingest("old-session", [
            SessionMessage("user", "Docker troubleshooting from long ago", old_time),
        ])

        # Ingest recent session (5 days ago)
        recent_time = datetime.now(timezone.utc) - timedelta(days=5)
        await store.ingest("recent-session", [
            SessionMessage("user", "Kubernetes setup help from yesterday", recent_time),
        ])

        # Cleanup with 30 day retention
        deleted = await store.cleanup(retention_days=30)
        
        assert deleted > 0

        # Recent session should still be searchable
        results = (await store.search("Kubernetes setup help"))["items"]
        assert len(results) > 0

        # Old session should be gone
        results = (await store.search("Docker troubleshooting", session_id="old-session"))["items"]
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_cleanup_preserves_recent(self, store):
        """Should not delete chunks within retention period."""
        # Ingest recent sessions
        recent_time = datetime.now(timezone.utc) - timedelta(days=5)
        await store.ingest("session-1", [
            SessionMessage("user", "Python programming tips", recent_time),
        ])
        await store.ingest("session-2", [
            SessionMessage("user", "Rust memory safety", recent_time),
        ])

        # Cleanup should not delete anything
        deleted = await store.cleanup(retention_days=30)
        assert deleted == 0

        # Both sessions should still be searchable
        results1 = (await store.search("Python programming", session_id="session-1"))["items"]
        results2 = (await store.search("Rust memory", session_id="session-2"))["items"]
        assert len(results1) > 0
        assert len(results2) > 0


class TestSessionStoreStats:
    """Tests for session statistics."""

    @pytest.fixture
    def store(self) -> SessionStore:
        """Create a SessionStore with mock embedding service."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)
        return SessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
        )

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, store):
        """Should return zero stats for empty store."""
        stats = await store.get_stats()
        
        assert stats["total_chunks"] == 0
        assert stats["total_sessions"] == 0
        assert stats["earliest_chunk"] is None
        assert stats["latest_chunk"] is None

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, store):
        """Should return accurate stats for populated store."""
        # Ingest multiple sessions
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        await store.ingest("session-1", [
            SessionMessage("user", "Message 1", base_time),
        ])
        await store.ingest("session-2", [
            SessionMessage("user", "Message 2", base_time + timedelta(hours=1)),
        ])
        await store.ingest("session-3", [
            SessionMessage("user", "Message 3", base_time + timedelta(hours=2)),
        ])

        stats = await store.get_stats()
        
        assert stats["total_chunks"] >= 3
        assert stats["total_sessions"] == 3
        assert stats["earliest_chunk"] is not None
        assert stats["latest_chunk"] is not None
        assert stats["earliest_chunk"] <= stats["latest_chunk"]
