"""Tests for LanceDB-backed session transcript storage.

TDD: RED → GREEN → REFACTOR

Tests persistent storage across store restarts, vector search,
and fallback to in-memory storage.
"""

import pytest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tribalmemory.services.session_store import (
    LanceDBSessionStore,
    InMemorySessionStore,
    SessionMessage,
)
from tribalmemory.services.vector_store import InMemoryVectorStore
from tribalmemory.testing.mocks import MockEmbeddingService


class TestLanceDBSessionStorePersistence:
    """Tests for persistent storage across store restarts."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for LanceDB storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def embedding_service(self):
        """Create a mock embedding service."""
        return MockEmbeddingService(embedding_dim=64)

    @pytest.fixture
    def vector_store(self, embedding_service):
        """Create an in-memory vector store."""
        return InMemoryVectorStore(embedding_service)

    @pytest.mark.asyncio
    async def test_chunks_persist_across_restarts(
        self, temp_db_path, embedding_service, vector_store
    ):
        """Should persist chunks across store restarts."""
        # Create first store instance and ingest data
        store1 = LanceDBSessionStore(
            instance_id="test-instance",
            embedding_service=embedding_service,
            vector_store=vector_store,
            db_path=temp_db_path,
        )

        messages = [
            SessionMessage("user", "What is Docker?", datetime(2024, 1, 1, 12, 0, 0)),
            SessionMessage(
                "assistant",
                "Docker is a container platform",
                datetime(2024, 1, 1, 12, 0, 30),
            ),
        ]

        result = await store1.ingest("session-1", messages)
        assert result["success"] is True
        assert result["chunks_created"] > 0

        # Create second store instance (simulating restart)
        store2 = LanceDBSessionStore(
            instance_id="test-instance",
            embedding_service=embedding_service,
            vector_store=vector_store,
            db_path=temp_db_path,
        )

        # Search should find the persisted chunks
        results = await store2.search("Docker")
        assert len(results) > 0
        assert "Docker" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_stats_persist_across_restarts(
        self, temp_db_path, embedding_service, vector_store
    ):
        """Should maintain accurate stats across restarts."""
        # First store instance
        store1 = LanceDBSessionStore(
            instance_id="test-instance",
            embedding_service=embedding_service,
            vector_store=vector_store,
            db_path=temp_db_path,
        )

        await store1.ingest("session-1", [
            SessionMessage("user", "Message 1", datetime(2024, 1, 1, 12, 0, 0)),
        ])
        await store1.ingest("session-2", [
            SessionMessage("user", "Message 2", datetime(2024, 1, 1, 13, 0, 0)),
        ])

        stats1 = await store1.get_stats()
        assert stats1["total_sessions"] == 2

        # Second store instance
        store2 = LanceDBSessionStore(
            instance_id="test-instance",
            embedding_service=embedding_service,
            vector_store=vector_store,
            db_path=temp_db_path,
        )

        stats2 = await store2.get_stats()
        assert stats2["total_sessions"] == 2
        assert stats2["total_chunks"] == stats1["total_chunks"]


class TestLanceDBSessionStoreVectorSearch:
    """Tests for LanceDB vector search functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for LanceDB storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a LanceDB session store."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)
        return LanceDBSessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
            db_path=temp_db_path,
        )

    @pytest.mark.asyncio
    async def test_search_across_sessions(self, store):
        """Should search across all sessions."""
        await store.ingest("session-1", [
            SessionMessage("user", "Docker container setup", datetime(2024, 1, 1, 12, 0, 0)),
        ])
        await store.ingest("session-2", [
            SessionMessage("user", "Kubernetes pod configuration", datetime(2024, 1, 2, 12, 0, 0)),
        ])

        results = await store.search("container", limit=10)

        # Should find Docker mention
        assert len(results) >= 1
        session_ids = {r["session_id"] for r in results}
        assert "session-1" in session_ids

    @pytest.mark.asyncio
    async def test_search_specific_session(self, store):
        """Should filter search to specific session."""
        await store.ingest("session-1", [
            SessionMessage("user", "Docker setup", datetime(2024, 1, 1, 12, 0, 0)),
        ])
        await store.ingest("session-2", [
            SessionMessage("user", "Docker troubleshooting", datetime(2024, 1, 2, 12, 0, 0)),
        ])

        results = await store.search("Docker", session_id="session-1")

        assert len(results) >= 1
        for result in results:
            assert result["session_id"] == "session-1"


class TestLanceDBSessionStoreCleanup:
    """Tests for session cleanup and deletion."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for LanceDB storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a LanceDB session store."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)
        return LanceDBSessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
            db_path=temp_db_path,
        )

    @pytest.mark.asyncio
    async def test_cleanup_deletes_expired_chunks(self, store):
        """Should delete chunks older than retention period."""
        # Ingest old session
        old_time = datetime.now(timezone.utc) - timedelta(days=35)
        await store.ingest("old-session", [
            SessionMessage("user", "Old Docker troubleshooting", old_time),
        ])

        # Ingest recent session
        recent_time = datetime.now(timezone.utc) - timedelta(days=5)
        await store.ingest("recent-session", [
            SessionMessage("user", "Recent Kubernetes setup", recent_time),
        ])

        # Cleanup with 30 day retention
        deleted = await store.cleanup(retention_days=30)

        assert deleted > 0

        # Recent session should still be searchable
        results = await store.search("Kubernetes")
        assert len(results) > 0

        # Old session should be gone
        results = await store.search("Old Docker", session_id="old-session")
        assert len(results) == 0


class TestInMemorySessionStoreFallback:
    """Tests for in-memory fallback when LanceDB is not available."""

    @pytest.fixture
    def store(self):
        """Create an in-memory session store."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)
        return InMemorySessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
        )

    @pytest.mark.asyncio
    async def test_in_memory_store_works(self, store):
        """Should function correctly as fallback."""
        messages = [
            SessionMessage("user", "What is Docker?", datetime(2024, 1, 1, 12, 0, 0)),
            SessionMessage(
                "assistant",
                "Docker is a container platform",
                datetime(2024, 1, 1, 12, 0, 30),
            ),
        ]

        result = await store.ingest("session-1", messages)
        assert result["success"] is True
        assert result["chunks_created"] > 0

        results = await store.search("Docker")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_in_memory_does_not_persist(self):
        """Should not persist data across instances (expected behavior)."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)

        # First instance
        store1 = InMemorySessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
        )

        await store1.ingest("session-1", [
            SessionMessage("user", "Test message", datetime(2024, 1, 1, 12, 0, 0)),
        ])

        # Second instance
        store2 = InMemorySessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
        )

        # Data should not be shared
        results = await store2.search("Test message")
        assert len(results) == 0


class TestLanceDBSessionStoreEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for LanceDB storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_create_without_db_path_raises_error(self):
        """Should raise error when neither db_path nor db_uri provided."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)

        store = LanceDBSessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
            db_path=None,
            db_uri=None,
        )

        with pytest.raises(ValueError, match="Either db_path or db_uri must be provided"):
            await store._ensure_initialized()

    @pytest.mark.asyncio
    async def test_invalid_session_id_sanitization(self, temp_db_path):
        """Should sanitize malicious session IDs to prevent injection."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)

        store = LanceDBSessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
            db_path=temp_db_path,
        )

        await store.ingest("session-1", [
            SessionMessage("user", "Test", datetime(2024, 1, 1, 12, 0, 0)),
        ])

        # Malicious session ID with SQL injection attempt
        malicious_id = "session-1' OR '1'='1"

        with pytest.raises(ValueError, match="Invalid ID format"):
            await store.search("test", session_id=malicious_id)

    @pytest.mark.asyncio
    async def test_empty_search_results(self, temp_db_path):
        """Should handle empty search results gracefully."""
        embedding = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding)

        store = LanceDBSessionStore(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
            db_path=temp_db_path,
        )

        # Search empty store
        results = await store.search("Docker")
        assert results == []
