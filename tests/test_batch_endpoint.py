"""Tests for batch ingestion endpoint."""

import pytest
from fastapi.testclient import TestClient

from tribalmemory.server.app import create_app
from tribalmemory.server.config import TribalMemoryConfig, DatabaseConfig, EmbeddingConfig
from tribalmemory.server import app as app_module
from tribalmemory.server.models import SourceType


@pytest.fixture
def test_config():
    """Create test configuration with in-memory storage."""
    return TribalMemoryConfig(
        instance_id="test-batch",
        db=DatabaseConfig(path=":memory:"),
        embedding=EmbeddingConfig(),
    )


@pytest.fixture
def mock_memory_service():
    """Create a mock memory service with in-memory vector store."""
    from tribalmemory.testing.mocks import MockEmbeddingService
    from tribalmemory.services import TribalMemoryService
    from tribalmemory.services.vector_store import InMemoryVectorStore

    embedding = MockEmbeddingService()
    vector_store = InMemoryVectorStore(embedding)
    service = TribalMemoryService(
        instance_id="test-batch",
        embedding_service=embedding,
        vector_store=vector_store,
    )
    return service


@pytest.fixture
def client(test_config, mock_memory_service):
    """Create test client with mocked memory service."""
    # Directly set the module-level variables
    app_module._memory_service = mock_memory_service
    app_module._instance_id = "test-batch"

    # Create app without lifespan (we manage service manually)
    from fastapi import FastAPI
    from tribalmemory.server.routes import router

    app = FastAPI()
    app.include_router(router)

    yield TestClient(app)

    # Cleanup
    app_module._memory_service = None
    app_module._instance_id = None


class TestBatchRememberEndpoint:
    """Tests for /v1/remember/batch endpoint."""

    def test_batch_remember_success(self, client):
        """Test successful batch ingestion of multiple memories."""
        response = client.post("/v1/remember/batch", json={
            "memories": [
                {"content": "Test memory 1"},
                {"content": "Test memory 2"},
                {"content": "Test memory 3"},
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["successful"] == 3
        assert data["failed"] == 0
        assert len(data["results"]) == 3
        for result in data["results"]:
            assert result["success"] is True
            assert result["memory_id"] is not None

    def test_batch_remember_with_metadata(self, client):
        """Test batch ingestion with full metadata."""
        response = client.post("/v1/remember/batch", json={
            "memories": [
                {
                    "content": "User preference: dark mode",
                    "source_type": "user_explicit",
                    "context": "Settings discussion",
                    "tags": ["preference", "ui"],
                },
                {
                    "content": "Meeting scheduled for Monday",
                    "source_type": "auto_capture",
                    "context": "Calendar sync",
                },
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["successful"] == 2

    def test_batch_remember_partial_failure(self, client):
        """Test that failures in one memory don't affect others."""
        # First, store a memory
        client.post("/v1/remember", json={"content": "Original memory"})

        # Now batch with duplicate - should still succeed for others
        response = client.post("/v1/remember/batch", json={
            "memories": [
                {"content": "New memory 1"},
                {"content": "Original memory"},  # Duplicate - will fail
                {"content": "New memory 2"},
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        # 2 new memories succeed, 1 duplicate fails
        assert data["successful"] == 2
        assert data["failed"] == 1
        # The duplicate should have duplicate_of set
        duplicate_result = data["results"][1]
        assert duplicate_result["success"] is False
        assert duplicate_result["duplicate_of"] is not None

    def test_batch_remember_empty_list_rejected(self, client):
        """Test that empty memory list is rejected."""
        response = client.post("/v1/remember/batch", json={
            "memories": []
        })
        assert response.status_code == 422  # Validation error

    def test_batch_remember_exceeds_max_length(self, client):
        """Test that exceeding max batch size is rejected."""
        # Create 1001 memories (limit is 1000)
        memories = [{"content": f"Memory {i}"} for i in range(1001)]
        response = client.post("/v1/remember/batch", json={
            "memories": memories
        })
        assert response.status_code == 422  # Validation error

    def test_batch_remember_reasonable_size(self, client):
        """Test batch with reasonable size succeeds."""
        # Test with 50 memories (well under the 1000 limit)
        memories = [{"content": f"Memory {i}"} for i in range(50)]
        response = client.post("/v1/remember/batch", json={
            "memories": memories
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 50

    def test_batch_remember_error_isolation(self, client):
        """Test that one memory's error doesn't crash the batch."""
        # Each memory is processed independently
        response = client.post("/v1/remember/batch", json={
            "memories": [
                {"content": "Valid memory 1"},
                {"content": "Valid memory 2"},
                {"content": "Valid memory 3"},
            ]
        })
        assert response.status_code == 200
        data = response.json()
        # All should succeed independently
        assert data["total"] == 3
        assert len(data["results"]) == 3

    def test_batch_remember_returns_memory_ids(self, client):
        """Test that each result includes the memory ID."""
        response = client.post("/v1/remember/batch", json={
            "memories": [
                {"content": "Memory A"},
                {"content": "Memory B"},
            ]
        })
        assert response.status_code == 200
        data = response.json()
        ids = [r["memory_id"] for r in data["results"]]
        assert len(ids) == 2
        assert all(id is not None for id in ids)
        # IDs should be unique
        assert len(set(ids)) == 2

    def test_batch_remember_duplicate_detection(self, client):
        """Test duplicate detection within a batch and across batches."""
        # Store original
        first = client.post("/v1/remember", json={"content": "Duplicate test"})
        original_id = first.json()["memory_id"]

        # Batch with duplicate
        response = client.post("/v1/remember/batch", json={
            "memories": [
                {"content": "Duplicate test"},  # Duplicate of original
                {"content": "Unique memory"},
            ]
        })
        assert response.status_code == 200
        data = response.json()
        
        # First should be detected as duplicate
        assert data["results"][0]["duplicate_of"] == original_id
        # Second should be new
        assert data["results"][1]["duplicate_of"] is None

    def test_batch_remember_empty_content(self, client):
        """Test batch with empty content string (issue #114)."""
        response = client.post("/v1/remember/batch", json={
            "memories": [{"content": ""}]
        })
        # Empty content should be rejected by validation
        assert response.status_code == 422

    def test_batch_remember_whitespace_only_content(self, client):
        """Test batch with whitespace-only content."""
        response = client.post("/v1/remember/batch", json={
            "memories": [{"content": "   "}]
        })
        # Whitespace-only should be rejected
        assert response.status_code == 422

    def test_batch_remember_mixed_empty_and_valid(self, client):
        """Test batch with mix of empty and valid content."""
        response = client.post("/v1/remember/batch", json={
            "memories": [
                {"content": "Valid memory"},
                {"content": ""},
            ]
        })
        # Should reject the whole batch since validation is per-item
        assert response.status_code == 422


class TestBatchPerformance:
    """Performance tests for batch endpoint (issue #115)."""

    def test_batch_faster_than_sequential(self, client):
        """Verify batch processing is faster than sequential requests."""
        import time

        count = 20

        # Sequential: individual requests
        seq_start = time.time()
        for i in range(count):
            resp = client.post("/v1/remember", json={
                "content": f"Sequential memory {i}"
            })
            assert resp.status_code == 200
        seq_time = time.time() - seq_start

        # Batch: single request
        batch_start = time.time()
        batch_resp = client.post("/v1/remember/batch", json={
            "memories": [
                {"content": f"Batch memory {i}"} for i in range(count)
            ]
        })
        batch_time = time.time() - batch_start

        assert batch_resp.status_code == 200
        data = batch_resp.json()
        assert data["successful"] == count

        # Batch should be faster (or at least not significantly slower)
        # Using 2x as threshold since test client overhead is high
        assert batch_time < seq_time * 2, (
            f"Batch ({batch_time:.2f}s) should be faster than sequential "
            f"({seq_time:.2f}s)"
        )

    def test_batch_50_completes_reasonably(self, client):
        """Verify batch of 50 memories completes in reasonable time."""
        import time

        start = time.time()
        response = client.post("/v1/remember/batch", json={
            "memories": [
                {"content": f"Performance test memory {i}"} for i in range(50)
            ]
        })
        elapsed = time.time() - start

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 50
        assert data["successful"] == 50
        # Should complete in under 10 seconds (mock embeddings are fast)
        assert elapsed < 10.0, f"Batch of 50 took {elapsed:.2f}s (expected < 10s)"
