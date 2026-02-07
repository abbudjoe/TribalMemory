"""Tests for the HTTP server."""

import pytest
from fastapi.testclient import TestClient

from tribalmemory.server.app import create_app
from tribalmemory.server.config import TribalMemoryConfig, DatabaseConfig, EmbeddingConfig
from tribalmemory.server import app as app_module


@pytest.fixture
def test_config():
    """Create test configuration with in-memory storage."""
    return TribalMemoryConfig(
        instance_id="test-instance",
        db=DatabaseConfig(path=":memory:"),
        embedding=EmbeddingConfig(),
    )


@pytest.fixture
def mock_memory_service():
    """Create a mock memory service."""
    from tribalmemory.testing.mocks import MockEmbeddingService
    from tribalmemory.services import TribalMemoryService
    from tribalmemory.services.vector_store import InMemoryVectorStore
    
    embedding = MockEmbeddingService()
    vector_store = InMemoryVectorStore(embedding)
    service = TribalMemoryService(
        instance_id="test-instance",
        embedding_service=embedding,
        vector_store=vector_store,
    )
    return service


@pytest.fixture
def client(test_config, mock_memory_service):
    """Create test client with mocked memory service."""
    # Directly set the module-level variables
    app_module._memory_service = mock_memory_service
    app_module._instance_id = "test-instance"
    
    # Create app without lifespan (we manage service manually)
    from fastapi import FastAPI
    from tribalmemory.server.routes import router
    
    app = FastAPI()
    app.include_router(router)
    
    @app.get("/")
    async def root():
        return {"service": "tribal-memory", "version": "0.1.0", "docs": "/docs"}
    
    yield TestClient(app)
    
    # Cleanup
    app_module._memory_service = None
    app_module._instance_id = None


class TestHealthEndpoint:
    """Tests for /v1/health endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return status ok."""
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["instance_id"] == "test-instance"
        assert "memory_count" in data


class TestRememberEndpoint:
    """Tests for /v1/remember endpoint."""

    def test_remember_stores_memory(self, client):
        """Remember endpoint should store a memory."""
        response = client.post("/v1/remember", json={
            "content": "Test memory content",
            "source_type": "user_explicit",
            "tags": ["test"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["memory_id"] is not None

    def test_remember_with_minimal_params(self, client):
        """Remember should work with just content."""
        response = client.post("/v1/remember", json={
            "content": "Minimal memory",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestRecallEndpoint:
    """Tests for /v1/recall endpoint."""

    def test_recall_returns_results(self, client):
        """Recall should return stored memories."""
        # Store a memory first
        client.post("/v1/remember", json={
            "content": "Joe likes Python programming",
        })
        
        # Recall it
        response = client.post("/v1/recall", json={
            "query": "What does Joe like?",
            "limit": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "query" in data
        assert "total_time_ms" in data

    def test_recall_empty_db(self, client):
        """Recall on empty db should return empty results."""
        # Create fresh service with empty store
        from tribalmemory.testing.mocks import MockEmbeddingService
        from tribalmemory.services import TribalMemoryService
        from tribalmemory.services.vector_store import InMemoryVectorStore
        
        embedding = MockEmbeddingService()
        vector_store = InMemoryVectorStore(embedding)
        empty_service = TribalMemoryService(
            instance_id="test-instance",
            embedding_service=embedding,
            vector_store=vector_store,
        )
        app_module._memory_service = empty_service
        
        response = client.post("/v1/recall", json={
            "query": "anything",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []


class TestForgetEndpoint:
    """Tests for /v1/forget endpoint."""

    def test_forget_deletes_memory(self, client):
        """Forget should delete a memory."""
        # Store a memory first
        response = client.post("/v1/remember", json={
            "content": "Memory to forget",
        })
        assert response.status_code == 200
        memory_id = response.json()["memory_id"]
        
        # Forget it
        response = client.delete(f"/v1/forget/{memory_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["memory_id"] == memory_id

    def test_forget_nonexistent_memory(self, client):
        """Forget should handle nonexistent memory gracefully."""
        response = client.delete("/v1/forget/nonexistent-id")
        assert response.status_code == 200
        # Should return success=False or True depending on implementation


class TestGetMemoryEndpoint:
    """Tests for /v1/memory/{id} endpoint."""

    def test_get_memory_returns_entry(self, client):
        """Get memory should return the stored entry."""
        # Store a memory first
        response = client.post("/v1/remember", json={
            "content": "Retrievable memory",
            "tags": ["test"],
        })
        assert response.status_code == 200
        memory_id = response.json()["memory_id"]
        
        # Get it back
        response = client.get(f"/v1/memory/{memory_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == memory_id
        assert data["content"] == "Retrievable memory"
        assert "test" in data["tags"]

    def test_get_memory_not_found(self, client):
        """Get memory should return 404 for nonexistent ID."""
        response = client.get("/v1/memory/nonexistent-id")
        assert response.status_code == 404


class TestCorrectEndpoint:
    """Tests for /v1/correct endpoint."""

    def test_correct_creates_correction(self, client):
        """Correct should create a new memory with supersedes link."""
        # Store original
        response = client.post("/v1/remember", json={
            "content": "Original content",
        })
        assert response.status_code == 200
        original_id = response.json()["memory_id"]
        
        # Correct it
        response = client.post("/v1/correct", json={
            "original_id": original_id,
            "corrected_content": "Corrected content",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["memory_id"] != original_id


    def test_correct_rejects_empty_content(self, client):
        """Correct should reject empty corrected content."""
        # Store original
        response = client.post("/v1/remember", json={
            "content": "Original content",
        })
        original_id = response.json()["memory_id"]

        # Correct with empty content
        response = client.post("/v1/correct", json={
            "original_id": original_id,
            "corrected_content": "",
        })
        assert response.status_code == 422

    def test_correct_rejects_whitespace_content(self, client):
        """Correct should reject whitespace-only corrected content."""
        response = client.post("/v1/remember", json={
            "content": "Original content",
        })
        original_id = response.json()["memory_id"]

        response = client.post("/v1/correct", json={
            "original_id": original_id,
            "corrected_content": "   ",
        })
        assert response.status_code == 422


class TestStatsEndpoint:
    """Tests for /v1/stats endpoint."""

    def test_stats_returns_counts(self, client):
        """Stats should return memory counts."""
        response = client.get("/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_memories" in data
        assert "by_source_type" in data
        assert "instance_id" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_info(self, client):
        """Root should return service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "tribal-memory"
        assert "version" in data
