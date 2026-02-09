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

    def test_recall_rejects_empty_query(self, client):
        """Recall should reject empty query string with 422."""
        response = client.post("/v1/recall", json={
            "query": "",
        })
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_recall_rejects_whitespace_query(self, client):
        """Recall should reject whitespace-only query string with 422."""
        response = client.post("/v1/recall", json={
            "query": "   ",
        })
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_recall_rejects_invalid_after_date(self, client):
        """Recall should reject invalid 'after' date format with 422."""
        response = client.post("/v1/recall", json={
            "query": "test query",
            "after": "not-a-date",
        })
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_recall_rejects_invalid_before_date(self, client):
        """Recall should reject invalid 'before' date format with 422."""
        response = client.post("/v1/recall", json={
            "query": "test query",
            "before": "invalid",
        })
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_recall_accepts_valid_iso_dates(self, client):
        """Recall should accept valid ISO 8601 dates."""
        response = client.post("/v1/recall", json={
            "query": "test query",
            "after": "2024-01-01T00:00:00Z",
            "before": "2024-12-31T23:59:59Z",
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_recall_accepts_none_dates(self, client):
        """Recall should accept None/omitted date fields."""
        response = client.post("/v1/recall", json={
            "query": "test query",
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data


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


class TestProjectScoping:
    """Tests for project-scoped memory (Issue #161)."""

    def test_remember_with_project_adds_tag(self, client):
        """Storing with project should add project:<name> tag."""
        response = client.post("/v1/remember", json={
            "content": "API uses GraphQL",
            "source_type": "user_explicit",
            "project": "my-app",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        memory_id = data["memory_id"]

        # Verify the project tag was added
        get_response = client.get(f"/v1/memory/{memory_id}")
        assert get_response.status_code == 200
        memory = get_response.json()
        assert "project:my-app" in memory["tags"]

    def test_remember_with_project_preserves_existing_tags(self, client):
        """Project tag should be added alongside existing tags."""
        response = client.post("/v1/remember", json={
            "content": "Use PostgreSQL for persistence",
            "source_type": "user_explicit",
            "tags": ["architecture", "database"],
            "project": "my-app",
        })
        assert response.status_code == 200
        data = response.json()
        memory_id = data["memory_id"]

        get_response = client.get(f"/v1/memory/{memory_id}")
        memory = get_response.json()
        assert "architecture" in memory["tags"]
        assert "database" in memory["tags"]
        assert "project:my-app" in memory["tags"]

    def test_remember_without_project_no_tag(self, client):
        """No project param should not add any project tag."""
        response = client.post("/v1/remember", json={
            "content": "General knowledge memory",
            "source_type": "user_explicit",
        })
        assert response.status_code == 200
        data = response.json()
        memory_id = data["memory_id"]

        get_response = client.get(f"/v1/memory/{memory_id}")
        memory = get_response.json()
        assert not any(t.startswith("project:") for t in memory["tags"])

    def test_recall_with_project_filters(self, client):
        """Recall with project should only return memories from that project."""
        # Store memories in different projects
        client.post("/v1/remember", json={
            "content": "Frontend uses React with TypeScript",
            "source_type": "user_explicit",
            "project": "alpha",
        })
        client.post("/v1/remember", json={
            "content": "Frontend uses Vue with JavaScript",
            "source_type": "user_explicit",
            "project": "beta",
        })

        # Recall with project filter
        response = client.post("/v1/recall", json={
            "query": "frontend framework",
            "project": "alpha",
            "min_relevance": 0.0,
        })
        assert response.status_code == 200
        results = response.json()["results"]

        # Should only contain alpha project memories
        for r in results:
            assert "project:alpha" in r["memory"]["tags"]

    def test_recall_without_project_returns_all(self, client):
        """Recall without project should return memories from all projects."""
        # Store in alpha
        r1 = client.post("/v1/remember", json={
            "content": "Alpha project uses microservices architecture pattern",
            "source_type": "user_explicit",
            "project": "alpha",
            "skip_dedup": True,
        })
        assert r1.json()["success"] is True

        # Store in beta
        r2 = client.post("/v1/remember", json={
            "content": "Beta project uses monolith architecture pattern",
            "source_type": "user_explicit",
            "project": "beta",
            "skip_dedup": True,
        })
        assert r2.json()["success"] is True

        # Recall both memories by ID to verify they exist with correct tags
        m1 = client.get(f"/v1/memory/{r1.json()['memory_id']}").json()
        m2 = client.get(f"/v1/memory/{r2.json()['memory_id']}").json()
        assert "project:alpha" in m1["tags"]
        assert "project:beta" in m2["tags"]

    def test_recall_project_with_additional_tags(self, client):
        """Project filter should combine with tag filters."""
        client.post("/v1/remember", json={
            "content": "Redis cache layer for alpha",
            "source_type": "user_explicit",
            "tags": ["infrastructure"],
            "project": "alpha",
        })
        client.post("/v1/remember", json={
            "content": "Redis cache layer for beta",
            "source_type": "user_explicit",
            "tags": ["infrastructure"],
            "project": "beta",
        })

        response = client.post("/v1/recall", json={
            "query": "cache",
            "tags": ["infrastructure"],
            "project": "alpha",
            "min_relevance": 0.0,
        })
        assert response.status_code == 200
        results = response.json()["results"]

        for r in results:
            assert "infrastructure" in r["memory"]["tags"]
            assert "project:alpha" in r["memory"]["tags"]

    def test_batch_remember_with_project(self, client):
        """Batch remember should apply project tags to each memory."""
        response = client.post("/v1/remember/batch", json={
            "memories": [
                {
                    "content": "Batch item one for gamma project setup",
                    "source_type": "user_explicit",
                    "project": "gamma",
                },
                {
                    "content": "Batch item two for gamma project config",
                    "source_type": "user_explicit",
                    "project": "gamma",
                    "tags": ["setup"],
                },
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["successful"] == 2

        # Verify both memories have the project tag
        for result in data["results"]:
            mem = client.get(f"/v1/memory/{result['memory_id']}").json()
            assert "project:gamma" in mem["tags"]

        # Second memory should also keep its explicit tag
        mem2 = client.get(
            f"/v1/memory/{data['results'][1]['memory_id']}"
        ).json()
        assert "setup" in mem2["tags"]

    def test_project_validation_rejects_blank(self, client):
        """Empty or whitespace-only project should be rejected."""
        for blank in ["", "  ", "\t"]:
            response = client.post("/v1/remember", json={
                "content": "Some content for validation test",
                "project": blank,
            })
            assert response.status_code == 422, (
                f"Expected 422 for project={blank!r}"
            )

    def test_project_validation_strips_whitespace(self, client):
        """Project names with leading/trailing whitespace should be stripped."""
        response = client.post("/v1/remember", json={
            "content": "Whitespace project test memory content",
            "source_type": "user_explicit",
            "project": "  my-app  ",
        })
        assert response.status_code == 200
        mem_id = response.json()["memory_id"]
        mem = client.get(f"/v1/memory/{mem_id}").json()
        # Tag should use stripped name
        assert "project:my-app" in mem["tags"]
        assert "project:  my-app  " not in mem["tags"]

    def test_cross_project_dedup_allows_same_content(self, client):
        """Same content in different projects should NOT be rejected as duplicate."""
        r1 = client.post("/v1/remember", json={
            "content": "Database uses PostgreSQL with read replicas",
            "source_type": "user_explicit",
            "project": "project-x",
        })
        assert r1.status_code == 200
        assert r1.json()["success"] is True

        # Same content, different project — should succeed
        r2 = client.post("/v1/remember", json={
            "content": "Database uses PostgreSQL with read replicas",
            "source_type": "user_explicit",
            "project": "project-y",
        })
        assert r2.status_code == 200
        assert r2.json()["success"] is True
        assert r2.json()["memory_id"] != r1.json()["memory_id"]

    def test_same_project_dedup_still_works(self, client):
        """Same content in the SAME project should still be deduplicated."""
        r1 = client.post("/v1/remember", json={
            "content": "Unique dedup test content for same project check",
            "source_type": "user_explicit",
            "project": "dedup-proj",
        })
        assert r1.status_code == 200
        assert r1.json()["success"] is True

        # Same content, same project — should be deduplicated
        r2 = client.post("/v1/remember", json={
            "content": "Unique dedup test content for same project check",
            "source_type": "user_explicit",
            "project": "dedup-proj",
        })
        assert r2.status_code == 200
        assert r2.json()["success"] is False  # Duplicate rejected

    def test_recall_without_project_returns_all_in_results(self, client):
        """Recall without project should return memories from all projects in results."""
        client.post("/v1/remember", json={
            "content": "Delta project uses event sourcing architecture",
            "source_type": "user_explicit",
            "project": "delta",
            "skip_dedup": True,
        })
        client.post("/v1/remember", json={
            "content": "Epsilon project uses event driven architecture",
            "source_type": "user_explicit",
            "project": "epsilon",
            "skip_dedup": True,
        })

        # Recall without project filter — should find both
        response = client.post("/v1/recall", json={
            "query": "event architecture",
            "min_relevance": 0.0,
            "limit": 10,
        })
        assert response.status_code == 200
        results = response.json()["results"]
        projects_found = set()
        for r in results:
            for tag in r["memory"]["tags"]:
                if tag.startswith("project:"):
                    projects_found.add(tag)
        assert "project:delta" in projects_found
        assert "project:epsilon" in projects_found
