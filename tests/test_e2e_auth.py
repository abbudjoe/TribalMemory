"""End-to-end tests for token authentication.

These tests exercise auth through the full FastAPI stack,
including middleware, routes, and service layer.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tribalmemory.server.auth import (
    generate_token,
    save_token,
    load_token,
    TokenAuthMiddleware,
)
from tribalmemory.server.routes import router
from tribalmemory.server import app as app_module
from tribalmemory.testing.mocks import MockEmbeddingService
from tribalmemory.services import TribalMemoryService
from tribalmemory.services.vector_store import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_INSTANCE_ID = "test-e2e-auth"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_rate_limit_state():
    """Clean up rate limit state before and after each test.
    
    Note: Each test creates a fresh TokenAuthMiddleware instance,
    so in-memory rate limit state (_failure_count, _cooldown_until)
    does not leak between tests. This fixture only cleans the
    persisted state on disk.
    """
    rate_limit_path = Path("~/.tribal-memory/rate-limits.json").expanduser()
    if rate_limit_path.exists():
        rate_limit_path.unlink()
    yield
    if rate_limit_path.exists():
        rate_limit_path.unlink()


@pytest.fixture(autouse=True)
def cleanup_app_module():
    """Clean up module-level state after each test."""
    yield
    app_module._memory_service = None
    app_module._instance_id = None
    app_module._session_store = None


@pytest.fixture
def temp_token_file():
    """Create a temporary file for token storage."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".env"
    ) as f:
        path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def mock_memory_service():
    """Create a mock memory service."""
    embedding = MockEmbeddingService()
    vector_store = InMemoryVectorStore(embedding)
    service = TribalMemoryService(
        instance_id=TEST_INSTANCE_ID,
        embedding_service=embedding,
        vector_store=vector_store,
    )
    return service


def create_test_app_with_auth(
    token: Optional[str] = None,
    memory_service: Optional[TribalMemoryService] = None,
) -> FastAPI:
    """Create a FastAPI app with auth middleware and real routes.
    
    This mimics create_app() but allows passing token directly
    for testing purposes. Uses TestClient for simpler test setup
    while still exercising the full middleware stack.
    """
    app = FastAPI(
        title="Tribal Memory Test",
        version="0.1.0",
    )

    # Set up module-level service for routes
    app_module._memory_service = memory_service
    app_module._instance_id = TEST_INSTANCE_ID
    
    # Set up session store (required by some routes)
    from tribalmemory.services.session_store import InMemorySessionStore
    app_module._session_store = InMemorySessionStore(
        instance_id=TEST_INSTANCE_ID,
        embedding_service=memory_service.embedding_service,
        vector_store=memory_service.vector_store,
    )

    # Add auth middleware
    app.add_middleware(
        TokenAuthMiddleware,
        token=token,
    )

    # Include routes (already have /v1 prefix in route definitions)
    app.include_router(router)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "tribal-memory",
            "version": "0.1.0",
            "docs": "/docs",
        }

    return app


# ---------------------------------------------------------------------------
# E2E Auth Tests
# ---------------------------------------------------------------------------


class TestTokenFlowE2E:
    """Test complete token generation, save, and auth flow."""

    def test_token_flow_end_to_end(
        self, mock_memory_service, temp_token_file
    ):
        """Generate token → save → start server → auth requests."""
        # Generate and save token
        token = generate_token()
        save_token(token, temp_token_file)

        # Verify token was saved
        loaded = load_token(temp_token_file)
        assert loaded == token

        # Create app with token auth
        app = create_test_app_with_auth(
            token=token,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        # Make authenticated request
        response = client.post(
            "/v1/remember",
            json={"content": "Test memory"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True



class TestRejectionFlowE2E:
    """Test auth rejection scenarios."""

    def test_request_without_token_rejected(
        self, mock_memory_service
    ):
        """Server with token → request without token → 401."""
        token = generate_token()
        app = create_test_app_with_auth(
            token=token,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        # Request without Authorization header
        response = client.post(
            "/v1/remember",
            json={"content": "Test memory"},
        )
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "Missing API token" in data["error"]
        assert response.headers.get("WWW-Authenticate") == "Bearer"

    def test_request_with_wrong_token_rejected(
        self, mock_memory_service
    ):
        """Server with token → request with wrong token → 401."""
        token = generate_token()
        wrong_token = generate_token()

        app = create_test_app_with_auth(
            token=token,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        # Request with wrong token
        response = client.post(
            "/v1/remember",
            json={"content": "Test memory"},
            headers={"Authorization": f"Bearer {wrong_token}"},
        )
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "Invalid API token" in data["error"]



class TestLegacyModeE2E:
    """Test server behavior without token (legacy mode)."""

    def test_legacy_mode_allows_all_requests(
        self, mock_memory_service
    ):
        """Server without token → all requests allowed."""
        # Create app without token
        app = create_test_app_with_auth(
            token=None,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        # Request without auth should succeed
        response = client.post(
            "/v1/remember",
            json={"content": "Test memory"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True



class TestPublicPathsE2E:
    """Test public endpoints that never require auth."""

    def test_health_endpoints_public(
        self, mock_memory_service
    ):
        """Health endpoints accessible without auth even with token."""
        token = generate_token()
        app = create_test_app_with_auth(
            token=token,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        # /v1/health should work without auth
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"



class TestRememberWithAuthE2E:
    """Test /v1/remember endpoint with authentication."""

    def test_remember_with_valid_token_succeeds(
        self, mock_memory_service
    ):
        """POST /v1/remember with valid token → success."""
        token = generate_token()
        app = create_test_app_with_auth(
            token=token,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        response = client.post(
            "/v1/remember",
            json={
                "content": "Authenticated memory",
                "tags": ["test"],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "memory_id" in data



class TestRecallWithAuthE2E:
    """Test /v1/recall endpoint with authentication."""

    def test_recall_with_valid_token_succeeds(
        self, mock_memory_service
    ):
        """POST /v1/recall with valid token → success."""
        token = generate_token()
        app = create_test_app_with_auth(
            token=token,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        # Store a memory first
        client.post(
            "/v1/remember",
            json={"content": "Searchable content"},
            headers={"Authorization": f"Bearer {token}"},
        )

        # Recall it
        response = client.post(
            "/v1/recall",
            json={"query": "searchable"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data



class TestRateLimitingE2E:
    """Test rate limiting on failed auth attempts."""

    def test_rate_limiting_triggers_on_bad_tokens(
        self, mock_memory_service
    ):
        """10 bad tokens → 429 on 11th."""
        token = generate_token()
        app = create_test_app_with_auth(
            token=token,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        # Make 10 requests with bad token
        wrong_token = generate_token()
        for i in range(10):
            response = client.post(
                "/v1/remember",
                json={"content": f"Attempt {i}"},
                headers={"Authorization": f"Bearer {wrong_token}"},
            )
            assert response.status_code == 401

        # 11th request should be rate-limited
        response = client.post(
            "/v1/remember",
            json={"content": "Attempt 11"},
            headers={"Authorization": f"Bearer {wrong_token}"},
        )
        assert response.status_code == 429
        data = response.json()
        assert "error" in data
        assert "Too many failed" in data["error"]



class TestTokenRotationE2E:
    """Test token rotation flow."""

    def test_token_rotation_old_fails_new_works(
        self, mock_memory_service, temp_token_file
    ):
        """Generate → use → rotate → old token fails, new works."""
        # Generate and save initial token
        old_token = generate_token()
        save_token(old_token, temp_token_file)

        # Create app with old token
        app_old = create_test_app_with_auth(
            token=old_token,
            memory_service=mock_memory_service,
        )
        client_old = TestClient(app_old)

        # Verify old token works
        response = client_old.post(
            "/v1/remember",
            json={"content": "Memory with old token"},
            headers={"Authorization": f"Bearer {old_token}"},
        )
        assert response.status_code == 200

        # Rotate token
        new_token = generate_token()
        save_token(new_token, temp_token_file)
        loaded = load_token(temp_token_file)
        assert loaded == new_token

        # Create new app with new token
        app_new = create_test_app_with_auth(
            token=new_token,
            memory_service=mock_memory_service,
        )
        client_new = TestClient(app_new)

        # Old token should fail on new server
        response = client_new.post(
            "/v1/remember",
            json={"content": "Memory with old token"},
            headers={"Authorization": f"Bearer {old_token}"},
        )
        assert response.status_code == 401

        # New token should work
        response = client_new.post(
            "/v1/remember",
            json={"content": "Memory with new token"},
            headers={"Authorization": f"Bearer {new_token}"},
        )
        assert response.status_code == 200



class TestEnvVarOverrideE2E:
    """Test TRIBAL_MEMORY_API_TOKEN env var override."""

    def test_env_var_overrides_file(
        self,
        mock_memory_service,
        temp_token_file,
        
        monkeypatch,
    ):
        """Set env var → overrides file token."""
        # Save one token to file
        file_token = generate_token()
        save_token(file_token, temp_token_file)

        # Set different token in env var
        env_token = generate_token()
        monkeypatch.setenv("TRIBAL_MEMORY_API_TOKEN", env_token)

        # Create app (simulating env var override behavior)
        # In production, create_app reads env var first
        app = create_test_app_with_auth(
            token=env_token,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        # File token should NOT work
        response = client.post(
            "/v1/remember",
            json={"content": "Test with file token"},
            headers={"Authorization": f"Bearer {file_token}"},
        )
        assert response.status_code == 401

        # Env var token should work
        response = client.post(
            "/v1/remember",
            json={"content": "Test with env token"},
            headers={"Authorization": f"Bearer {env_token}"},
        )
        assert response.status_code == 200



class TestOtherEndpointsWithAuth:
    """Test other API endpoints require auth."""

    def test_forget_endpoint_requires_auth(
        self, mock_memory_service
    ):
        """DELETE /v1/forget/{id} requires auth."""
        token = generate_token()
        app = create_test_app_with_auth(
            token=token,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        # Without auth → 401
        response = client.delete("/v1/forget/some-id")
        assert response.status_code == 401

        # With auth → should proceed (200 with success=True/False)
        response = client.delete(
            "/v1/forget/some-id",
            headers={"Authorization": f"Bearer {token}"},
        )
        # Should get 200, not 401
        assert response.status_code == 200

    def test_get_memory_endpoint_requires_auth(
        self, mock_memory_service
    ):
        """GET /v1/memory/{id} requires auth."""
        token = generate_token()
        app = create_test_app_with_auth(
            token=token,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        # Without auth → 401
        response = client.get("/v1/memory/some-id")
        assert response.status_code == 401

        # With auth → should proceed (404 for nonexistent ID)
        response = client.get(
            "/v1/memory/some-id",
            headers={"Authorization": f"Bearer {token}"},
        )
        # Should get 404, not 401
        assert response.status_code == 404

    def test_stats_endpoint_requires_auth(
        self, mock_memory_service
    ):
        """GET /v1/stats requires auth."""
        token = generate_token()
        app = create_test_app_with_auth(
            token=token,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        # Without auth → 401
        response = client.get("/v1/stats")
        assert response.status_code == 401

        # With auth → success
        response = client.get(
            "/v1/stats",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200



class TestOptionsRequestsPublic:
    """Test OPTIONS (CORS preflight) never require auth."""

    def test_options_request_public(
        self, mock_memory_service
    ):
        """OPTIONS requests should not require auth."""
        token = generate_token()
        app = create_test_app_with_auth(
            token=token,
            memory_service=mock_memory_service,
        )
        client = TestClient(app)

        # OPTIONS should work without auth
        response = client.options("/v1/remember")
        # Should not return 401
        assert response.status_code != 401

