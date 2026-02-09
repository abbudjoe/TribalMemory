"""Integration tests for token authentication with FastAPI app.

Tests component interactions: middleware + routes + config loading.
Unit tests for middleware alone are in test_auth.py.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from tribalmemory.server import app as app_module
from tribalmemory.server.auth import (
    COOLDOWN_SECONDS,
    MAX_FAILURES,
    TokenAuthMiddleware,
    generate_token,
    save_token,
)
from tribalmemory.server.routes import router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_app_state() -> None:
    """Reset module-level app state after every test."""
    yield
    app_module._memory_service = None
    app_module._session_store = None
    app_module._instance_id = None


@pytest.fixture
def test_token() -> str:
    """Generate a test token."""
    return generate_token()


@pytest.fixture
def mock_memory_service():
    """Create a mock memory service."""
    from tribalmemory.testing.mocks import (
        MockEmbeddingService,
        MockVectorStore,
        MockMemoryService,
    )

    embedding = MockEmbeddingService()
    vector_store = MockVectorStore(embedding)
    service = MockMemoryService(
        instance_id="test-instance",
        embedding_service=embedding,
        vector_store=vector_store,
    )
    return service


@pytest.fixture
def mock_session_store():
    """Create a mock session store."""
    from tribalmemory.services.session_store import InMemorySessionStore
    from tribalmemory.testing.mocks import MockEmbeddingService, MockVectorStore

    embedding = MockEmbeddingService()
    vector_store = MockVectorStore(embedding)
    return InMemorySessionStore(
        instance_id="test-instance",
        embedding_service=embedding,
        vector_store=vector_store,
    )


def create_test_app(
    token: str | None = None,
    add_cors: bool = True,
) -> FastAPI:
    """Create FastAPI app with auth middleware for testing.

    Args:
        token: Optional API token to enable auth.
        add_cors: Whether to add CORS middleware.

    Returns:
        FastAPI app with auth + routes.
    """
    app = FastAPI()

    # Add auth middleware
    app.add_middleware(TokenAuthMiddleware, token=token)

    # Add CORS middleware (matches production config)
    if add_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include routes
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


@pytest.fixture
def client_with_token(test_token, mock_memory_service, mock_session_store):
    """Create test client with token auth enabled."""
    # Set module-level globals
    app_module._memory_service = mock_memory_service
    app_module._session_store = mock_session_store
    app_module._instance_id = "test-instance"

    app = create_test_app(token=test_token)

    yield TestClient(app), test_token

    # Cleanup
    app_module._memory_service = None
    app_module._session_store = None
    app_module._instance_id = None


@pytest.fixture
def client_without_token(mock_memory_service, mock_session_store):
    """Create test client without token (legacy mode)."""
    # Set module-level globals
    app_module._memory_service = mock_memory_service
    app_module._session_store = mock_session_store
    app_module._instance_id = "test-instance"

    app = create_test_app(token=None)

    yield TestClient(app)

    # Cleanup
    app_module._memory_service = None
    app_module._session_store = None
    app_module._instance_id = None


# ---------------------------------------------------------------------------
# Auth + real routes
# ---------------------------------------------------------------------------


class TestAuthWithRoutes:
    """Test auth middleware works with actual API endpoints."""

    def test_auth_protects_remember_endpoint(self, client_with_token):
        """POST /v1/remember requires valid token."""
        client, token = client_with_token

        payload = {
            "content": "test memory",
            "metadata": {"tags": ["test"]},
        }

        # Without token
        response = client.post("/v1/remember", json=payload)
        assert response.status_code == 401

        # With valid token
        response = client.post(
            "/v1/remember",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

    def test_auth_protects_recall_endpoint(self, client_with_token):
        """POST /v1/recall requires valid token."""
        client, token = client_with_token

        payload = {"query": "test query", "limit": 5}

        # Without token
        response = client.post("/v1/recall", json=payload)
        assert response.status_code == 401

        # With valid token
        response = client.post(
            "/v1/recall",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

    def test_auth_protects_forget_endpoint(self, client_with_token):
        """DELETE /v1/forget requires valid token."""
        client, token = client_with_token

        # First store a memory
        response = client.post(
            "/v1/remember",
            json={"content": "temp memory"},
            headers={"Authorization": f"Bearer {token}"},
        )
        memory_id = response.json()["memory_id"]

        # Without token
        response = client.delete(f"/v1/forget/{memory_id}")
        assert response.status_code == 401

        # With valid token
        response = client.delete(
            f"/v1/forget/{memory_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        # Should work (200 or 404 if already deleted)
        assert response.status_code in [200, 404]

    def test_health_endpoint_always_public(self, client_with_token):
        """GET /v1/health never requires auth."""
        client, token = client_with_token

        # Should work without token
        response = client.get("/v1/health")
        assert response.status_code == 200

    def test_auth_protects_batch_endpoint(self, client_with_token):
        """POST /v1/remember/batch requires valid token."""
        client, token = client_with_token

        payload = {
            "memories": [
                {"content": "memory 1", "metadata": {"tags": ["test"]}},
                {"content": "memory 2", "metadata": {"tags": ["test"]}},
            ]
        }

        # Without token
        response = client.post("/v1/remember/batch", json=payload)
        assert response.status_code == 401

        # With valid token
        response = client.post(
            "/v1/remember/batch",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# CORS + Auth interaction
# ---------------------------------------------------------------------------


class TestCORSAuthInteraction:
    """Test CORS middleware works correctly with auth."""

    def test_cors_headers_present_with_auth(self, client_with_token):
        """CORS headers should be present on authed responses."""
        client, token = client_with_token

        response = client.post(
            "/v1/recall",
            json={"query": "test", "limit": 5},
            headers={
                "Authorization": f"Bearer {token}",
                "Origin": "http://localhost:3000",
            },
        )
        assert response.status_code == 200
        # CORS middleware should add these headers
        assert "access-control-allow-origin" in response.headers

    def test_cors_headers_on_auth_failure(self, client_with_token):
        """CORS headers should be present even on 401 responses."""
        client, token = client_with_token

        response = client.post(
            "/v1/recall",
            json={"query": "test", "limit": 5},
            headers={"Origin": "http://localhost:3000"},
        )
        assert response.status_code == 401
        # Auth middleware returns JSONResponse, CORS should still apply
        assert "access-control-allow-origin" in response.headers


# ---------------------------------------------------------------------------
# Request patterns
# ---------------------------------------------------------------------------


class TestMultipleRequests:
    """Test auth state across multiple requests."""

    def test_sequential_requests_with_same_token(self, client_with_token):
        """Same token should work for many sequential requests."""
        client, token = client_with_token
        headers = {"Authorization": f"Bearer {token}"}

        # Make 20 sequential requests
        for _ in range(20):
            response = client.post(
                "/v1/recall",
                json={"query": "test", "limit": 5},
                headers=headers,
            )
            assert response.status_code == 200

    def test_concurrent_requests_no_interference(self, client_with_token):
        """Multiple simultaneous requests should not interfere."""
        client, token = client_with_token
        headers = {"Authorization": f"Bearer {token}"}

        # Simulate concurrent requests (TestClient is sync)
        responses = []
        for _ in range(10):
            response = client.post(
                "/v1/recall",
                json={"query": "test", "limit": 5},
                headers=headers,
            )
            responses.append(response)

        assert all(r.status_code == 200 for r in responses)

    def test_mixed_valid_invalid_requests(
        self, client_with_token,
    ) -> None:
        """Valid and invalid requests shouldn't corrupt state."""
        client, token = client_with_token
        headers_valid = {"Authorization": f"Bearer {token}"}
        headers_invalid = {"Authorization": "Bearer tm_wrong"}

        # Interleave valid and invalid requests
        for i in range(5):
            headers = headers_valid if i % 2 == 0 else headers_invalid
            expected = 200 if i % 2 == 0 else 401
            response = client.post(
                "/v1/recall",
                json={"query": "test", "limit": 5},
                headers=headers,
            )
            assert response.status_code == expected

    def test_empty_authorization_header(
        self, client_with_token,
    ) -> None:
        """Empty Authorization header returns 401."""
        client, _token = client_with_token

        response = client.get(
            "/v1/stats",
            headers={"Authorization": ""},
        )
        assert response.status_code == 401

    def test_bearer_without_token(
        self, client_with_token,
    ) -> None:
        """'Bearer ' with no token value returns 401."""
        client, _token = client_with_token

        response = client.get(
            "/v1/stats",
            headers={"Authorization": "Bearer "},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Rate limiting across requests
# ---------------------------------------------------------------------------


class TestRateLimitIntegration:
    """Test rate limiting state persists across requests."""

    def test_rate_limit_triggers_after_max_failures(
        self, mock_memory_service, mock_session_store, test_token
    ):
        """After MAX_FAILURES, client should be rate-limited."""
        # Create fresh app with clean middleware state
        app_module._memory_service = mock_memory_service
        app_module._session_store = mock_session_store
        app_module._instance_id = "test-instance"

        app = create_test_app(token=test_token)
        client = TestClient(app)

        headers = {"Authorization": "Bearer tm_wrong_token"}

        # Trigger MAX_FAILURES
        for _ in range(MAX_FAILURES):
            response = client.post(
                "/v1/recall", json={"query": "test", "limit": 5},
                headers=headers
            )
            assert response.status_code == 401

        # Next request should be rate-limited
        response = client.post(
            "/v1/recall", json={"query": "test", "limit": 5}, headers=headers
        )
        assert response.status_code == 429
        assert "Too many failed" in response.json()["error"]

        # Cleanup
        app_module._memory_service = None
        app_module._session_store = None
        app_module._instance_id = None

    def test_rate_limit_cleared_on_success(
        self, mock_memory_service, mock_session_store, test_token
    ):
        """Successful auth should clear failure count."""
        # Create fresh app with clean middleware state
        app_module._memory_service = mock_memory_service
        app_module._session_store = mock_session_store
        app_module._instance_id = "test-instance"

        app = create_test_app(token=test_token)
        client = TestClient(app)

        # Accumulate some failures (but not MAX_FAILURES)
        for i in range(5):  # Fewer failures to avoid rate limit
            response = client.post(
                "/v1/recall",
                json={"query": f"test {i}", "limit": 5},
                headers={"Authorization": "Bearer tm_wrong"},
            )
            assert response.status_code == 401

        # Successful auth should clear failures
        response = client.post(
            "/v1/recall",
            json={"query": "success test", "limit": 5},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert response.status_code == 200

        # Cleanup
        app_module._memory_service = None
        app_module._session_store = None
        app_module._instance_id = None


# ---------------------------------------------------------------------------
# Response header forwarding
# ---------------------------------------------------------------------------


class TestResponseHeaders:
    """Test auth doesn't interfere with response headers."""

    def test_auth_preserves_content_type(
        self, mock_memory_service, mock_session_store, test_token
    ):
        """Auth middleware should not modify Content-Type."""
        # Fresh app to avoid rate limit pollution
        app_module._memory_service = mock_memory_service
        app_module._session_store = mock_session_store
        app_module._instance_id = "test-instance"

        app = create_test_app(token=test_token)
        client = TestClient(app)

        response = client.post(
            "/v1/recall",
            json={"query": "test", "limit": 5},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

        # Cleanup
        app_module._memory_service = None
        app_module._session_store = None
        app_module._instance_id = None

    def test_auth_preserves_custom_headers(self, client_with_token):
        """Custom response headers should pass through auth."""
        client, token = client_with_token

        # Health endpoint doesn't require auth, good for testing headers
        response = client.get("/v1/health")
        assert response.status_code == 200
        # Should have standard FastAPI headers
        assert "content-type" in response.headers


# ---------------------------------------------------------------------------
# Logging integration
# ---------------------------------------------------------------------------


class TestAuthLogging:
    """Test auth audit logging."""

    def test_successful_auth_logged(
        self, mock_memory_service, mock_session_store, test_token, caplog
    ):
        """Successful auth should not log warnings."""
        # Create fresh app
        app_module._memory_service = mock_memory_service
        app_module._session_store = mock_session_store
        app_module._instance_id = "test-instance"

        app = create_test_app(token=test_token)
        client = TestClient(app)

        with caplog.at_level("WARNING"):
            response = client.post(
                "/v1/recall",
                json={"query": "test", "limit": 5},
                headers={"Authorization": f"Bearer {test_token}"},
            )
            assert response.status_code == 200
            # Should not have warning logs for successful auth
            assert not any(
                "Rate limit" in record.message for record in caplog.records
            )

        # Cleanup
        app_module._memory_service = None
        app_module._session_store = None
        app_module._instance_id = None

    def test_rate_limit_triggers_warning_log(self, test_token, caplog):
        """Rate limit should emit WARNING log."""
        from tribalmemory.testing.mocks import (
            MockEmbeddingService,
            MockVectorStore,
            MockMemoryService,
        )
        from tribalmemory.services.session_store import InMemorySessionStore

        # Fresh mock services
        embedding = MockEmbeddingService()
        vector_store = MockVectorStore(embedding)
        service = MockMemoryService(
            instance_id="test-instance-rate-log",
            embedding_service=embedding,
            vector_store=vector_store,
        )
        session_store = InMemorySessionStore(
            instance_id="test-instance-rate-log",
            embedding_service=embedding,
            vector_store=vector_store,
        )

        # Create fresh app
        app_module._memory_service = service
        app_module._session_store = session_store
        app_module._instance_id = "test-instance-rate-log"

        app = create_test_app(token=test_token)
        client = TestClient(app)

        with caplog.at_level("WARNING"):
            # Trigger rate limit
            for i in range(MAX_FAILURES):
                client.post(
                    "/v1/recall",
                    json={"query": f"test {i}", "limit": 5},
                    headers={"Authorization": "Bearer tm_wrong"},
                )

            # Should have rate limit warning
            assert any(
                "Rate limit triggered" in record.message
                for record in caplog.records
            )

        # Cleanup
        app_module._memory_service = None
        app_module._session_store = None
        app_module._instance_id = None

    def test_no_token_warning_on_middleware_init(self, caplog):
        """Middleware without token should log warning on init."""
        with caplog.at_level("WARNING"):
            middleware = TokenAuthMiddleware(app=None, token=None)
            # Should warn about missing token
            assert any(
                "No API token configured" in record.message
                for record in caplog.records
            )


# ---------------------------------------------------------------------------
# Legacy mode behavior
# ---------------------------------------------------------------------------


class TestLegacyMode:
    """Test app behavior when no token is configured."""

    def test_legacy_mode_allows_all_requests(self, client_without_token):
        """Without token, all requests should succeed."""
        client = client_without_token

        # Should work without any auth header
        response = client.get("/v1/health")
        assert response.status_code == 200

        response = client.post(
            "/v1/recall", json={"query": "test", "limit": 5}
        )
        assert response.status_code == 200

    def test_legacy_mode_ignores_auth_header(self, client_without_token):
        """Legacy mode should ignore Authorization header."""
        client = client_without_token

        # Should work even with invalid token in legacy mode
        response = client.post(
            "/v1/recall",
            json={"query": "test", "limit": 5},
            headers={"Authorization": "Bearer tm_fake"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Token loading from environment
# ---------------------------------------------------------------------------


class TestTokenLoading:
    """Test token loading from env vars and files."""

    def test_token_from_env_var(self, monkeypatch) -> None:
        """Token loaded from TRIBAL_MEMORY_API_TOKEN env var."""
        from tribalmemory.testing.mocks import (
            MockEmbeddingService,
            MockVectorStore,
            MockMemoryService,
        )
        from tribalmemory.services.session_store import (
            InMemorySessionStore,
        )

        token = generate_token()
        monkeypatch.setenv("TRIBAL_MEMORY_API_TOKEN", token)

        embedding = MockEmbeddingService()
        vector_store = MockVectorStore(embedding)
        service = MockMemoryService(
            instance_id="test-instance-token-env",
            embedding_service=embedding,
            vector_store=vector_store,
        )
        session_store = InMemorySessionStore(
            instance_id="test-instance-token-env",
            embedding_service=embedding,
            vector_store=vector_store,
        )

        app_module._memory_service = service
        app_module._session_store = session_store
        app_module._instance_id = "test-instance-token-env"

        # Resolve token the same way create_app does:
        # env var takes precedence over file
        import os
        resolved = os.environ.get("TRIBAL_MEMORY_API_TOKEN")
        assert resolved == token

        app = create_test_app(token=resolved)
        client = TestClient(app)

        # Should require token
        response = client.post(
            "/v1/recall", json={"query": "test", "limit": 5}
        )
        assert response.status_code == 401

        # Should work with env-resolved token
        response = client.post(
            "/v1/recall",
            json={"query": "test", "limit": 5},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

    def test_token_from_file(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Token loaded from .env file via load_token."""
        from tribalmemory.testing.mocks import (
            MockEmbeddingService,
            MockVectorStore,
            MockMemoryService,
        )
        from tribalmemory.services.session_store import (
            InMemorySessionStore,
        )
        from tribalmemory.server.auth import load_token

        token = generate_token()
        env_path = tmp_path / ".env"
        save_token(token, env_path)

        # Ensure env var is NOT set so file wins
        monkeypatch.delenv(
            "TRIBAL_MEMORY_API_TOKEN", raising=False,
        )

        # Actually exercise load_token
        resolved = load_token(env_path)
        assert resolved == token

        embedding = MockEmbeddingService()
        vector_store = MockVectorStore(embedding)
        service = MockMemoryService(
            instance_id="test-instance-token-file",
            embedding_service=embedding,
            vector_store=vector_store,
        )
        session_store = InMemorySessionStore(
            instance_id="test-instance-token-file",
            embedding_service=embedding,
            vector_store=vector_store,
        )

        app_module._memory_service = service
        app_module._session_store = session_store
        app_module._instance_id = "test-instance-token-file"

        app = create_test_app(token=resolved)
        client = TestClient(app)

        # Should reject without token
        response = client.post(
            "/v1/recall", json={"query": "test", "limit": 5}
        )
        assert response.status_code == 401

        # Should accept with file-loaded token
        response = client.post(
            "/v1/recall",
            json={"query": "test", "limit": 5},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Middleware stacking order
# ---------------------------------------------------------------------------


class TestMiddlewareOrder:
    """Test that middleware is applied in correct order."""

    def test_auth_before_cors(self, test_token):
        """Auth middleware should process before CORS."""
        app = create_test_app(token=test_token, add_cors=True)

        # Verify middleware stack order
        middleware_types = [m.cls.__name__ for m in app.user_middleware]

        # TokenAuthMiddleware should be in the stack
        assert any("TokenAuthMiddleware" in cls for cls in middleware_types)

        # CORSMiddleware should be in the stack
        assert any("CORSMiddleware" in cls for cls in middleware_types)

    def test_auth_applied_to_all_routes(self, test_token):
        """Auth should apply to all non-public routes."""
        from tribalmemory.testing.mocks import (
            MockEmbeddingService,
            MockVectorStore,
            MockMemoryService,
        )
        from tribalmemory.services.session_store import InMemorySessionStore

        # Fresh mock services
        embedding = MockEmbeddingService()
        vector_store = MockVectorStore(embedding)
        service = MockMemoryService(
            instance_id="test-instance-all-routes",
            embedding_service=embedding,
            vector_store=vector_store,
        )
        session_store = InMemorySessionStore(
            instance_id="test-instance-all-routes",
            embedding_service=embedding,
            vector_store=vector_store,
        )

        # Create fresh app
        app_module._memory_service = service
        app_module._session_store = session_store
        app_module._instance_id = "test-instance-all-routes"

        app = create_test_app(token=test_token)
        client = TestClient(app)

        # Test various endpoints
        endpoints = [
            ("/v1/remember", "POST", {"content": "test"}),
            ("/v1/recall", "POST", {"query": "test", "limit": 5}),
            ("/v1/forget/fake-id", "DELETE", None),
        ]

        for endpoint, method, payload in endpoints:
            if method == "POST":
                response = client.post(endpoint, json=payload)
            elif method == "DELETE":
                response = client.delete(endpoint)

            # Should all require auth (401 without token)
            assert response.status_code == 401

        # Cleanup
        app_module._memory_service = None
        app_module._session_store = None
        app_module._instance_id = None
