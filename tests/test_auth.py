"""Tests for token authentication middleware and utilities."""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tribalmemory.server.auth import (
    TOKEN_PREFIX,
    COOLDOWN_SECONDS,
    MAX_FAILURES,
    TokenAuthMiddleware,
    _constant_time_compare,
    generate_token,
    load_token,
    save_token,
)


# ---------------------------------------------------------------------------
# Token generation and storage
# ---------------------------------------------------------------------------


class TestTokenGeneration:
    """Tests for token generation."""

    def test_generate_token_has_prefix(self):
        token = generate_token()
        assert token.startswith(TOKEN_PREFIX)

    def test_generate_token_length(self):
        token = generate_token()
        # tm_ (3 chars) + 64 hex chars (32 bytes) = 67 chars
        assert len(token) == 67

    def test_generate_token_unique(self):
        tokens = {generate_token() for _ in range(100)}
        assert len(tokens) == 100

    def test_generate_token_hex_chars(self):
        token = generate_token()
        hex_part = token[len(TOKEN_PREFIX):]
        assert all(c in "0123456789abcdef" for c in hex_part)


class TestTokenStorage:
    """Tests for token save/load."""

    def test_save_and_load(self, tmp_path):
        env_path = tmp_path / ".env"
        token = "tm_test123"
        save_token(token, env_path)
        loaded = load_token(env_path)
        assert loaded == token

    def test_save_sets_permissions(self, tmp_path):
        env_path = tmp_path / ".env"
        save_token("tm_test", env_path)
        mode = oct(env_path.stat().st_mode)[-3:]
        assert mode == "600"

    def test_save_preserves_other_vars(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("OTHER_VAR=hello\n")
        save_token("tm_test", env_path)
        content = env_path.read_text()
        assert "OTHER_VAR=hello" in content
        assert "TRIBAL_MEMORY_API_TOKEN=tm_test" in content

    def test_save_overwrites_existing_token(self, tmp_path):
        env_path = tmp_path / ".env"
        save_token("tm_old", env_path)
        save_token("tm_new", env_path)
        loaded = load_token(env_path)
        assert loaded == "tm_new"

    def test_load_nonexistent_returns_none(self, tmp_path):
        env_path = tmp_path / "nonexistent" / ".env"
        assert load_token(env_path) is None

    def test_load_no_token_in_file(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("OTHER_VAR=hello\n")
        assert load_token(env_path) is None

    def test_save_creates_parent_dirs(self, tmp_path):
        env_path = tmp_path / "deep" / "nested" / ".env"
        save_token("tm_test", env_path)
        assert load_token(env_path) == "tm_test"


class TestConstantTimeCompare:
    """Tests for timing-safe comparison."""

    def test_equal_strings(self):
        assert _constant_time_compare("abc", "abc") is True

    def test_unequal_strings(self):
        assert _constant_time_compare("abc", "xyz") is False

    def test_empty_strings(self):
        assert _constant_time_compare("", "") is True

    def test_different_lengths(self):
        assert _constant_time_compare("short", "longer_string") is False


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


def _create_test_app(token=None):
    """Create a minimal FastAPI app with auth middleware for testing."""
    app = FastAPI()
    app.add_middleware(TokenAuthMiddleware, token=token)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/health")
    async def v1_health():
        return {"status": "ok"}

    @app.get("/")
    async def root():
        return {"service": "tribal-memory"}

    @app.get("/v1/stats")
    async def stats():
        return {"total_memories": 42}

    @app.post("/v1/remember")
    async def remember():
        return {"success": True}

    @app.post("/v1/recall")
    async def recall():
        return {"results": []}

    return app


class TestAuthMiddleware:
    """Tests for TokenAuthMiddleware."""

    def test_no_token_configured_allows_all(self):
        """Legacy mode: no token = no auth required."""
        app = _create_test_app(token=None)
        client = TestClient(app)
        response = client.get("/v1/stats")
        assert response.status_code == 200

    def test_public_paths_no_auth(self):
        """Public paths should work without token."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)

        for path in ["/health", "/v1/health", "/"]:
            response = client.get(path)
            assert response.status_code == 200, f"Failed for {path}"

    def test_protected_path_requires_token(self):
        """Protected paths should reject requests without token."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)
        response = client.get("/v1/stats")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_valid_token_allows_access(self):
        """Valid bearer token should grant access."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)
        response = client.get(
            "/v1/stats",
            headers={"Authorization": "Bearer tm_secret"},
        )
        assert response.status_code == 200

    def test_invalid_token_rejected(self):
        """Wrong token should be rejected."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)
        response = client.get(
            "/v1/stats",
            headers={"Authorization": "Bearer tm_wrong"},
        )
        assert response.status_code == 401

    def test_missing_bearer_prefix(self):
        """Token without 'Bearer ' prefix should be rejected."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)
        response = client.get(
            "/v1/stats",
            headers={"Authorization": "tm_secret"},
        )
        assert response.status_code == 401

    def test_empty_authorization_header(self):
        """Empty Authorization header should be rejected."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)
        response = client.get(
            "/v1/stats",
            headers={"Authorization": ""},
        )
        assert response.status_code == 401

    def test_post_requires_token(self):
        """POST endpoints should also require token."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)

        # Without token
        response = client.post("/v1/remember", json={"content": "test"})
        assert response.status_code == 401

        # With token
        response = client.post(
            "/v1/remember",
            json={"content": "test"},
            headers={"Authorization": "Bearer tm_secret"},
        )
        assert response.status_code == 200

    def test_rate_limiting_after_failures(self):
        """Should rate-limit after MAX_FAILURES failed attempts."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)

        # Exhaust failure limit
        for _ in range(MAX_FAILURES):
            client.get(
                "/v1/stats",
                headers={"Authorization": "Bearer tm_wrong"},
            )

        # Next request should be rate-limited (even with correct token)
        response = client.get(
            "/v1/stats",
            headers={"Authorization": "Bearer tm_secret"},
        )
        assert response.status_code == 429
        assert "Too many" in response.json()["error"]

    def test_valid_auth_clears_failure_count(self):
        """Successful auth should reset the failure counter."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)

        # Some failures (but below limit)
        for _ in range(MAX_FAILURES - 2):
            client.get(
                "/v1/stats",
                headers={"Authorization": "Bearer tm_wrong"},
            )

        # Successful auth clears counter
        response = client.get(
            "/v1/stats",
            headers={"Authorization": "Bearer tm_secret"},
        )
        assert response.status_code == 200

        # More failures (would exceed limit if counter wasn't cleared)
        for _ in range(MAX_FAILURES - 2):
            client.get(
                "/v1/stats",
                headers={"Authorization": "Bearer tm_wrong"},
            )

        # Still not rate-limited because counter was reset
        response = client.get(
            "/v1/stats",
            headers={"Authorization": "Bearer tm_secret"},
        )
        assert response.status_code == 200

    def test_options_always_allowed(self):
        """OPTIONS requests (CORS preflight) should bypass auth."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)
        response = client.options("/v1/stats")
        # OPTIONS may return 405 if not handled, but should NOT return 401
        assert response.status_code != 401
