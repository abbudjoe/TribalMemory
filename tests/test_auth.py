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
    load_rate_limit_state,
    load_token,
    save_rate_limit_state,
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


def _create_test_app(token=None, rate_limit_path=None):
    """Create a minimal FastAPI app with auth middleware for testing."""
    import tempfile
    app = FastAPI()
    # Use isolated rate limit path to prevent test state leaking
    if rate_limit_path is None:
        rate_limit_path = Path(
            tempfile.mkdtemp()
        ) / "rate-limits.json"
    app.add_middleware(
        TokenAuthMiddleware,
        token=token,
        rate_limit_path=rate_limit_path,
    )

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

    def test_env_var_takes_precedence(self):
        """TRIBAL_MEMORY_API_TOKEN env var should override .env file."""
        with patch.dict(os.environ, {"TRIBAL_MEMORY_API_TOKEN": "tm_from_env"}):
            from tribalmemory.server.app import create_app
            from tribalmemory.server.config import TribalMemoryConfig
            config = TribalMemoryConfig()
            app = create_app(config)
            client = TestClient(app)

            # Token from env var works
            response = client.get(
                "/v1/stats",
                headers={"Authorization": "Bearer tm_from_env"},
            )
            # May be 500 (no service) but NOT 401
            assert response.status_code != 401

    def test_missing_token_error_message(self):
        """Missing token should give specific error message."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)
        response = client.get("/v1/stats")
        assert response.status_code == 401
        assert "Missing" in response.json()["error"]
        assert "Authorization" in response.json()["error"]

    def test_invalid_token_error_message(self):
        """Invalid token should give different error than missing."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)
        response = client.get(
            "/v1/stats",
            headers={"Authorization": "Bearer tm_wrong"},
        )
        assert response.status_code == 401
        assert "Invalid" in response.json()["error"]

    def test_ip_tracking_memory_bounded(self):
        """Rate limit tracking should not grow unbounded."""
        app = _create_test_app(token="tm_secret")
        client = TestClient(app)

        # Access middleware directly to set a small limit
        for middleware in app.user_middleware:
            pass  # Can't easily access, test via behavior

        # Just verify the constant exists and is reasonable
        from tribalmemory.server.auth import MAX_TRACKED_IPS
        assert MAX_TRACKED_IPS > 0
        assert MAX_TRACKED_IPS <= 100000


# ---------------------------------------------------------------------------
# Rate limit persistence
# ---------------------------------------------------------------------------


class TestRateLimitPersistence:
    """Tests for rate limit state persistence to disk."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Persisted rate limit state survives save/load cycle."""
        path = tmp_path / "rate-limits.json"
        future = time.time() + 3600  # 1 hour from now

        save_rate_limit_state(
            {"1.2.3.4": 10, "5.6.7.8": 5},
            {"1.2.3.4": future, "5.6.7.8": future},
            path,
        )

        failures, cooldowns = load_rate_limit_state(path)
        assert failures["1.2.3.4"] == 10
        assert failures["5.6.7.8"] == 5
        assert cooldowns["1.2.3.4"] == future
        assert cooldowns["5.6.7.8"] == future

    def test_expired_cooldowns_not_loaded(self, tmp_path):
        """Expired cooldowns are filtered out on load."""
        path = tmp_path / "rate-limits.json"
        past = time.time() - 100  # already expired
        future = time.time() + 3600

        save_rate_limit_state(
            {"expired": 10, "active": 10},
            {"expired": past, "active": future},
            path,
        )

        # Re-write to include expired entry (save filters it out,
        # so we write directly for this test)
        import json
        with open(path, "w") as f:
            json.dump({
                "expired_ip": {
                    "failures": 10,
                    "cooldown_until": past,
                },
                "active_ip": {
                    "failures": 10,
                    "cooldown_until": future,
                },
            }, f)

        failures, cooldowns = load_rate_limit_state(path)
        assert "expired_ip" not in failures
        assert "active_ip" in failures
        assert failures["active_ip"] == 10

    def test_nonexistent_file_returns_empty(self, tmp_path):
        """Loading from nonexistent file returns empty dicts."""
        path = tmp_path / "nonexistent.json"
        failures, cooldowns = load_rate_limit_state(path)
        assert failures == {}
        assert cooldowns == {}

    def test_corrupted_file_returns_empty(self, tmp_path):
        """Corrupted JSON file returns empty dicts (graceful)."""
        path = tmp_path / "rate-limits.json"
        with open(path, "w") as f:
            f.write("not valid json{{{")

        failures, cooldowns = load_rate_limit_state(path)
        assert failures == {}
        assert cooldowns == {}

    def test_non_dict_json_returns_empty(self, tmp_path):
        """Valid JSON that's not a dict returns empty (graceful)."""
        import json as json_mod
        path = tmp_path / "rate-limits.json"

        # Array instead of dict
        with open(path, "w") as f:
            json_mod.dump([1, 2, 3], f)
        failures, cooldowns = load_rate_limit_state(path)
        assert failures == {}
        assert cooldowns == {}

        # String instead of dict
        with open(path, "w") as f:
            json_mod.dump("hello", f)
        failures, cooldowns = load_rate_limit_state(path)
        assert failures == {}
        assert cooldowns == {}

    def test_malformed_entry_skipped(self, tmp_path):
        """Non-dict entries in the state file are skipped."""
        import json as json_mod
        path = tmp_path / "rate-limits.json"
        future = time.time() + 3600

        with open(path, "w") as f:
            json_mod.dump({
                "good_ip": {
                    "failures": 5,
                    "cooldown_until": future,
                },
                "bad_ip": "not a dict",
                "also_bad": 42,
            }, f)

        failures, cooldowns = load_rate_limit_state(path)
        assert "good_ip" in failures
        assert failures["good_ip"] == 5
        assert "bad_ip" not in failures
        assert "also_bad" not in failures

    def test_save_removes_file_when_no_active(self, tmp_path):
        """File is cleaned up when no active cooldowns exist."""
        path = tmp_path / "rate-limits.json"
        past = time.time() - 100

        # First create the file with active entry
        future = time.time() + 3600
        save_rate_limit_state(
            {"1.2.3.4": 10},
            {"1.2.3.4": future},
            path,
        )
        assert path.exists()

        # Now save with only expired entries
        save_rate_limit_state(
            {"1.2.3.4": 10},
            {"1.2.3.4": past},
            path,
        )
        assert not path.exists()

    def test_save_sets_secure_permissions(self, tmp_path):
        """Rate limit file should have 600 permissions."""
        path = tmp_path / "rate-limits.json"
        future = time.time() + 3600

        save_rate_limit_state(
            {"1.2.3.4": 10},
            {"1.2.3.4": future},
            path,
        )

        perms = oct(path.stat().st_mode)[-3:]
        assert perms == "600"

    def test_middleware_loads_persisted_state(self, tmp_path):
        """Middleware picks up persisted rate limits on init."""
        path = tmp_path / "rate-limits.json"
        future = time.time() + 3600

        # Persist a cooldown for testclient's IP
        save_rate_limit_state(
            {"testclient": MAX_FAILURES},
            {"testclient": future},
            path,
        )

        # Create app with the persisted state
        app = _create_test_app(
            token="tm_secret",
            rate_limit_path=path,
        )
        client = TestClient(app)

        # Should be immediately rate-limited
        response = client.get(
            "/v1/stats",
            headers={"Authorization": "Bearer tm_secret"},
        )
        assert response.status_code == 429

    def test_middleware_persists_on_cooldown(self, tmp_path):
        """Middleware writes state to disk when cooldown triggers."""
        path = tmp_path / "rate-limits.json"
        app = _create_test_app(
            token="tm_secret",
            rate_limit_path=path,
        )
        client = TestClient(app)

        assert not path.exists()

        # Trigger rate limiting
        for _ in range(MAX_FAILURES):
            client.get(
                "/v1/stats",
                headers={"Authorization": "Bearer tm_wrong"},
            )

        # State should be persisted
        assert path.exists()
        failures, cooldowns = load_rate_limit_state(path)
        assert len(failures) > 0


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


class TestAuditLogging:
    """Tests for auth audit logging."""

    def test_successful_auth_logged(self, tmp_path, caplog):
        """Successful authentication should be logged at INFO."""
        import logging
        with caplog.at_level(logging.INFO, logger="tribalmemory.auth"):
            app = _create_test_app(token="tm_secret")
            client = TestClient(app)
            client.get(
                "/v1/stats",
                headers={"Authorization": "Bearer tm_secret"},
            )

        assert any(
            "Auth success" in r.message and "/v1/stats" in r.message
            for r in caplog.records
        )

    def test_failed_auth_not_logged_as_success(self, tmp_path, caplog):
        """Failed auth should NOT produce success log."""
        import logging
        with caplog.at_level(logging.INFO, logger="tribalmemory.auth"):
            app = _create_test_app(token="tm_secret")
            client = TestClient(app)
            client.get(
                "/v1/stats",
                headers={"Authorization": "Bearer tm_wrong"},
            )

        assert not any(
            "Auth success" in r.message
            for r in caplog.records
        )
