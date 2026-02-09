"""Token-based authentication middleware for TribalMemory API.

Security properties:
- 32-byte random token (256 bits entropy), prefixed with 'tm_'
- Constant-time comparison (no timing attacks)
- Rate limiting on failed auth attempts (persisted to disk)
- Token stored in ~/.tribal-memory/.env with 600 permissions
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("tribalmemory.auth")

# Token prefix for easy identification
TOKEN_PREFIX = "tm_"

# Rate limiting: max failures before cooldown
MAX_FAILURES = 10
COOLDOWN_SECONDS = 60
MAX_TRACKED_IPS = 10000  # Prevent unbounded memory growth


def generate_token() -> str:
    """Generate a new API token.

    Returns:
        Token string with 'tm_' prefix and 32 random hex bytes.
    """
    return f"{TOKEN_PREFIX}{secrets.token_hex(32)}"


def save_token(token: str, env_path: Optional[Path] = None) -> Path:
    """Save token to .env file with secure permissions.

    Args:
        token: The API token to save.
        env_path: Path to .env file. Defaults to ~/.tribal-memory/.env.

    Returns:
        Path where the token was saved.
    """
    if env_path is None:
        env_path = Path("~/.tribal-memory/.env").expanduser()

    env_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing env vars (preserve other settings)
    existing: dict[str, str] = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    existing[key.strip()] = value.strip()

    # Update token
    existing["TRIBAL_MEMORY_API_TOKEN"] = token

    # Write back
    with open(env_path, "w") as f:
        f.write("# TribalMemory configuration\n")
        f.write("# This file contains sensitive credentials.\n")
        for key, value in existing.items():
            f.write(f"{key}={value}\n")

    # Set secure permissions (owner read/write only)
    os.chmod(env_path, 0o600)

    return env_path


def load_token(env_path: Optional[Path] = None) -> Optional[str]:
    """Load token from .env file.

    Args:
        env_path: Path to .env file. Defaults to ~/.tribal-memory/.env.

    Returns:
        Token string, or None if not found.
    """
    if env_path is None:
        env_path = Path("~/.tribal-memory/.env").expanduser()

    if not env_path.exists():
        return None

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("TRIBAL_MEMORY_API_TOKEN="):
                return line.split("=", 1)[1].strip()

    return None


def _default_rate_limit_path() -> Path:
    """Return default path for rate limit state persistence."""
    return Path("~/.tribal-memory/rate-limits.json").expanduser()


def save_rate_limit_state(
    failure_count: dict[str, int],
    cooldown_until: dict[str, float],
    path: Optional[Path] = None,
) -> None:
    """Persist rate limit state to disk.

    Only saves IPs that are currently in cooldown to avoid
    persisting stale data.

    Args:
        failure_count: Per-IP failure counts.
        cooldown_until: Per-IP cooldown expiry timestamps.
        path: File path. Defaults to ~/.tribal-memory/rate-limits.json.
    """
    if path is None:
        path = _default_rate_limit_path()

    now = time.time()
    # Only persist IPs with active cooldowns
    active: dict[str, dict] = {}
    for ip, until in cooldown_until.items():
        if until > now:
            active[ip] = {
                "failures": failure_count.get(ip, 0),
                "cooldown_until": until,
            }

    if not active:
        # Remove file if no active cooldowns
        if path.exists():
            path.unlink()
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(active, f)
    os.chmod(path, 0o600)


def load_rate_limit_state(
    path: Optional[Path] = None,
) -> tuple[dict[str, int], dict[str, float]]:
    """Load rate limit state from disk.

    Returns only entries with active (non-expired) cooldowns.

    Args:
        path: File path. Defaults to ~/.tribal-memory/rate-limits.json.

    Returns:
        Tuple of (failure_count, cooldown_until) dicts.
    """
    if path is None:
        path = _default_rate_limit_path()

    failure_count: dict[str, int] = {}
    cooldown_until: dict[str, float] = {}

    if not path.exists():
        return failure_count, cooldown_until

    try:
        with open(path) as f:
            data = json.load(f)

        now = time.time()
        for ip, entry in data.items():
            until = entry.get("cooldown_until", 0)
            if until > now:
                failure_count[ip] = entry.get("failures", 0)
                cooldown_until[ip] = until
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load rate limit state: %s", e)

    return failure_count, cooldown_until


def _constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for bearer token authentication.

    Checks Authorization header on all requests except:
    - GET /health (always public)
    - GET /docs, /openapi.json (API docs)
    - OPTIONS (CORS preflight)

    When no token is configured (legacy mode), logs a warning
    but allows requests through.
    """

    # Paths that never require auth
    PUBLIC_PATHS = frozenset({
        "/health",
        "/v1/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/",
    })

    def __init__(
        self,
        app,
        token: Optional[str] = None,
        rate_limit_path: Optional[Path] = None,
    ):
        super().__init__(app)
        self.token = token
        self._rate_limit_path = rate_limit_path
        self._max_tracked_ips = MAX_TRACKED_IPS

        # Load persisted rate limit state
        self._failure_count, self._cooldown_until = (
            load_rate_limit_state(rate_limit_path)
        )
        if self._failure_count:
            logger.info(
                "Loaded %d rate-limited IPs from disk",
                len(self._failure_count),
            )

        if not token:
            logger.warning(
                "No API token configured. Server is running without authentication. "
                "Run 'tribalmemory token generate' to secure your instance."
            )

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no auth required)."""
        return path in self.PUBLIC_PATHS

    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate-limited due to failed attempts."""
        cooldown = self._cooldown_until.get(client_ip, 0)
        if time.time() < cooldown:
            return True

        # Clear expired cooldown
        if cooldown > 0 and time.time() >= cooldown:
            self._failure_count.pop(client_ip, None)
            self._cooldown_until.pop(client_ip, None)

        return False

    def _record_failure(self, client_ip: str) -> None:
        """Record a failed auth attempt and apply rate limiting if needed."""
        # Evict oldest entry if tracking too many IPs (memory safety)
        at_capacity = len(self._failure_count) >= self._max_tracked_ips
        is_new_ip = client_ip not in self._failure_count
        if at_capacity and is_new_ip:
            oldest_ip = next(iter(self._failure_count))
            self._failure_count.pop(oldest_ip, None)
            self._cooldown_until.pop(oldest_ip, None)

        count = self._failure_count.get(client_ip, 0) + 1
        self._failure_count[client_ip] = count

        if count >= MAX_FAILURES:
            self._cooldown_until[client_ip] = (
                time.time() + COOLDOWN_SECONDS
            )
            logger.warning(
                "Rate limit triggered for %s (%d failures). "
                "Cooldown for %ds.",
                client_ip,
                count,
                COOLDOWN_SECONDS,
            )
            # Persist to disk on cooldown trigger
            save_rate_limit_state(
                self._failure_count,
                self._cooldown_until,
                self._rate_limit_path,
            )

    def _clear_failures(self, client_ip: str) -> None:
        """Clear failure count on successful auth."""
        self._failure_count.pop(client_ip, None)
        self._cooldown_until.pop(client_ip, None)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Check authentication before passing to route handler."""
        # Allow preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # Allow public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # No token configured = legacy mode (allow all)
        if not self.token:
            return await call_next(request)

        # Check rate limiting
        client_ip = request.client.host if request.client else "unknown"
        if self._is_rate_limited(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many failed authentication attempts. "
                    f"Try again in {COOLDOWN_SECONDS} seconds."
                },
            )

        # Extract token from Authorization header
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            provided_token = auth_header[7:]
        else:
            provided_token = ""

        # Validate token
        if not provided_token:
            self._record_failure(client_ip)
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Missing API token. "
                    "Include 'Authorization: Bearer <token>' header."
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not _constant_time_compare(provided_token, self.token):
            self._record_failure(client_ip)
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API token."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Auth success
        self._clear_failures(client_ip)
        logger.info(
            "Auth success: %s %s from %s",
            request.method,
            request.url.path,
            client_ip,
        )
        return await call_next(request)
