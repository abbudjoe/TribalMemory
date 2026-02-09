# Authentication

TribalMemory supports bearer token authentication to protect your
memory API from unauthorized access.

## Quick Start

```bash
# Generate a token
tribalmemory token generate

# Start the server (token is loaded automatically)
tribalmemory serve
```

The server reads the token from `~/.tribal-memory/.env` on startup.
All non-public endpoints require the token in the `Authorization`
header:

```bash
curl -H "Authorization: Bearer tm_abc123..." \
  http://127.0.0.1:18790/v1/recall \
  -d '{"query": "meeting notes"}'
```

## Token Management

### Generate a Token

```bash
tribalmemory token generate
```

This creates a `tm_`-prefixed token with 256 bits of cryptographic
entropy and saves it to `~/.tribal-memory/.env` with `600`
permissions (owner read/write only).

> **Important:** The full token is displayed only once at generation
> time. Copy it immediately and store it securely.

### Rotate a Token

```bash
tribalmemory token rotate
```

Generates a new token and replaces the old one. The server must be
restarted to pick up the new token. Update all clients with the new
token after rotation.

### View Current Token

```bash
tribalmemory token show
```

Displays a masked version of the token (`tm_...last8chars`) for
verification. The full token is never shown after initial generation.

## Token Storage

Tokens are stored in **`~/.tribal-memory/.env`** as:

```
TRIBAL_MEMORY_API_TOKEN=tm_<64 hex chars>
```

The file is created with `600` permissions (owner read/write only).
Other environment variables in the file are preserved during token
operations.

### Environment Variable Override

Set `TRIBAL_MEMORY_API_TOKEN` as an environment variable to override
the file-based token. This is useful for containerized deployments
(12-factor app pattern):

```bash
export TRIBAL_MEMORY_API_TOKEN=tm_abc123...
tribalmemory serve
```

The server logs which source the token was loaded from (environment
variable or file).

## Legacy Mode

If no token is configured, the server runs in **legacy mode**:

- All requests are allowed without authentication
- A warning is logged on startup
- This preserves backward compatibility for existing deployments

To secure your instance, run `tribalmemory token generate`.

## Public Paths

The following paths never require authentication:

| Path | Purpose |
|------|---------|
| `/` | Root endpoint |
| `/health` | Health check |
| `/v1/health` | Versioned health check |
| `/docs` | Swagger UI |
| `/redoc` | ReDoc UI |
| `/openapi.json` | OpenAPI schema |

`OPTIONS` requests are also allowed without auth for CORS preflight.

## Rate Limiting

Failed authentication attempts are rate-limited per client IP:

- **Threshold:** 10 failed attempts
- **Cooldown:** 60 seconds
- **Scope:** Per IP address, in-memory

After 10 failures from the same IP, the server returns `429 Too Many
Requests` for 60 seconds. A successful authentication clears the
failure count for that IP.

Rate limit state is persisted to `~/.tribal-memory/rate-limits.json`
(600 permissions) and survives server restarts. Only active cooldowns
are saved; expired entries are auto-cleaned on load, and the file is
removed when no cooldowns exist. The server tracks up to 10,000
unique IPs to prevent unbounded memory growth.

## HTTPS / Transport Security

TribalMemory binds to `127.0.0.1` by default, so the API is only
accessible from the local machine. For remote access, **always use
encrypted transport**.

### Recommended: Tailscale Serve

[Tailscale Serve](https://tailscale.com/kb/1242/tailscale-serve)
provides automatic HTTPS with no configuration:

```bash
# Expose TribalMemory over Tailscale with HTTPS
tailscale serve https / http://127.0.0.1:18790
```

This gives you:
- Automatic TLS certificates (via Let's Encrypt)
- Access only from your Tailscale network
- No port forwarding or firewall changes needed

### Alternative: Reverse Proxy

Use nginx or Caddy as a reverse proxy with TLS termination:

```nginx
server {
    listen 443 ssl;
    server_name memory.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:18790;
        proxy_set_header Authorization $http_authorization;
    }
}
```

> **Warning:** Never expose TribalMemory directly on a public
> network without TLS. Bearer tokens sent over plain HTTP can be
> intercepted.

## MCP Client Configuration

When using TribalMemory as an MCP server (Claude Code, Codex), the
token is passed via environment variables in the MCP config:

```json
{
  "mcpServers": {
    "tribal-memory": {
      "command": "tribalmemory-mcp",
      "env": {
        "TRIBAL_MEMORY_API_TOKEN": "tm_abc123..."
      }
    }
  }
}
```

For HTTP-based MCP clients, include the token in the server URL
or pass it as a header per your client's configuration.

## Security Properties

- **Token entropy:** 256 bits (cryptographically secure via
  `secrets.token_hex`)
- **Comparison:** Constant-time via `hmac.compare_digest` (prevents
  timing attacks)
- **Storage:** File permissions `600` (owner-only access)
- **Prefix:** `tm_` for easy identification in logs and configs
- **Audit:** Successful and failed auth attempts are logged
