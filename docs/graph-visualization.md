# Knowledge Graph Visualization

TribalMemory includes a built-in web UI for exploring the
knowledge graph that forms as your agents store memories.

## Quick Start

1. **Start the server** (with or without auth):

   ```bash
   tribalmemory serve
   ```

2. **Open the graph UI** in your browser:

   ```
   http://127.0.0.1:18790/graph
   ```

3. **If token auth is enabled**, enter your token in the
   login form. The token is never sent via URL — only
   through the login form for security.

## Features

- **Entity search** — Find entities by name
- **Type filtering** — Filter by entity type
  (e.g., person, technology, service)
- **Neighborhood exploration** — Click a node to see its
  connections; double-click to expand further
- **Multi-hop traversal** — Explore up to 3 hops from
  any entity
- **Dark theme** — Matches modern developer tools
- **No external dependencies** — Cytoscape.js is bundled
  locally, no CDN required

## API Endpoints

The visualization is powered by three REST endpoints.
Entity names in URL paths must be URL-encoded
(e.g., `Albert Einstein` → `Albert%20Einstein`).

**Response codes (all endpoints):**
- `200 OK` — Success
- `401 Unauthorized` — Invalid or missing bearer token
- `422 Unprocessable Entity` — Invalid query parameters
  (e.g., limit out of range, search too long)
- `400 Bad Request` — Empty or oversized entity name
  (neighborhood endpoint only)
- `404 Not Found` — Entity not found (neighborhood only)

### GET /v1/graph/stats

Returns high-level graph statistics.

```json
{
  "entity_count": 142,
  "relationship_count": 89,
  "entity_types": {"person": 12, "technology": 45},
  "relationship_types": {"uses": 30, "knows": 15}
}
```

### GET /v1/graph/entities

Paginated entity listing with optional filters.

**Parameters:**
- `offset` (int, default 0) — Pagination offset
- `limit` (int, 1–100, default 50) — Values outside
  this range return 422
- `entity_type` (string, optional) — Filter by type
- `search` (string, optional, max 200 chars) — Name
  substring search; returns 422 if exceeded

```json
{
  "entities": [
    {"name": "Python", "entity_type": "technology",
     "memory_count": 12}
  ],
  "total": 142,
  "offset": 0,
  "limit": 50
}
```

The `memory_count` field indicates how many memories
reference this entity.

### GET /v1/graph/neighborhood/{entity_name}

Explore the neighborhood around a specific entity.

**Parameters:**
- `hops` (int, 1–3, default 1) — Traversal depth

```json
{
  "focal": "Python",
  "hops": 1,
  "nodes": [
    {"name": "Python", "entity_type": "technology",
     "memory_count": 12},
    {"name": "FastAPI", "entity_type": "technology",
     "memory_count": 5}
  ],
  "edges": [
    {"source": "FastAPI", "target": "Python",
     "relation_type": "uses"}
  ]
}
```

## Authentication

When token auth is configured, the graph UI and API
endpoints require a valid bearer token:

- The `/graph` page itself is a **public path** (serves
  the HTML/JS)
- All `/v1/graph/*` API calls require the token via
  `Authorization: Bearer <token>` header
- The UI handles this through a login form — enter your
  token once and it's stored in memory for the session
- **Tokens are never passed in URLs** to prevent leakage
  via Referer headers or browser history

## Architecture

```
Browser
  ├── GET /graph (HTML + Cytoscape.js)
  ├── GET /v1/graph/stats (auth required)
  ├── GET /v1/graph/entities (auth required)
  └── GET /v1/graph/neighborhood/{name} (auth required)
        ↓
    GraphStore (SQLite)
        ↓
    entities, relationships, entity_memories tables
```

All database queries run in `asyncio.to_thread()` to avoid
blocking the event loop. Batch queries prevent N+1 problems
when loading memory counts for neighborhood nodes.
