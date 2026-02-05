# TribalMemory Cloud — Architecture Spec

*Created: 2026-02-05*

---

## Vision

Any agent, anywhere, shares one brain. Local agents sync to the cloud. Cloud agents connect directly. Everything works offline and syncs when online.

```
     ┌─────────────┐         ┌──────────────────────┐
     │ Claude Code  │── MCP ──┤                      │
     │ (Joe's Mac)  │         │                      │
     └─────────────┘         │  TribalMemory Cloud   │
     ┌─────────────┐         │                      │
     │  Clawdio     │── API ──┤  api.tribalmemory.io  │
     │ (VPS)        │         │                      │
     └─────────────┘         │  • Auth (API keys)   │
     ┌─────────────┐         │  • Multi-tenant      │
     │  Codex       │── MCP ──┤  • Hosted embeddings │
     │ (Joe's Mac)  │         │  • Postgres+pgvector │
     └─────────────┘         │  • Sync protocol     │
     ┌─────────────┐         │                      │
     │  Future Agent │── API ──┤                      │
     └─────────────┘         └──────────────────────┘
```

---

## Architecture Layers

### 1. API Gateway

The cloud is just the same TribalMemory HTTP API with auth bolted on.

```
POST /v1/remember    →  store memory
POST /v1/recall      →  semantic search
GET  /v1/health      →  status
POST /v1/sessions/*  →  session indexing
GET  /v1/stats       →  usage metrics
POST /v1/export      →  bulk export
POST /v1/import      →  bulk import
```

**Auth**: Bearer token per API key. Each key scoped to a workspace.

```
Authorization: Bearer tm_live_abc123...
```

**New endpoints for cloud**:
```
POST /v1/auth/keys         →  create API key
GET  /v1/auth/keys         →  list keys
DELETE /v1/auth/keys/:id   →  revoke key
GET  /v1/usage             →  metering (memories, queries, storage)
```

### 2. Multi-Tenancy

Each user gets isolated workspaces. Agents connect to a workspace via API key.

```
┌─────────────────────────────────────────┐
│  User: joe@example.com                  │
│                                         │
│  Workspace: "default"                   │
│  ├── API Key: tm_live_abc... (Clawdio)  │
│  ├── API Key: tm_live_def... (Claude)   │
│  └── API Key: tm_live_ghi... (Codex)    │
│                                         │
│  Workspace: "work-project"              │
│  ├── API Key: tm_live_jkl... (Claude)   │
│  └── API Key: tm_live_mno... (Codex)    │
└─────────────────────────────────────────┘
```

**Data isolation**: Row-level security on `workspace_id`. Each memory belongs to exactly one workspace. Agents within the same workspace share all memories.

**Instance tracking**: Existing `source_instance` field tells you *which agent* stored each memory. Cloud just adds `workspace_id` on top.

### 3. Storage: Postgres + pgvector

Replace LanceDB with Postgres for cloud. Keep LanceDB for local/self-hosted.

```sql
-- Core memories table
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    content TEXT NOT NULL,
    embedding vector(384) NOT NULL,  -- pgvector
    source_instance TEXT NOT NULL,
    source_type TEXT NOT NULL,
    tags TEXT[] DEFAULT '{}',
    context JSONB,
    confidence FLOAT DEFAULT 1.0,
    supersedes UUID REFERENCES memories(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_memories_workspace ON memories(workspace_id);
CREATE INDEX idx_memories_embedding ON memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
CREATE INDEX idx_memories_created ON memories(workspace_id, created_at);
CREATE INDEX idx_memories_fts ON memories
    USING gin (to_tsvector('english', content));

-- Graph entities (existing GraphStore, adapted)
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    memory_id UUID REFERENCES memories(id)
);

-- API keys
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL,
    key_hash TEXT NOT NULL,  -- bcrypt hash, never store raw
    label TEXT,
    scopes TEXT[] DEFAULT '{read,write}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ
);
```

**Why Postgres over managed LanceDB**:
- Battle-tested at scale, everyone knows it
- pgvector handles vector search + metadata filtering in one query
- Full-text search via `tsvector` (replaces our SQLite FTS5)
- Row-level security for multi-tenancy
- Managed offerings everywhere (Supabase, Neon, RDS, etc.)

### 4. Hosted Embeddings

Cloud users shouldn't need to run their own embedding model. The cloud provides embeddings as part of the service.

**Options** (pick one or offer both):
- **FastEmbed on server**: Run ONNX model server-side. Zero external API cost. ~50ms/embedding.
- **OpenAI API**: Higher quality, but per-call cost we'd eat or pass through.

**Recommendation**: FastEmbed on server (bge-small-en-v1.5). Same model local users get, so embeddings are compatible. If a user migrates from local to cloud, vectors transfer 1:1 since dimensions match.

**Embedding as a service** (optional future):
```
POST /v1/embed  →  { "text": "...", "embedding": [0.1, 0.2, ...] }
```

### 5. Sync Protocol (Hybrid Mode)

The killer feature: local TribalMemory syncs with cloud. Work offline, sync when online.

```
Local TribalMemory ←──sync──→ Cloud TribalMemory
   (fast, offline)              (shared, backed up)
```

**Sync strategy**: Event-based, eventual consistency.

```
# Local → Cloud: push new memories
POST /v1/sync/push
{
    "since": "2026-02-05T00:00:00Z",
    "memories": [ ... ],
    "entities": [ ... ]
}

# Cloud → Local: pull new memories from other agents
POST /v1/sync/pull
{
    "since": "2026-02-05T00:00:00Z"
}
→ { "memories": [ ... ], "entities": [ ... ] }
```

**Conflict resolution**: Last-write-wins on content, merge on tags. `supersedes` field handles explicit overwrites. Same pattern as import/export (already built).

**Offline-first**: Local always works. Sync is best-effort. Agent never blocks on cloud availability.

---

## Implementation Plan

### Phase A: Auth + Multi-Tenant API (1-2 weeks)

Add auth to the existing FastAPI server. Minimal changes to current code.

1. **API key middleware** — validate Bearer token, resolve workspace
2. **Workspace model** — `workspace_id` added to all data models
3. **Key management endpoints** — create, list, revoke
4. **Postgres adapter** — new storage backend alongside LanceDB
   - `PostgresVectorStore` implementing `IVectorStore`
   - `PostgresGraphStore` implementing graph interface
   - `PostgresFTSStore` for full-text search
5. **Config**: `db.provider: postgres` with connection string

```yaml
# Cloud config
instance_id: cloud
db:
  provider: postgres
  uri: postgresql://user:pass@host:5432/tribal
embedding:
  provider: fastembed  # server-side
server:
  host: 0.0.0.0
  port: 18790
auth:
  enabled: true
  require_key: true
```

### Phase B: Deployment (1 week)

1. **Docker image** — already have Dockerfile, add Postgres deps
2. **Docker Compose** — app + Postgres + pgvector
3. **Fly.io / Railway** — one-click deploy
4. **Managed Postgres** — Supabase or Neon for zero-ops DB

### Phase C: Sync Protocol (2 weeks)

1. **Change tracking** — `sync_version` column, monotonic counter per workspace
2. **Push endpoint** — accept batch of memories, dedup against existing
3. **Pull endpoint** — return memories since version N
4. **Local sync client** — background thread in local server
5. **Config**: `sync.cloud_url` + `sync.api_key` + `sync.interval`

```yaml
# Local config with cloud sync
instance_id: clawdio
db:
  provider: lancedb
  path: ~/.tribal-memory/lancedb
embedding:
  provider: fastembed
sync:
  enabled: true
  cloud_url: https://api.tribalmemory.io
  api_key: tm_live_abc123
  interval: 60  # seconds
```

### Phase D: Dashboard + Billing (3-4 weeks)

1. **Web dashboard** — view memories, search, manage keys
2. **Usage metering** — track memories stored, queries, storage bytes
3. **Billing** — Stripe integration, per-workspace plans
4. **Team management** — invite members, shared workspaces

---

## Client Configuration

### For Joe's agents right now (Phase A)

Claude Code (`~/.claude.json`):
```json
{
    "mcpServers": {
        "tribal-memory": {
            "command": "tribalmemory-mcp",
            "env": {
                "TRIBAL_MEMORY_CLOUD_URL": "https://api.tribalmemory.io",
                "TRIBAL_MEMORY_API_KEY": "tm_live_abc123"
            }
        }
    }
}
```

Clawdio (OpenClaw plugin config):
```json
{
    "memory-tribal": {
        "config": {
            "serverUrl": "https://api.tribalmemory.io",
            "apiKey": "tm_live_abc123",
            "autoRecall": true,
            "autoCapture": true
        }
    }
}
```

Codex (`~/.codex/config.toml`):
```toml
[mcp_servers.tribal-memory]
command = "tribalmemory-mcp"

[mcp_servers.tribal-memory.env]
TRIBAL_MEMORY_CLOUD_URL = "https://api.tribalmemory.io"
TRIBAL_MEMORY_API_KEY = "tm_live_abc123"
```

### CLI init with cloud

```bash
tribalmemory init --cloud
# Prompts for API key, writes to ~/.tribal-memory/.env
# Configures MCP to point at cloud instead of localhost
```

---

## Pricing (Draft)

| Tier | Price | Memories | Queries | Agents |
|------|-------|----------|---------|--------|
| Free | $0/mo | 1,000 | 10,000/mo | 3 |
| Pro | $12/mo | 50,000 | unlimited | 10 |
| Team | $29/mo | 250,000 | unlimited | unlimited |
| Enterprise | Custom | unlimited | unlimited | unlimited |

Storage costs are low (text + 384-dim vectors). The expensive part is compute for embeddings — but FastEmbed on server is essentially free.

---

## Security

- **API keys**: Hashed with bcrypt, never stored in plaintext
- **Transport**: TLS only (HTTPS required)
- **Data isolation**: Row-level security on workspace_id
- **Encryption at rest**: Postgres-level (managed DB handles this)
- **Data residency**: Configurable region (US, EU)
- **Audit log**: Track who stored/recalled what, when
- **Data export**: Full export always available (already built)
- **Data deletion**: Hard delete on request (GDPR compliance)

---

## What We Already Have (Reusable)

| Component | Status | Cloud Adaptation |
|-----------|--------|-----------------|
| HTTP API (FastAPI) | ✅ Built | Add auth middleware |
| Remember/Recall | ✅ Built | Add workspace_id filter |
| Graph store | ✅ Built | Postgres adapter |
| FTS (BM25) | ✅ Built | Postgres tsvector |
| Session indexing | ✅ Built | Add workspace_id |
| Import/Export | ✅ Built | Per-workspace scoping |
| Dedup | ✅ Built | Works as-is |
| Temporal reasoning | ✅ Built | Works as-is |
| Reranking | ✅ Built | Works as-is |
| FastEmbed | ✅ Built | Run server-side |
| CLI init | ✅ Built | Add --cloud flag |
| MCP server | ✅ Built | Add cloud URL env |
| OpenClaw plugin | ✅ Built | Point at cloud URL |

**Estimated effort to MVP (auth + Postgres + deploy)**: 2-3 weeks.

---

## Immediate Next Steps

1. **File issues** for Phase A milestones
2. **Postgres adapter** — the biggest piece of new code
3. **Auth middleware** — API key validation + workspace resolution
4. **Docker Compose** with Postgres + pgvector
5. **Deploy to Fly.io** — get a live endpoint
6. **Connect Clawdio + Joe's agents** — dogfood the cloud
