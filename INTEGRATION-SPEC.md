# Tribal Memory ↔ OpenClaw Integration Spec

**Version:** 1.0  
**Created:** 2026-01-31  
**Status:** Draft

---

## Overview

Integrate tribal-memory as an OpenClaw extension (`memory-tribal`) that replaces `memory-lancedb` while preserving `memory-core` (file-based) for workspace context.

```
┌─────────────────────────────────────────────────────────────────┐
│                         OpenClaw Gateway                        │
├─────────────────────────────────────────────────────────────────┤
│  memory-core          │  memory-tribal (NEW)                    │
│  ├─ memory_search     │  ├─ memory_recall                       │
│  └─ memory_get        │  ├─ memory_store                        │
│  (file-based)         │  ├─ memory_correct                      │
│                       │  └─ memory_forget                       │
│                       │                                         │
│                       │  Lifecycle hooks:                       │
│                       │  ├─ before_agent_start (auto-recall)    │
│                       │  └─ agent_end (auto-capture)            │
└───────────────────────┴─────────────────────────────────────────┘
                              │
                              │ HTTP (localhost:18790)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    tribal-memory service                        │
│                    (Python / FastAPI)                           │
├─────────────────────────────────────────────────────────────────┤
│  MemorySystem                                                   │
│  ├─ remember()     → StoreResult                                │
│  ├─ recall()       → list[RecallResult]                         │
│  ├─ correct()      → StoreResult                                │
│  ├─ forget()       → bool                                       │
│  └─ get()          → MemoryEntry                                │
├─────────────────────────────────────────────────────────────────┤
│  Providers:                                                     │
│  ├─ OpenAI Embeddings                                           │
│  ├─ LanceDB (local or Cloud)                                    │
│  ├─ Deduplication                                               │
│  └─ Timestamp (RFC 3161) [future]                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ (future: multi-instance sync)
                              ▼
                    ┌─────────────────┐
                    │  LanceDB Cloud  │
                    │  (shared store) │
                    └─────────────────┘
```

---

## Sync Architecture (Option C: Local + Async Write-Through)

Tribal-memory uses a hybrid sync model for multi-instance support:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Clawdio-0                                │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ tribal-memory   │───▶│  Local LanceDB  │                     │
│  │    service      │    │  (primary)      │                     │
│  └─────────────────┘    └────────┬────────┘                     │
│                                  │ async push                   │
└──────────────────────────────────┼──────────────────────────────┘
                                   ▼
                    ┌─────────────────────────┐
                    │     LanceDB Cloud       │
                    │   (shared sync target)  │
                    └─────────────────────────┘
                                   ▲
                                   │ pull on startup + periodic
┌──────────────────────────────────┼──────────────────────────────┐
│                        Clawdio-1                                │
│  ┌─────────────────┐    ┌────────┴────────┐                     │
│  │ tribal-memory   │───▶│  Local LanceDB  │                     │
│  │    service      │    │  (primary)      │                     │
│  └─────────────────┘    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### Sync Behavior

| Operation | Behavior |
|-----------|----------|
| **Write (remember/correct)** | Local first → async push to cloud |
| **Read (recall)** | Local only (fast) |
| **Startup** | Pull from cloud to catch up |
| **Periodic** | Full bidirectional sync every N minutes |
| **Offline** | Queues writes, syncs when reconnected |

### Sync Configuration

```yaml
# ~/.tribal-memory/config.yaml
sync:
  enabled: true
  cloud_uri: "db://..."              # LanceDB Cloud URI
  interval_minutes: 5                # Periodic sync interval
  push_on_write: true                # Async push after each write
  pull_on_startup: true              # Sync on service start
  conflict_resolution: "last_write_wins"  # By updated_at timestamp
  offline_queue_max: 1000            # Max queued writes when offline
```

### Conflict Resolution

Memories are append-mostly, so conflicts are rare. When they occur:

1. **Last-write-wins**: Compare `updated_at` timestamps
2. **Corrections preserve chains**: If both sides have corrections, keep both in chain
3. **Dedup on sync**: If identical content synced from both sides, merge to single entry

### Sync Events

```python
class SyncEvent(Enum):
    PUSH_SUCCESS = "push_success"      # Memory pushed to cloud
    PUSH_FAILED = "push_failed"        # Push failed, queued for retry
    PULL_NEW = "pull_new"              # New memory from cloud
    PULL_CONFLICT = "pull_conflict"    # Conflict resolved
    SYNC_COMPLETE = "sync_complete"    # Periodic sync finished
```

---

## Component 1: tribal-memory HTTP Service

A lightweight FastAPI server exposing tribal-memory functionality.

### Location
```
/home/clawdio/clawd/projects/tribal-memory/src/tribal_memory/server/
├── __init__.py
├── app.py          # FastAPI app
├── routes.py       # API routes
└── models.py       # Pydantic request/response models
```

### API Endpoints

```
POST /v1/remember
  Request:  { content, source_type?, context?, tags?, skip_dedup? }
  Response: { success, memory_id?, duplicate_of?, error? }

POST /v1/recall
  Request:  { query, limit?, min_relevance?, tags? }
  Response: { results: [{ memory, similarity_score, retrieval_time_ms }] }

POST /v1/correct
  Request:  { original_id, corrected_content, context? }
  Response: { success, memory_id?, error? }

DELETE /v1/forget/{memory_id}
  Response: { success }

GET /v1/memory/{memory_id}
  Response: { memory } | 404

GET /v1/health
  Response: { status: "ok", instance_id, memory_count }

GET /v1/stats
  Response: { total_memories, by_source_type, by_tag, ... }
```

### Configuration

```yaml
# ~/.tribal-memory/config.yaml
instance_id: "clawdio-0"
db:
  provider: "lancedb"
  path: "~/.tribal-memory/lancedb"        # local
  # uri: "db://..."                        # cloud (future)
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  api_key: "${OPENAI_API_KEY}"
server:
  host: "127.0.0.1"
  port: 18790
```

### Startup

```bash
# As systemd user service (recommended)
tribal-memory serve --config ~/.tribal-memory/config.yaml

# Or via OpenClaw managed service
openclaw service start tribal-memory
```

---

## Component 2: OpenClaw Extension (`memory-tribal`)

TypeScript plugin following OpenClaw extension conventions.

### Location
```
/home/clawdio/clawd/projects/tribal-memory/extensions/memory-tribal/
├── index.ts              # Plugin entry
├── client.ts             # HTTP client for tribal-memory service
├── config.ts             # Config schema
├── openclaw.plugin.json  # Plugin manifest
└── package.json
```

### Plugin Manifest

```json
{
  "id": "memory-tribal",
  "name": "Memory (Tribal)",
  "description": "Tribal Memory - provenance-aware long-term memory with multi-instance sync",
  "kind": "memory",
  "version": "0.1.0",
  "requires": {
    "openclaw": ">=2025.1.0"
  }
}
```

### Config Schema

```typescript
export const tribalMemoryConfigSchema = {
  parse(value: unknown): TribalMemoryConfig {
    // ...
    return {
      serviceUrl: cfg.serviceUrl ?? "http://127.0.0.1:18790",
      instanceId: cfg.instanceId ?? "default",
      autoCapture: cfg.autoCapture !== false,
      autoRecall: cfg.autoRecall !== false,
      captureThreshold: cfg.captureThreshold ?? 0.7,  // importance threshold
      recallLimit: cfg.recallLimit ?? 5,
      recallMinScore: cfg.recallMinScore ?? 0.3,
    };
  },
};
```

### Tools

```typescript
// memory_recall - Search long-term memory
api.registerTool({
  name: "memory_recall",
  description: "Search through long-term memories with provenance tracking.",
  parameters: Type.Object({
    query: Type.String({ description: "Search query" }),
    limit: Type.Optional(Type.Number({ default: 5 })),
    tags: Type.Optional(Type.Array(Type.String())),
  }),
  async execute(_id, params) {
    const results = await client.recall(params.query, params.limit, params.tags);
    // Format with provenance info
    const text = results.map((r, i) => 
      `${i+1}. ${r.memory.content}\n   [${r.memory.source_type}] ${r.similarity_score.toFixed(0)}%`
    ).join("\n\n");
    return { content: [{ type: "text", text }] };
  },
});

// memory_store - Explicitly store a memory
api.registerTool({
  name: "memory_store",
  description: "Save important information with provenance tracking.",
  parameters: Type.Object({
    content: Type.String({ description: "Information to remember" }),
    tags: Type.Optional(Type.Array(Type.String())),
    context: Type.Optional(Type.String()),
  }),
  async execute(_id, params) {
    const result = await client.remember(params.content, {
      source_type: "user_explicit",
      tags: params.tags,
      context: params.context,
    });
    if (result.duplicate_of) {
      return { content: [{ type: "text", text: `Already remembered (${result.duplicate_of.slice(0,8)}...)` }] };
    }
    return { content: [{ type: "text", text: `Remembered: "${params.content.slice(0, 80)}..."` }] };
  },
});

// memory_correct - Correct existing memory
api.registerTool({
  name: "memory_correct",
  description: "Correct or update an existing memory. Creates a correction chain.",
  parameters: Type.Object({
    original_id: Type.String({ description: "ID of memory to correct" }),
    corrected_content: Type.String({ description: "Corrected information" }),
    context: Type.Optional(Type.String()),
  }),
  async execute(_id, params) {
    const result = await client.correct(params.original_id, params.corrected_content, params.context);
    return { content: [{ type: "text", text: `Corrected → ${result.memory_id?.slice(0,8)}...` }] };
  },
});

// memory_forget - GDPR-compliant deletion
api.registerTool({
  name: "memory_forget",
  description: "Forget a specific memory (GDPR-compliant deletion).",
  parameters: Type.Object({
    query: Type.Optional(Type.String({ description: "Search to find memory" })),
    memory_id: Type.Optional(Type.String({ description: "Specific memory ID" })),
  }),
  async execute(_id, params) {
    // Similar logic to memory-lancedb: search or direct delete
  },
});
```

### Lifecycle Hooks

```typescript
// Auto-recall: inject relevant memories before agent starts
if (cfg.autoRecall) {
  api.on("before_agent_start", async (event) => {
    if (!event.prompt || event.prompt.length < 10) return;
    
    const results = await client.recall(event.prompt, cfg.recallLimit);
    if (results.length === 0) return;
    
    const memoryContext = results.map(r => 
      `- [${r.memory.source_type}] ${r.memory.content}`
    ).join("\n");
    
    return {
      prependContext: `<tribal-memory>\n${memoryContext}\n</tribal-memory>`,
    };
  });
}

// Auto-capture: analyze conversation for important info
if (cfg.autoCapture) {
  api.on("agent_end", async (event) => {
    if (!event.success || !event.messages) return;
    
    // Extract capturable content (similar to memory-lancedb triggers)
    const toCapture = extractCapturableContent(event.messages);
    
    for (const item of toCapture.slice(0, 3)) {
      await client.remember(item.content, {
        source_type: "auto_capture",
        context: item.context,
        tags: item.tags,
      });
    }
  });
}
```

---

## Component 3: Service Management

### Option A: Systemd User Service (Recommended)

```ini
# ~/.config/systemd/user/tribal-memory.service
[Unit]
Description=Tribal Memory Service
After=network.target

[Service]
Type=simple
ExecStart=/home/clawdio/clawd/projects/tribal-memory/.venv/bin/python -m tribal_memory.server
Restart=on-failure
Environment=OPENAI_API_KEY=${OPENAI_API_KEY}

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable tribal-memory
systemctl --user start tribal-memory
```

### Option B: OpenClaw Managed Service

OpenClaw can spawn and manage the service:

```typescript
api.registerService({
  id: "tribal-memory-server",
  start: async () => {
    // Spawn Python process
    const proc = spawn("python", ["-m", "tribal_memory.server"], {
      cwd: TRIBAL_MEMORY_PATH,
      env: { ...process.env },
    });
    // Health check loop
    await waitForHealthy("http://127.0.0.1:18790/v1/health");
  },
  stop: async () => {
    // Graceful shutdown via /v1/shutdown or SIGTERM
  },
});
```

---

## Migration Path

### Phase 1: Parallel Running (Week 1)
1. Build tribal-memory HTTP server
2. Build memory-tribal extension
3. Run alongside memory-lancedb
4. Compare results, tune parameters

### Phase 2: Switchover (Week 2)
1. Disable memory-lancedb in config
2. Enable memory-tribal
3. Optional: migrate existing memories from lancedb → tribal

### Phase 3: Multi-Instance (Week 4+)
1. Deploy Clawdio-1
2. Point both instances at LanceDB Cloud
3. Enable cross-instance propagation
4. Test sync behavior

---

## Configuration Example

```yaml
# openclaw.yaml
extensions:
  memory-core:
    enabled: true  # Keep file-based memory_search/memory_get
  
  memory-lancedb:
    enabled: false  # Disable old vector memory
  
  memory-tribal:
    enabled: true
    serviceUrl: "http://127.0.0.1:18790"
    instanceId: "clawdio-0"
    autoCapture: true
    autoRecall: true
    recallLimit: 5
    recallMinScore: 0.3
```

---

## Success Criteria

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| Service startup | Time to healthy | < 5s |
| Recall latency | p95 response time | < 500ms |
| Auto-recall injection | Memories injected | > 0 when relevant |
| Deduplication | False positive rate | < 5% |
| Provenance accuracy | Source tracking | 100% |

---

## Open Questions

1. **Memory migration**: Import existing memory-lancedb data into tribal-memory?
2. **Fallback behavior**: What if tribal-memory service is down?
3. **Rate limiting**: Cap auto-capture to prevent runaway storage?
4. **Context budget**: How much memory context to inject (tokens)?

---

## Next Steps

1. [ ] Build `tribal_memory.server` (FastAPI)
2. [ ] Write OpenClaw extension `memory-tribal`
3. [ ] Systemd service file
4. [ ] Integration tests
5. [ ] Documentation

---

*Spec by Clawdio · 2026-01-31*
