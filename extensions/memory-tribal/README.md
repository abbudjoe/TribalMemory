# Memory-Tribal Plugin

Learned retrieval layer for OpenClaw's memory system with query caching, expansion, and feedback loops.

## Overview

Memory-tribal is a **slot plugin** that replaces OpenClaw's default `memory-core` with an enhanced retrieval system designed to improve memory recall accuracy over time.

### Features

- **Query Caching** — Remembers which queries successfully retrieved which facts
- **Query Expansion** — Converts natural questions to keyword searches
- **Feedback Tracking** — Learns from which retrievals were actually useful
- **Fact Anchoring** — Builds stable mappings between queries and memory locations
- **Graceful Fallback** — Falls back to built-in search if tribal server unavailable

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    memory_search()                       │
├─────────────────────────────────────────────────────────┤
│  1. Query Cache     → Check for known-good mapping      │
│  2. Query Expander  → Convert question → keywords       │
│  3. Tribal Client   → Call HTTP server (or fallback)    │
│  4. Feedback Track  → Record what was useful            │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   SQLite Persistence    │
              │  ~/.openclaw/memory-    │
              │     tribal.sqlite       │
              └─────────────────────────┘
```

## Tools Provided

| Tool | Description |
|------|-------------|
| `memory_search` | Semantically search memory with learned enhancements |
| `memory_get` | Read memory file content by path |
| `memory_feedback` | Record which memories were useful (optional) |
| `memory_stats` | View retrieval accuracy statistics (optional) |

## Installation

See [SETUP.md](./SETUP.md) for detailed installation instructions.

**Quick start:**

```bash
cd extensions/memory-tribal
npm install
openclaw plugins install -l .
openclaw config set plugins.slots.memory=memory-tribal
openclaw gateway restart
```

## Configuration

```json
{
  "plugins": {
    "slots": {
      "memory": "memory-tribal"
    },
    "entries": {
      "memory-tribal": {
        "enabled": true,
        "config": {
          "tribalServerUrl": "http://localhost:18790",
          "queryCacheEnabled": true,
          "queryExpansionEnabled": true,
          "feedbackEnabled": true,
          "minCacheSuccesses": 3
        }
      }
    }
  }
}
```

## How It Works

### Query Cache

When a query successfully retrieves useful information, the mapping is cached:

```
"What coffee does Joe like?" → memory/USER.md:15-20
```

Next time a similar query comes in, the cache is checked first.

### Query Expansion

Natural language questions are expanded to keywords:

| Question | Expanded |
|----------|----------|
| "What's Joe's favorite coffee?" | `joe favorite coffee preference` |
| "When did we deploy v2?" | `deploy v2 release date when` |
| "What allergies does he have?" | `allergies allergy food medical` |

### Feedback Loop

After retrieval, the agent can report which results were useful:

```typescript
memory_feedback({
  queryId: "abc123",
  useful: ["chunk_1", "chunk_3"],
  notUseful: ["chunk_2"]
})
```

This reinforces good retrievals and penalizes bad ones.

## Persistence

All learned state is stored in SQLite at `~/.openclaw/memory-tribal.sqlite`:

| Table | Purpose |
|-------|---------|
| `query_cache` | Cached query→fact mappings |
| `feedback_weights` | Reinforcement weights per chunk |
| `usage_history` | Query frequency and success rates |
| `learned_expansions` | Custom expansion rules |
| `fact_anchors` | Stable query→location mappings |

## Tribal Memory Server (Optional)

For cross-instance memory sharing, connect to a tribal-memory HTTP server:

```bash
# Start the server (separate process)
tribal-memory serve --port 18790

# Plugin connects automatically via tribalServerUrl config
```

Without the server, the plugin falls back to OpenClaw's built-in memory search.

## Development

```bash
# Run in development mode (linked install)
openclaw plugins install -l ./extensions/memory-tribal

# After changes, restart gateway
openclaw gateway restart

# Check for errors
openclaw gateway logs | grep memory-tribal
```

## Testing

Run the evaluation harness to measure retrieval accuracy:

```bash
cd eval/memory-test
python harness.py generate L3  # Memory search level
# ... run agent with queries ...
python harness.py score results/test-L3-*.json
```

Baseline: 61.5% accuracy with tuned config. Goal: >80% with learned retrieval.

## License

MIT

## Author

Clawdio 🦝
