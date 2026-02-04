# memory-tribal — OpenClaw Plugin for Tribal Memory

Cross-agent long-term memory plugin that connects OpenClaw to a
[Tribal Memory](https://github.com/abbudjoe/TribalMemory) server.

## Features

- **Auto-recall**: Injects relevant memories before agent responds
  (`before_agent_start` lifecycle hook)
- **Auto-capture**: Stores learnings after agent turns
  (`agent_end` lifecycle hook)
- **Hybrid search**: Vector similarity + BM25 keyword search
- **Safeguards**: Token budgets, circuit breaker, smart triggers,
  session deduplication
- **Query learning**: Cache, expansion, and feedback for better recall
- **Cross-agent**: Memories shared across all connected agents

## Prerequisites

1. A running Tribal Memory server:
   ```bash
   pip install tribalmemory
   tribalmemory init --local  # or with OpenAI key
   tribalmemory serve
   ```

2. OpenClaw v2026.1.0 or later

## Installation

Copy the `extensions/memory-tribal/` directory into your OpenClaw
extensions folder:

```bash
cp -r extensions/memory-tribal/ ~/.openclaw/extensions/memory-tribal/
cd ~/.openclaw/extensions/memory-tribal/
npm install
```

## Configuration

Add to your `openclaw.json`:

```json5
{
  plugins: {
    slots: {
      memory: "memory-tribal"   // Replace default memory plugin
    },
    entries: {
      "memory-tribal": {
        enabled: true,
        config: {
          serverUrl: "http://localhost:18790",
          autoRecall: true,       // Inject memories before agent
          autoCapture: true       // Store learnings after agent
        }
      }
    }
  }
}
```

### Full Config Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `serverUrl` | string | `http://localhost:18790` | Tribal Memory server URL |
| `autoRecall` | bool | `true` | Inject relevant memories before agent |
| `autoCapture` | bool | `true` | Store learnings after agent turns |
| `queryCacheEnabled` | bool | `true` | Cache successful query→result mappings |
| `queryCacheMinSuccesses` | number | `3` | Min hits before caching a query |
| `queryExpansionEnabled` | bool | `true` | Expand queries for better matching |
| `feedbackEnabled` | bool | `true` | Learn from which memories get used |
| `maxTokensPerRecall` | number | `500` | Token cap per recall |
| `maxTokensPerTurn` | number | `750` | Token cap per agent turn |
| `maxTokensPerSession` | number | `5000` | Token cap per session |
| `maxTokensPerSnippet` | number | `100` | Token cap per snippet |
| `turnMaxAgeMs` | number | `1800000` | Stale turn cleanup threshold |
| `circuitBreakerMaxEmpty` | number | `5` | Empty recalls before tripping |
| `circuitBreakerCooldownMs` | number | `300000` | Cooldown after trip (5 min) |
| `smartTriggerEnabled` | bool | `true` | Skip low-value queries |
| `smartTriggerMinQueryLength` | number | `2` | Min query length |
| `smartTriggerSkipEmojiOnly` | bool | `true` | Skip emoji-only queries |
| `sessionDedupEnabled` | bool | `true` | Deduplicate within session |
| `sessionDedupCooldownMs` | number | `300000` | Dedup cooldown (5 min) |

### Migrating from v0.1

Config names were standardized. Old names still work but emit warnings:

| Old Name | New Name |
|----------|----------|
| `tribalServerUrl` | `serverUrl` |
| `minCacheSuccesses` | `queryCacheMinSuccesses` |
| `maxConsecutiveEmpty` | `circuitBreakerMaxEmpty` |
| `smartTriggersEnabled` | `smartTriggerEnabled` |
| `minQueryLength` | `smartTriggerMinQueryLength` |
| `skipEmojiOnly` | `smartTriggerSkipEmojiOnly` |
| `dedupCooldownMs` | `sessionDedupCooldownMs` |

## CLI Commands

```bash
openclaw tribal-memory status    # Check server connection
openclaw tribal-memory stats     # Show memory statistics
openclaw tribal-memory search <query>  # Search memories
```

## Tools

| Tool | Description |
|------|-------------|
| `memory_search` | Search memories with learned retrieval |
| `memory_get` | Read memory file content by path |
| `memory_feedback` | Record which memories were useful |
| `memory_metrics` | View safeguard metrics snapshot |

## Architecture

```
OpenClaw Agent
  ├── before_agent_start → auto-recall (search memories)
  ├── Agent processes with injected memory context
  └── agent_end → auto-capture (store learnings)
                        │
                        ▼
              Tribal Memory Server (HTTP)
              ├── Vector search (LanceDB)
              ├── BM25 search (SQLite FTS5)
              └── Deduplication + provenance
```
