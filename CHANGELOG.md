# Changelog

All notable changes to TribalMemory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] - 2026-02-07 ([PyPI](https://pypi.org/project/tribalmemory/0.6.2/))

### Added

#### Batch Ingestion Endpoint
- **`POST /v1/remember/batch`** — Store up to 1000 memories in a single request
- Concurrent processing with `asyncio.gather` (chunk size: 50)
- Returns list of memory IDs for all stored items
- Ideal for bulk ingestion, migrations, and benchmarking

#### Auto-Temporal Query Extraction
- **Date parsing from natural language queries** — "What did I say last Saturday?" automatically filters to that date
- Powered by `dateparser` with relative date support
- Extracts date ranges from query text and applies as temporal filters
- Works with all recall endpoints (HTTP, MCP, Python API)

---

## [0.6.1] - 2026-02-06 ([PyPI](https://pypi.org/project/tribalmemory/0.6.1/))

### Fixed

- **FastEmbed is now a core dependency** — No longer requires `pip install tribalmemory[fastembed]`
- Removed unused `openai` dependency from core requirements
- Simplified install: just `pip install tribalmemory`

---

## [0.6.0] - 2026-02-06 ([PyPI](https://pypi.org/project/tribalmemory/0.6.0/))

### ⚠️ Breaking Changes

**FastEmbed is now the ONLY embedding provider**

This release removes OpenAI and Ollama embedding support entirely. FastEmbed provides
local, zero-cloud embeddings with no API keys required.

**Migration for existing users:**

1. **New installs**: Automatic — FastEmbed (384 dims) is the default
2. **Existing installs with OpenAI/Ollama config**: Must migrate:
   ```bash
   # 1. Backup your data if needed
   # 2. Delete old embeddings (incompatible dimensions)
   rm -rf ~/.tribal-memory/lancedb/
   
   # 3. Reinstall with FastEmbed
   pip install --upgrade tribalmemory[fastembed]
   
   # 4. Reinitialize config
   tribalmemory init --force
   
   # 5. Re-ingest your memories
   ```

**Why this change?**
- Zero cloud dependencies (fully offline)
- No API keys required
- Simpler codebase (~2400 lines removed)
- Consistent behavior across all deployments

### Changed

#### Default Embeddings: FastEmbed (Zero Cloud)
- **FastEmbed is now the only embedding provider**
- Model: `BAAI/bge-small-en-v1.5` (384 dimensions)
- Zero cloud dependencies — fully local embeddings out of the box
- No API key required

#### OpenClaw Plugin Installation
- Improved installation docs with prominent `npm install` warning
- Added `install.sh` script for one-line installation
- Clearer error handling for missing dependencies

### Removed

- **OpenAI embedding support** — Use FastEmbed instead
- **Ollama embedding support** — FastEmbed handles local embeddings
- `--openai`, `--ollama`, `--fastembed`, and `--local` CLI flags removed
- `OpenAIEmbeddingService` class removed
- `a21/` experimental directory removed
- API key parameters removed from `create_memory_service()`

## [0.5.1] - 2026-02-06 ([PyPI](https://pypi.org/project/tribalmemory/0.5.1/))

### Added

#### Lazy spaCy — 70x Faster Ingest
- **`lazy_spacy` config option** (default: `true`) — dramatically improves ingest performance
- Fast regex entity extraction on ingest (~2-3 seconds per conversation)
- spaCy NER only runs on recall queries (once per search, not per message)
- Maintains full accuracy for personal conversation retrieval
- Config: `search.lazy_spacy: true` in config.yaml

#### spaCy Entity Extraction
- **spaCy NER integration** for personal conversations (PERSON, GPE, ORG, DATE entities)
- `HybridEntityExtractor` combines regex patterns with spaCy for best coverage
- Optional dependency: `pip install tribalmemory[spacy]` + `python -m spacy download en_core_web_sm`
- Falls back gracefully to regex-only when spaCy unavailable

### Performance

- **Ingest**: ~70x faster with lazy spaCy (50-200s → 2-3s per conversation)
- **Recall**: Unchanged — spaCy runs once on query text for accurate entity matching
- **LongMemEval**: Full 500-question benchmark in progress

---

## [0.5.0] - 2026-02-05 ([PyPI](https://pypi.org/project/tribalmemory/0.5.0/))

### Added

#### Service Management
- **`tribalmemory service` subcommand** — install, uninstall, status, logs
- systemd on Linux (user services, no root required)
- launchd on macOS (LaunchAgents)
- Auto-restart on failure, boot persistence
- Cross-platform service file generation

#### OpenClaw Plugin (extensions/memory-tribal)
- **`tribal_store` tool** — deliberate memory storage with tags and context
- **`tribal_recall` tool** — query with full control (tags, temporal, relevance)
- Fixed: `TribalClient.search()` now passes tags/after/before to recall (was silently dropping)
- 230 plugin tests passing

### Fixed

- LoCoMo benchmark verified: **100% accuracy on 1986 questions** across all categories (open-domain, adversarial, temporal, single-hop, multi-hop)

---

## [0.4.2] - 2026-02-05 ([PyPI](https://pypi.org/project/tribalmemory/0.4.2/))

### Fixed

- **uv tool environment support** — `tribalmemory init` now detects uv-managed environments (which lack pip) and uses `uv pip install --python <exe> fastembed` instead. Falls back gracefully with context-aware error messages. Cross-platform path detection for Windows compatibility.

## [0.4.1] - 2026-02-05 ([PyPI](https://pypi.org/project/tribalmemory/0.4.1/))

### Fixed

- **Auto-install FastEmbed during init** — `tribalmemory init` now detects missing FastEmbed and offers to install it automatically via `pip install fastembed`. Works in uv tool environments, regular venvs, and system Python. Non-interactive mode auto-installs without prompting.
- **Accurate percentile calculation** — Benchmark stats now use `statistics.quantiles` with linear interpolation instead of nearest-rank. More accurate p50/p95/p99 for small sample sizes (50–100 queries). Includes 7 unit tests for edge cases.

## [0.4.0] - 2026-02-05 ([PyPI](https://pypi.org/project/tribalmemory/0.4.0/))

### Migrating from 0.3.0

The default embedding provider changed from OpenAI to **FastEmbed**. Existing users with OpenAI configs are **not affected** — your `config.yaml` already specifies `provider: openai` and will continue to work. This only changes what `tribalmemory init` generates for new setups.

If you want to switch an existing install to FastEmbed:
```bash
pip install "tribalmemory[fastembed]"
tribalmemory init --force   # overwrites config with FastEmbed defaults
```

To keep OpenAI:
```bash
tribalmemory init --openai --force   # prompts for key, saves to .env
```

### Added

#### FastEmbed Local Embeddings
- **FastEmbed provider** — local ONNX embeddings via `BAAI/bge-small-en-v1.5` (384 dims)
- Zero cloud, zero API keys, ~130MB model auto-download on first use
- Optional dependency: `pip install "tribalmemory[fastembed]"`
- `FastEmbedService` with async `embed()` and `embed_batch()`
- Provider auto-detection via `provider_name` property on embedding services

#### Zero-Friction CLI Init
- **FastEmbed is now the default** — `tribalmemory init` generates FastEmbed config
- `--openai` flag: prompts for API key interactively, saves to `~/.tribal-memory/.env` (600 permissions)
- `--ollama` flag: generates Ollama config template
- `--local` kept as deprecated alias for `--ollama`
- FastEmbed import validation at init time (exits with helpful message if not installed)
- `load_env_file()` loads `.env` at server/MCP startup (won't overwrite explicit env vars)

#### Temporal Recall Filtering
- `after` and `before` parameters on `recall()` for date-range queries
- Temporal extraction via `dateparser` — resolves relative/absolute dates
- `TemporalExtractor` service with `TemporalEntity` dataclasses
- GraphStore: `temporal_facts` table, date range queries
- MCP tool and HTTP API support for temporal filtering

#### Persistent Session Storage
- **LanceDB-backed SessionStore** — session chunks survive server restarts
- Delta ingestion state persisted via chunk metadata
- Cosine metric for vector search, filter-based cleanup
- `InMemorySessionStore` preserved as fallback

#### Session Search Pagination
- Offset-based pagination on session search: `offset`, `limit`, `has_more`
- `_PAGINATION_POOL_CAP = 1000` to bound memory usage
- `total_count` in response for UI page controls

#### Connection Pooling
- Persistent SQLite connection in GraphStore (WAL mode, RLock)
- Eliminates ~6.5ms per-operation connection overhead
- Context manager support for clean shutdown

### Changed
- Default embedding provider changed from OpenAI to FastEmbed
- `tribalmemory init` no longer requires manual config.yaml editing
- OpenAI API keys stored in `.env` file instead of config.yaml (security)
- README rewritten: FastEmbed-first, Codex setup documented, integrations after Quick Start

### Fixed
- FTS5 BM25 search fails on punctuation (#56) — phrase quoting for special chars
- Unicode month names fallback in temporal extraction (#60)
- Temporal extraction batching for scale (#62)

### Testing
- 679 tests passing (up from ~550 in v0.3.0)
- Session edge case tests: pagination, unicode/CJK, large payloads, timestamps, load testing
- Session integration tests: HTTP + MCP + concurrent (21 tests)
- MemoryBench LoCoMo: 100% accuracy on 10-question sample (FastEmbed)

---

## [0.3.0] - 2026-02-04

### Added

#### Graph-Enriched Search
- **Entity extraction** at store time using pattern-based extraction
  - Recognizes service names (kebab-case: `auth-service`, `user-db`)
  - Recognizes 40+ technology names (PostgreSQL, Redis, Kafka, etc.)
  - Extracts relationships (`uses`, `connects_to`, `stores_in`, etc.)
- **GraphStore** with SQLite backend for entity/relationship storage
  - No external dependencies (Neo4j, etc.) — local-first
  - Multi-hop traversal via `find_connected(entity, hops)`
  - Memory-to-entity associations with provenance

#### Entity-Centric Queries
- `recall_entity(name, hops, limit)` for entity-focused recall
  - "Tell me everything about auth-service"
  - Traverses relationship graph to find connected memories
- `get_entity_graph(name, hops)` for visualization/debugging
- `tribal_recall_entity` MCP tool for Claude Code integration
- `tribal_entity_graph` MCP tool for graph exploration

#### Graph-Aware Hybrid Recall
- `recall()` now supports `graph_expansion` parameter (default: True)
- Extracts entities from query, expands candidates via graph
- Scoring: 1-hop = 0.85, 2-hop = 0.70 (configurable via class constants)
- Respects `min_relevance` for graph results
- Batch fetching with `asyncio.gather()` for performance

#### Retrieval Method Tracking
- `RecallResult.retrieval_method` field indicates result source:
  - `"vector"`: Pure vector similarity
  - `"hybrid"`: Vector + BM25 merge
  - `"graph"`: Entity graph traversal
  - `"entity"`: Direct entity match
- `RetrievalMethod` Literal type for type safety

### Changed
- `IMemoryService.recall()` interface updated with `graph_expansion` parameter
- Factory function `create_memory_service()` now initializes GraphStore
- Graph cleanup integrated into `forget()` method

### Performance
- Capped graph expansion to `limit * GRAPH_EXPANSION_BUFFER` to prevent memory issues
- Concurrent memory fetching for graph results

## [0.2.0] - 2026-02-04

### Added

#### Hybrid Search (BM25 + Vector)
- **SQLite FTS5 integration** for keyword-based BM25 search
- `FTSStore` class for full-text indexing alongside vector embeddings
- `hybrid_merge()` algorithm with min-max normalized scoring
- Configurable `vector_weight` and `text_weight` in `SearchConfig`
- `candidate_multiplier` for controlling retrieval pool size
- Native `get_stats()` for InMemoryVectorStore and LanceDB

#### Result Reranking
- `IReranker` protocol for pluggable reranking strategies
- `HeuristicReranker`: recency decay + tag boost + length normalization
- `CrossEncoderReranker`: optional sentence-transformers integration
- `NoopReranker`: passthrough for benchmarking
- Factory with automatic fallback (cross-encoder → heuristic)
- Configurable `rerank_pool_multiplier` in `SearchConfig`
- Batched BM25-only fetches via `asyncio.gather` for performance

#### Session Indexing
- `SessionStore` for ingesting conversation transcripts
- ~400 token chunking with overlap for context preservation
- Delta ingestion: only new turns get embedded
- Background cleanup task (configurable retention, 6-hour interval)
- `tribal_sessions_ingest` MCP tool for transcript ingestion
- `tribal_recall` now accepts `sources` param: `"memories"` | `"sessions"` | `"all"`
- HTTP endpoints: `POST /sessions/ingest`, `GET /sessions/search`

#### OpenClaw Plugin SDK
- Full rewrite to `OpenClawPluginApi` interface
- `before_agent_start` lifecycle hook for auto-recall
- `agent_end` lifecycle hook for auto-capture
- `api.registerService()` and `api.registerCli()` integration
- `kind: "memory"` manifest type

### Changed
- `SearchConfig` now includes `hybrid_enabled`, `reranking`, `recency_decay_days`, `tag_boost_weight`
- Recall pipeline: vector search → optional BM25 merge → rerank → return
- Session search results include `chunk_index` and `session_id` metadata

### Fixed
- `datetime.now(timezone.utc)` replaces deprecated `datetime.utcnow()` (Python 3.12+)
- Consistent `limit` naming across all request models

## [0.1.3] - 2026-02-04

### Added
- `--auto-capture` flag for `tribalmemory init` (generates CLAUDE.md/AGENTS.md instructions)
- Extended auto-capture support for Codex CLI (AGENTS.md)

## [0.1.2] - 2026-02-04

### Fixed
- Full path resolution for `tribalmemory-mcp` binary in generated configs

## [0.1.1] - 2026-02-04

### Fixed
- Dual config path support for Claude Code CLI

## [0.1.0] - 2026-02-04

### Added
- Initial public release
- Core memory service with vector search (LanceDB)
- Local embedding support via Ollama (`nomic-embed-text`)
- OpenAI embeddings support
- MCP server with `tribal_store`, `tribal_recall`, `tribal_stats` tools
- HTTP REST API
- CLI: `tribalmemory init`, `tribalmemory serve`, `tribalmemory mcp`
- Docker support
- Import/export functionality
- Token budget management with circuit breaker
- Session deduplication
- Embedding portability metadata

[0.4.2]: https://github.com/abbudjoe/TribalMemory/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/abbudjoe/TribalMemory/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/abbudjoe/TribalMemory/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/abbudjoe/TribalMemory/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/abbudjoe/TribalMemory/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/abbudjoe/TribalMemory/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/abbudjoe/TribalMemory/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/abbudjoe/TribalMemory/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/abbudjoe/TribalMemory/releases/tag/v0.1.0
