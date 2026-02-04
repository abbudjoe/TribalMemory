# Changelog

All notable changes to TribalMemory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/abbudjoe/TribalMemory/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/abbudjoe/TribalMemory/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/abbudjoe/TribalMemory/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/abbudjoe/TribalMemory/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/abbudjoe/TribalMemory/releases/tag/v0.1.0
