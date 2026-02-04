# Changelog

All notable changes to TribalMemory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.3.0]: https://github.com/abbudjoe/TribalMemory/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/abbudjoe/TribalMemory/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/abbudjoe/TribalMemory/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/abbudjoe/TribalMemory/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/abbudjoe/TribalMemory/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/abbudjoe/TribalMemory/releases/tag/v0.1.0
