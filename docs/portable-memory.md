# Portable Memory Plugin Spec (Draft)

**Goals**
- Provide a model-agnostic memory API usable by any agent or model runtime.
- Let a user carry memory across models and devices with consistent behavior.
- Support corrections, provenance, and safe sharing with explicit scopes.
- Keep local-first operation with optional sync.

**Non-Goals**
- Real-time multi-user collaboration conflicts beyond simple last-write resolution.
- Full knowledge-graph reasoning or ontology management.
- Replacing model reasoning; this is retrieval and context management.

**Architecture Overview**
- Memory Server: an HTTP service that exposes a stable API and owns storage.
- Client SDK: lightweight library (TypeScript and Python) used by any model runtime.
- Storage: local vector store + metadata store, with export and import.
- Optional Sync: push and pull bundles or deltas to a remote endpoint.

**Core API (Versioned)**
All endpoints are under `v1` and return JSON. Requests should include `user_id` and `workspace_id`.

- `POST /v1/remember`
- `POST /v1/recall`
- `POST /v1/correct`
- `DELETE /v1/forget/{memory_id}`
- `GET /v1/memory/{memory_id}`
- `GET /v1/stats`
- `GET /v1/health`
- `POST /v1/export`
- `POST /v1/import`

**Data Model**
MemoryEntry fields:
- `id`: UUID
- `content`: string
- `embedding`: float[]
- `source_instance`: string
- `source_type`: enum
- `created_at`: RFC3339 timestamp
- `updated_at`: RFC3339 timestamp
- `tags`: string[]
- `context`: string | null
- `confidence`: float 0.0-1.0
- `supersedes`: string | null
- `related_to`: string[]
- `scope`: enum
- `user_id`: string
- `workspace_id`: string
- `model_id`: string | null

Scope values:
- `personal`: Only this user can access across all models
- `shared`: Shared among authorized users in a workspace
- `model_specific`: Only this user and this model instance

**Recall Semantics**
- Default retrieval is semantic by embedding similarity.
- Corrections should supersede originals during recall.
- Optional filters: `tags`, `scope`, `model_id`, `source_instance`.
- Optional ranking modifiers: recency decay and confidence weighting.

**Portability and Bundles**
- Export creates a portable bundle containing:
  - `manifest.json` with schema version and metadata.
  - `memory.db` for metadata and corrections.
  - `vectors.db` for vector index (or a standard format with rebuild script).
- Import validates schema version, merges by `id`, and resolves conflicts by `updated_at`.

**Sync (Optional)**
- Local-first by default; sync is opt-in.
- Sync model:
  - Periodic export of deltas by time window.
  - Server-side merge with conflict resolution by `updated_at` and `supersedes`.
- Sync security:
  - End-to-end encryption for content and embeddings when enabled.

**Client SDK Expectations**
- Consistent request format across models.
- Built-in retry and timeout defaults.
- Helper methods for correction chains and scoped queries.
- Optional adapters for different model runtimes.

**Privacy and Security**
- Encrypt memory at rest on disk.
- Explicit opt-in for `shared` scope.
- Ability to delete memories by `id` or by `scope`.
- Redaction support in `remember` for sensitive fields.

**Migration and Compatibility**
- Schema version is required in all responses.
- Deprecations follow a documented timeline.
- Import should accept earlier schemas with a conversion step.

**Open Questions**
- Should embeddings be stored per model or re-embedded on import?
- Do we allow multiple embedding models in a single store?
- How should conflicts be resolved when two corrections race?

