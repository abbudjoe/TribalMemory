# Episode Memories

**Episodes** group related memories spanning multiple sessions into cohesive narratives with auto-generated summaries.

## Why Episodes?

Typical memory systems treat each memory as isolated. But real work unfolds across multiple sessions:

- **Project work** spans days or weeks — a bug fix might involve investigation, prototyping, testing, and deployment
- **Learning** builds over time — concepts connect across multiple conversations
- **Decisions evolve** — initial ideas are refined through iterative discussion

Episodes capture the **full narrative arc**, not just isolated snapshots.

### Key Benefits

- **Narrative continuity**: Memories are grouped into coherent stories
- **Automatic detection**: No manual tagging — episodes are detected at write time
- **Progressive summarization**: Summaries update incrementally as you work
- **Surfaces in recall**: Episode summaries are indexed as memories, so standard `recall()` queries automatically include them

---

## Quick Start

### 1. Enable Episodes

Add to `~/.tribal-memory/config.yaml`:

```yaml
episodes:
  enabled: true
  summarizer_model: gpt-4o-mini
  summarizer_provider: openai
```

### 2. Set API Key (Required)

**⚠️ Episodes require an LLM for summarization.** Without an API key, episode detection and summarization will fail silently. Set your API key in `~/.tribal-memory/.env`:

```bash
# For OpenAI
OPENAI_API_KEY=sk-...

# For Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# For Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Restart the Server

```bash
tribalmemory serve
```

That's it! Episodes will now be automatically detected and summarized as you store memories.

---

## How It Works

### Write-Time Detection

When you store a memory with `tribal_store()` or `/v1/remember`, the episode detector runs:

1. **Fast path (embedding similarity)**:
   - Compares the new memory to active episode summaries
   - If similarity > 0.75 (configurable), auto-joins the episode
   - **Cost**: ~0ms (local embedding computation)

2. **Slow path (LLM classification)**:
   - For borderline cases, an LLM classifies whether the memory belongs
   - Only runs when embedding similarity is ambiguous
   - **Cost**: ~1-2 cents per classification (gpt-4o-mini)

3. **Episode creation**:
   - If no match, a new episode is created with a generated title

### Progressive Summarization

As memories are added to an episode, the summary updates **incrementally**:

- **Cheap updates**: "Here's the new memory, update the summary" (~1-2 cents)
- **Full regeneration**: Every 10 memories (configurable), rebuild from scratch (~5-10 cents)
- **Final summary**: When an episode closes, a comprehensive summary is generated

This balances accuracy with cost — most updates are cheap, with periodic full regenerations for quality.

### Surfaces in Recall

Episode summaries are stored as regular memories with:
- `source_type = EPISODE_SUMMARY`
- Tagged with all constituent memory tags
- Full vector embedding for semantic search

This means **you don't need special queries** — standard `recall()` automatically includes episode summaries in results.

---

## Configuration

All configuration fields with descriptions and defaults:

```yaml
episodes:
  # ===== Core Settings =====
  enabled: true                          # Enable episode detection
  
  # ===== Detection Strategy =====
  detector_strategy: hybrid              # "embedding", "llm", or "hybrid"
                                         # - embedding: Fast, similarity-based only
                                         # - llm: Slow, LLM classification only
                                         # - hybrid: Fast path first, LLM fallback
  
  embedding_similarity_threshold: 0.75   # Auto-join threshold (0.0-1.0)
                                         # Higher = stricter matching
  
  # ===== Episode Management =====
  active_window_days: 14                 # Consider episodes active for N days
                                         # Older episodes auto-close
  
  max_active_episodes: 20                # Max concurrent active episodes
                                         # Oldest are closed when limit reached
  
  # ===== Summarization =====
  summarizer_model: gpt-4o-mini          # Model for summary generation
                                         # Supports: gpt-4o-mini, gpt-4o,
                                         # claude-3-5-sonnet-latest, etc.
  
  summarizer_provider: openai            # "openai", "anthropic", or "ollama"
  
  summarizer_temperature: 0.3            # LLM temperature (0.0-2.0)
                                         # Lower = more focused, higher = creative
  
  # ===== Cost Management =====
  full_regen_interval: 10                # Full summary regeneration every N memories
                                         # Lower = more accurate, higher = cheaper
  
  max_llm_calls_per_memory: 2            # Max LLM calls when storing one memory
                                         # Prevents runaway costs
  
  monthly_cost_ceiling: 5.0              # Max monthly cost in USD
                                         # Disables summaries when exceeded
```

---

## MCP Tools

When connected via MCP (Claude Code, Codex), these tools are available:

### `tribal_episodes_list`

List episodes with optional filtering.

**Parameters:**
- `status` (optional): Filter by status — `"active"`, `"closed"`, or `"archived"`
- `limit` (optional): Max results (1-100, default 50)

**Returns:**
```json
{
  "episodes": [
    {
      "id": "ep-abc123...",
      "title": "Bug Fix: Auth Race Condition",
      "summary": "Initial investigation revealed...",
      "status": "active",
      "memory_count": 7,
      "created_at": "2026-02-08T14:23:00Z",
      "updated_at": "2026-02-09T16:45:00Z",
      "closed_at": null
    }
  ],
  "count": 1,
  "status": "active"
}
```

**Example:**
```python
# List all active episodes
tribal_episodes_list(status="active")

# Get recent 10 episodes
tribal_episodes_list(limit=10)
```

---

### `tribal_episode_get`

Get full episode details with constituent memory IDs.

**Parameters:**
- `episode_id` (required): Episode UUID

**Returns:**
```json
{
  "episode": {
    "id": "ep-abc123...",
    "title": "Bug Fix: Auth Race Condition",
    "summary": "Detailed narrative spanning investigation, root cause analysis, fix implementation, testing, and deployment...",
    "summary_memory_id": "mem-summary-xyz...",
    "status": "closed",
    "memory_count": 12,
    "created_at": "2026-02-08T14:23:00Z",
    "updated_at": "2026-02-10T09:15:00Z",
    "closed_at": "2026-02-10T09:15:00Z",
    "metadata": {}
  },
  "memory_ids": [
    "mem-1...",
    "mem-2...",
    "..."
  ]
}
```

**Example:**
```python
tribal_episode_get(episode_id="ep-abc123...")
```

---

### `tribal_episode_create`

Manually create an episode from existing memories.

**Parameters:**
- `title` (required): Episode title (non-empty string)
- `memory_ids` (required): List of memory UUIDs to include

**Returns:**
```json
{
  "success": true,
  "episode_id": "ep-abc123...",
  "memory_count": 5
}
```

**Example:**
```python
# Group related memories into an episode
tribal_episode_create(
  title="Migration to PostgreSQL",
  memory_ids=["mem-1...", "mem-2...", "mem-3..."]
)
```

---

### `tribal_episode_add`

Add a memory to an existing episode.

**Parameters:**
- `episode_id` (required): Episode UUID
- `memory_id` (required): Memory UUID to add

**Returns:**
```json
{
  "success": true
}
```

**Behavior:**
- Adds the memory to the episode
- Triggers progressive summary update
- Idempotent — adding twice is safe

**Example:**
```python
tribal_episode_add(
  episode_id="ep-abc123...",
  memory_id="mem-new..."
)
```

---

### `tribal_episode_remove`

Remove a memory from an episode.

**Parameters:**
- `episode_id` (required): Episode UUID
- `memory_id` (required): Memory UUID to remove

**Returns:**
```json
{
  "success": true
}
```

**Example:**
```python
tribal_episode_remove(
  episode_id="ep-abc123...",
  memory_id="mem-wrong..."
)
```

---

### `tribal_episode_close`

Close an episode (marks it complete).

**Parameters:**
- `episode_id` (required): Episode UUID

**Returns:**
```json
{
  "success": true
}
```

**Behavior:**
- Sets status to `"closed"`
- Records `closed_at` timestamp
- Triggers **full summary regeneration** (expensive but comprehensive)

**Example:**
```python
# Mark project complete
tribal_episode_close(episode_id="ep-abc123...")
```

---

### `tribal_episode_regenerate`

Force full summary regeneration (expensive).

**Parameters:**
- `episode_id` (required): Episode UUID

**Returns:**
```json
{
  "success": true
}
```

**When to use:**
- After bulk memory changes
- To improve summary quality
- When progressive updates drift

**Cost:** ~5-10 cents for a 20-memory episode with gpt-4o-mini.

**Example:**
```python
tribal_episode_regenerate(episode_id="ep-abc123...")
```

---

## HTTP API

All routes are under `/v1/episodes` and require the same authentication as other routes (see [authentication.md](authentication.md)).

### `GET /v1/episodes`

List episodes with optional filtering.

**Query Parameters:**
- `status` (optional): `active`, `closed`, or `archived`
- `limit` (optional): Max results (1-100, default 50)

**Example:**
```bash
curl -H "Authorization: Bearer tm_abc123..." \
  "http://localhost:18790/v1/episodes?status=active&limit=10"
```

**Response:**
```json
{
  "episodes": [...],
  "count": 10,
  "status": "active"
}
```

---

### `GET /v1/episodes/{id}`

Get full episode details.

**Example:**
```bash
curl -H "Authorization: Bearer tm_abc123..." \
  http://localhost:18790/v1/episodes/ep-abc123...
```

**Response:**
```json
{
  "episode": {
    "id": "ep-abc123...",
    "title": "Bug Fix: Auth Race Condition",
    "summary": "...",
    "status": "closed",
    "memory_count": 12,
    ...
  },
  "memory_ids": ["mem-1...", "mem-2...", ...]
}
```

---

### `POST /v1/episodes`

Create an episode manually.

**Request Body:**
```json
{
  "title": "Migration to PostgreSQL",
  "memory_ids": ["mem-1...", "mem-2...", "mem-3..."]
}
```

**Example:**
```bash
curl -X POST \
  -H "Authorization: Bearer tm_abc123..." \
  -H "Content-Type: application/json" \
  -d '{"title": "New Feature", "memory_ids": ["mem-1", "mem-2"]}' \
  http://localhost:18790/v1/episodes
```

**Response:**
```json
{
  "success": true,
  "episode_id": "ep-abc123...",
  "memory_count": 2
}
```

---

### `POST /v1/episodes/{id}/memories`

Add a memory to an episode.

**Request Body:**
```json
{
  "memory_id": "mem-new..."
}
```

**Example:**
```bash
curl -X POST \
  -H "Authorization: Bearer tm_abc123..." \
  -H "Content-Type: application/json" \
  -d '{"memory_id": "mem-xyz..."}' \
  http://localhost:18790/v1/episodes/ep-abc123.../memories
```

**Response:**
```json
{
  "success": true
}
```

---

### `DELETE /v1/episodes/{id}/memories/{memory_id}`

Remove a memory from an episode.

**Example:**
```bash
curl -X DELETE \
  -H "Authorization: Bearer tm_abc123..." \
  http://localhost:18790/v1/episodes/ep-abc123.../memories/mem-xyz...
```

**Response:**
```json
{
  "success": true
}
```

---

### `POST /v1/episodes/{id}/close`

Close an episode (triggers final summary).

**Example:**
```bash
curl -X POST \
  -H "Authorization: Bearer tm_abc123..." \
  http://localhost:18790/v1/episodes/ep-abc123.../close
```

**Response:**
```json
{
  "success": true
}
```

---

### `POST /v1/episodes/{id}/regenerate`

Force full summary regeneration (expensive).

**Example:**
```bash
curl -X POST \
  -H "Authorization: Bearer tm_abc123..." \
  http://localhost:18790/v1/episodes/ep-abc123.../regenerate
```

**Response:**
```json
{
  "success": true
}
```

---

## Troubleshooting

### Episodes Not Detecting

**Symptom:** Memories are stored but episodes aren't created automatically.

**Common causes:**
1. **Episodes not enabled** — Check `episodes.enabled: true` in config
2. **No API key** — Episodes require an LLM for summarization (see [API key setup](#2-set-api-key))
3. **Service not restarted** — Config changes require server restart
4. **Similarity threshold too high** — Try lowering `embedding_similarity_threshold` to 0.65
5. **Max active episodes reached** — Old episodes are auto-closed when limit hit

**Debug:**
```bash
# Check server logs for episode-related errors
tribalmemory service logs

# Verify episodes enabled
curl http://localhost:18790/v1/episodes
# Should NOT return "Episode feature not enabled"
```

---

### Summaries Not Updating

**Symptom:** Episode created but summary is empty or stale.

**Common causes:**
1. **LLM API key missing or invalid**
2. **Monthly cost ceiling exceeded** — Check `monthly_cost_ceiling` in config
3. **Summarizer model unavailable** — Try switching to a different model
4. **Network issues** — Check LLM API connectivity

**Debug:**
```bash
# Check server logs for summarization errors
tribalmemory service logs | grep -i summary

# Force regeneration to trigger summarization
curl -X POST \
  -H "Authorization: Bearer tm_abc123..." \
  http://localhost:18790/v1/episodes/{id}/regenerate
```

---

### High LLM Costs

**Symptom:** Monthly cost ceiling reached quickly.

**Solutions:**
1. **Increase `full_regen_interval`** — Less frequent full regenerations
   ```yaml
   episodes:
     full_regen_interval: 20  # Was: 10
   ```

2. **Lower `max_llm_calls_per_memory`** — Fewer LLM calls per store
   ```yaml
   episodes:
     max_llm_calls_per_memory: 1  # Was: 2
   ```

3. **Use cheaper model** — Switch to gpt-4o-mini or local Ollama
   ```yaml
   episodes:
     summarizer_model: gpt-4o-mini
     summarizer_provider: openai
   ```

4. **Disable LLM detection** — Use embedding-only detection
   ```yaml
   episodes:
     detector_strategy: embedding  # Was: hybrid
   ```

5. **Use local LLM (Ollama)**:
   ```yaml
   episodes:
     summarizer_provider: ollama
     summarizer_model: llama3.2:latest
   ```
   ```bash
   # In ~/.tribal-memory/.env
   OLLAMA_BASE_URL=http://localhost:11434
   ```

---

### Authentication Errors

**Symptom:** Episode routes return 401 Unauthorized.

Episode routes require the same authentication as all other `/v1/` routes.

**Solution:** Include bearer token in requests (see [authentication.md](authentication.md)):

```bash
# Generate token if not already done
tribalmemory token generate

# Use token in requests
curl -H "Authorization: Bearer tm_abc123..." \
  http://localhost:18790/v1/episodes
```

---

## Cost Estimation

Approximate costs with **gpt-4o-mini** (~$0.01 per 1K input tokens, ~$0.03 per 1K output tokens):

| Operation | Avg Cost (per operation) | When It Happens |
|-----------|--------------------------|-----------------|
| Episode detection (LLM path) | $0.01-0.02 | When embedding similarity is ambiguous |
| Progressive summary update | $0.01-0.02 | Every memory added to episode |
| Full summary regeneration | $0.05-0.10 | Every 10 memories, or when closing |

**Example monthly usage:**
- 500 memories stored
- 50 episodes created
- 100 LLM classifications (hybrid mode)
- 50 progressive updates
- 5 full regenerations

**Total: ~$3-4/month** with default settings.

**Cost controls:**
- `monthly_cost_ceiling: 5.0` — Hard limit, disables summaries when exceeded
- `max_llm_calls_per_memory: 2` — Prevents runaway costs per write
- Switch to local Ollama for zero cost (quality tradeoff)

---

## Advanced Usage

### Manual Episode Management

You can manually group memories into episodes for:
- Organizing historical memories
- Correcting auto-detection mistakes
- Creating thematic collections

```python
# Find related memories
results = tribal_recall(query="docker debugging", limit=20)

# Extract IDs
memory_ids = [r["memory_id"] for r in results["results"]]

# Create episode
tribal_episode_create(
  title="Docker Debugging Session",
  memory_ids=memory_ids
)
```

### Hybrid Search with Episodes

Episode summaries surface in standard recall:

```python
# This returns both regular memories AND episode summaries
results = tribal_recall(query="authentication work")

# Episode summaries have source_type = "EPISODE_SUMMARY"
for r in results["results"]:
    if r["memory"]["source_type"] == "EPISODE_SUMMARY":
        print(f"📖 Episode: {r['memory']['content']}")
    else:
        print(f"💭 Memory: {r['memory']['content']}")
```

### Closing Episodes Automatically

Configure `active_window_days` to auto-close stale episodes:

```yaml
episodes:
  active_window_days: 7  # Close episodes inactive for 7+ days
```

This keeps the active episode list focused on current work.

---

## Design Details

For implementation details, see [design/episode-memories.md](design/episode-memories.md).

---

## See Also

- [Authentication](authentication.md) — Required for episode HTTP routes
- [Import/Export Guide](import-export-guide.md) — Episodes are included in exports
- [Graph Visualization](graph-visualization.md) — Explore episode relationships visually
