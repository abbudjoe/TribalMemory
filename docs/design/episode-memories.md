# Episode Memories: Write-time Narrative Synthesis for Multi-hop Retrieval

**Issue:** #190  
**Author:** Clawdio (AI assistant)  
**Status:** Draft  
**Date:** 2026-02-10  

---

## 1. Problem Statement

### The Multi-hop Recall Gap

Consider a user working with an AI assistant over several weeks while house hunting. Across separate sessions they mention:

- **Session A (Jan 15):** "Viewed the Elm Street bungalow today. 3 bed, needs new roof. $385k."
- **Session B (Jan 18):** "Toured 44 Oak Avenue — gorgeous kitchen but tiny yard. $420k."
- **Session C (Jan 22):** "Saw the Brookside townhouse, loved it. 2 bed, modern finishes. $395k."
- **Session D (Jan 25):** "Also checked out a duplex on River Road. Not great. $450k."
- **Session E (Jan 28):** "Put in an offer on the Brookside townhouse!"

A week later the user asks:

> "How many properties did I view before making an offer on the Brookside townhouse?"

The answer is **3** (Elm Street, Oak Avenue, River Road — excluding Brookside itself which was the offer target). This requires connecting 5 sessions that share a common *activity* but have almost zero semantic overlap with the query.

### Why Current Architecture Fails

TribalMemory's retrieval pipeline is powerful but read-optimized:

1. **Vector search** embeds the query and finds semantically similar memories. But "How many properties did I view before making an offer" doesn't embed anywhere near "Viewed the Elm Street bungalow" or "Toured 44 Oak Avenue." The query is *about* the collection of events, not *similar to* any individual event.

2. **BM25/FTS** matches keywords. The query mentions "Brookside townhouse" so it might find Session C and E, but misses A, B, and D entirely — they share no keywords with the query.

3. **Graph expansion** via entity extraction could help if all properties were linked to a "house hunting" entity, but the current `EntityExtractor` is designed for software/technology entities and person names. It won't detect "Elm Street bungalow" as related to "Brookside townhouse" — they're different entities with no graph edges between them.

4. **Temporal filtering** could narrow by date range if the user said "in January," but the query doesn't include temporal language.

The fundamental issue: **no single memory is the answer**. The answer requires *aggregating* across multiple memories that are only connected by a shared human *goal* — not by any textual, semantic, or entity overlap that our retrieval can detect.

### Why Query-time Solutions Are the Wrong Layer

The obvious alternative: build an agentic RAG pipeline that decomposes the query, runs multiple searches, and synthesizes an answer. This fails for three reasons:

1. **Decomposition requires knowledge we don't have.** To decompose "How many properties before Brookside?" the query planner needs to know that house hunting happened across multiple sessions. But it only knows this *after* finding those sessions — a chicken-and-egg problem.

2. **Cost and latency.** Each query would require 3-5 LLM calls for decomposition, sub-queries, and synthesis. At ~500ms per call, that's 1.5-2.5 seconds of latency plus $0.01-0.05 per query. For a memory system called on every message, this is untenable.

3. **Wrong level of abstraction.** The *query* shouldn't need to be clever. The *memory system* should already know that these events form a coherent episode. Write-time reasoning amortizes the cost: you pay once at ingest, and every subsequent query benefits.

**Our proposal:** detect episodes at write time and generate progressive summary memories that make multi-hop questions answerable with a single vector search.

After episode processing, the system would contain a summary memory like:

> "House hunting episode (Jan 15 – Jan 28): Viewed 4 properties — Elm Street bungalow ($385k, needs roof), 44 Oak Avenue ($420k, small yard), Brookside townhouse ($395k, modern finishes), River Road duplex ($450k, not great). Made an offer on Brookside townhouse on Jan 28."

This summary embeds close to "How many properties before Brookside?" because it literally contains the narrative. Standard vector search handles the rest.

---

## 2. Design: Episode Detection

### Approaches Compared

| Approach | Accuracy | Latency | Cost | Complexity |
|----------|----------|---------|------|------------|
| Embedding similarity to episode summaries | Medium | <10ms | Free | Low |
| LLM classification | High | 300-800ms | $0.001-0.005/call | Medium |
| Hybrid (embedding + LLM edge cases) | High | 10-800ms | $0.001 amortized | Medium-High |
| Entity overlap (existing extraction) | Low-Medium | <5ms | Free | Low |

**Embedding similarity** is fast and cheap: embed the incoming memory, compare against existing episode summary embeddings, join if similarity exceeds a threshold. Problem: it's unreliable for the exact case we're solving. "Viewed Elm Street bungalow" and "Put in offer on Brookside" have low embedding similarity (~0.3-0.4) despite being the same episode. Embedding similarity works for topically identical content, not thematically related activities.

**LLM classification** is accurate: send the memory + a list of active episodes to a fast model and ask "Does this belong to any of these episodes? If so, which one? If not, should it start a new episode?" This handles nuance well but adds 300-800ms latency and $0.001-0.005 per memory stored. For a system that ingests 50-200 memories/day, that's $0.05-1.00/day — acceptable, but the latency hit on every `remember()` call is concerning.

**Entity overlap** using existing extraction: check if the incoming memory shares entities with any active episode's constituent memories. Cheap and fast, but our entity extractor is tuned for software/tech entities and person names. "Elm Street bungalow" won't be extracted as an entity by `EntityExtractor` or even `SpacyEntityExtractor` reliably. This approach would require significant entity extraction improvements first.

### Recommendation: Hybrid with Async LLM

Use a **two-phase hybrid approach**:

1. **Fast path (synchronous, <10ms):** Embedding similarity against active episode summaries. If similarity > 0.75, auto-join the episode. This catches obvious cases (e.g., "Viewed another property on Oak Avenue" when there's already a "house hunting" episode summary).

2. **Slow path (asynchronous, background):** For memories that don't match any episode via embedding (similarity < 0.75), queue them for LLM classification. A background worker processes the queue, classifying memories against active episodes and optionally creating new ones.

The key insight: **episode detection doesn't need to be synchronous.** The memory is stored immediately via the normal `remember()` flow. Episode assignment happens in the background. A 5-30 second delay before the episode summary updates is perfectly acceptable — the user won't query about an episode they just added to.

```python
# Pseudocode: Episode detection in remember() flow

async def remember(self, content, ...):
    # Normal remember flow (unchanged)
    result = await self._store_memory(content, ...)
    
    if not result.success or not self.episodes_enabled:
        return result
    
    # Fast path: embedding similarity to active episodes
    episode_match = await self._fast_episode_match(result.memory_id, embedding)
    
    if episode_match and episode_match.similarity > 0.75:
        # High confidence: join episode synchronously
        await self._join_episode(result.memory_id, episode_match.episode_id)
        await self._update_episode_summary(episode_match.episode_id)
    else:
        # Low confidence or no match: queue for LLM classification
        await self._queue_episode_classification(result.memory_id, content, embedding)
    
    return result
```

### Episode Creation vs. Joining

**Creating a new episode** should require LLM involvement — we don't want to auto-create episodes from every memory. The LLM classifier determines:

1. Does this memory belong to an existing active episode? → **Join**
2. Does this memory represent a new multi-session activity/goal? → **Create** a new episode
3. Is this memory standalone (a preference, a fact, a one-off event)? → **Skip** — not everything is an episode

Creation heuristics for the LLM prompt:
- Episodes represent **goal-directed activities** spanning multiple sessions (house hunting, job searching, planning a trip, debugging a production issue)
- Single facts or preferences are NOT episodes ("Joe prefers TypeScript" → skip)
- One-off events are NOT episodes unless they connect to prior context ("Had lunch at Olive Garden" is standalone unless there's a "trying restaurants" episode)

### Active Episode Window

Episodes have an **activity window** — they can't stay "active" forever. An episode is considered active if it received a new memory within the last **14 days** (configurable). After that, it's closed and no longer checked during detection. Closed episodes' summaries remain in the vector store permanently — they just stop accepting new members.

```python
# Pseudocode: LLM classification prompt

EPISODE_CLASSIFY_PROMPT = """You are classifying a memory for episode detection.

Active episodes:
{episodes_with_summaries}

New memory:
"{content}"

Respond with JSON:
- If this memory belongs to an existing episode:
  {{"action": "join", "episode_id": "<id>", "reason": "<brief reason>"}}
- If this memory starts a new goal-directed activity:
  {{"action": "create", "title": "<episode title>", "reason": "<brief reason>"}}
- If this memory is standalone (a fact, preference, or one-off event):
  {{"action": "skip", "reason": "<brief reason>"}}

Rules:
- Episodes are GOAL-DIRECTED activities spanning multiple sessions
- Single facts, preferences, or one-off events are NOT episodes
- When in doubt, skip — we can always retroactively assign later
"""
```

---

## 3. Design: Episode Summarization

### Progressive Summarization Protocol

When a memory joins an episode, the episode summary must be updated. Regenerating from scratch (re-reading all constituent memories) is expensive and doesn't scale. Instead, use **progressive summarization**:

1. **On episode creation:** Generate an initial summary from the first 1-2 memories.
2. **On memory join:** Send the *current summary* + the *new memory* to the LLM. The LLM produces an updated summary that incorporates the new information.
3. **Periodic regeneration:** Every N memories (e.g., 10) or on explicit request, regenerate the summary from scratch using all constituent memories. This corrects drift from progressive updates.

```python
# Pseudocode: Progressive summary update

async def _update_episode_summary(self, episode_id: str):
    episode = await self.episode_store.get(episode_id)
    new_memories = await self._get_unsummarized_memories(episode_id)
    
    if not new_memories:
        return
    
    if episode.memory_count % 10 == 0:
        # Full regeneration every 10 memories
        all_memories = await self._get_episode_memories(episode_id)
        new_summary = await self._generate_full_summary(episode, all_memories)
    else:
        # Progressive update: current summary + new memories
        new_summary = await self._generate_progressive_summary(
            episode.summary, new_memories
        )
    
    # Update episode and re-embed the summary
    episode.summary = new_summary
    episode.summary_embedding = await self.embedding_service.embed(new_summary)
    await self.episode_store.update(episode)
    
    # Update the episode's summary memory in the vector store
    await self._update_summary_memory(episode)
```

### LLM Model Selection

**Recommendation: `gpt-4o-mini` (or equivalent fast/cheap model), configurable.**

Rationale:
- Summarization is a well-understood task — smaller models handle it well
- `gpt-4o-mini` costs ~$0.15/1M input tokens, $0.60/1M output tokens
- Average episode summary update: ~500 input tokens (current summary + new memory) → ~200 output tokens
- Cost per update: ~$0.0002 (negligible)
- Latency: 300-600ms (acceptable for async background processing)

The model should be configurable via `episodes.summarizer_model` to allow local models (Ollama) or other providers.

### Summary Format/Schema

Episode summaries should be **narrative prose**, not structured data. The whole point is that they embed well for semantic search. A structured format (JSON, bullet points) would reduce retrieval quality.

However, summaries should follow a consistent template:

```
{title} ({date_range}): {narrative_summary}

Key details: {important facts, names, numbers, decisions}

Status: {ongoing|completed|abandoned}
```

Example:
```
House hunting (Jan 15 – Jan 28, 2026): Viewed 4 properties in the $385k-$450k range.
Started with the Elm Street bungalow (3 bed, needs new roof, $385k), then toured
44 Oak Avenue (gorgeous kitchen but tiny yard, $420k), the Brookside townhouse
(2 bed, modern finishes, $395k), and a duplex on River Road ($450k, not great).
Made an offer on the Brookside townhouse on Jan 28.

Key details: 4 properties viewed, offer made on Brookside townhouse at $395k.

Status: ongoing
```

### Summary Generation Prompts

```python
PROGRESSIVE_SUMMARY_PROMPT = """Update this episode summary with new information.

Current summary:
{current_summary}

New memory (added {timestamp}):
"{new_memory_content}"

Write an updated narrative summary that:
1. Incorporates the new information naturally
2. Maintains chronological order
3. Preserves all specific details (names, numbers, dates, decisions)
4. Updates counts and aggregates as needed
5. Keeps the same format: title, date range, narrative, key details, status

Updated summary:"""

FULL_SUMMARY_PROMPT = """Create a narrative summary of this episode from all constituent memories.

Episode title: {title}
Memories (chronological):
{all_memories_with_timestamps}

Write a comprehensive narrative summary that:
1. Tells the story chronologically
2. Preserves all specific details (names, numbers, dates, decisions)
3. Includes counts and aggregates where relevant
4. Notes the current status (ongoing, completed, abandoned)

Format:
{title} ({date_range}): {narrative}

Key details: {important facts}

Status: {status}"""
```

### When to Regenerate vs. Append

| Trigger | Action |
|---------|--------|
| 1st memory added | Generate initial summary |
| 2nd-9th memory | Progressive update (summary + new memory → updated summary) |
| Every 10th memory | Full regeneration from all constituent memories |
| Manual request via MCP tool | Full regeneration |
| Episode closed (14 days inactive) | Final full regeneration |

### Cost Analysis

Per memory stored (worst case, assuming every memory triggers episode processing):

| Operation | LLM Calls | Cost (gpt-4o-mini) | Latency |
|-----------|-----------|---------------------|---------|
| Episode detection (fast path hit) | 0 | $0 | <10ms |
| Episode detection (LLM classification) | 1 | ~$0.0003 | 400ms |
| Progressive summary update | 1 | ~$0.0002 | 400ms |
| Full regeneration (every 10th) | 1 | ~$0.001 | 600ms |
| **Amortized per memory** | **~1.1** | **~$0.0005** | **async** |

At 100 memories/day: **~$0.05/day**, **~$1.50/month**. Acceptable.

---

## 4. Design: Storage & Data Model

### Episode Table Schema

Extend the existing `GraphStore` SQLite pattern with new tables:

```sql
-- Core episode table
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,           -- UUID
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    summary_memory_id TEXT,        -- FK to the episode's summary in vector store
    status TEXT NOT NULL DEFAULT 'active',  -- active, closed, archived
    created_at TEXT NOT NULL,      -- ISO timestamp
    updated_at TEXT NOT NULL,      -- ISO timestamp (last memory added)
    closed_at TEXT,                -- ISO timestamp (when episode became inactive)
    memory_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT DEFAULT '{}'
);

-- Episode ↔ memory association (one-to-many with episode as parent)
CREATE TABLE IF NOT EXISTS episode_memories (
    episode_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    added_at TEXT NOT NULL,        -- ISO timestamp
    summarized BOOLEAN NOT NULL DEFAULT 0,  -- has this memory been incorporated into summary?
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE,
    PRIMARY KEY (episode_id, memory_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_episode_status ON episodes(status);
CREATE INDEX IF NOT EXISTS idx_episode_updated ON episodes(updated_at);
CREATE INDEX IF NOT EXISTS idx_episode_memories_memory ON episode_memories(memory_id);
```

### Relationship: Episodes ↔ Memories

**One-to-many** (one episode has many memories, each memory belongs to at most one episode).

Rationale: While many-to-many sounds more flexible, it dramatically complicates summary generation — if a memory belongs to two episodes, which summary should it update? And in practice, a memory's activity context is almost always singular. A "viewed Elm Street bungalow" memory is part of "house hunting," period.

If we later need many-to-many (e.g., sub-episodes), we can add a separate `episode_links` table for episode-to-episode relationships without changing the core model.

### Episode Summary as First-class Memory

The episode summary is stored as a regular `MemoryEntry` in the vector store with:
- `source_type = MemorySource.EPISODE_SUMMARY` (new enum value)
- `tags = ["episode:<episode_id>", "episode_summary"]`
- `content = <the narrative summary>`
- `embedding = <embedding of the summary>`

This means episode summaries **automatically surface in `recall()`** with zero changes to the retrieval pipeline. They're just memories that happen to be rich, aggregated narratives.

### Graph Edges

Add edges in the existing graph store:
- Each episode summary memory gets an entity: `Entity(name=episode.title, entity_type="episode")`
- Graph edges: `episode_summary_memory → constituent_memory` (via `related_to` field)
- This enables graph expansion to find constituent memories when the episode summary is retrieved

---

## 5. Integration Points

### Hook into `remember()` Flow

Episode detection hooks into `remember()` **after** all existing processing (embedding, dedup, storage, FTS indexing, entity extraction, temporal extraction). It's the last step, and it's non-blocking:

```python
# In TribalMemoryService.remember(), after all existing steps:

# ... existing code: embedding, dedup, store, FTS, graph, temporal ...

# Episode detection (async, non-blocking)
if result.success and self.episode_detector:
    try:
        await self._detect_episode(entry, embedding)
    except Exception as e:
        logger.warning("Episode detection failed for %s: %s", entry.id, e)
        # Never fail remember() due to episode detection failure
```

The `_detect_episode` method:
1. Computes embedding similarity against active episode summaries
2. If high similarity (>0.75): joins episode synchronously, queues summary update
3. If low similarity: queues for async LLM classification

### How Episode Summaries Surface in `recall()`

**Zero changes needed.** Episode summaries are regular memories with `source_type=EPISODE_SUMMARY`. They participate in:
- Vector search (they have embeddings)
- BM25/FTS search (they have text content)
- Graph expansion (they have entity associations)
- Temporal filtering (they span date ranges)

The only addition: when an episode summary is returned in recall results, the response could optionally include a flag indicating it's an episode summary, allowing the caller to fetch constituent memories if desired.

### Configuration Schema

```python
@dataclass
class EpisodeConfig:
    """Configuration for episode detection and summarization."""
    
    enabled: bool = False                          # Feature flag
    
    # Detection
    detector_strategy: str = "hybrid"              # "embedding", "llm", "hybrid"
    embedding_similarity_threshold: float = 0.75   # For fast-path matching
    active_window_days: int = 14                   # Days before episode auto-closes
    max_active_episodes: int = 20                  # Cap on concurrent active episodes
    
    # Summarization
    summarizer_model: str = "gpt-4o-mini"          # LLM model for summaries
    summarizer_provider: str = "openai"            # "openai", "anthropic", "ollama"
    full_regen_interval: int = 10                  # Full regeneration every N memories
    
    # Cost controls
    max_llm_calls_per_memory: int = 2              # Max LLM calls triggered per remember()
    monthly_cost_ceiling: float = 5.0              # Pause episode processing if exceeded
```

This maps to the MCP server YAML config:

```yaml
episodes:
  enabled: true
  detector_strategy: hybrid
  embedding_similarity_threshold: 0.75
  active_window_days: 14
  summarizer_model: gpt-4o-mini
  monthly_cost_ceiling: 5.0
```

### MCP Tools

New MCP tools for episode management:

| Tool | Description |
|------|-------------|
| `tribal_episodes_list` | List episodes (active, closed, all) with summaries |
| `tribal_episode_get` | Get episode details + constituent memory IDs |
| `tribal_episode_create` | Manually create an episode from existing memories |
| `tribal_episode_add` | Manually add a memory to an episode |
| `tribal_episode_remove` | Remove a memory from an episode |
| `tribal_episode_close` | Manually close an episode |
| `tribal_episode_regenerate` | Force full summary regeneration |

---

## 6. Implementation Plan

### Phase 1: Data Model & Storage (PR #1)

**Can be built independently.** No LLM dependency, no changes to existing flows.

- Add `EpisodeStore` class extending the SQLite `GraphStore` pattern
- Episode CRUD operations (create, update, close, list, get)
- Episode-memory association management
- Add `MemorySource.EPISODE_SUMMARY` enum value
- Schema migration (new tables, idempotent CREATE IF NOT EXISTS)
- Unit tests for all store operations

**Estimated effort:** 1-2 days  
**Dependencies:** None

### Phase 2: Episode Detection (PR #2)

**Depends on Phase 1** for storage.

- `EpisodeDetector` class with embedding similarity fast path
- LLM classifier for edge cases (with configurable model/provider)
- Background queue for async classification (simple asyncio.Queue)
- Hook into `remember()` flow (non-blocking, failure-tolerant)
- `EpisodeConfig` dataclass and config loading
- Integration tests with mock LLM responses

**Estimated effort:** 2-3 days  
**Dependencies:** Phase 1

### Phase 3: Episode Summarization (PR #3)

**Depends on Phase 2** for detection triggering.

- `EpisodeSummarizer` class with progressive and full regeneration
- Summary prompt templates
- Summary-as-memory storage in vector store
- Cost tracking (LLM calls per day, monthly spend)
- Episode auto-close on inactivity timeout
- Integration tests with mock LLM responses
- End-to-end test: ingest house-hunting memories → query → verify episode summary surfaces

**Estimated effort:** 2-3 days  
**Dependencies:** Phase 2

### Phase 4: MCP Tools & Polish (PR #4)

**Depends on Phase 3** for full functionality.

- MCP tool implementations (list, get, create, add, remove, close, regenerate)
- Configuration via YAML config file
- Documentation update (README, API docs)
- Performance benchmarking (latency impact on `remember()`)
- LongMemEval benchmark test cases

**Estimated effort:** 1-2 days  
**Dependencies:** Phase 3

### Testing Strategy

- **Unit tests:** Each class in isolation with mocked dependencies
- **Integration tests:** Full pipeline with mock LLM responses (no real API calls in CI)
- **LongMemEval benchmark:** The house-hunting scenario as a regression test — must pass after Phase 3
- **Performance tests:** Verify `remember()` latency doesn't regress (episode detection is async)
- **Cost tests:** Verify LLM call counts stay within configured limits

### Migration: Existing Memories

When episodes are first enabled on an existing database:

1. **No automatic retroactive processing.** Existing memories are not retroactively assigned to episodes. This would require processing every memory through the LLM classifier — expensive and of uncertain value.
2. **Manual episode creation** via MCP tools: users can create episodes and assign existing memories manually.
3. **Future enhancement:** A batch job that processes recent memories (last 30 days) through the episode detector. This is Phase 5, not in scope for initial release.

---

## 7. Open Questions

### Should episodes have a maximum size/duration?

**Recommendation: Soft limit of 50 memories per episode, no hard duration cap.**

Beyond ~50 memories, progressive summarization starts to drift significantly from ground truth. Full regeneration at that point requires sending 50+ memory texts to the LLM, which is expensive and may exceed context windows. Pragmatically, if an episode has 50+ memories, it's probably too broad and should be split.

Implementation: when an episode hits 50 memories, log a warning and continue accepting memories but regenerate the summary every 5 additions instead of every 10.

### How to handle episode splitting?

A "house hunting" episode might have natural sub-phases: "searching," "making offers," "mortgage process," "closing." For v1, **we don't split.** The episode summary captures the full narrative, and sub-episodes add complexity without clear retrieval benefit.

Future enhancement: allow hierarchical episodes (parent-child). A child episode's summary references the parent. This would require a `parent_episode_id` column but no architectural changes.

### How to handle episode merging?

Two episodes might turn out to be the same activity (e.g., "looking at apartments" and "house hunting" are the same episode discovered from different angles). For v1, **manual merge via MCP tool** (move memories from one episode to another, regenerate summary, delete the empty episode). Automatic merge detection is a future enhancement.

### Offline batch processing vs. real-time?

**Recommendation: Hybrid.** Fast-path embedding match is synchronous (real-time). LLM classification and summarization are asynchronous (near-real-time, 5-30 second delay). Batch processing is available as a manual MCP tool for retroactive episode assignment.

### Cost ceiling: max LLM spend per memory operation?

**Recommendation: 2 LLM calls max per `remember()` invocation** (1 for classification, 1 for summary update). The `monthly_cost_ceiling` config parameter ($5.00 default) pauses all episode processing when exceeded, preventing runaway costs. At $0.0005/memory amortized, $5.00 supports ~10,000 memories/month — well above typical usage.

### How does this interact with session transcripts?

`SessionStore` ingests conversation chunks — these are raw transcript windows, not semantic memories. Episode detection should operate on **memories** (from `remember()`), not session chunks. However, a future enhancement could detect episodes from session patterns (e.g., the same topic discussed across multiple sessions).

### What about episode summaries and deduplication?

Episode summaries are stored as regular memories, so they participate in deduplication. When an episode summary is updated, the old summary memory should be **replaced** (same memory ID, updated content and embedding), not stored as a new memory. This prevents summary versions from accumulating and confusing recall.

---

## Appendix: End-to-End Example

### Ingest Flow

```
Day 1: remember("Viewed Elm Street bungalow, 3 bed, needs roof, $385k")
  → stored as memory M1
  → embedding similarity to active episodes: no match (no episodes exist)
  → queued for LLM classification
  → LLM: "skip" (standalone event, not yet a pattern)

Day 2: remember("Toured 44 Oak Avenue, gorgeous kitchen, tiny yard, $420k")
  → stored as memory M2
  → embedding similarity: no match
  → queued for LLM classification
  → LLM: "create" episode "House hunting" (second property viewing suggests a pattern)
  → Episode E1 created, M1 and M2 retroactively assigned
  → Summary generated: "House hunting (Jan 15-18): Viewed 2 properties..."

Day 3: remember("Saw Brookside townhouse, loved it, 2 bed, modern, $395k")
  → stored as memory M3
  → embedding similarity to E1 summary: 0.82 (house viewing → house hunting)
  → fast-path join to E1
  → progressive summary update: "House hunting (Jan 15-22): Viewed 3 properties..."

Day 4: remember("Checked duplex on River Road, not great, $450k")
  → stored as memory M4
  → embedding similarity to E1 summary: 0.79
  → fast-path join to E1
  → progressive summary update: "House hunting (Jan 15-25): Viewed 4 properties..."

Day 5: remember("Put in offer on Brookside townhouse!")
  → stored as memory M5
  → embedding similarity to E1 summary: 0.71 (below threshold — "offer" ≠ "viewing")
  → queued for LLM classification
  → LLM: "join" E1 (same house hunting activity)
  → progressive summary update: "House hunting (Jan 15-28): Viewed 4 properties... Made offer on Brookside."
```

### Recall Flow

```
recall("How many properties did I view before making an offer on Brookside?")
  → embedding generated
  → vector search returns: E1 summary (similarity 0.83) ← this is the money shot
  → also returns: M5 (0.72), M3 (0.68)
  → E1 summary contains the complete narrative with all 4 properties
  → agent can answer: "You viewed 3 properties before making the offer"
```

No query decomposition. No multi-hop reasoning. No agentic RAG. Just a well-structured memory that was prepared at write time.
