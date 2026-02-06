# Learned Retrieval Layer - Architecture Proposal

**Created:** 2026-01-31
**Status:** Draft
**Context:** Baseline tests show 60% retrieval accuracy with tuned config. Core issue: question-to-fact semantic gap.

---

## Problem Statement

Current RAG retrieval fails because:
1. **Semantic gap**: Questions ("What coffee do I like?") don't embed close to facts ("Coffee: Oat milk latte")
2. **Chunk granularity**: 40-line chunks bury specific facts in noise
3. **Static embeddings**: No learning from successful/failed retrievals

**Baseline measurement:** 60% on easy queries with tuned hybrid search (minScore 0.1, 60/40 vector/BM25).

---

## Current PRD Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Tribal Memory PRD                     │
├─────────────────────────────────────────────────────────┤
│  IMemoryService                                          │
│    ├── capture(memory) → stores raw fact                 │
│    ├── recall(query) → vector similarity search          │
│    ├── search(query, filters) → hybrid search            │
│    └── relate(id, id) → manual linking                   │
│                                                          │
│  IVectorStore (LanceDB)                                  │
│    └── Pure cosine similarity on embeddings              │
│                                                          │
│  IEmbedding (FastEmbed BAAI/bge-small-en-v1.5)           │
│    └── Static embeddings, no feedback                    │
└─────────────────────────────────────────────────────────┘
```

**Gap:** No mechanism to learn from retrieval success/failure.

---

## Proposed Enhancement: Learned Retrieval Layer

```
┌─────────────────────────────────────────────────────────┐
│              Enhanced Tribal Memory                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │         LEARNED RETRIEVAL LAYER (NEW)           │    │
│  ├─────────────────────────────────────────────────┤    │
│  │                                                  │    │
│  │  Query Cache                                     │    │
│  │    └── query_hash → [fact_ids, success_count]   │    │
│  │                                                  │    │
│  │  Query Expansion                                 │    │
│  │    └── "What coffee?" → ["coffee preference",   │    │
│  │         "favorite drink", "morning beverage"]   │    │
│  │                                                  │    │
│  │  Retrieval Feedback                              │    │
│  │    └── Track: query → retrieved → used/ignored  │    │
│  │    └── Reinforce successful query→fact links    │    │
│  │                                                  │    │
│  │  Fact Anchoring                                  │    │
│  │    └── Facts tagged with successful queries     │    │
│  │    └── "Coffee: Oat milk" anchored to           │    │
│  │        ["what coffee", "coffee preference"]     │    │
│  │                                                  │    │
│  └─────────────────────────────────────────────────┘    │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │         EXISTING MEMORY SERVICE                  │    │
│  │  IMemoryService + IVectorStore + IEmbedding     │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. Query Cache (Fast Path)

**Purpose:** Bypass embedding search for known-good query→fact mappings.

```python
class QueryCache:
    def lookup(self, query: str) -> Optional[List[FactId]]:
        """Return cached fact IDs if query seen before with success."""
        normalized = self.normalize(query)
        if normalized in self.cache:
            entry = self.cache[normalized]
            if entry.success_rate > 0.8:
                return entry.fact_ids
        return None  # Fall through to embedding search
    
    def record_success(self, query: str, fact_ids: List[FactId]):
        """Record successful retrieval for future cache hits."""
        normalized = self.normalize(query)
        self.cache[normalized].fact_ids = fact_ids
        self.cache[normalized].success_count += 1
    
    def normalize(self, query: str) -> str:
        """Normalize query for fuzzy matching."""
        # Lowercase, remove punctuation, stem words
        return stemmed_lowercase(query)
```

**Impact:** Instant retrieval for repeated/similar queries.

### 2. Query Expansion (Improved Recall)

**Purpose:** Transform question-style queries into keyword-style for better embedding match.

```python
class QueryExpander:
    def expand(self, query: str) -> List[str]:
        """Generate query variants for broader search."""
        variants = [query]
        
        # Rule-based expansions
        if query.startswith("What"):
            # "What coffee do I like?" → "coffee preference"
            variants.append(extract_topic(query) + " preference")
        
        # LLM-based expansion (if available)
        if self.llm:
            variants.extend(self.llm.generate_variants(query))
        
        # Historical expansions (learned)
        if query in self.expansion_cache:
            variants.extend(self.expansion_cache[query])
        
        return variants
```

**Impact:** Bridge question-to-fact gap without changing embeddings.

### 3. Retrieval Feedback (Learning Signal)

**Purpose:** Track what retrievals actually get used vs ignored.

```python
class RetrievalFeedback:
    def record_retrieval(self, query: str, retrieved: List[Fact], 
                         context_id: str):
        """Log retrieval event for later feedback."""
        self.pending[context_id] = {
            "query": query,
            "retrieved": retrieved,
            "timestamp": now()
        }
    
    def record_usage(self, context_id: str, used_facts: List[FactId]):
        """Record which retrieved facts were actually used in response."""
        if context_id in self.pending:
            retrieval = self.pending[context_id]
            for fact in retrieval["retrieved"]:
                if fact.id in used_facts:
                    self.reinforce(retrieval["query"], fact.id)
                else:
                    self.penalize(retrieval["query"], fact.id)
    
    def reinforce(self, query: str, fact_id: FactId):
        """Strengthen query→fact link."""
        self.query_fact_weights[(query, fact_id)] += REINFORCE_DELTA
    
    def penalize(self, query: str, fact_id: FactId):
        """Weaken query→fact link."""
        self.query_fact_weights[(query, fact_id)] -= PENALIZE_DELTA
```

**Impact:** System learns from actual usage, not just retrieval.

### 4. Fact Anchoring (Bidirectional Links)

**Purpose:** Attach successful query patterns directly to facts.

```python
class FactAnchoring:
    def anchor(self, fact_id: FactId, query_patterns: List[str]):
        """Link query patterns to a fact for future retrieval."""
        self.fact_anchors[fact_id].extend(query_patterns)
    
    def search_by_anchors(self, query: str) -> List[FactId]:
        """Find facts whose anchors match the query."""
        matches = []
        for fact_id, anchors in self.fact_anchors.items():
            for anchor in anchors:
                if semantic_match(query, anchor) > 0.8:
                    matches.append(fact_id)
        return matches
```

**Impact:** Facts become findable by the questions that surface them.

---

## Integration with Existing PRD

### Modified IMemoryService Interface

```python
class IMemoryService(Protocol):
    # Existing methods (unchanged)
    def capture(self, memory: Memory) -> MemoryId: ...
    def recall(self, query: str) -> List[Memory]: ...
    def search(self, query: str, filters: SearchFilters) -> List[Memory]: ...
    def relate(self, source: MemoryId, target: MemoryId) -> None: ...
    
    # NEW: Learned retrieval methods
    def recall_with_feedback(self, query: str, context_id: str) -> List[Memory]:
        """Recall with feedback tracking enabled."""
        ...
    
    def record_usage_feedback(self, context_id: str, used_ids: List[MemoryId]):
        """Record which memories were actually used."""
        ...
    
    def get_retrieval_stats(self) -> RetrievalStats:
        """Return learned retrieval statistics."""
        ...
```

### Storage Schema Extension

```sql
-- Query cache table
CREATE TABLE query_cache (
    query_hash TEXT PRIMARY KEY,
    query_normalized TEXT,
    fact_ids TEXT,  -- JSON array
    success_count INTEGER DEFAULT 0,
    last_success TIMESTAMP
);

-- Query-fact weights (learned)
CREATE TABLE query_fact_weights (
    query_hash TEXT,
    fact_id TEXT,
    weight REAL DEFAULT 0.0,
    updated_at TIMESTAMP,
    PRIMARY KEY (query_hash, fact_id)
);

-- Fact anchors
CREATE TABLE fact_anchors (
    fact_id TEXT,
    anchor_pattern TEXT,
    source TEXT,  -- 'manual' | 'learned' | 'llm'
    confidence REAL,
    created_at TIMESTAMP
);
```

---

## Implementation Timeline

| Week | Current PRD | + Learned Retrieval |
|------|-------------|---------------------|
| 3-4 | OpenClaw integration | Add feedback hooks |
| 5-6 | LanceDB deployment | Query cache + schema |
| 7-8 | Multi-instance | Feedback aggregation |
| 9-10 | TSA integration | Query expansion |
| 11-12 | Accumulate memories | Fact anchoring |
| 13-16 | Evaluation | A/B test learned vs baseline |

**Incremental:** Each component can be added independently without blocking existing work.

---

## Success Metrics (Additions to PRD)

| Metric | Baseline | Target | Method |
|--------|----------|--------|--------|
| Query Cache Hit Rate | 0% | >40% | Repeated query tracking |
| Retrieval-to-Usage Ratio | Unknown | >80% | Feedback tracking |
| Query Expansion Lift | 60% | >75% | A/B test expanded vs raw |
| Learning Curve Slope | 0 | Positive | Weekly improvement tracking |

---

## Risk Analysis

| Risk | Mitigation |
|------|------------|
| Cache pollution (wrong mappings cached) | Require 3+ successes before caching |
| Feedback latency (delayed signals) | Async processing, batch updates |
| Cold start (no learned data) | Fall back to pure embedding search |
| Compute overhead | Cache + expansion adds <50ms |

---

## Recommendation

**Add Learned Retrieval Layer as Phase 2.5** — between current implementation (Weeks 3-9) and evaluation (Weeks 10-16).

**Rationale:**
1. Addresses root cause of retrieval failures we measured today
2. Incremental addition, doesn't block existing work
3. Creates measurable improvement signal before Week 12 review
4. If RAG+learning fails, provides data for Phase 3 (LoRA) decision

---

## OpenClaw Plugin Architecture

### Slot-Based Replacement

The learned retrieval layer will be implemented as an OpenClaw memory plugin that replaces `memory-core`:

```jsonc
// openclaw.json
{
  "plugins": {
    "slots": {
      "memory": "memory-tribal"  // replaces "memory-core"
    }
  }
}
```

### Plugin Structure

```
~/.openclaw/extensions/memory-tribal/
├── index.ts              # Plugin entry point
├── manifest.json         # Plugin manifest
├── src/
│   ├── tools/
│   │   ├── memory-search.ts    # Enhanced search with learned layer
│   │   ├── memory-get.ts       # Compatible file read
│   │   └── memory-feedback.ts  # NEW: usage feedback
│   ├── learned/
│   │   ├── query-cache.ts      # Query→fact cache
│   │   ├── query-expander.ts   # Question→keyword expansion
│   │   ├── feedback-tracker.ts # Usage tracking
│   │   └── fact-anchoring.ts   # Bidirectional links
│   └── tribal-client.ts        # HTTP client to tribal-memory server
└── package.json
```

### Tool Interface (Drop-in Compatible)

```typescript
// memory_search - enhanced with learned layer
api.registerTool({
  name: "memory_search",
  description: "Search memory with learned retrieval",
  parameters: Type.Object({
    query: Type.String(),
    maxResults: Type.Optional(Type.Number()),
    minScore: Type.Optional(Type.Number()),
  }),
  async execute(id, params, context) {
    // 1. Check query cache for known-good mappings
    const cached = await queryCache.lookup(params.query);
    if (cached) return formatResults(cached);
    
    // 2. Expand query for better matching
    const expanded = await queryExpander.expand(params.query);
    
    // 3. Search with expanded queries
    const results = await tribalClient.search(expanded, params);
    
    // 4. Record retrieval for feedback tracking
    feedbackTracker.recordRetrieval(context.sessionId, params.query, results);
    
    return formatResults(results);
  }
});

// memory_feedback - NEW tool for recording usage
api.registerTool({
  name: "memory_feedback",
  description: "Record which memories were useful",
  parameters: Type.Object({
    sessionId: Type.String(),
    usedMemoryIds: Type.Array(Type.String()),
  }),
  async execute(id, params) {
    await feedbackTracker.recordUsage(params.sessionId, params.usedMemoryIds);
    return { success: true };
  }
}, { optional: true });
```

### A/B Testing Support

```jsonc
// Per-agent slot override for A/B testing
{
  "agents": {
    "list": [
      {
        "id": "main",
        "plugins": {
          "slots": { "memory": "memory-tribal" }  // Test group
        }
      },
      {
        "id": "control",
        "plugins": {
          "slots": { "memory": "memory-core" }    // Control group
        }
      }
    ]
  }
}
```

---

## Implementation Plan

### Phase 1: Basic Plugin (Week 3-4)
- [ ] Create plugin scaffold
- [ ] Implement `memory_search` passthrough to tribal-memory server
- [ ] Implement `memory_get` (compatible with memory-core)
- [ ] Test drop-in replacement

### Phase 2: Query Cache (Week 4-5)
- [ ] SQLite storage for query→fact mappings
- [ ] Cache hit logic with success threshold
- [ ] Cache invalidation on memory updates

### Phase 3: Query Expansion (Week 5-6)
- [ ] Rule-based expansion (question→keywords)
- [ ] Historical expansion from successful retrievals
- [ ] Optional LLM-based expansion

### Phase 4: Feedback Loop (Week 6-7)
- [ ] Retrieval tracking per session
- [ ] Usage signal collection
- [ ] Reinforce/penalize weight updates

### Phase 5: Fact Anchoring (Week 7-8)
- [ ] Anchor storage schema
- [ ] Automatic anchor generation from feedback
- [ ] Anchor-based retrieval path

---

## Next Steps

1. ✅ Complete baseline measurement (61.5% on full suite)
2. **NOW:** Create plugin scaffold and basic passthrough
3. Implement query cache
4. Measure improvement before expanding scope
