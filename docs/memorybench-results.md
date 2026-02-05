# MemoryBench Results — TribalMemory v0.3.0

**Date:** 2026-02-05
**Benchmark:** LoCoMo (Long Conversation Memory)
**Provider:** TribalMemory via HTTP API
**Questions:** 10 (sampled from 1986 total)

## Summary

| Metric | Value |
|--------|-------|
| **Accuracy** | 0% (0/10) |
| **Hit@10** | 10% |
| **MRR** | 0.020 |
| **Precision@10** | 1.3% |
| **Recall@10** | 10% |

## Latency

| Phase | Median | p95 | p99 |
|-------|--------|-----|-----|
| Ingest | 184s | 216s | 216s |
| Search | 442ms | 500ms | 500ms |
| Answer | 539ms | 2632ms | 2632ms |
| **Total** | 186s | 218s | 218s |

## By Question Type

| Type | Total | Correct | Hit@10 |
|------|-------|---------|--------|
| multi-hop | 6 | 0 | 17% |
| temporal | 1 | 0 | 0% |
| single-hop | 3 | 0 | 0% |

## Failure Analysis

### Issue 1: Metadata Parsing Bug 🐛

The TribalMemory provider stores context as a JSON string, but the search method spreads it without parsing:

```typescript
// Bug: spreading a string gives character-by-character keys
metadata: {
  ...r.memory.context,  // "0": "{", "1": "\"", ...
}
```

**Impact:** LLM sees corrupted context, can't extract dates/metadata properly.

**Fix:** Parse context before spreading:
```typescript
...(r.memory.context ? JSON.parse(r.memory.context) : {})
```

### Issue 2: Temporal Reasoning Failure

**Question 0:** "When did Caroline go to the LGBTQ support group?"
- Ground truth: "7 May 2023"
- Model answer: "May 8, 2023" ← One day off!

The relevant memory says "I went to a LGBTQ support group **yesterday**" with timestamp `2023-05-08`. The model used the message date instead of reasoning that "yesterday" = May 7.

**Impact:** Even correct retrieval fails if dates are relative.

**Fix consideration:** Store absolute resolved dates, or include date context more explicitly.

### Issue 3: Cross-Question Isolation

Each question uses a separate `containerTag` for isolation. This means:
- q0's memories only match q0's searches
- 9/10 questions had empty search results because their container had no memories

LoCoMo expects all prior sessions to be available for multi-hop reasoning across the full conversation history.

**Fix consideration:** Ingest all sessions into a shared container, or remove container filtering.

### Issue 4: FTS5 Syntax Errors

Queries with punctuation (`?`, `'`, `,`) cause BM25 search failures:
```
FTS5 search error: fts5: syntax error near "?"
```

**Tracked:** Issue #56

## Comparison Context

- **Mem0 on LoCoMo:** ~15-25% accuracy (reported)
- **TribalMemory:** 0% (but retrieval is partially working)

The 0% is largely due to the metadata bug corrupting context. With fixes, we expect significant improvement.

## Next Steps

1. [ ] Fix metadata parsing bug in provider
2. [ ] Fix FTS5 query escaping (Issue #56)
3. [ ] Re-run benchmark after fixes
4. [ ] Consider shared container mode for full-context benchmarks
5. [ ] Add temporal resolution (relative → absolute dates)

## Raw Data

Run ID: `run-20260205-022627`
Report: `memorybench/data/runs/run-20260205-022627/report.json`
