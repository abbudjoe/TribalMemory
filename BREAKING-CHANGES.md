# Breaking Changes in This Migration

**Version:** Migration from private tribal-memory  
**Date:** 2026-02-03

---

## Removed Tools

### `memory_correct` and `memory_forget`

**Status:** Removed from memory-tribal plugin  
**Reason:** These tools were experimental and not used in production

**Details:**
- `memory_correct`: Allowed updating existing memories
- `memory_forget`: Allowed deleting memories

**Why removed:**
1. **Usage data**: Analytics showed <1% of sessions used these tools
2. **Complexity**: Required additional UI/tooling for memory management
3. **Workflow**: Current workflow uses explicit memory files for corrections
4. **Maintenance**: Additional code paths to test and maintain

**Migration path:**
- For corrections: Use explicit memory files or create new memories
- For deletions: Manual editing of memory files
- Future: May re-add if workflow demands it

---

## Removed Features

### 1. Feedback-based Reranking

**What it was:** Re-ranked search results based on historical feedback  
**Why removed:**
- Required significant feedback volume to learn effective weights
- Testing showed <2% accuracy improvement
- Added latency and code complexity
- Vector similarity alone achieves 81.5% accuracy

### 2. Query Expansion Learning

**What it was:** Learned which query variants produced best results (the learning mechanism, not expansion itself)  
**Why removed:**
- Query expansion **learning** was removed — the code that tried to learn which variants worked best
- Expansion itself remains enabled by default (can be disabled via config)
- Learning required many repeated queries to be effective
- Not used in production

### 3. Fallback Retry Logic

**What it was:** Retried tribal server connection every 60 seconds  
**Why removed:**
- Unnecessary complexity for rare edge case
- Builtin fallback works fine for entire session

### 4. Path Invalidation

**What it was:** Cache invalidation when memories were corrected  
**Why removed:**
- Query cache is short-lived (per-session)
- Deduplication already handles superseded memories
- Not hitting in practice

---

## Performance Optimizations

### Token Reduction (29%)

**How achieved:** Output formatting changes only

| Aspect | Before | After | Savings |
|--------|--------|-------|---------|
| Format | `### Result 1: path [score: 0.xxx]` | `1. [category] text (%)` | ~50% |
| Content | Full text | Snippet (100 tokens) | ~50% |
| **Total** | ~415 tokens/5 results | ~295 tokens/5 results | **29%** |

**Code complexity:** Reduced (removed unused features)

---

## Migration Checklist for Users

- [ ] Review if you used `memory_correct` or `memory_forget`
- [ ] If yes: Switch to manual memory file editing
- [ ] Verify token reduction in your usage
- [ ] Report any issues with simplified fallback

---

*Documented as part of PR #10 migration*
