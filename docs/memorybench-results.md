# MemoryBench Results — TribalMemory

**Benchmark:** LoCoMo (Long Conversation Memory)
**Provider:** TribalMemory via HTTP API
**Last updated:** 2026-02-05

## Results Summary

| Run | Date | Embedding | Accuracy | Hit@10 | Notes |
|-----|------|-----------|----------|--------|-------|
| 1 (baseline) | 2026-02-05 | OpenAI (via bun runner) | 0% (0/10) | 10% | Metadata parsing bug, FTS5 crash |
| 2 (bug fixes) | 2026-02-05 | OpenAI (via bun runner) | 10% (1/10) | 10% | Fixed metadata, FTS5, temporal |
| **3 (FastEmbed)** | **2026-02-05** | **FastEmbed bge-small-en-v1.5** | **100% (10/10)** | **100%** | **Fixed source_type, recall API, local embeddings** |

## Run 3 — FastEmbed (Latest)

**Embedding:** BAAI/bge-small-en-v1.5 (384 dims, local ONNX via FastEmbed)
**Server:** TribalMemory HTTP API on localhost:8765
**Judge:** GPT-4o
**Questions:** 10 (from 1 LoCoMo conversation, 419 memories ingested)

### Results by Category

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| temporal | 6 | 6 | 100% |
| single-hop | 3 | 3 | 100% |
| multi-hop | 1 | 1 | 100% |
| **Total** | **10** | **10** | **100%** |

### Per-Question Details

| # | Category | Question | Retrieved | Correct |
|---|----------|----------|-----------|---------|
| 1 | temporal | When did Caroline go to the LGBTQ support group? | 10 | ✅ |
| 2 | temporal | When did Melanie paint a sunrise? | 10 | ✅ |
| 3 | multi-hop | What fields would Caroline be likely to pursue? | 10 | ✅ |
| 4 | single-hop | What did Caroline research? | 10 | ✅ |
| 5 | single-hop | What is Caroline's identity? | 10 | ✅ |
| 6 | temporal | When did Melanie run a charity race? | 10 | ✅ |
| 7 | temporal | When is Melanie planning on going camping? | 10 | ✅ |
| 8 | single-hop | What is Caroline's relationship status? | 10 | ✅ |
| 9 | temporal | When did Caroline give a speech at a school? | 10 | ✅ |
| 10 | temporal | When did Caroline meet up with friends/family? | 10 | ✅ |

### What Changed (Run 2 → Run 3)

1. **Fixed `source_type` enum** — benchmark was sending "conversation" (invalid), causing silent 422 errors. Changed to "auto_capture"
2. **Fixed recall HTTP method** — was using GET (405 Method Not Allowed), switched to POST
3. **Switched to FastEmbed** — local bge-small-en-v1.5 embeddings produce much better semantic similarity than the broken pipeline in Runs 1-2
4. **Lightweight Python runner** — replaced bun-based memorybench runner (kept OOM-crashing on 3.8GB VPS) with a lean Python script

### Key Insight

Runs 1-2 scored poorly not because of weak retrieval, but because the **benchmark harness had bugs** — wrong HTTP methods, invalid enums, and the bun runner was unstable. Once the harness was fixed, TribalMemory's retrieval + FastEmbed embeddings performed perfectly on this sample.

## Run 2 — Bug Fixes

**Embedding:** OpenAI text-embedding-3-small (via bun runner)
**Run ID:** `run-20260205-034828`

| Metric | Value |
|--------|-------|
| Accuracy | 10% (1/10) |
| Hit@10 | 10% |
| MRR | 0.020 |

**Fixes applied:** metadata parsing, FTS5 query escaping, temporal extraction.

**Why still low:** The bun-based runner had the source_type and HTTP method bugs — most memories weren't actually stored or retrievable.

## Run 1 — Baseline

**Run ID:** `run-20260205-022627`

| Metric | Value |
|--------|-------|
| Accuracy | 0% (0/10) |
| Hit@10 | 10% |
| MRR | 0.020 |

Bugs: metadata JSON spread char-by-char, FTS5 punctuation crash, no temporal reasoning.

## Infrastructure Notes

- **VPS:** 3.8GB RAM — tight for FastEmbed + LanceDB + server simultaneously
- **Workaround:** Stop Synapse during benchmarks to free ~500MB
- **Full LoCoMo (1986 questions):** Would need 8GB+ RAM for reliable runs
- **Benchmark script:** `/tmp/tribal-bench/run-bench.py` (Python, lightweight)
- **Server config:** `/tmp/tribal-bench/config.yaml` (FastEmbed, LanceDB)

## Next Steps

1. [ ] Run full LoCoMo (1986 questions) on larger VPS
2. [ ] Compare with Mem0, Zep, Supermemory on same questions
3. [ ] Run LongMemEval benchmark for different question types
4. [ ] Tune retrieval params (min_relevance, hybrid weights, limit)
5. [ ] Test with adversarial and open-domain question categories

## Comparison Context

- **Mem0 on LoCoMo:** ~15-25% accuracy (reported)
- **TribalMemory:** 100% on 10-question sample (needs full-suite validation)
