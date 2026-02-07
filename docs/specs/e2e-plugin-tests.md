# E2E Plugin Test Spec — memory-tribal

## Problem

The `tribal_store` tool shipped with `sourceType: "deliberate"` — an invalid enum value — and was broken since day one. No test caught it because all existing tests mock the HTTP layer. We need E2E tests that hit a real TribalMemory server.

## Scope

Test every plugin component against a real TribalMemory server instance:

| Component | Type | Priority |
|-----------|------|----------|
| `tribal_store` | Tool | P0 — was broken |
| `tribal_recall` | Tool | P0 |
| `before_agent_start` (auto-recall) | Hook | P0 |
| `agent_end` (auto-capture) | Hook | P1 |
| `/remember` command | Hook | P1 |
| `memory_search` | Tool (proxy) | P2 |
| `memory_get` | Tool (proxy) | P2 |
| `memory_feedback` | Tool | P2 |
| `memory_metrics` | Tool | P2 |
| CLI: `tribal-memory status` | CLI | P2 |
| CLI: `tribal-memory stats` | CLI | P2 |
| CLI: `tribal-memory search` | CLI | P2 |

## Architecture

```
tests/e2e/
├── setup.ts          # Start/stop TribalMemory server, shared fixtures
├── tribal-store.test.ts
├── tribal-recall.test.ts
├── auto-recall.test.ts
├── auto-capture.test.ts
├── remember-command.test.ts
├── memory-tools.test.ts      # memory_search, memory_get, feedback, metrics
└── cli-commands.test.ts
```

### Test Server
- Start a real `tribalmemory serve` on a random port before tests
- Fresh empty database per test suite (tmp directory)
- Config: `lazy_spacy: true` (fast), `host: 127.0.0.1`
- Tear down after all tests complete
- Use `beforeAll`/`afterAll` in Bun test runner

### Test Approach
- Import and call plugin tool `execute()` functions directly (not through OpenClaw)
- Use real `TribalClient` pointing at test server
- Verify both the tool response AND the server state (query back what was stored)

---

## Sprint: 5 Issues

### Issue #1: E2E test infrastructure + tribal_store tests (P0)
**Est: 2-3 hours**

Setup:
- [ ] `tests/e2e/setup.ts` — helper to spawn `tribalmemory serve` on random port, wait for health check, return client + cleanup function
- [ ] Fresh tmp database per suite
- [ ] `bun test` configuration for e2e directory

tribal_store tests:
- [ ] Store a memory → verify 200 + memory_id returned
- [ ] Store with tags → verify tags preserved on recall
- [ ] Store with context → verify context preserved
- [ ] Store duplicate → verify dedup response (duplicate_of)
- [ ] Store empty content → verify graceful error (not crash)
- [ ] Store with invalid sourceType → verify 422 + error message (regression test for the "deliberate" bug)
- [ ] Store with `"deliberate"` sourceType explicitly → verify it still fails with 422 (true e2e regression: sends the bad value to a real server, prevents re-introduction of the bug)
- [ ] Store very long content (10KB+) → verify success
- [ ] Verify stored memory retrievable via `/v1/recall`

### Issue #2: tribal_recall E2E tests (P0)
**Est: 2 hours**

Setup: seed server with 10+ diverse memories first.

Tests:
- [ ] Recall by semantic query → verify relevant results returned
- [ ] Recall with limit → verify respects limit param
- [ ] Recall with min_relevance → verify filters low scores
- [ ] Recall with tags filter → verify only tagged results
- [ ] Recall with after/before temporal filters → verify date filtering
- [ ] Recall empty query → verify graceful handling
- [ ] Recall with no matches → verify empty result + friendly message
- [ ] Recall against empty database → verify no crash
- [ ] Verify result format: score, snippet, tags all present

### Issue #3: Auto-recall + /remember hook E2E tests (P0/P1)
**Est: 3 hours**

Tests for `before_agent_start` hook:
- [ ] Send prompt → verify memories injected as `<relevant-memories>` XML
- [ ] Short prompt (<5 chars) → verify skipped (no recall)
- [ ] Emoji-only prompt → verify smart trigger skips it
- [ ] Verify circuit breaker trips after repeated empty recalls
- [ ] Verify circuit breaker resets after successful recall
- [ ] Verify session dedup suppresses duplicate injections within cooldown

Tests for `/remember` command:
- [ ] `/remember my favorite color is blue` → verify stored as user_explicit
- [ ] `/remember` with channel prefix `[Telegram Joe...] /remember X` → verify extracted correctly
- [ ] `/remember` with empty content → verify graceful handling
- [ ] Verify stored memory retrievable via recall after `/remember`
- [ ] Verify agent gets informed of storage (prependContext response)

### Issue #4: Auto-capture + memory proxy tools E2E tests (P1/P2)
**Est: 2-3 hours**

Auto-capture (`agent_end` hook):
- [ ] Simulate agent_end with conversation → verify memory auto-captured
- [ ] Verify sourceType is "auto_capture"
- [ ] Short/trivial conversation → verify smart trigger skips capture
- [ ] Verify session dedup prevents re-capturing same content
- [ ] Verify token budget limits captured content size

Memory proxy tools:
- [ ] `memory_search` → verify proxies to TribalMemory recall
- [ ] `memory_get` → verify reads memory file content
- [ ] `memory_feedback` → verify records feedback (used/unused paths)
- [ ] `memory_metrics` → verify returns safeguard metrics

### Issue #5: CLI commands + error handling E2E tests (P2)
**Est: 1-2 hours**

CLI commands:
- [ ] `tribal-memory status` → verify shows connection info
- [ ] `tribal-memory stats` → verify shows memory count + breakdown
- [ ] `tribal-memory search <query>` → verify returns results with scores

Error handling (all tools):
- [ ] Server unreachable → verify circuit breaker + graceful error (not crash)
- [ ] Server returns 500 → verify error message propagated
- [ ] Malformed response → verify doesn't throw unhandled exception
- [ ] Concurrent requests → verify no race conditions

---

## Enum Validation Guard (Bonus)

Add a compile-time or runtime check that `sourceType` values match the server's `SourceType` enum:

```typescript
// In tribal-client.ts or types.ts
const VALID_SOURCE_TYPES = [
  "user_explicit", "auto_capture", "correction",
  "cross_instance", "legacy", "unknown",
] as const;
type SourceType = typeof VALID_SOURCE_TYPES[number];
```

This prevents future "deliberate"-style bugs at the TypeScript level.

---

## Definition of Done

- All P0 tests pass against a real TribalMemory server
- Tests run in CI (can be gated behind a `--e2e` flag since they need a server)
- No mocked HTTP — every test hits real endpoints
- Each test verifies both the tool response AND the server state
- Invalid enum values are caught at compile time (TypeScript) not runtime (422)
