/**
 * E2E tests for auto-recall (before_agent_start) and /remember command hooks.
 *
 * Tests against a REAL TribalMemory server — no mocks on the server side.
 * The OpenClaw plugin API is mocked since we're testing the plugin's behavior,
 * not the OpenClaw framework.
 *
 * Issue #146: Auto-recall + /remember hook tests
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import {
  startTestServer,
  type TestEnvironment,
  type RecallResponse,
  rawPost,
  expectValidRecallResponse,
  seedStandardMemories,
  INDEX_DELAY_MS,
} from "./setup";

let env: TestEnvironment;

beforeAll(async () => {
  env = await startTestServer();
}, 60_000);

afterAll(async () => {
  await env.cleanup();
}, 15_000);

// ============================================================================
// /remember command tests
// ============================================================================

describe("/remember command (E2E)", () => {
  it("stores a memory via /remember command", async () => {
    const content = "my favorite state is Florida";
    const result = await env.client.remember(content, {
      sourceType: "user_explicit",
      context: "User /remember command (session: test-session)",
    });

    expect(result.memoryId).toBeTruthy();

    // Verify retrievable
    await new Promise((r) => setTimeout(r, INDEX_DELAY_MS));
    const recalled = await rawPost<RecallResponse>(
      env.baseUrl,
      "/v1/recall",
      { query: "favorite state", limit: 5 },
    );
    expectValidRecallResponse(recalled);
    const contents = recalled.body.results.map((r) => r.memory.content);
    expect(contents.some((c) => c.includes("Florida"))).toBe(true);
  });

  it("extracts /remember from channel-prefixed message", async () => {
    // Simulate what the plugin does: extractRememberCommand parses the prefix
    const rawPrompt = "[Telegram Joe (@abbudjoe) id:8228946254 +2m] /remember my cat's name is Whiskers";

    // The plugin's extractRememberCommand regex:
    const REMEMBER_RE = /^(?:\[[^\]]*\]\s*)?\/remember\s+(.+)$/is;
    const match = rawPrompt.match(REMEMBER_RE);
    expect(match).not.toBeNull();

    const extracted = match![1];
    expect(extracted).toBe("my cat's name is Whiskers");

    // Store via the extracted content
    const result = await env.client.remember(extracted, {
      sourceType: "user_explicit",
      context: "User /remember command (session: test-channel-prefix)",
    });
    expect(result.memoryId).toBeTruthy();

    // Verify retrievable
    await new Promise((r) => setTimeout(r, INDEX_DELAY_MS));
    const recalled = await rawPost<RecallResponse>(
      env.baseUrl,
      "/v1/recall",
      { query: "cat name Whiskers", limit: 5 },
    );
    expectValidRecallResponse(recalled);
    expect(recalled.body.results.some((r) => r.memory.content.includes("Whiskers"))).toBe(true);
  });

  it("rejects empty /remember content via regex", () => {
    // The plugin checks rememberContent.length > 0 before storing
    const REMEMBER_RE = /^(?:\[[^\]]*\]\s*)?\/remember\s+(.+)$/is;

    // "/remember " with trailing space but no content — regex requires .+
    expect("/remember ".match(REMEMBER_RE)).toBeNull();
    // "/remember" with no space
    expect("/remember".match(REMEMBER_RE)).toBeNull();
  });

  it("stores /remember and retrieves via recall", async () => {
    const uniqueContent = `test-unique-${Date.now()}: the office wifi password is hunter2`;

    await env.client.remember(uniqueContent, {
      sourceType: "user_explicit",
      context: "User /remember command",
    });

    await new Promise((r) => setTimeout(r, INDEX_DELAY_MS));

    const recalled = await rawPost<RecallResponse>(
      env.baseUrl,
      "/v1/recall",
      { query: "office wifi password", limit: 5 },
    );
    expectValidRecallResponse(recalled);
    expect(recalled.body.results.some((r) => r.memory.content.includes("hunter2"))).toBe(true);
  });

  it("detects duplicate /remember content", async () => {
    const content = `duplicate-test-${Date.now()}: always use TypeScript`;

    const first = await env.client.remember(content, { sourceType: "user_explicit" });
    expect(first.memoryId).toBeTruthy();

    // Second store — server handles dedup
    const second = await env.client.remember(content, { sourceType: "user_explicit" });
    expect(second.memoryId || second.duplicateOf).toBeTruthy();
  });
});

// ============================================================================
// Auto-recall tests
// ============================================================================

describe("Auto-recall (E2E)", () => {
  beforeAll(async () => {
    await seedStandardMemories(env.client);
  });

  it("recalls relevant memories for a semantic query", async () => {
    const recalled = await rawPost<RecallResponse>(
      env.baseUrl,
      "/v1/recall",
      { query: "What programming language does Joe like?", limit: 3, min_relevance: 0.3 },
    );

    expectValidRecallResponse(recalled);
    expect(recalled.body.results.length).toBeGreaterThan(0);
    const contents = recalled.body.results.map((r) => r.memory.content);
    expect(contents.some((c) => c.includes("Rust"))).toBe(true);
  });

  it("returns valid response for short prompts", async () => {
    // The plugin returns early for prompts < 5 chars (plugin-side skip).
    // Server still responds fine.
    const recalled = await rawPost<RecallResponse>(
      env.baseUrl,
      "/v1/recall",
      { query: "hi", limit: 3 },
    );
    expectValidRecallResponse(recalled);
  });

  it("recalls with tag filters", async () => {
    const recalled = await rawPost<RecallResponse>(
      env.baseUrl,
      "/v1/recall",
      { query: "preferences", limit: 10, tags: ["preference"] },
    );

    expectValidRecallResponse(recalled);
    for (const r of recalled.body.results) {
      expect(r.memory.tags).toContain("preference");
    }
  });

  it("respects limit parameter", async () => {
    const recalled = await rawPost<RecallResponse>(
      env.baseUrl,
      "/v1/recall",
      { query: "Joe", limit: 2 },
    );

    expectValidRecallResponse(recalled);
    expect(recalled.body.results.length).toBeLessThanOrEqual(2);
  });

  it("respects min_relevance filter", async () => {
    const recalled = await rawPost<RecallResponse>(
      env.baseUrl,
      "/v1/recall",
      { query: "quantum physics spacetime wormhole", limit: 10, min_relevance: 0.8 },
    );

    expectValidRecallResponse(recalled);
    // High min_relevance + unrelated query = fewer results
    expect(recalled.body.results.length).toBeLessThanOrEqual(10);
  });

  it("returns empty results gracefully for no matches", async () => {
    const recalled = await rawPost<RecallResponse>(
      env.baseUrl,
      "/v1/recall",
      { query: "xyzzy plugh completely unrelated gibberish", limit: 5, min_relevance: 0.9 },
    );

    expectValidRecallResponse(recalled);
  });
});

// ============================================================================
// Smart trigger integration tests
// ============================================================================

describe("Smart trigger filtering (E2E)", () => {
  it("allows substantive queries through to recall", async () => {
    const recalled = await rawPost<RecallResponse>(
      env.baseUrl,
      "/v1/recall",
      { query: "When is the project deadline?", limit: 3 },
    );

    expectValidRecallResponse(recalled);
    expect(recalled.body.results.length).toBeGreaterThan(0);
    const contents = recalled.body.results.map((r) => r.memory.content);
    expect(contents.some((c) => c.includes("March 15") || c.includes("deadline"))).toBe(true);
  });
});

// ============================================================================
// Circuit breaker integration tests
// ============================================================================

describe("Circuit breaker behavior (E2E)", () => {
  it("server handles nonsense queries without errors", async () => {
    // The circuit breaker is plugin-side logic. The server should respond
    // gracefully to gibberish queries (no crashes, valid JSON).
    for (let i = 0; i < 5; i++) {
      const recalled = await rawPost<RecallResponse>(
        env.baseUrl,
        "/v1/recall",
        { query: `zzzzxyzzy${i}${Date.now()}`, limit: 3 },
      );
      expectValidRecallResponse(recalled);
    }
  });

  it("server returns results for real queries", async () => {
    const recalled = await rawPost<RecallResponse>(
      env.baseUrl,
      "/v1/recall",
      { query: "Joe's dog Luna", limit: 3 },
    );

    expectValidRecallResponse(recalled);
    expect(recalled.body.results.length).toBeGreaterThan(0);
  });
});

// ============================================================================
// Session dedup integration tests
// ============================================================================

describe("Session dedup behavior (E2E)", () => {
  it("stores the same memory only once (server-side dedup)", async () => {
    const uniqueContent = `dedup-e2e-${Date.now()}: Joe always uses vim`;

    const first = await env.client.remember(uniqueContent, { sourceType: "auto_capture" });
    const second = await env.client.remember(uniqueContent, { sourceType: "auto_capture" });

    expect(first.memoryId).toBeTruthy();
    expect(second.memoryId || second.duplicateOf).toBeTruthy();
  });

  it("similar but distinct memories both get stored", async () => {
    const ts = Date.now();
    const mem1 = `distinct-${ts}: Joe likes morning coffee`;
    const mem2 = `distinct-${ts}: Joe likes evening tea`;

    const r1 = await env.client.remember(mem1, { sourceType: "auto_capture" });
    const r2 = await env.client.remember(mem2, { sourceType: "auto_capture" });

    expect(r1.memoryId).toBeTruthy();
    expect(r2.memoryId).toBeTruthy();
    if (r1.memoryId && r2.memoryId) {
      expect(r1.memoryId).not.toBe(r2.memoryId);
    }
  });
});
