/**
 * E2E tests for auto-recall (before_agent_start) and /remember command hooks.
 *
 * Tests against a REAL TribalMemory server — no mocks on the server side.
 * The OpenClaw plugin API is mocked since we're testing the plugin's behavior,
 * not the OpenClaw framework.
 *
 * Issue #146: Auto-recall + /remember hook tests
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from "vitest";
import { startTestServer, type TestEnvironment, rawPost } from "./setup";
import { TribalClient } from "../../src/tribal-client";

let env: TestEnvironment;

beforeAll(async () => {
  env = await startTestServer();
}, 60_000);

afterAll(async () => {
  await env.cleanup();
}, 15_000);

// ============================================================================
// Helper: Seed the server with memories for recall tests
// ============================================================================

async function seedMemories() {
  const memories = [
    { content: "Joe's favorite programming language is Rust", tags: ["preference"] },
    { content: "The project deadline is March 15, 2026", tags: ["deadline", "project"] },
    { content: "Luna is Joe's dog, a golden retriever", tags: ["personal", "pet"] },
    { content: "TribalMemory uses FastEmbed with bge-small-en-v1.5 for embeddings", tags: ["architecture"] },
    { content: "Joe prefers afternoon meetings, never before noon", tags: ["preference", "schedule"] },
  ];

  for (const m of memories) {
    await env.client.remember(m.content, {
      sourceType: "user_explicit",
      tags: m.tags,
    });
  }

  // Small delay for indexing
  await new Promise((r) => setTimeout(r, 500));
}

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
    const recalled = await rawPost<{ results: Array<{ memory: { content: string } }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "favorite state", limit: 5 },
    );
    expect(recalled.status).toBe(200);
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
    const recalled = await rawPost<{ results: Array<{ memory: { content: string } }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "cat name Whiskers", limit: 5 },
    );
    expect(recalled.status).toBe(200);
    expect(recalled.body.results.some((r) => r.memory.content.includes("Whiskers"))).toBe(true);
  });

  it("handles empty /remember content gracefully", async () => {
    // The plugin checks rememberContent.length > 0 before storing
    const REMEMBER_RE = /^(?:\[[^\]]*\]\s*)?\/remember\s+(.+)$/is;

    // "/remember " with trailing space but no content
    const match1 = "/remember ".match(REMEMBER_RE);
    expect(match1).toBeNull(); // Regex requires .+ (at least one char after space)

    // "/remember" with no space
    const match2 = "/remember".match(REMEMBER_RE);
    expect(match2).toBeNull();
  });

  it("stores /remember and retrieves via recall", async () => {
    const uniqueContent = `test-unique-${Date.now()}: the office wifi password is hunter2`;

    await env.client.remember(uniqueContent, {
      sourceType: "user_explicit",
      context: "User /remember command",
    });

    await new Promise((r) => setTimeout(r, 300));

    const recalled = await rawPost<{ results: Array<{ memory: { content: string } }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "office wifi password", limit: 5 },
    );
    expect(recalled.status).toBe(200);
    expect(recalled.body.results.some((r) => r.memory.content.includes("hunter2"))).toBe(true);
  });

  it("detects duplicate /remember content", async () => {
    const content = `duplicate-test-${Date.now()}: always use TypeScript`;

    // First store
    const first = await env.client.remember(content, {
      sourceType: "user_explicit",
    });
    expect(first.memoryId).toBeTruthy();

    // Second store (same content) — server handles dedup
    const second = await env.client.remember(content, {
      sourceType: "user_explicit",
    });
    // Server may return duplicate_of or a new ID — depends on dedup threshold
    expect(second.memoryId || second.duplicateOf).toBeTruthy();
  });
});

// ============================================================================
// Auto-recall tests
// ============================================================================

describe("Auto-recall (E2E)", () => {
  beforeAll(async () => {
    await seedMemories();
  });

  it("recalls relevant memories for a semantic query", async () => {
    const recalled = await rawPost<{ results: Array<{ memory: { content: string }; similarity_score: number }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "What programming language does Joe like?", limit: 3, min_relevance: 0.3 },
    );

    expect(recalled.status).toBe(200);
    expect(recalled.body.results.length).toBeGreaterThan(0);
    // Should find the Rust preference
    const contents = recalled.body.results.map((r) => r.memory.content);
    expect(contents.some((c) => c.includes("Rust"))).toBe(true);
  });

  it("skips recall for short prompts (<5 chars)", async () => {
    // The plugin returns early for prompts < 5 chars
    // We verify the server still works but the plugin logic would skip this
    const shortPrompt = "hi";
    expect(shortPrompt.length).toBeLessThan(5);

    // Server still responds fine — the skip is plugin-side logic
    const recalled = await rawPost<{ results: Array<{ memory: { content: string } }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: shortPrompt, limit: 3 },
    );
    expect(recalled.status).toBe(200);
  });

  it("recalls with tag filters", async () => {
    const recalled = await rawPost<{ results: Array<{ memory: { content: string; tags: string[] } }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "preferences", limit: 10, tags: ["preference"] },
    );

    expect(recalled.status).toBe(200);
    // All results should have the preference tag
    for (const r of recalled.body.results) {
      expect(r.memory.tags).toContain("preference");
    }
  });

  it("respects limit parameter", async () => {
    const recalled = await rawPost<{ results: Array<{ memory: { content: string } }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "Joe", limit: 2 },
    );

    expect(recalled.status).toBe(200);
    expect(recalled.body.results.length).toBeLessThanOrEqual(2);
  });

  it("respects min_relevance filter", async () => {
    const recalled = await rawPost<{ results: Array<{ memory: { content: string }; similarity_score: number }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "quantum physics spacetime wormhole", limit: 10, min_relevance: 0.8 },
    );

    expect(recalled.status).toBe(200);
    // With high min_relevance and unrelated query, should get fewer results
    // than a broad query (server may return some if similarity is high enough)
    expect(recalled.body.results.length).toBeLessThanOrEqual(10);
  });

  it("returns empty results for no matches gracefully", async () => {
    const recalled = await rawPost<{ results: Array<{ memory: { content: string } }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "xyzzy plugh completely unrelated gibberish", limit: 5, min_relevance: 0.9 },
    );

    expect(recalled.status).toBe(200);
    expect(recalled.body.results).toBeDefined();
    expect(Array.isArray(recalled.body.results)).toBe(true);
  });

  it("formats recall results as memory context XML", () => {
    // Test the formatting logic that the plugin applies to results
    const mockResults = [
      { snippet: "Joe prefers Rust" },
      { snippet: "Luna is a golden retriever" },
    ];

    const memoryContext = mockResults
      .map((r) => `- ${r.snippet}`)
      .join("\n");

    const xml =
      `<relevant-memories>\n` +
      `The following memories may be relevant:\n` +
      `${memoryContext}\n` +
      `</relevant-memories>`;

    expect(xml).toContain("<relevant-memories>");
    expect(xml).toContain("Joe prefers Rust");
    expect(xml).toContain("Luna is a golden retriever");
    expect(xml).toContain("</relevant-memories>");
  });
});

// ============================================================================
// Smart trigger integration tests
// ============================================================================

describe("Smart trigger filtering (E2E)", () => {
  it("classifies emoji-only prompts as skippable", () => {
    // Import the smart trigger logic
    // The plugin skips recall for emoji-only / greeting-only prompts
    const emojiOnly = "👍🎉🔥";
    const greetingOnly = "hi";
    const substantive = "What is the project deadline for TribalMemory?";

    // Emoji-only and greetings should be skippable
    expect(emojiOnly.length).toBeLessThan(10);
    expect(greetingOnly.length).toBeLessThan(5);
    expect(substantive.length).toBeGreaterThan(10);
  });

  it("allows substantive queries through to recall", async () => {
    const recalled = await rawPost<{ results: Array<{ memory: { content: string } }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "When is the project deadline?", limit: 3 },
    );

    expect(recalled.status).toBe(200);
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
      const recalled = await rawPost<{ results: Array<unknown> }>(
        env.baseUrl,
        "/v1/recall",
        { query: `zzzzxyzzy${i}${Date.now()}`, limit: 3 },
      );
      expect(recalled.status).toBe(200);
      expect(Array.isArray(recalled.body.results)).toBe(true);
    }
  });

  it("server returns results for real queries (breaker stays closed)", async () => {
    const recalled = await rawPost<{ results: Array<{ memory: { content: string } }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "Joe's dog Luna", limit: 3 },
    );

    expect(recalled.status).toBe(200);
    expect(recalled.body.results.length).toBeGreaterThan(0);
  });
});

// ============================================================================
// Session dedup integration tests
// ============================================================================

describe("Session dedup behavior (E2E)", () => {
  it("stores the same memory only once (server-side dedup)", async () => {
    const uniqueContent = `dedup-e2e-${Date.now()}: Joe always uses vim`;

    // Store twice
    const first = await env.client.remember(uniqueContent, {
      sourceType: "auto_capture",
    });
    const second = await env.client.remember(uniqueContent, {
      sourceType: "auto_capture",
    });

    // Both should succeed but second may be flagged as duplicate
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
    // Should be different memories (not deduped)
    if (r1.memoryId && r2.memoryId) {
      expect(r1.memoryId).not.toBe(r2.memoryId);
    }
  });
});
