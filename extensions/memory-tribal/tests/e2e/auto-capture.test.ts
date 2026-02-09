/**
 * E2E tests for auto-capture (agent_end hook) and memory proxy tools.
 *
 * Tests against a REAL TribalMemory server — no mocks on the server side.
 *
 * Issue #147: Auto-capture + memory proxy tools tests
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { startTestServer, type TestEnvironment, rawPost } from "./setup";

let env: TestEnvironment;

beforeAll(async () => {
  env = await startTestServer();
}, 60_000);

afterAll(async () => {
  await env.cleanup();
}, 15_000);

// ============================================================================
// Auto-capture tests (agent_end hook behavior)
// ============================================================================

describe("Auto-capture (E2E)", () => {
  it("stores a captured conversation memory", async () => {
    const content = "User said they prefer React over Vue for frontend work";
    const result = await env.client.remember(content, {
      sourceType: "auto_capture",
      context: "Auto-captured from conversation (session: test-capture)",
    });

    expect(result.memoryId).toBeTruthy();

    // Verify retrievable
    const recalled = await rawPost<{ results: Array<{ memory: { content: string } }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "frontend framework preference React Vue", limit: 5 },
    );
    expect(recalled.status).toBe(200);
    expect(recalled.body.results.some((r) => r.memory.content.includes("React"))).toBe(true);
  });

  it("stores with auto_capture sourceType", async () => {
    const content = `auto-capture-type-test-${Date.now()}: the API uses REST not GraphQL`;
    const result = await env.client.remember(content, {
      sourceType: "auto_capture",
    });

    expect(result.memoryId).toBeTruthy();
    // sourceType "auto_capture" is a valid enum — should succeed without 422
  });

  it("rejects invalid sourceType", async () => {
    const response = await rawPost(env.baseUrl, "/v1/remember", {
      content: "test content for invalid source",
      source_type: "deliberate", // Invalid — the original bug
    });

    expect(response.status).toBe(422);
  });

  it("handles short/trivial content", async () => {
    // The plugin's shouldCapture() filters short content (<10 chars)
    // But the server itself should still accept it
    const result = await env.client.remember("ok", {
      sourceType: "auto_capture",
    });
    // Server accepts any content — filtering is plugin-side
    expect(result.memoryId).toBeTruthy();
  });

  it("handles long captured content", async () => {
    const longContent = "User discussed their architecture decision: ".padEnd(2000, "They want microservices with event sourcing. ");
    const result = await env.client.remember(longContent, {
      sourceType: "auto_capture",
      context: "Auto-captured from long conversation",
    });

    expect(result.memoryId).toBeTruthy();
  });

  it("deduplicates identical auto-captured content", async () => {
    const content = `dedup-capture-${Date.now()}: the deploy script uses Docker Compose`;

    const first = await env.client.remember(content, {
      sourceType: "auto_capture",
    });
    const second = await env.client.remember(content, {
      sourceType: "auto_capture",
    });

    expect(first.memoryId).toBeTruthy();
    expect(second.memoryId || second.duplicateOf).toBeTruthy();
  });

  it("captures multiple distinct memories from one conversation", async () => {
    const ts = Date.now();
    const memories = [
      `multi-${ts}: Joe's birthday is in October`,
      `multi-${ts}: The team standup is at 2pm Mountain Time`,
      `multi-${ts}: Use pytest for Python tests, vitest for TypeScript`,
    ];

    const ids: string[] = [];
    for (const content of memories) {
      const result = await env.client.remember(content, {
        sourceType: "auto_capture",
        context: "Auto-captured batch",
      });
      expect(result.memoryId).toBeTruthy();
      if (result.memoryId) ids.push(result.memoryId);
    }

    // All should be distinct
    const uniqueIds = new Set(ids);
    expect(uniqueIds.size).toBe(3);
  });
});

// ============================================================================
// Memory proxy tools: memory_search (proxies to /v1/recall)
// ============================================================================

describe("memory_search proxy (E2E)", () => {
  beforeAll(async () => {
    // Seed some memories for search tests
    const memories = [
      "The database uses PostgreSQL 15 in production",
      "Staging environment runs on port 3001",
      "CI/CD pipeline uses GitHub Actions with 4 parallel jobs",
      "The API rate limit is 100 requests per minute",
    ];

    for (const content of memories) {
      await env.client.remember(content, {
        sourceType: "user_explicit",
        tags: ["infrastructure"],
      });
    }
    await new Promise((r) => setTimeout(r, 500));
  });

  it("searches and returns relevant results", async () => {
    const recalled = await rawPost<{ results: Array<{ memory: { content: string }; similarity_score: number }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "what database do we use", limit: 5 },
    );

    expect(recalled.status).toBe(200);
    expect(recalled.body.results.length).toBeGreaterThan(0);
    expect(recalled.body.results.some((r) => r.memory.content.includes("PostgreSQL"))).toBe(true);
  });

  it("returns scores with results", async () => {
    const recalled = await rawPost<{ results: Array<{ similarity_score: number }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "database", limit: 5 },
    );

    expect(recalled.status).toBe(200);
    for (const r of recalled.body.results) {
      expect(typeof r.similarity_score).toBe("number");
      expect(r.similarity_score).toBeGreaterThan(0);
    }
  });

  it("filters by tags", async () => {
    const recalled = await rawPost<{ results: Array<{ memory: { content: string; tags: string[] } }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "infrastructure", limit: 10, tags: ["infrastructure"] },
    );

    expect(recalled.status).toBe(200);
    for (const r of recalled.body.results) {
      expect(r.memory.tags).toContain("infrastructure");
    }
  });
});

// ============================================================================
// Memory proxy tools: memory_feedback
// ============================================================================

describe("memory_feedback (E2E)", () => {
  it("server accepts recall requests (feedback is plugin-side)", async () => {
    // memory_feedback is tracked in the plugin, not the server
    // But the server should handle recall without issues
    const recalled = await rawPost<{ results: Array<{ memory: { id: string } }> }>(
      env.baseUrl,
      "/v1/recall",
      { query: "test feedback query", limit: 3 },
    );

    expect(recalled.status).toBe(200);
    // Each result should have a memory_id that could be used for feedback
    for (const r of recalled.body.results) {
      expect(r.memory.id).toBeTruthy();
    }
  });
});

// ============================================================================
// Memory proxy tools: memory_metrics (safeguard metrics)
// ============================================================================

describe("memory_metrics (E2E)", () => {
  it("server health endpoint responds", async () => {
    const res = await fetch(`${env.baseUrl}/v1/health`);
    expect(res.ok).toBe(true);
    const data = (await res.json()) as { status: string };
    expect(data.status).toBe("ok");
  });

  it("server stats endpoint responds", async () => {
    const res = await fetch(`${env.baseUrl}/v1/stats`);
    expect(res.ok).toBe(true);
    const data = (await res.json()) as { total_memories: number };
    expect(typeof data.total_memories).toBe("number");
    expect(data.total_memories).toBeGreaterThanOrEqual(0);
  });
});

// ============================================================================
// Token budget integration tests
// ============================================================================

describe("Token budget behavior (E2E)", () => {
  it("stores content of various sizes", async () => {
    // Token budget is enforced plugin-side (truncation before sending to server)
    // Verify server handles various content sizes
    const sizes = [10, 100, 500, 1000];

    for (const size of sizes) {
      const content = `token-budget-${size}-${Date.now()}: ${"a".repeat(size)}`;
      const result = await env.client.remember(content, {
        sourceType: "auto_capture",
      });
      expect(result.memoryId).toBeTruthy();
    }
  });
});

// ============================================================================
// Concurrent operations
// ============================================================================

describe("Concurrent operations (E2E)", () => {
  it("handles concurrent stores without errors", async () => {
    const promises = Array.from({ length: 10 }, (_, i) =>
      env.client.remember(`concurrent-store-${Date.now()}-${i}: item ${i}`, {
        sourceType: "auto_capture",
      }),
    );

    const results = await Promise.all(promises);
    for (const r of results) {
      expect(r.memoryId).toBeTruthy();
    }
  });

  it("handles concurrent recalls without errors", async () => {
    const queries = [
      "database",
      "API rate limit",
      "CI pipeline",
      "staging environment",
      "deployment",
    ];

    const promises = queries.map((q) =>
      rawPost<{ results: Array<unknown> }>(env.baseUrl, "/v1/recall", {
        query: q,
        limit: 3,
      }),
    );

    const results = await Promise.all(promises);
    for (const r of results) {
      expect(r.status).toBe(200);
      expect(Array.isArray(r.body.results)).toBe(true);
    }
  });

  it("handles mixed concurrent store + recall", async () => {
    const storePromises = Array.from({ length: 5 }, (_, i) =>
      env.client.remember(`mixed-${Date.now()}-${i}: mixed operation ${i}`, {
        sourceType: "auto_capture",
      }),
    );

    const recallPromises = Array.from({ length: 5 }, (_, i) =>
      rawPost<{ results: Array<unknown> }>(env.baseUrl, "/v1/recall", {
        query: `mixed operation ${i}`,
        limit: 3,
      }),
    );

    const allResults = await Promise.all([...storePromises, ...recallPromises]);
    // All should succeed without errors
    expect(allResults.length).toBe(10);
  });
});
