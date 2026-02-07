/**
 * E2E tests for error handling and edge cases.
 *
 * Validates that TribalMemory server handles errors gracefully across all
 * endpoints: malformed requests, validation errors, edge cases, and
 * concurrent operations.
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import {
  startTestServer,
  rawPost,
  type RecallResult,
  type TestEnvironment,
} from "./setup";

describe("Error Handling & Edge Cases E2E", () => {
  let env: TestEnvironment;

  beforeAll(async () => {
    env = await startTestServer();
  }, 60000);

  afterAll(async () => {
    await env.cleanup();
  }, 10000);

  // ==========================================================================
  // Server Error Handling
  // ==========================================================================

  it("should handle server unreachable gracefully", async () => {
    const wrongPort = env.port + 1000; // Wrong port
    const wrongUrl = `http://127.0.0.1:${wrongPort}`;

    // Should throw a clean error (ECONNREFUSED, timeout, etc.)
    await expect(
      fetch(`${wrongUrl}/v1/health`, {
        signal: AbortSignal.timeout(2000),
      }),
    ).rejects.toThrow();
  });

  it("should reject malformed JSON to /v1/remember with 400 or 422", async () => {
    const res = await fetch(`${env.baseUrl}/v1/remember`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{invalid json payload",
    });

    expect([400, 422]).toContain(res.status);
  });

  it("should reject malformed JSON to /v1/recall with 400 or 422", async () => {
    const res = await fetch(`${env.baseUrl}/v1/recall`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: '{"query": "test", invalid}',
    });

    expect([400, 422]).toContain(res.status);
  });

  it("should reject missing required field 'content' on /v1/remember with 422", async () => {
    const res = await rawPost(env.baseUrl, "/v1/remember", {
      source_type: "user_explicit",
      // Missing 'content' field
    });

    expect(res.status).toBe(422);
    expect(res.body.detail).toBeDefined();
  });

  it("should reject missing required field 'query' on /v1/recall with 422", async () => {
    const res = await rawPost(env.baseUrl, "/v1/recall", {
      limit: 5,
      // Missing 'query' field
    });

    expect(res.status).toBe(422);
    expect(res.body.detail).toBeDefined();
  });

  it("should ignore extra unknown fields on /v1/remember", async () => {
    const res = await rawPost(env.baseUrl, "/v1/remember", {
      content: "Test memory with extra fields",
      source_type: "user_explicit",
      unknown_field: "should be ignored",
      another_unknown: 12345,
    });

    // Should succeed, ignoring unknown fields
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.memory_id).toBeDefined();
  });

  it("should ignore extra unknown fields on /v1/recall", async () => {
    const res = await rawPost(env.baseUrl, "/v1/recall", {
      query: "test query",
      limit: 5,
      extra_field: "ignored",
      random_number: 999,
    });

    // Should succeed, ignoring unknown fields
    expect(res.status).toBe(200);
    expect(res.body.results).toBeDefined();
  });

  it("should reject GET request to /v1/remember with 405", async () => {
    const res = await fetch(`${env.baseUrl}/v1/remember`, {
      method: "GET",
    });

    expect(res.status).toBe(405);
  });

  it("should reject GET request to /v1/recall with 405", async () => {
    const res = await fetch(`${env.baseUrl}/v1/recall`, {
      method: "GET",
    });

    expect(res.status).toBe(405);
  });

  it("should handle Content-Type text/plain with appropriate error", async () => {
    const res = await fetch(`${env.baseUrl}/v1/remember`, {
      method: "POST",
      headers: { "Content-Type": "text/plain" },
      body: "This is plain text, not JSON",
    });

    // Should reject with 4xx error (likely 400 or 422)
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(res.status).toBeLessThan(500);
  });

  // ==========================================================================
  // Edge Cases
  // ==========================================================================

  it("should handle Unicode content (emoji, CJK, RTL)", async () => {
    const unicodeTests = [
      { label: "emoji", content: "Test with emoji: 🚀 🎉 ❤️ 🌟" },
      { label: "CJK", content: "中文测试 日本語テスト 한글테스트" },
      { label: "RTL", content: "اختبار العربية עברית בדיקה" },
      { label: "mixed", content: "Mixed: Hello 世界 🌍 مرحبا" },
    ];

    for (const test of unicodeTests) {
      const storeRes = await rawPost(env.baseUrl, "/v1/remember", {
        content: test.content,
        source_type: "user_explicit",
        tags: ["unicode-test", test.label],
      });

      expect(storeRes.status).toBe(200);
      expect(storeRes.body.success).toBe(true);
      expect(storeRes.body.memory_id).toBeDefined();

      // Verify recall preserves Unicode
      const recallRes = await rawPost(env.baseUrl, "/v1/recall", {
        query: test.content.substring(0, 20),
        limit: 5,
        min_relevance: 0.0,
      });

      expect(recallRes.status).toBe(200);
      const results = recallRes.body.results as RecallResult[];
      const found = results.some((r) =>
        r.memory.content.includes(test.content),
      );
      expect(found).toBe(true);
    }
  }, 60000);

  it("should handle very long query string (10KB+)", async () => {
    // Store a memory first
    await rawPost(env.baseUrl, "/v1/remember", {
      content: "Findable memory for long query test",
      source_type: "user_explicit",
      tags: ["long-query-test"],
    });

    // Create 10KB+ query (mostly padding with search terms)
    const padding = "x".repeat(10240);
    const longQuery = `Findable memory long query test ${padding}`;

    const res = await rawPost(env.baseUrl, "/v1/recall", {
      query: longQuery,
      limit: 5,
    });

    // Should handle gracefully (either succeed or reject with validation error)
    expect([200, 422]).toContain(res.status);

    if (res.status === 200) {
      expect(res.body.results).toBeDefined();
      expect(Array.isArray(res.body.results)).toBe(true);
    }
  }, 30000);

  it("should handle concurrent store requests without corruption", async () => {
    const concurrency = 5;
    const timestamp = Date.now();
    const storePromises = Array.from({ length: concurrency }, (_, i) =>
      rawPost(env.baseUrl, "/v1/remember", {
        content: `Concurrent store test ${timestamp}-${i}-${Math.random().toString(36).slice(2)}`,
        source_type: "user_explicit",
        tags: ["concurrency-test", `batch-${timestamp}`],
        skip_dedup: true, // Skip duplicate detection for concurrency test
      }),
    );

    const results = await Promise.all(storePromises);

    // All should succeed (improved diagnostics)
    for (let i = 0; i < results.length; i++) {
      const res = results[i];
      expect(
        res.status,
        `Request ${i + 1}/${concurrency} should return 200`,
      ).toBe(200);
      expect(
        res.body.success,
        `Request ${i + 1}/${concurrency} should succeed. Error: ${res.body.error || "none"}`,
      ).toBe(true);
      expect(
        res.body.memory_id,
        `Request ${i + 1}/${concurrency} should have memory_id`,
      ).toBeDefined();
    }

    // All memory IDs should be unique
    const memoryIds = results.map((r) => r.body.memory_id);
    const uniqueIds = new Set(memoryIds);
    expect(uniqueIds.size).toBe(concurrency);
  }, 30000);

  it("should handle concurrent recall requests", async () => {
    // Store a memory to recall
    await rawPost(env.baseUrl, "/v1/remember", {
      content: "Concurrent recall test memory",
      source_type: "user_explicit",
      tags: ["concurrent-recall"],
    });

    const concurrency = 5;
    const recallPromises = Array.from({ length: concurrency }, () =>
      rawPost(env.baseUrl, "/v1/recall", {
        query: "concurrent recall",
        limit: 5,
      }),
    );

    const results = await Promise.all(recallPromises);

    // All should succeed and return valid results
    for (const res of results) {
      expect(res.status).toBe(200);
      expect(res.body.results).toBeDefined();
      expect(Array.isArray(res.body.results)).toBe(true);
    }
  }, 30000);

  it("should ensure store-then-immediate-recall consistency", async () => {
    const uniqueContent = `Consistency test ${Date.now()}: immediate recall`;

    // Store
    const storeRes = await rawPost(env.baseUrl, "/v1/remember", {
      content: uniqueContent,
      source_type: "user_explicit",
      tags: ["consistency-test"],
    });

    expect(storeRes.status).toBe(200);
    expect(storeRes.body.success).toBe(true);
    const memoryId = storeRes.body.memory_id;

    // Immediate recall
    const recallRes = await rawPost(env.baseUrl, "/v1/recall", {
      query: uniqueContent,
      limit: 10,
      min_relevance: 0.0,
    });

    expect(recallRes.status).toBe(200);
    const results = recallRes.body.results as RecallResult[];
    const found = results.some(
      (r) =>
        r.memory.memory_id === memoryId ||
        r.memory.content.includes("immediate recall"),
    );
    expect(found).toBe(true);
  }, 30000);

  it("should reject empty string content with 422", async () => {
    const res = await rawPost(env.baseUrl, "/v1/remember", {
      content: "",
      source_type: "user_explicit",
    });

    expect(res.status).toBe(422);
    expect(res.body.detail).toBeDefined();
  });

  it("should reject whitespace-only content with 422", async () => {
    const res = await rawPost(env.baseUrl, "/v1/remember", {
      content: "   \n\t  ",
      source_type: "user_explicit",
    });

    expect(res.status).toBe(422);
    expect(res.body.detail).toBeDefined();
  });

  it("should handle null fields with validation errors", async () => {
    // Null content
    const res1 = await fetch(`${env.baseUrl}/v1/remember`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: null,
        source_type: "user_explicit",
      }),
    });

    expect(res1.status).toBe(422);

    // Null query
    const res2 = await fetch(`${env.baseUrl}/v1/recall`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: null,
        limit: 5,
      }),
    });

    expect(res2.status).toBe(422);
  });

  it("should handle undefined fields appropriately", async () => {
    // JavaScript undefined becomes missing field in JSON
    const payload = {
      content: "Test",
      source_type: "user_explicit",
      tags: undefined, // Optional field, should be fine
    };

    const res = await rawPost(env.baseUrl, "/v1/remember", payload);

    // Should succeed (undefined optional fields are OK)
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
  });

  // ==========================================================================
  // Health & Stats Endpoints
  // ==========================================================================

  it("should return 200 with status 'ok' from GET /v1/health", async () => {
    const res = await fetch(`${env.baseUrl}/v1/health`);

    expect(res.status).toBe(200);

    const data = (await res.json()) as {
      status: string;
      instance_id: string;
      memory_count: number;
    };

    expect(data.status).toBe("ok");
    expect(data.instance_id).toBeDefined();
    expect(typeof data.memory_count).toBe("number");
    expect(data.memory_count).toBeGreaterThanOrEqual(0);
  });

  it("should return 200 with valid structure from GET /v1/stats", async () => {
    const res = await fetch(`${env.baseUrl}/v1/stats`);

    expect(res.status).toBe(200);

    const data = (await res.json()) as {
      total_memories: number;
      by_source_type: Record<string, number>;
      by_tag: Record<string, number>;
      instance_id: string;
    };

    expect(typeof data.total_memories).toBe("number");
    expect(data.total_memories).toBeGreaterThanOrEqual(0);
    expect(typeof data.by_source_type).toBe("object");
    expect(typeof data.by_tag).toBe("object");
    expect(data.instance_id).toBeDefined();
    expect(typeof data.instance_id).toBe("string");
  });

  it("should reject POST to /v1/health with 405", async () => {
    const res = await fetch(`${env.baseUrl}/v1/health`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });

    expect(res.status).toBe(405);
  });

  it("should reject POST to /v1/stats with 405", async () => {
    const res = await fetch(`${env.baseUrl}/v1/stats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });

    expect(res.status).toBe(405);
  });

  // ==========================================================================
  // Additional Edge Cases
  // ==========================================================================

  it("should handle invalid recall limit with 422", async () => {
    // Limit too high (max 50)
    const res1 = await rawPost(env.baseUrl, "/v1/recall", {
      query: "test",
      limit: 100,
    });

    expect(res1.status).toBe(422);
    expect(res1.body.detail).toBeDefined();

    // Limit too low (min 1)
    const res2 = await rawPost(env.baseUrl, "/v1/recall", {
      query: "test",
      limit: 0,
    });

    expect(res2.status).toBe(422);
    expect(res2.body.detail).toBeDefined();

    // Negative limit
    const res3 = await rawPost(env.baseUrl, "/v1/recall", {
      query: "test",
      limit: -5,
    });

    expect(res3.status).toBe(422);
    expect(res3.body.detail).toBeDefined();
  });

  it("should handle invalid min_relevance with 422", async () => {
    // min_relevance > 1.0
    const res1 = await rawPost(env.baseUrl, "/v1/recall", {
      query: "test",
      min_relevance: 1.5,
    });

    expect(res1.status).toBe(422);
    expect(res1.body.detail).toBeDefined();

    // min_relevance < 0.0
    const res2 = await rawPost(env.baseUrl, "/v1/recall", {
      query: "test",
      min_relevance: -0.1,
    });

    expect(res2.status).toBe(422);
    expect(res2.body.detail).toBeDefined();
  });

  it("should reject invalid UUID format on GET /v1/memory/{id} with 404 or 422", async () => {
    const invalidUuids = [
      "not-a-uuid",
      "12345",
      "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz",
    ];

    for (const invalidUuid of invalidUuids) {
      const res = await fetch(`${env.baseUrl}/v1/memory/${invalidUuid}`);
      // Should return 404 (not found) or 422 (invalid format)
      expect(
        [404, 422],
        `Invalid UUID "${invalidUuid}" should return 404 or 422, got ${res.status}`,
      ).toContain(res.status);
    }
  });

  it("should handle non-existent valid UUID on GET /v1/memory/{id} with 404", async () => {
    const fakeId = "00000000-0000-0000-0000-000000000000";
    const res = await fetch(`${env.baseUrl}/v1/memory/${fakeId}`);

    // Should return 404
    expect(res.status).toBe(404);

    const data = await res.json();
    expect(data.detail).toBeDefined();
  });

  it("should handle array instead of string for content field with 422", async () => {
    const res = await fetch(`${env.baseUrl}/v1/remember`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: ["array", "not", "string"],
        source_type: "user_explicit",
      }),
    });

    expect(res.status).toBe(422);
  });

  it("should handle object instead of string for query field with 422", async () => {
    const res = await fetch(`${env.baseUrl}/v1/recall`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: { object: "not string" },
        limit: 5,
      }),
    });

    expect(res.status).toBe(422);
  });
});
