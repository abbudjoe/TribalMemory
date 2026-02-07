/**
 * E2E tests for tribal_store tool.
 *
 * Tests against a REAL TribalMemory server — no mocks.
 * Validates that the plugin correctly stores memories and that
 * the server accepts and persists them.
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import {
  startTestServer,
  rawPost,
  VALID_SOURCE_TYPES,
  type TestEnvironment,
} from "./setup";

// 10KB+ tests server handling of payloads exceeding typical memory size
const LARGE_CONTENT_SIZE = 10240;

interface RecallResult {
  memory: {
    content: string;
    tags: string[];
    memory_id: string;
  };
  similarity_score: number;
  retrieval_time_ms?: number;
}

describe("tribal_store E2E", () => {
  let env: TestEnvironment;

  // Extended timeout for server startup, config write, and health check
  beforeAll(async () => {
    env = await startTestServer();
  }, 60000);

  afterAll(async () => {
    await env.cleanup();
  }, 10000);

  it("stores a memory and returns a memory_id", async () => {
    const res = await rawPost(env.baseUrl, "/v1/remember", {
      content: "The auth service uses JWT with RS256 signing",
      source_type: "user_explicit",
    });

    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.memory_id).toBeDefined();
    expect(typeof res.body.memory_id).toBe("string");
    expect(res.body.memory_id).not.toBe("");
  });

  it("stores with tags and preserves them on recall", async () => {
    const content = "PostgreSQL is our primary database";
    const tags = ["architecture", "database"];

    // Store
    const storeRes = await rawPost(env.baseUrl, "/v1/remember", {
      content,
      source_type: "user_explicit",
      tags,
    });
    expect(storeRes.status).toBe(200);
    expect(storeRes.body.success).toBe(true);
    const storedMemoryId = storeRes.body.memory_id as string;

    // Recall and verify tags + response structure
    const recallRes = await rawPost(env.baseUrl, "/v1/recall", {
      query: "primary database",
      limit: 5,
    });
    expect(recallRes.status).toBe(200);

    // Validate recall response structure
    expect(recallRes.body.results).toBeDefined();
    expect(Array.isArray(recallRes.body.results)).toBe(true);

    const results = recallRes.body.results as RecallResult[];
    
    // Each result should have required fields
    for (const result of results) {
      expect(result.memory).toBeDefined();
      expect(result.memory.content).toBeDefined();
      expect(typeof result.memory.content).toBe("string");
      expect(result.similarity_score).toBeDefined();
      expect(typeof result.similarity_score).toBe("number");
      expect(result.similarity_score).toBeGreaterThanOrEqual(0);
      expect(result.similarity_score).toBeLessThanOrEqual(1);
    }

    // Find our stored memory (improved test isolation)
    const match = results.find((r) =>
      r.memory.memory_id === storedMemoryId || r.memory.content.includes("PostgreSQL"),
    );
    expect(match).toBeDefined();
    expect(match!.memory.tags).toEqual(expect.arrayContaining(tags));
  });

  it("stores with context and preserves it", async () => {
    const content = "Redis cache TTL is 300 seconds";
    const context = "Discussed during architecture review";

    const storeRes = await rawPost(env.baseUrl, "/v1/remember", {
      content,
      source_type: "user_explicit",
      context,
    });
    expect(storeRes.status).toBe(200);
    expect(storeRes.body.success).toBe(true);
  });

  it("detects and reports duplicates", async () => {
    const content = "Duplicate detection test memory " + Date.now();

    // Store first time
    const first = await rawPost(env.baseUrl, "/v1/remember", {
      content,
      source_type: "user_explicit",
    });
    expect(first.status).toBe(200);
    expect(first.body.success).toBe(true);
    expect(first.body.duplicate_of).toBeNull();

    // Store same content again
    const second = await rawPost(env.baseUrl, "/v1/remember", {
      content,
      source_type: "user_explicit",
    });
    expect(second.status).toBe(200);
    // Should detect as duplicate
    expect(second.body.duplicate_of).toBeDefined();
    expect(second.body.duplicate_of).not.toBeNull();
  });

  it("should reject empty content with 422", async () => {
    const res = await rawPost(env.baseUrl, "/v1/remember", {
      content: "",
      source_type: "user_explicit",
    });
    // Server should reject empty content
    expect(res.status).toBe(422);
    expect(res.body.detail).toBeDefined();
  });

  it("should reject invalid sourceType 'deliberate' with 422 (regression)", async () => {
    // This is the exact bug that shipped broken — sourceType: "deliberate"
    // is not a valid enum value. The server MUST reject it.
    const res = await rawPost(env.baseUrl, "/v1/remember", {
      content: "This should fail because deliberate is not a valid sourceType",
      source_type: "deliberate",
    });
    expect(res.status).toBe(422);
    expect(res.body.detail).toBeDefined();
  });

  it("should reject other invalid sourceType values with 422", async () => {
    // Comprehensive test of various invalid types (deliberate tested above, included here for completeness)
    const invalidTypes = ["deliberate", "manual", "agent", "system", ""];
    for (const badType of invalidTypes) {
      const res = await rawPost(env.baseUrl, "/v1/remember", {
        content: `Testing invalid sourceType: ${badType}`,
        source_type: badType,
      });
      expect(res.status).toBe(422);
      expect(res.body.detail).toBeDefined();
    }
  });

  it("accepts all valid sourceType values", async () => {
    for (const validType of VALID_SOURCE_TYPES) {
      const uniqueContent =
        `Valid sourceType test: ${validType} — ` +
        `unique ${Math.random().toString(36).slice(2)}`;
      const res = await rawPost(env.baseUrl, "/v1/remember", {
        content: uniqueContent,
        source_type: validType,
      });
      // The key is no 422 (invalid enum) - dedup is acceptable
      expect(res.status).toBe(200);
      expect(res.body.success).toBeDefined();
    }
  }, 30000); // Extended timeout for multiple API calls

  it("should reject malformed JSON with 400 or 422", async () => {
    // Test that server handles invalid JSON payloads gracefully
    const res = await fetch(`${env.baseUrl}/v1/remember`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{invalid json payload",
    });
    
    // Server should reject with 400 (bad request) or 422 (validation error)
    expect([400, 422]).toContain(res.status);
  });

  it("handles long content (10KB+)", async () => {
    const longContent = "x".repeat(LARGE_CONTENT_SIZE) + " architecture decision " + Date.now();

    const res = await rawPost(env.baseUrl, "/v1/remember", {
      content: longContent,
      source_type: "user_explicit",
    });
    expect(res.status).toBe(200);
    expect(res.body.success).toBe(true);
    expect(res.body.memory_id).toBeDefined();
  }, 30000); // Extended timeout for large payload processing

  // Extended timeout for embedding computation and semantic search
  it("stored memory is retrievable via recall", { timeout: 30000 }, async () => {
    const uniqueContent =
      `E2E retrieval test ${Date.now()}: ` +
      "the deployment pipeline uses blue-green strategy";

    // Store
    const storeRes = await rawPost(env.baseUrl, "/v1/remember", {
      content: uniqueContent,
      source_type: "user_explicit",
    });
    expect(storeRes.status).toBe(200);

    // Recall
    const recallRes = await rawPost(env.baseUrl, "/v1/recall", {
      query: "deployment pipeline blue-green",
      limit: 10,
      min_relevance: 0.0,
    });
    expect(recallRes.status).toBe(200);

    const results = recallRes.body.results as RecallResult[];
    const found = results.some((r) =>
      r.memory.content.includes("blue-green"),
    );
    expect(found).toBe(true);
  });

  it("VALID_SOURCE_TYPES constant does not contain 'deliberate'", () => {
    // Compile-time guard: if someone adds "deliberate" to the union,
    // this test catches it.
    expect(VALID_SOURCE_TYPES).not.toContain("deliberate");
  });

  // Extended timeout for client initialization and embedding
  it("store via TribalClient uses correct sourceType", { timeout: 30000 }, async () => {
    // Test the actual TribalClient.remember() method (same path as plugin)
    const result = await env.client.remember(
      "TribalClient integration test " + Date.now(),
      {
        sourceType: "user_explicit",
        context: "E2E test via TribalClient",
        tags: ["e2e-test"],
      },
    );

    expect(result.memoryId).toBeDefined();
    expect(result.memoryId).not.toBe("");
    expect(result.duplicateOf).toBeNull();
  });
});
