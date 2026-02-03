/**
 * Unit tests for TribalClient
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { TribalClient } from "../src/tribal-client";

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("TribalClient", () => {
  let client: TribalClient;

  beforeEach(() => {
    client = new TribalClient("http://localhost:18790");
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("constructor", () => {
    it("removes trailing slash from baseUrl", () => {
      const c = new TribalClient("http://localhost:18790/");
      // @ts-ignore - accessing private for test
      expect(c.baseUrl).toBe("http://localhost:18790");
    });
  });

  describe("recall()", () => {
    it("calls /v1/recall with correct parameters", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ results: [] }),
      });

      await client.recall("test query", { maxResults: 5, minScore: 0.3 });

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:18790/v1/recall",
        expect.objectContaining({
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: "test query",
            limit: 5,
            min_relevance: 0.3,
            tags: undefined,
          }),
        })
      );
    });

    it("transforms server response to SearchResult format", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          results: [{
            memory: {
              id: "abc123",
              content: "Test content",
              source_type: "user_explicit",
              tags: ["test"],
            },
            similarity_score: 0.85,
            retrieval_time_ms: 10,
          }],
        }),
      });

      const results = await client.recall("test");

      expect(results).toHaveLength(1);
      expect(results[0]).toMatchObject({
        id: "abc123",
        snippet: "Test content",
        score: 0.85,
        source: "user_explicit",
      });
    });

    it("uses full UUID in path (no truncation)", async () => {
      const fullUuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          results: [{
            memory: {
              id: fullUuid,
              content: "Test content",
              source_type: "user_explicit",
              tags: [],
            },
            similarity_score: 0.9,
            retrieval_time_ms: 5,
          }],
        }),
      });

      const results = await client.recall("test");

      expect(results).toHaveLength(1);
      expect(results[0].path).toBe(`tribal-memory:${fullUuid}`);
    });
  });

  describe("remember()", () => {
    it("calls /v1/remember with correct parameters", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, memory_id: "new123" }),
      });

      const result = await client.remember("New memory", {
        sourceType: "user_explicit",
        tags: ["test"],
      });

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:18790/v1/remember",
        expect.objectContaining({
          method: "POST",
          body: expect.stringContaining("New memory"),
        })
      );
      expect(result.success).toBe(true);
      expect(result.memoryId).toBe("new123");
    });

    it("handles duplicate detection", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: false, duplicate_of: "existing123" }),
      });

      const result = await client.remember("Duplicate content");

      expect(result.success).toBe(false);
      expect(result.duplicateOf).toBe("existing123");
    });
  });

  describe("get()", () => {
    it("calls /v1/memory/{id}", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: "abc123", content: "Test" }),
      });

      const result = await client.get("abc123");

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:18790/v1/memory/abc123",
        expect.any(Object)
      );
      expect(result.id).toBe("abc123");
    });

    it("returns null for 404", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      const result = await client.get("nonexistent");
      expect(result).toBeNull();
    });
  });

  describe("forget()", () => {
    it("calls DELETE /v1/forget/{id}", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });

      const result = await client.forget("abc123");

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:18790/v1/forget/abc123",
        expect.objectContaining({ method: "DELETE" })
      );
      expect(result).toBe(true);
    });
  });

  describe("health()", () => {
    it("returns ok: true when server is healthy", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: "ok",
          instance_id: "test-instance",
          memory_count: 42,
        }),
      });

      const result = await client.health();

      expect(result.ok).toBe(true);
      expect(result.instanceId).toBe("test-instance");
      expect(result.memoryCount).toBe(42);
    });

    it("returns ok: false when server is down", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Connection refused"));

      const result = await client.health();

      expect(result.ok).toBe(false);
    });
  });

  describe("deprecated methods", () => {
    it("capture() calls remember() with deprecation warning", async () => {
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, memory_id: "new123" }),
      });

      const result = await client.capture("Old API content", { tags: ["test"] });

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining("deprecated")
      );
      expect(result).toBe("new123");
      warnSpy.mockRestore();
    });

    it("searchSingle() calls recall() with deprecation warning", async () => {
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ results: [] }),
      });

      await client.searchSingle("query", { maxResults: 5 });

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining("deprecated")
      );
      warnSpy.mockRestore();
    });
  });
});
