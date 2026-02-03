/**
 * Tests for Issue #15 (config naming) and Issue #16 (result types).
 *
 * Verifies that PluginConfig and MemoryResult interfaces
 * are consistent, well-named, and properly typed.
 */

import { describe, it, expect } from "vitest";

// Import the types we're testing
import type { MemoryResult } from "../src/types";

describe("MemoryResult type", () => {
  it("accepts a fully populated result", () => {
    const result: MemoryResult = {
      id: "abc-123",
      path: "tribal-memory:abc-123",
      score: 0.85,
      snippet: "Test snippet content",
      startLine: 1,
      endLine: 10,
    };
    expect(result.id).toBe("abc-123");
    expect(result.score).toBe(0.85);
  });

  it("accepts a minimal result (all optional)", () => {
    const result: MemoryResult = {};
    expect(result.id).toBeUndefined();
    expect(result.snippet).toBeUndefined();
  });

  it("accepts text as alternative to snippet", () => {
    const result: MemoryResult = {
      text: "Alternative content field",
    };
    expect(result.text).toBe("Alternative content field");
  });

  it("supports tags array", () => {
    const result: MemoryResult = {
      id: "tagged",
      tags: ["work", "important"],
    };
    expect(result.tags).toEqual(["work", "important"]);
  });

  it("supports source and supersedes fields", () => {
    const result: MemoryResult = {
      id: "corrected",
      source: "user_explicit",
      supersedes: "original-id",
      sourceQuery: "expanded query",
    };
    expect(result.supersedes).toBe("original-id");
    expect(result.sourceQuery).toBe("expanded query");
  });
});
