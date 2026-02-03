/**
 * Unit tests for SnippetTruncator
 */

import { describe, it, expect, beforeEach } from "vitest";
import { SnippetTruncator, DEFAULT_TRUNCATION_CONFIG } from "../src/safeguards/truncation";

describe("SnippetTruncator", () => {
  let truncator: SnippetTruncator;

  beforeEach(() => {
    truncator = new SnippetTruncator();
  });

  describe("truncate()", () => {
    it("leaves short snippets unchanged", () => {
      const text = "This is a short snippet with only a few words.";
      expect(truncator.truncate(text)).toBe(text);
    });

    it("truncates long snippets and adds ellipsis", () => {
      // Generate a snippet over 100 tokens (100 / 0.75 = 133 words)
      const words = new Array(150).fill("word").map((w, i) => `${w}${i}`);
      const longText = words.join(" ");
      
      const result = truncator.truncate(longText);
      
      expect(result).toContain("...");
      expect(result).not.toBe(longText);
      expect(result.length).toBeLessThan(longText.length);
    });

    it("truncates to approximately the token limit", () => {
      // Generate a snippet over 100 tokens
      const words = new Array(200).fill("word");
      const longText = words.join(" ");
      
      const result = truncator.truncate(longText);
      
      // Result should be roughly 100 tokens (133 words)
      const resultWords = result.replace("...", "").split(/\s+/).filter(w => w.length > 0);
      expect(resultWords.length).toBeLessThanOrEqual(135); // ~100 tokens with buffer
      expect(resultWords.length).toBeGreaterThan(125); // Close to target
    });

    it("handles empty strings", () => {
      expect(truncator.truncate("")).toBe("");
    });

    it("handles null/undefined gracefully", () => {
      expect(truncator.truncate(null as any)).toBe(null);
      expect(truncator.truncate(undefined as any)).toBe(undefined);
    });
  });

  describe("truncateResults()", () => {
    it("truncates snippet fields in results", () => {
      const longSnippet = new Array(200).fill("word").join(" ");
      
      const results = [
        { snippet: longSnippet, score: 0.9 },
        { snippet: "short", score: 0.8 },
      ];

      truncator.truncateResults(results);

      expect(results[0].snippet).toContain("...");
      expect(results[0].snippet.length).toBeLessThan(longSnippet.length);
      expect(results[1].snippet).toBe("short");
    });

    it("truncates text fields as fallback", () => {
      const longText = new Array(200).fill("word").join(" ");
      
      const results = [
        { text: longText, score: 0.9 },
        { text: "short", score: 0.8 },
      ];

      truncator.truncateResults(results);

      expect(results[0].text).toContain("...");
      expect(results[1].text).toBe("short");
    });

    it("modifies results in place and returns them", () => {
      const results = [
        { snippet: new Array(200).fill("word").join(" ") },
      ];

      const returned = truncator.truncateResults(results);

      expect(returned).toBe(results); // Same reference
      expect(returned[0].snippet).toContain("...");
    });

    it("handles empty result arrays", () => {
      const results: any[] = [];
      expect(truncator.truncateResults(results)).toBe(results);
    });

    it("handles results with no text fields", () => {
      const results = [
        { score: 0.9, path: "/some/path" },
      ];

      truncator.truncateResults(results);
      
      // Should not throw, results unchanged
      expect(results[0].score).toBe(0.9);
    });
  });

  describe("custom config", () => {
    it("respects custom max tokens per snippet", () => {
      const customTruncator = new SnippetTruncator({
        maxTokensPerSnippet: 20, // Very low limit
      });

      // 50 words = ~37.5 tokens, should be truncated at 20 tokens (~26 words)
      const text = new Array(50).fill("word").join(" ");
      const result = customTruncator.truncate(text);

      expect(result).toContain("...");
      const resultWords = result.replace("...", "").split(/\s+/).filter(w => w.length > 0);
      expect(resultWords.length).toBeLessThanOrEqual(28); // ~20 tokens with buffer
    });
  });

  describe("getConfig()", () => {
    it("returns config copy", () => {
      const config = truncator.getConfig();
      expect(config.maxTokensPerSnippet).toBe(DEFAULT_TRUNCATION_CONFIG.maxTokensPerSnippet);
      
      // Verify it's a copy
      config.maxTokensPerSnippet = 999;
      expect(truncator.getConfig().maxTokensPerSnippet).toBe(DEFAULT_TRUNCATION_CONFIG.maxTokensPerSnippet);
    });
  });
});
