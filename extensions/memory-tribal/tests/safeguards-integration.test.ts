/**
 * Integration tests for safeguards working together
 */

import { describe, it, expect, beforeEach } from "vitest";
import { TokenBudget } from "../src/safeguards/token-budget";
import { SnippetTruncator } from "../src/safeguards/truncation";
import { CircuitBreaker } from "../src/safeguards/circuit-breaker";

describe("Safeguards Integration", () => {
  let tokenBudget: TokenBudget;
  let truncator: SnippetTruncator;
  let breaker: CircuitBreaker;

  beforeEach(() => {
    tokenBudget = new TokenBudget();
    truncator = new SnippetTruncator();
    breaker = new CircuitBreaker();
  });

  describe("Full memory_search flow simulation", () => {
    it("applies truncation before token budget", () => {
      // Simulate long snippets that need truncation
      const longSnippet = new Array(200).fill("word").join(" ");
      
      const results = [
        { snippet: longSnippet, score: 0.9 },
        { snippet: longSnippet, score: 0.8 },
      ];

      // Step 1: Truncate (done before budget)
      truncator.truncateResults(results);

      // Step 2: Count tokens after truncation
      const tokens1 = tokenBudget.countTokens(results[0].snippet);
      const tokens2 = tokenBudget.countTokens(results[1].snippet);

      // Each snippet should be ~100 tokens max after truncation
      expect(tokens1).toBeLessThanOrEqual(105); // Small buffer for "..."
      expect(tokens2).toBeLessThanOrEqual(105);

      // Step 3: Check budget
      expect(tokenBudget.canUseForRecall(tokens1 + tokens2)).toBe(true);
    });

    it("respects per-recall token budget", () => {
      const sessionId = "session1";
      const turnId = "turn1";

      // Create results that would exceed per-recall cap (500 tokens)
      const results: any[] = [];
      
      // Each result has ~100 tokens
      for (let i = 0; i < 10; i++) {
        const words = new Array(135).fill("word").join(" "); // ~100 tokens
        results.push({ snippet: words, score: 0.9 - i * 0.05 });
      }

      // Simulate budget filtering (like in index.ts)
      const filteredResults: any[] = [];
      let recallTokens = 0;

      for (const result of results) {
        const tokens = tokenBudget.countTokens(result.snippet);

        if (recallTokens + tokens > 500) { // Per-recall cap
          break;
        }

        filteredResults.push(result);
        recallTokens += tokens;
      }

      // Should only get ~5 results (500 tokens / 100 tokens each)
      expect(filteredResults.length).toBeLessThanOrEqual(5);
      expect(recallTokens).toBeLessThanOrEqual(500);
    });

    it("tracks circuit breaker across multiple searches", () => {
      const sessionId = "session1";

      // Simulate 5 consecutive empty searches
      for (let i = 0; i < 5; i++) {
        expect(breaker.isTripped(sessionId)).toBe(false);
        breaker.recordResult(sessionId, 0); // Empty result
      }

      // Breaker should now be tripped
      expect(breaker.isTripped(sessionId)).toBe(true);

      // Subsequent searches should be blocked
      const status = breaker.getStatus(sessionId);
      expect(status.tripped).toBe(true);
      expect(status.consecutiveEmpty).toBe(5);
    });

    it("resets circuit breaker on successful search", () => {
      const sessionId = "session1";

      // Build up to threshold
      for (let i = 0; i < 4; i++) {
        breaker.recordResult(sessionId, 0);
      }

      expect(breaker.isTripped(sessionId)).toBe(false);

      // Successful search resets the counter
      breaker.recordResult(sessionId, 3);

      const status = breaker.getStatus(sessionId);
      expect(status.consecutiveEmpty).toBe(0);
      expect(status.tripped).toBe(false);

      // Can accumulate again from scratch
      for (let i = 0; i < 3; i++) {
        breaker.recordResult(sessionId, 0);
      }

      expect(breaker.getStatus(sessionId).consecutiveEmpty).toBe(3);
      expect(breaker.isTripped(sessionId)).toBe(false);
    });

    it("handles multi-turn scenario with session budget", () => {
      const sessionId = "session1";
      const turn1 = "turn1";
      const turn2 = "turn2";
      const turn3 = "turn3";

      // Turn 1: Use 2000 tokens
      tokenBudget.recordUsage(sessionId, turn1, 2000);
      expect(tokenBudget.canUseForSession(sessionId, 2500)).toBe(true); // 4500 total

      // Turn 2: Use 2000 more tokens
      tokenBudget.recordUsage(sessionId, turn2, 2000);
      expect(tokenBudget.canUseForSession(sessionId, 1500)).toBe(false); // Would exceed 5000

      // Turn 3: Only 1000 tokens left in session budget
      expect(tokenBudget.canUseForSession(sessionId, 1000)).toBe(true);
      tokenBudget.recordUsage(sessionId, turn3, 1000);

      const usage = tokenBudget.getUsage(sessionId, turn3);
      expect(usage.session).toBe(5000);
      expect(usage.sessionRemaining).toBe(0);
    });
  });

  describe("Edge cases", () => {
    it("handles empty result set gracefully", () => {
      const results: any[] = [];
      
      truncator.truncateResults(results);
      expect(results).toEqual([]);
      
      breaker.recordResult("session1", results.length);
      expect(breaker.getStatus("session1").consecutiveEmpty).toBe(1);
    });

    it("handles session isolation for all safeguards", () => {
      // Circuit breaker isolation
      breaker.recordResult("session1", 0);
      breaker.recordResult("session1", 0);
      expect(breaker.getStatus("session1").consecutiveEmpty).toBe(2);
      expect(breaker.getStatus("session2").consecutiveEmpty).toBe(0);

      // Token budget isolation
      tokenBudget.recordUsage("session1", "turn1", 1000);
      expect(tokenBudget.getUsage("session1", "turn1").session).toBe(1000);
      expect(tokenBudget.getUsage("session2", "turn1").session).toBe(0);
    });
  });
});
