/**
 * Unit tests for TokenBudget
 */

import { describe, it, expect, beforeEach } from "vitest";
import { TokenBudget, DEFAULT_TOKEN_BUDGET } from "../src/safeguards/token-budget";

describe("TokenBudget", () => {
  let budget: TokenBudget;

  beforeEach(() => {
    budget = new TokenBudget();
  });

  describe("countTokens()", () => {
    it("counts tokens approximately (0.75 tokens per word)", () => {
      // 4 words * 0.75 = 3 tokens
      const tokens = budget.countTokens("this is a test");
      expect(tokens).toBe(3);
    });

    it("handles empty strings", () => {
      expect(budget.countTokens("")).toBe(0);
    });

    it("handles multi-whitespace", () => {
      // 3 words * 0.75 = 2.25 -> ceil to 3
      const tokens = budget.countTokens("word1   word2    word3");
      expect(tokens).toBe(3);
    });

    it("rounds up fractional tokens", () => {
      // 1 word * 0.75 = 0.75 -> ceil to 1
      const tokens = budget.countTokens("word");
      expect(tokens).toBe(1);
    });
  });

  describe("canUseForRecall()", () => {
    it("allows tokens under per-recall cap", () => {
      expect(budget.canUseForRecall(100)).toBe(true);
      expect(budget.canUseForRecall(DEFAULT_TOKEN_BUDGET.perRecallCap)).toBe(true);
    });

    it("rejects tokens over per-recall cap", () => {
      expect(budget.canUseForRecall(DEFAULT_TOKEN_BUDGET.perRecallCap + 1)).toBe(false);
    });
  });

  describe("canUseForTurn()", () => {
    it("allows tokens under per-turn cap", () => {
      expect(budget.canUseForTurn("turn1", 100)).toBe(true);
    });

    it("tracks cumulative usage per turn", () => {
      budget.recordUsage("session1", "turn1", 400);
      
      // 400 + 300 = 700, under 750 cap
      expect(budget.canUseForTurn("turn1", 300)).toBe(true);
      
      // 400 + 400 = 800, over 750 cap
      expect(budget.canUseForTurn("turn1", 400)).toBe(false);
    });

    it("isolates turns from each other", () => {
      budget.recordUsage("session1", "turn1", 700);
      
      // turn1 is near cap, but turn2 is fresh
      expect(budget.canUseForTurn("turn1", 100)).toBe(false);
      expect(budget.canUseForTurn("turn2", 100)).toBe(true);
    });
  });

  describe("canUseForSession()", () => {
    it("allows tokens under per-session cap", () => {
      expect(budget.canUseForSession("session1", 1000)).toBe(true);
    });

    it("tracks cumulative usage per session", () => {
      budget.recordUsage("session1", "turn1", 2000);
      budget.recordUsage("session1", "turn2", 2000);
      
      // 4000 + 500 = 4500, under 5000 cap
      expect(budget.canUseForSession("session1", 500)).toBe(true);
      
      // 4000 + 1500 = 5500, over 5000 cap
      expect(budget.canUseForSession("session1", 1500)).toBe(false);
    });

    it("isolates sessions from each other", () => {
      budget.recordUsage("session1", "turn1", 4900);
      
      // session1 is near cap, but session2 is fresh
      expect(budget.canUseForSession("session1", 200)).toBe(false);
      expect(budget.canUseForSession("session2", 200)).toBe(true);
    });
  });

  describe("recordUsage()", () => {
    it("accumulates usage for turn and session", () => {
      budget.recordUsage("session1", "turn1", 100);
      budget.recordUsage("session1", "turn1", 200);
      
      const usage = budget.getUsage("session1", "turn1");
      expect(usage.turn).toBe(300);
      expect(usage.session).toBe(300);
    });

    it("tracks multiple turns in same session", () => {
      budget.recordUsage("session1", "turn1", 100);
      budget.recordUsage("session1", "turn2", 200);
      
      const usage1 = budget.getUsage("session1", "turn1");
      const usage2 = budget.getUsage("session1", "turn2");
      
      expect(usage1.turn).toBe(100);
      expect(usage1.session).toBe(300); // Shared session total
      expect(usage2.turn).toBe(200);
      expect(usage2.session).toBe(300); // Same session total
    });
  });

  describe("getUsage()", () => {
    it("returns zero for unused session/turn", () => {
      const usage = budget.getUsage("session1", "turn1");
      expect(usage.turn).toBe(0);
      expect(usage.session).toBe(0);
      expect(usage.turnRemaining).toBe(DEFAULT_TOKEN_BUDGET.perTurnCap);
      expect(usage.sessionRemaining).toBe(DEFAULT_TOKEN_BUDGET.perSessionCap);
    });

    it("calculates remaining budget correctly", () => {
      budget.recordUsage("session1", "turn1", 100);
      
      const usage = budget.getUsage("session1", "turn1");
      expect(usage.turnRemaining).toBe(DEFAULT_TOKEN_BUDGET.perTurnCap - 100);
      expect(usage.sessionRemaining).toBe(DEFAULT_TOKEN_BUDGET.perSessionCap - 100);
    });
  });

  describe("cleanupOldTurns()", () => {
    it("keeps recent turns and drops old ones", () => {
      // Create 150 turns
      for (let i = 0; i < 150; i++) {
        budget.recordUsage("session1", `turn${i}`, 10);
      }

      budget.cleanupOldTurns(100);

      // Should keep only last 100
      const oldUsage = budget.getUsage("session1", "turn0");
      const recentUsage = budget.getUsage("session1", "turn149");
      
      expect(oldUsage.turn).toBe(0); // Dropped
      expect(recentUsage.turn).toBe(10); // Kept
    });

    it("does nothing if under threshold", () => {
      budget.recordUsage("session1", "turn1", 100);
      budget.recordUsage("session1", "turn2", 200);
      
      budget.cleanupOldTurns(100);
      
      const usage = budget.getUsage("session1", "turn1");
      expect(usage.turn).toBe(100); // Still there
    });
  });

  describe("resetSession()", () => {
    it("clears session usage", () => {
      budget.recordUsage("session1", "turn1", 1000);
      budget.resetSession("session1");
      
      const usage = budget.getUsage("session1", "turn1");
      expect(usage.session).toBe(0);
      expect(usage.turn).toBe(1000); // Turn usage not affected
    });
  });

  describe("custom config", () => {
    it("respects custom budget limits", () => {
      const customBudget = new TokenBudget({
        perRecallCap: 100,
        perTurnCap: 200,
        perSessionCap: 1000,
      });

      expect(customBudget.canUseForRecall(101)).toBe(false);
      
      customBudget.recordUsage("s1", "t1", 150);
      expect(customBudget.canUseForTurn("t1", 100)).toBe(false);
      
      customBudget.recordUsage("s1", "t2", 900);
      expect(customBudget.canUseForSession("s1", 200)).toBe(false);
    });
  });

  describe("edge cases", () => {
    it("should handle negative token counts gracefully", () => {
      expect(budget.canUseForRecall(-10)).toBe(true);
      expect(budget.canUseForTurn("t1", -10)).toBe(true);
      expect(budget.canUseForSession("s1", -10)).toBe(true);
    });

    it("should handle extremely large token counts", () => {
      expect(budget.canUseForRecall(Number.MAX_SAFE_INTEGER)).toBe(false);
      expect(budget.canUseForTurn("t1", Number.MAX_SAFE_INTEGER)).toBe(false);
      expect(budget.canUseForSession("s1", Number.MAX_SAFE_INTEGER)).toBe(false);
    });

    it("should report turn count via getTurnCount()", () => {
      expect(budget.getTurnCount()).toBe(0);
      budget.recordUsage("s1", "t1", 10);
      budget.recordUsage("s1", "t2", 20);
      expect(budget.getTurnCount()).toBe(2);
    });
  });
});
