/**
 * Smart Triggers - Tests
 *
 * Phase 2 of Issue #11: Skip memory recall for low-value queries
 * like "hi", "ok", "thanks", etc. to save 30-50% of unnecessary recalls.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { SmartTrigger, SmartTriggerConfig, DEFAULT_SMART_TRIGGER_CONFIG } from "../src/safeguards/smart-triggers";

describe("SmartTrigger", () => {
  let trigger: SmartTrigger;

  beforeEach(() => {
    trigger = new SmartTrigger();
  });

  describe("default skip patterns", () => {
    it("should skip common greetings", () => {
      expect(trigger.shouldSkip("hi")).toBe(true);
      expect(trigger.shouldSkip("hello")).toBe(true);
      expect(trigger.shouldSkip("hey")).toBe(true);
      expect(trigger.shouldSkip("hi there")).toBe(true);
    });

    it("should skip acknowledgments", () => {
      expect(trigger.shouldSkip("ok")).toBe(true);
      expect(trigger.shouldSkip("okay")).toBe(true);
      expect(trigger.shouldSkip("sure")).toBe(true);
      expect(trigger.shouldSkip("yes")).toBe(true);
      expect(trigger.shouldSkip("no")).toBe(true);
      expect(trigger.shouldSkip("yep")).toBe(true);
      expect(trigger.shouldSkip("nope")).toBe(true);
      expect(trigger.shouldSkip("yeah")).toBe(true);
    });

    it("should skip thanks/pleasantries", () => {
      expect(trigger.shouldSkip("thanks")).toBe(true);
      expect(trigger.shouldSkip("thank you")).toBe(true);
      expect(trigger.shouldSkip("thanks!")).toBe(true);
      expect(trigger.shouldSkip("thx")).toBe(true);
      expect(trigger.shouldSkip("ty")).toBe(true);
      expect(trigger.shouldSkip("cool")).toBe(true);
      expect(trigger.shouldSkip("nice")).toBe(true);
      expect(trigger.shouldSkip("great")).toBe(true);
      expect(trigger.shouldSkip("awesome")).toBe(true);
      expect(trigger.shouldSkip("got it")).toBe(true);
      expect(trigger.shouldSkip("sounds good")).toBe(true);
    });

    it("should skip farewells", () => {
      expect(trigger.shouldSkip("bye")).toBe(true);
      expect(trigger.shouldSkip("goodbye")).toBe(true);
      expect(trigger.shouldSkip("see you")).toBe(true);
      expect(trigger.shouldSkip("later")).toBe(true);
      expect(trigger.shouldSkip("gn")).toBe(true);
      expect(trigger.shouldSkip("good night")).toBe(true);
    });
  });

  describe("case insensitivity", () => {
    it("should be case insensitive", () => {
      expect(trigger.shouldSkip("Hi")).toBe(true);
      expect(trigger.shouldSkip("THANKS")).toBe(true);
      expect(trigger.shouldSkip("Ok")).toBe(true);
      expect(trigger.shouldSkip("GoOd NiGhT")).toBe(true);
    });
  });

  describe("punctuation handling", () => {
    it("should skip queries with trailing punctuation", () => {
      expect(trigger.shouldSkip("hi!")).toBe(true);
      expect(trigger.shouldSkip("thanks!!")).toBe(true);
      expect(trigger.shouldSkip("ok.")).toBe(true);
      expect(trigger.shouldSkip("hey...")).toBe(true);
      expect(trigger.shouldSkip("bye!")).toBe(true);
    });

    it("should skip queries with leading/trailing whitespace", () => {
      expect(trigger.shouldSkip("  hi  ")).toBe(true);
      expect(trigger.shouldSkip("\tthanks\n")).toBe(true);
    });
  });

  describe("should NOT skip meaningful queries", () => {
    it("should not skip questions", () => {
      expect(trigger.shouldSkip("What is the project structure?")).toBe(false);
      expect(trigger.shouldSkip("How do I configure the server?")).toBe(false);
      expect(trigger.shouldSkip("Who was at the meeting yesterday?")).toBe(false);
    });

    it("should not skip contextual queries", () => {
      expect(trigger.shouldSkip("Tell me about the database schema")).toBe(false);
      expect(trigger.shouldSkip("What did we decide about the API?")).toBe(false);
      expect(trigger.shouldSkip("Find the deployment instructions")).toBe(false);
    });

    it("should not skip short but meaningful queries", () => {
      expect(trigger.shouldSkip("search API docs")).toBe(false);
      expect(trigger.shouldSkip("find config")).toBe(false);
      expect(trigger.shouldSkip("list todos")).toBe(false);
    });

    it("should not skip queries containing skip words within longer text", () => {
      expect(trigger.shouldSkip("hi, can you help me find the config?")).toBe(false);
      expect(trigger.shouldSkip("thanks for that, now what about the API?")).toBe(false);
      expect(trigger.shouldSkip("ok so how does the memory system work?")).toBe(false);
    });
  });

  describe("short query threshold", () => {
    it("should skip very short queries by default", () => {
      // Single character queries are too short to be meaningful
      expect(trigger.shouldSkip("a")).toBe(true);
      expect(trigger.shouldSkip("x")).toBe(true);
      expect(trigger.shouldSkip("")).toBe(true);
    });

    it("should not skip short but meaningful words", () => {
      // 3+ character words that aren't in skip list should pass through
      expect(trigger.shouldSkip("API")).toBe(false);
      expect(trigger.shouldSkip("MCP")).toBe(false);
      expect(trigger.shouldSkip("bug")).toBe(false);
    });
  });

  describe("emoji-only queries", () => {
    it("should skip emoji-only queries", () => {
      expect(trigger.shouldSkip("👍")).toBe(true);
      expect(trigger.shouldSkip("😂")).toBe(true);
      expect(trigger.shouldSkip("🎉🎉🎉")).toBe(true);
      expect(trigger.shouldSkip("👍👍")).toBe(true);
    });

    it("should not skip queries with emoji mixed with text", () => {
      expect(trigger.shouldSkip("search for 🔑 config")).toBe(false);
    });
  });

  describe("custom configuration", () => {
    it("should allow custom skip keywords", () => {
      const custom = new SmartTrigger({
        skipKeywords: ["skip-this", "and-this"],
      });
      expect(custom.shouldSkip("skip-this")).toBe(true);
      expect(custom.shouldSkip("and-this")).toBe(true);
      // Default keywords should still work if extended
    });

    it("should allow custom minimum query length", () => {
      const custom = new SmartTrigger({ minQueryLength: 5 });
      expect(custom.shouldSkip("abc")).toBe(true);  // Under 5 chars
      expect(custom.shouldSkip("abcde")).toBe(false); // Exactly 5 chars
    });

    it("should allow disabling emoji skip", () => {
      const custom = new SmartTrigger({ skipEmojiOnly: false });
      expect(custom.shouldSkip("👍")).toBe(false);
    });
  });

  describe("skip reason", () => {
    it("should return reason for skip", () => {
      const result = trigger.classify("hi");
      expect(result.skip).toBe(true);
      expect(result.reason).toBeDefined();
      expect(result.reason).toContain("keyword");
    });

    it("should return reason for emoji skip", () => {
      const result = trigger.classify("👍");
      expect(result.skip).toBe(true);
      expect(result.reason).toContain("emoji");
    });

    it("should return reason for short query skip", () => {
      const result = trigger.classify("");
      expect(result.skip).toBe(true);
      expect(result.reason).toContain("short");
    });

    it("should return no-skip for meaningful queries", () => {
      const result = trigger.classify("What is the project structure?");
      expect(result.skip).toBe(false);
      expect(result.reason).toBeNull();
    });
  });

  describe("skip stats", () => {
    it("should track skip and pass counts", () => {
      trigger.shouldSkip("hi");
      trigger.shouldSkip("What is the API?");
      trigger.shouldSkip("thanks");
      trigger.shouldSkip("Find the config");

      const stats = trigger.getStats();
      expect(stats.totalChecked).toBe(4);
      expect(stats.totalSkipped).toBe(2);
      expect(stats.totalPassed).toBe(2);
      expect(stats.skipRate).toBeCloseTo(0.5);
    });

    it("should start with zero stats", () => {
      const stats = trigger.getStats();
      expect(stats.totalChecked).toBe(0);
      expect(stats.totalSkipped).toBe(0);
      expect(stats.totalPassed).toBe(0);
      expect(stats.skipRate).toBe(0);
    });

    it("should allow resetting stats", () => {
      trigger.shouldSkip("hi");
      trigger.shouldSkip("thanks");
      trigger.resetStats();
      const stats = trigger.getStats();
      expect(stats.totalChecked).toBe(0);
    });
  });

  describe("getConfig", () => {
    it("should return current config", () => {
      const cfg = trigger.getConfig();
      expect(cfg.minQueryLength).toBeDefined();
      expect(cfg.skipKeywords).toBeDefined();
      expect(Array.isArray(cfg.skipKeywords)).toBe(true);
      expect(cfg.skipEmojiOnly).toBe(true);
    });
  });
});
