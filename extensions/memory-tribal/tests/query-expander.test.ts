/**
 * Unit tests for QueryExpander
 */

import { describe, it, expect, beforeEach } from "vitest";
import { QueryExpander } from "../src/learned/query-expander";

describe("QueryExpander", () => {
  let expander: QueryExpander;

  beforeEach(() => {
    expander = new QueryExpander();
  });

  describe("expand()", () => {
    it("always includes the original query", () => {
      const result = expander.expand("test query");
      expect(result).toContain("test query");
    });

    it("limits output to 8 variants max", () => {
      // A query that would generate many variants
      const result = expander.expand("What is my favorite medical care partner?");
      expect(result.length).toBeLessThanOrEqual(8);
    });

    describe("rule-based expansion", () => {
      it("expands 'What is my X?' pattern", () => {
        const result = expander.expand("What is my favorite food?");
        expect(result).toContain("favorite food");
        expect(result).toContain("my favorite food");
      });

      it("expands 'Who is my X?' pattern", () => {
        const result = expander.expand("Who is my manager?");
        expect(result).toContain("my manager");
        expect(result).toContain("manager");
      });

      it("expands 'When is X?' pattern", () => {
        const result = expander.expand("When is the meeting?");
        expect(result).toContain("meeting date");
        expect(result).toContain("meeting time");
      });

      it("expands 'Where is X?' pattern", () => {
        const result = expander.expand("Where is the office?");
        expect(result).toContain("office location");
        expect(result).toContain("office address");
      });
    });

    describe("semantic synonym expansion", () => {
      it("expands 'medical care' to include 'doctor'", () => {
        const result = expander.expand("Where is medical care?");
        expect(result).toContain("doctor");
      });

      it("expands 'life partner' to include 'spouse'", () => {
        const result = expander.expand("Who is my life partner?");
        expect(result).toContain("spouse");
      });

      it("expands 'animals' to include 'pet'", () => {
        const result = expander.expand("Any animals in the household?");
        expect(result).toContain("pet");
      });

      it("expands 'code editor' to include 'IDE'", () => {
        const result = expander.expand("What code editor do you use?");
        expect(result).toContain("IDE");
      });

      it("expands 'food restrictions' to include 'allergies'", () => {
        const result = expander.expand("What are the food restrictions?");
        expect(result).toContain("allergies");
      });
    });

    describe("keyword extraction fallback", () => {
      it("extracts meaningful words from unmatched queries", () => {
        const result = expander.expand("Tell me about the database configuration");
        // Should extract words longer than 3 chars, excluding stop words
        expect(result.some(v => v.includes("database"))).toBe(true);
      });
    });
  });

  describe("learnExpansion()", () => {
    it("stores learned expansions", () => {
      expander.learnExpansion("favorite color", "colour preference");
      const result = expander.expand("favorite color");
      expect(result).toContain("colour preference");
    });

    it("limits stored expansions to 5 per query", () => {
      for (let i = 0; i < 10; i++) {
        expander.learnExpansion("test query", `expansion ${i}`);
      }
      const result = expander.expand("test query");
      // Original + up to 5 learned + other expansions
      expect(result.filter(v => v.startsWith("expansion")).length).toBeLessThanOrEqual(5);
    });
  });

  describe("ruleCount()", () => {
    it("returns count of rules plus learned expansions", () => {
      const initialCount = expander.ruleCount();
      expect(initialCount).toBeGreaterThan(0);

      expander.learnExpansion("new query", "new expansion");
      expect(expander.ruleCount()).toBe(initialCount + 1);
    });
  });
});
