/**
 * Session Deduplication Tests
 *
 * Ensures the same memory snippet is not injected twice within a session.
 * Uses a cooldown window so memories can reappear after enough time passes.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { SessionDedup, SessionDedupConfig } from "../src/safeguards/session-dedup";

describe("SessionDedup", () => {
  let dedup: SessionDedup;

  beforeEach(() => {
    dedup = new SessionDedup();
  });

  describe("basic deduplication", () => {
    it("should allow a result the first time it appears", () => {
      const results = [
        { path: "memory/2026-01-01.md", startLine: 1, endLine: 10, snippet: "foo", score: 0.9 },
      ];
      const filtered = dedup.filter("session-1", results);
      expect(filtered).toHaveLength(1);
    });

    it("should filter out a result seen in the same session", () => {
      const results = [
        { path: "memory/2026-01-01.md", startLine: 1, endLine: 10, snippet: "foo", score: 0.9 },
      ];
      dedup.filter("session-1", results);
      const second = dedup.filter("session-1", results);
      expect(second).toHaveLength(0);
    });

    it("should allow the same result in a different session", () => {
      const results = [
        { path: "memory/2026-01-01.md", startLine: 1, endLine: 10, snippet: "foo", score: 0.9 },
      ];
      dedup.filter("session-1", results);
      const other = dedup.filter("session-2", results);
      expect(other).toHaveLength(1);
    });

    it("should not dedup results seen only in other sessions", () => {
      const r1 = [{ path: "a.md", startLine: 1, endLine: 5, snippet: "x", score: 0.9 }];
      const r2 = [{ path: "b.md", startLine: 1, endLine: 5, snippet: "y", score: 0.8 }];
      dedup.filter("s1", r1); // s1 sees r1
      dedup.filter("s2", r2); // s2 sees r2
      // s1 should still accept r2 (never seen it before)
      expect(dedup.filter("s1", r2)).toHaveLength(1);
    });

    it("should filter duplicates but keep unique results", () => {
      const batch1 = [
        { path: "MEMORY.md", startLine: 5, endLine: 15, snippet: "a", score: 0.8 },
      ];
      const batch2 = [
        { path: "MEMORY.md", startLine: 5, endLine: 15, snippet: "a", score: 0.8 },
        { path: "MEMORY.md", startLine: 20, endLine: 30, snippet: "b", score: 0.7 },
      ];
      dedup.filter("s1", batch1);
      const filtered = dedup.filter("s1", batch2);
      expect(filtered).toHaveLength(1);
      expect(filtered[0].snippet).toBe("b");
    });
  });

  describe("identity key", () => {
    it("should use path + startLine + endLine as identity", () => {
      const r1 = [{ path: "a.md", startLine: 1, endLine: 5, snippet: "x", score: 0.9 }];
      const r2 = [{ path: "a.md", startLine: 6, endLine: 10, snippet: "y", score: 0.8 }];
      dedup.filter("s1", r1);
      const filtered = dedup.filter("s1", r2);
      expect(filtered).toHaveLength(1); // different line range → not a dupe
    });

    it("should fall back to path + snippet hash when no line info", () => {
      const r1 = [{ path: "a.md", snippet: "hello world", score: 0.9 }];
      dedup.filter("s1", r1);
      const r2 = [{ path: "a.md", snippet: "hello world", score: 0.8 }];
      const filtered = dedup.filter("s1", r2);
      expect(filtered).toHaveLength(0); // same path + snippet → dupe
    });

    it("should not treat different snippets at same path as duplicates", () => {
      const r1 = [{ path: "a.md", snippet: "hello", score: 0.9 }];
      dedup.filter("s1", r1);
      const r2 = [{ path: "a.md", snippet: "world", score: 0.8 }];
      const filtered = dedup.filter("s1", r2);
      expect(filtered).toHaveLength(1);
    });
  });

  describe("cooldown window", () => {
    it("should re-allow results after cooldown expires", () => {
      vi.useFakeTimers();
      try {
        const short = new SessionDedup({ cooldownMs: 1000 });
        const results = [
          { path: "a.md", startLine: 1, endLine: 5, snippet: "x", score: 0.9 },
        ];
        short.filter("s1", results);
        expect(short.filter("s1", results)).toHaveLength(0);

        vi.advanceTimersByTime(1001);
        expect(short.filter("s1", results)).toHaveLength(1);
      } finally {
        vi.useRealTimers();
      }
    });

    it("should use default 5-minute cooldown", () => {
      const d = new SessionDedup();
      expect(d.getConfig().cooldownMs).toBe(5 * 60 * 1000);
    });
  });

  describe("session cleanup", () => {
    it("should clear tracking for a specific session", () => {
      const results = [
        { path: "a.md", startLine: 1, endLine: 5, snippet: "x", score: 0.9 },
      ];
      dedup.filter("s1", results);
      expect(dedup.filter("s1", results)).toHaveLength(0);
      dedup.resetSession("s1");
      expect(dedup.filter("s1", results)).toHaveLength(1);
    });

    it("should not affect other sessions when clearing one", () => {
      const results = [
        { path: "a.md", startLine: 1, endLine: 5, snippet: "x", score: 0.9 },
      ];
      dedup.filter("s1", results);
      dedup.filter("s2", results);
      dedup.resetSession("s1");
      expect(dedup.filter("s1", results)).toHaveLength(1);
      expect(dedup.filter("s2", results)).toHaveLength(0);
    });
  });

  describe("stats", () => {
    it("should track total and deduplicated counts", () => {
      const results = [
        { path: "a.md", startLine: 1, endLine: 5, snippet: "x", score: 0.9 },
        { path: "b.md", startLine: 1, endLine: 5, snippet: "y", score: 0.8 },
      ];
      dedup.filter("s1", results);
      dedup.filter("s1", results); // both duped
      const stats = dedup.getStats();
      expect(stats.totalSeen).toBe(4);
      expect(stats.totalDeduped).toBe(2);
    });

    it("should start with zero stats", () => {
      const stats = dedup.getStats();
      expect(stats.totalSeen).toBe(0);
      expect(stats.totalDeduped).toBe(0);
    });

    it("should expose session state for debugging", () => {
      expect(dedup.getSessionState("s1")).toBeNull();
      const results = [
        { path: "a.md", startLine: 1, endLine: 5, snippet: "x", score: 0.9 },
        { path: "b.md", startLine: 10, endLine: 20, snippet: "y", score: 0.8 },
      ];
      dedup.filter("s1", results);
      const state = dedup.getSessionState("s1");
      expect(state).not.toBeNull();
      expect(state!.keyCount).toBe(2);
      expect(state!.keys).toHaveLength(2);
    });
  });

  describe("memory cleanup", () => {
    it("should prune sessions with maxSessions config", () => {
      const small = new SessionDedup({ maxSessions: 2 });
      const r = [{ path: "a.md", startLine: 1, endLine: 5, snippet: "x", score: 0.9 }];
      small.filter("s1", r);
      small.filter("s2", r);
      small.filter("s3", r); // evicts s1 (oldest), order: [s2, s3]
      // s1 was evicted, so result is allowed again
      expect(small.filter("s1", r)).toHaveLength(1);
      // s3 should still be tracked (was not evicted)
      expect(small.filter("s3", r)).toHaveLength(0);
    });

    it("should verify all non-evicted sessions survive", () => {
      const small = new SessionDedup({ maxSessions: 3 });
      const r = [{ path: "a.md", startLine: 1, endLine: 5, snippet: "x", score: 0.9 }];
      small.filter("s1", r);
      small.filter("s2", r);
      small.filter("s3", r);
      small.filter("s4", r); // evicts s1, order: [s2, s3, s4]
      // Check retained sessions first (won't cause evictions)
      expect(small.filter("s2", r)).toHaveLength(0); // retained
      expect(small.filter("s3", r)).toHaveLength(0); // retained
      expect(small.filter("s4", r)).toHaveLength(0); // retained
      // Use getSessionState to verify s1 was evicted without side effects
      expect(small.getSessionState("s1")).toBeNull();
    });

    it("should move touched sessions to end of LRU queue", () => {
      const small = new SessionDedup({ maxSessions: 2 });
      const r = [{ path: "a.md", startLine: 1, endLine: 5, snippet: "x", score: 0.9 }];
      small.filter("s1", r); // order: [s1]
      small.filter("s2", r); // order: [s1, s2]
      small.filter("s1", r); // touch s1, order: [s2, s1]
      small.filter("s3", r); // evicts s2 (now oldest), order: [s1, s3]
      // Check retained sessions first
      expect(small.filter("s1", r)).toHaveLength(0); // s1 retained (touched)
      expect(small.filter("s3", r)).toHaveLength(0); // s3 retained
      // Verify s2 was evicted
      expect(small.getSessionState("s2")).toBeNull();
    });
  });
});
