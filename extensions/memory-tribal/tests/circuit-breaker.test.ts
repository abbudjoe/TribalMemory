/**
 * Unit tests for CircuitBreaker
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { CircuitBreaker, DEFAULT_CIRCUIT_BREAKER_CONFIG } from "../src/safeguards/circuit-breaker";

describe("CircuitBreaker", () => {
  let breaker: CircuitBreaker;

  beforeEach(() => {
    breaker = new CircuitBreaker();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe("isTripped()", () => {
    it("returns false for new session", () => {
      expect(breaker.isTripped("session1")).toBe(false);
    });

    it("returns true after max consecutive empty results", () => {
      const sessionId = "session1";
      
      // Record 5 empty results (default threshold)
      for (let i = 0; i < DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty; i++) {
        breaker.recordResult(sessionId, 0);
      }

      expect(breaker.isTripped(sessionId)).toBe(true);
    });

    it("auto-resets after cooldown period", () => {
      const sessionId = "session1";
      
      // Trip the breaker
      for (let i = 0; i < DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty; i++) {
        breaker.recordResult(sessionId, 0);
      }
      
      expect(breaker.isTripped(sessionId)).toBe(true);

      // Fast-forward past cooldown (5 minutes)
      vi.advanceTimersByTime(DEFAULT_CIRCUIT_BREAKER_CONFIG.cooldownMs + 1000);

      expect(breaker.isTripped(sessionId)).toBe(false);
    });

    it("remains tripped during cooldown", () => {
      const sessionId = "session1";
      
      // Trip the breaker
      for (let i = 0; i < DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty; i++) {
        breaker.recordResult(sessionId, 0);
      }

      // Fast-forward halfway through cooldown
      vi.advanceTimersByTime(DEFAULT_CIRCUIT_BREAKER_CONFIG.cooldownMs / 2);

      expect(breaker.isTripped(sessionId)).toBe(true);
    });
  });

  describe("recordResult()", () => {
    it("increments consecutive empty count on zero results", () => {
      const sessionId = "session1";
      
      breaker.recordResult(sessionId, 0);
      breaker.recordResult(sessionId, 0);
      
      const status = breaker.getStatus(sessionId);
      expect(status.consecutiveEmpty).toBe(2);
    });

    it("resets consecutive empty count on successful result", () => {
      const sessionId = "session1";
      
      breaker.recordResult(sessionId, 0);
      breaker.recordResult(sessionId, 0);
      breaker.recordResult(sessionId, 5); // Success
      
      const status = breaker.getStatus(sessionId);
      expect(status.consecutiveEmpty).toBe(0);
    });

    it("trips breaker after threshold consecutive empty results", () => {
      const sessionId = "session1";
      
      for (let i = 0; i < DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty - 1; i++) {
        breaker.recordResult(sessionId, 0);
      }
      
      expect(breaker.isTripped(sessionId)).toBe(false);
      
      breaker.recordResult(sessionId, 0); // One more to trip
      
      expect(breaker.isTripped(sessionId)).toBe(true);
    });

    it("does not trip if successful result resets count", () => {
      const sessionId = "session1";
      
      for (let i = 0; i < DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty - 1; i++) {
        breaker.recordResult(sessionId, 0);
      }
      
      breaker.recordResult(sessionId, 3); // Success before threshold
      
      // Add more empty results
      for (let i = 0; i < DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty - 1; i++) {
        breaker.recordResult(sessionId, 0);
      }
      
      expect(breaker.isTripped(sessionId)).toBe(false);
    });
  });

  describe("reset()", () => {
    it("manually resets breaker state", () => {
      const sessionId = "session1";
      
      // Trip the breaker
      for (let i = 0; i < DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty; i++) {
        breaker.recordResult(sessionId, 0);
      }
      
      expect(breaker.isTripped(sessionId)).toBe(true);
      
      breaker.reset(sessionId);
      
      expect(breaker.isTripped(sessionId)).toBe(false);
      expect(breaker.getStatus(sessionId).consecutiveEmpty).toBe(0);
    });
  });

  describe("getStatus()", () => {
    it("returns status for new session", () => {
      const status = breaker.getStatus("session1");
      
      expect(status.consecutiveEmpty).toBe(0);
      expect(status.tripped).toBe(false);
      expect(status.tripTime).toBe(null);
      expect(status.cooldownRemaining).toBe(null);
    });

    it("returns status for tripped breaker", () => {
      const sessionId = "session1";
      
      // Trip the breaker
      for (let i = 0; i < DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty; i++) {
        breaker.recordResult(sessionId, 0);
      }
      
      const status = breaker.getStatus(sessionId);
      
      expect(status.consecutiveEmpty).toBe(DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty);
      expect(status.tripped).toBe(true);
      expect(status.tripTime).toBeGreaterThan(0);
      expect(status.cooldownRemaining).toBeGreaterThan(0);
    });

    it("updates cooldown remaining as time passes", () => {
      const sessionId = "session1";
      
      // Trip the breaker
      for (let i = 0; i < DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty; i++) {
        breaker.recordResult(sessionId, 0);
      }
      
      const status1 = breaker.getStatus(sessionId);
      const remaining1 = status1.cooldownRemaining ?? 0;
      
      // Fast-forward 1 minute
      vi.advanceTimersByTime(60 * 1000);
      
      const status2 = breaker.getStatus(sessionId);
      const remaining2 = status2.cooldownRemaining ?? 0;
      
      expect(remaining2).toBeLessThan(remaining1);
      expect(remaining1 - remaining2).toBeCloseTo(60 * 1000, -2); // Within ~100ms
    });
  });

  describe("session isolation", () => {
    it("isolates breaker state between sessions", () => {
      // Trip breaker for session1
      for (let i = 0; i < DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty; i++) {
        breaker.recordResult("session1", 0);
      }
      
      expect(breaker.isTripped("session1")).toBe(true);
      expect(breaker.isTripped("session2")).toBe(false);
      
      // session2 should work independently
      breaker.recordResult("session2", 0);
      breaker.recordResult("session2", 0);
      
      expect(breaker.getStatus("session2").consecutiveEmpty).toBe(2);
      expect(breaker.isTripped("session2")).toBe(false);
    });
  });

  describe("custom config", () => {
    it("respects custom max consecutive empty", () => {
      const customBreaker = new CircuitBreaker({
        maxConsecutiveEmpty: 3,
        cooldownMs: 60000,
      });

      const sessionId = "session1";
      
      customBreaker.recordResult(sessionId, 0);
      customBreaker.recordResult(sessionId, 0);
      expect(customBreaker.isTripped(sessionId)).toBe(false);
      
      customBreaker.recordResult(sessionId, 0); // Third empty
      expect(customBreaker.isTripped(sessionId)).toBe(true);
    });

    it("respects custom cooldown duration", () => {
      const customBreaker = new CircuitBreaker({
        maxConsecutiveEmpty: 2,
        cooldownMs: 10000, // 10 seconds
      });

      const sessionId = "session1";
      
      // Trip breaker
      customBreaker.recordResult(sessionId, 0);
      customBreaker.recordResult(sessionId, 0);
      
      expect(customBreaker.isTripped(sessionId)).toBe(true);
      
      // Fast-forward 9 seconds (still tripped)
      vi.advanceTimersByTime(9000);
      expect(customBreaker.isTripped(sessionId)).toBe(true);
      
      // Fast-forward past cooldown
      vi.advanceTimersByTime(2000);
      expect(customBreaker.isTripped(sessionId)).toBe(false);
    });
  });

  describe("getConfig()", () => {
    it("returns config copy", () => {
      const config = breaker.getConfig();
      expect(config.maxConsecutiveEmpty).toBe(DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty);
      expect(config.cooldownMs).toBe(DEFAULT_CIRCUIT_BREAKER_CONFIG.cooldownMs);
      
      // Verify it's a copy
      config.maxConsecutiveEmpty = 999;
      expect(breaker.getConfig().maxConsecutiveEmpty).toBe(DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConsecutiveEmpty);
    });
  });
});
