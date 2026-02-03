/**
 * Safeguard Metrics Tests (Phase 4 — Issue #11)
 *
 * Central metrics collector + alerting for all safeguards:
 * - Aggregates stats from token budget, circuit breaker, smart triggers, session dedup
 * - Fires alerts when thresholds are crossed
 * - Provides a unified snapshot for the memory_metrics tool
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  SafeguardMetrics,
  MetricsSnapshot,
  Alert,
  AlertLevel,
  AlertCallback,
} from "../src/safeguards/metrics";
import { TokenBudget } from "../src/safeguards/token-budget";
import { CircuitBreaker } from "../src/safeguards/circuit-breaker";
import { SmartTrigger } from "../src/safeguards/smart-triggers";
import { SessionDedup } from "../src/safeguards/session-dedup";

describe("SafeguardMetrics", () => {
  let tokenBudget: TokenBudget;
  let circuitBreaker: CircuitBreaker;
  let smartTrigger: SmartTrigger;
  let sessionDedup: SessionDedup;
  let metrics: SafeguardMetrics;

  beforeEach(() => {
    tokenBudget = new TokenBudget({
      perRecallCap: 500,
      perTurnCap: 750,
      perSessionCap: 5000,
    });
    circuitBreaker = new CircuitBreaker({
      maxConsecutiveEmpty: 5,
      cooldownMs: 300_000,
    });
    smartTrigger = new SmartTrigger();
    sessionDedup = new SessionDedup();
    metrics = new SafeguardMetrics({
      tokenBudget,
      circuitBreaker,
      smartTrigger,
      sessionDedup,
    });
  });

  describe("snapshot", () => {
    it("should return a complete metrics snapshot", () => {
      const snapshot = metrics.snapshot("session-1", "turn-1");
      expect(snapshot).toBeDefined();
      expect(snapshot.tokenBudget).toBeDefined();
      expect(snapshot.circuitBreaker).toBeDefined();
      expect(snapshot.smartTrigger).toBeDefined();
      expect(snapshot.sessionDedup).toBeDefined();
      expect(typeof snapshot.timestamp).toBe("number");
    });

    it("should reflect token budget usage in snapshot", () => {
      tokenBudget.recordUsage("s1", "t1", 300);
      const snapshot = metrics.snapshot("s1", "t1");
      expect(snapshot.tokenBudget.turnUsed).toBe(300);
      expect(snapshot.tokenBudget.sessionUsed).toBe(300);
      expect(snapshot.tokenBudget.turnRemaining).toBe(450); // 750 - 300
      expect(snapshot.tokenBudget.sessionRemaining).toBe(4700); // 5000 - 300
    });

    it("should reflect circuit breaker state in snapshot", () => {
      const snapshot = metrics.snapshot("s1", "t1");
      expect(snapshot.circuitBreaker.tripped).toBe(false);
      expect(snapshot.circuitBreaker.consecutiveEmpty).toBe(0);
    });

    it("should reflect smart trigger stats in snapshot", () => {
      smartTrigger.shouldSkip("hi"); // skipped
      smartTrigger.shouldSkip("what is the architecture of TribalMemory?"); // passed
      const snapshot = metrics.snapshot("s1", "t1");
      expect(snapshot.smartTrigger.totalChecked).toBe(2);
      expect(snapshot.smartTrigger.totalSkipped).toBe(1);
      expect(snapshot.smartTrigger.skipRate).toBe(0.5);
    });

    it("should reflect session dedup stats in snapshot", () => {
      const results = [
        { path: "a.md", startLine: 1, endLine: 5, snippet: "x", score: 0.9 },
      ];
      sessionDedup.filter("s1", results);
      sessionDedup.filter("s1", results); // deduped
      const snapshot = metrics.snapshot("s1", "t1");
      expect(snapshot.sessionDedup.totalSeen).toBe(2);
      expect(snapshot.sessionDedup.totalDeduped).toBe(1);
    });

    it("should include budget utilization percentages", () => {
      tokenBudget.recordUsage("s1", "t1", 400);
      const snapshot = metrics.snapshot("s1", "t1");
      // 400/750 turn, 400/5000 session
      expect(snapshot.tokenBudget.turnUtilization).toBeCloseTo(400 / 750, 2);
      expect(snapshot.tokenBudget.sessionUtilization).toBeCloseTo(400 / 5000, 2);
    });
  });

  describe("alerts", () => {
    it("should fire alert when session budget exceeds 80%", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      // Spread usage across multiple turns so session > 80% but current turn < 80%
      tokenBudget.recordUsage("s1", "t1", 500);
      tokenBudget.recordUsage("s1", "t2", 500);
      tokenBudget.recordUsage("s1", "t3", 500);
      tokenBudget.recordUsage("s1", "t4", 500);
      tokenBudget.recordUsage("s1", "t5", 500);
      tokenBudget.recordUsage("s1", "t6", 500);
      tokenBudget.recordUsage("s1", "t7", 500);
      tokenBudget.recordUsage("s1", "t8", 500);
      tokenBudget.recordUsage("s1", "t9", 100); // total: 4100 = 82% of 5000
      metrics.checkAlerts("s1", "t9");

      expect(alerts).toHaveLength(1);
      expect(alerts[0].level).toBe("warning");
      expect(alerts[0].source).toBe("tokenBudget");
      expect(alerts[0].message).toContain("session budget");
    });

    it("should fire alert when turn budget exceeds 80%", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      // Use 81% of turn budget: 750 * 0.81 = 607.5 → 608
      tokenBudget.recordUsage("s1", "t1", 608);
      metrics.checkAlerts("s1", "t1");

      const turnAlerts = alerts.filter((a) => a.message.includes("turn budget"));
      expect(turnAlerts.length).toBeGreaterThanOrEqual(1);
      expect(turnAlerts[0].level).toBe("warning");
    });

    it("should fire alert when circuit breaker trips", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      // Trip the circuit breaker: 5 consecutive empty results
      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t1");

      const cbAlerts = alerts.filter((a) => a.source === "circuitBreaker");
      expect(cbAlerts).toHaveLength(1);
      expect(cbAlerts[0].level).toBe("warning");
      expect(cbAlerts[0].message).toContain("circuit breaker");
    });

    it("should not fire alert when thresholds are not crossed", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      tokenBudget.recordUsage("s1", "t1", 100); // well under 80%
      metrics.checkAlerts("s1", "t1");

      expect(alerts).toHaveLength(0);
    });

    it("should support multiple alert listeners", () => {
      const a1: Alert[] = [];
      const a2: Alert[] = [];
      metrics.onAlert((a) => a1.push(a));
      metrics.onAlert((a) => a2.push(a));

      // Trip circuit breaker (fires exactly 1 alert)
      for (let i = 0; i < 5; i++) circuitBreaker.recordResult("s1", 0);
      metrics.checkAlerts("s1", "t1");

      expect(a1.length).toBeGreaterThan(0);
      expect(a2.length).toBeGreaterThan(0);
    });

    it("should include session and turn IDs in alerts", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      // Trip circuit breaker for a clean single alert
      for (let i = 0; i < 5; i++) circuitBreaker.recordResult("s1", 0);
      metrics.checkAlerts("s1", "t1");

      expect(alerts[0].sessionId).toBe("s1");
      expect(alerts[0].turnId).toBe("t1");
    });

    it("should not fire duplicate alerts for the same condition in the same turn", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      // Spread across turns to only trigger session alert
      for (let i = 1; i <= 8; i++) {
        tokenBudget.recordUsage("s1", `t${i}`, 500);
      }
      tokenBudget.recordUsage("s1", "t9", 100); // total 4100 = 82%
      metrics.checkAlerts("s1", "t9");
      metrics.checkAlerts("s1", "t9"); // second check same turn

      // Should only fire once per turn per condition
      const sessionAlerts = alerts.filter((a) => a.message.includes("session budget"));
      expect(sessionAlerts).toHaveLength(1);
    });

    it("should fire again in a new turn for ongoing conditions", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      // Spread across turns to only trigger session alert
      for (let i = 1; i <= 8; i++) {
        tokenBudget.recordUsage("s1", `t${i}`, 500);
      }
      tokenBudget.recordUsage("s1", "t9", 100); // total 4100 = 82%
      metrics.checkAlerts("s1", "t9");

      // New turn — condition persists, should fire again
      tokenBudget.recordUsage("s1", "t10", 50);
      metrics.checkAlerts("s1", "t10");

      const sessionAlerts = alerts.filter((a) => a.message.includes("session budget"));
      expect(sessionAlerts).toHaveLength(2);
    });

    it("should support custom alert thresholds", () => {
      const customMetrics = new SafeguardMetrics({
        tokenBudget,
        circuitBreaker,
        smartTrigger,
        sessionDedup,
        thresholds: {
          sessionBudgetWarning: 0.5, // Alert at 50% instead of 80%
          turnBudgetWarning: 0.5,
        },
      });
      const alerts: Alert[] = [];
      customMetrics.onAlert((a) => alerts.push(a));

      // Spread across turns: 51% of session (2550) but current turn under 50%
      for (let i = 1; i <= 7; i++) {
        tokenBudget.recordUsage("s1", `t${i}`, 350);
      }
      tokenBudget.recordUsage("s1", "t8", 100); // total: 2550 = 51% of 5000
      customMetrics.checkAlerts("s1", "t8");

      const sessionAlerts = alerts.filter((a) => a.message.includes("session budget"));
      expect(sessionAlerts).toHaveLength(1);
    });
  });

  describe("removeAlertListener", () => {
    it("should remove a specific listener", () => {
      const alerts: Alert[] = [];
      const cb: AlertCallback = (a) => alerts.push(a);
      metrics.onAlert(cb);
      metrics.removeAlertListener(cb);

      tokenBudget.recordUsage("s1", "t1", 4050);
      metrics.checkAlerts("s1", "t1");

      expect(alerts).toHaveLength(0);
    });
  });

  describe("formatSnapshot", () => {
    it("should format snapshot as human-readable text", () => {
      tokenBudget.recordUsage("s1", "t1", 200);
      smartTrigger.shouldSkip("hi");
      smartTrigger.shouldSkip("tell me about X");

      const text = metrics.formatSnapshot("s1", "t1");
      expect(typeof text).toBe("string");
      expect(text).toContain("Token Budget");
      expect(text).toContain("Circuit Breaker");
      expect(text).toContain("Smart Triggers");
      expect(text).toContain("Session Dedup");
    });

    it("should show utilization percentages", () => {
      tokenBudget.recordUsage("s1", "t1", 375); // 50% of turn
      const text = metrics.formatSnapshot("s1", "t1");
      expect(text).toContain("50"); // should show 50%
    });
  });

  describe("reset", () => {
    it("should clear alert dedup state", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      // Trip circuit breaker for a clean single alert
      for (let i = 0; i < 5; i++) circuitBreaker.recordResult("s1", 0);
      metrics.checkAlerts("s1", "t1");
      expect(alerts).toHaveLength(1);

      metrics.resetAlertState();
      metrics.checkAlerts("s1", "t1"); // should fire again after reset
      expect(alerts).toHaveLength(2);
    });
  });
});
