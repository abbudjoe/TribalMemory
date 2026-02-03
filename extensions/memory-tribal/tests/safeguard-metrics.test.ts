/**
 * Safeguard Metrics Tests (Phase 4 — Issue #11)
 *
 * Central metrics collector + transition-based alerting for all
 * safeguards: token budget, circuit breaker, smart triggers,
 * session dedup.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  SafeguardMetrics,
  MetricsSnapshot,
  Alert,
  AlertLevel,
  AlertCondition,
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

  /* ============================================================== */
  /*  snapshot                                                       */
  /* ============================================================== */

  describe("snapshot", () => {
    it("should return a complete metrics snapshot", () => {
      const snap = metrics.snapshot("s1", "t1");
      expect(snap).toBeDefined();
      expect(snap.tokenBudget).toBeDefined();
      expect(snap.circuitBreaker).toBeDefined();
      expect(snap.smartTrigger).toBeDefined();
      expect(snap.sessionDedup).toBeDefined();
      expect(typeof snap.timestamp).toBe("number");
    });

    it("should reflect token budget usage", () => {
      tokenBudget.recordUsage("s1", "t1", 300);
      const snap = metrics.snapshot("s1", "t1");
      expect(snap.tokenBudget.turnUsed).toBe(300);
      expect(snap.tokenBudget.sessionUsed).toBe(300);
      expect(snap.tokenBudget.turnRemaining).toBe(450);
      expect(snap.tokenBudget.sessionRemaining).toBe(4700);
    });

    it("should include actual caps from config", () => {
      const snap = metrics.snapshot("s1", "t1");
      expect(snap.tokenBudget.turnCap).toBe(750);
      expect(snap.tokenBudget.sessionCap).toBe(5000);
    });

    it("should include utilization percentages", () => {
      tokenBudget.recordUsage("s1", "t1", 400);
      const snap = metrics.snapshot("s1", "t1");
      expect(snap.tokenBudget.turnUtilization)
        .toBeCloseTo(400 / 750, 4);
      expect(snap.tokenBudget.sessionUtilization)
        .toBeCloseTo(400 / 5000, 4);
    });

    it("should reflect circuit breaker state", () => {
      const snap = metrics.snapshot("s1", "t1");
      expect(snap.circuitBreaker.tripped).toBe(false);
      expect(snap.circuitBreaker.consecutiveEmpty).toBe(0);
    });

    it("should reflect smart trigger stats", () => {
      smartTrigger.shouldSkip("hi");
      smartTrigger.shouldSkip("architecture of TribalMemory?");
      const snap = metrics.snapshot("s1", "t1");
      expect(snap.smartTrigger.totalChecked).toBe(2);
      expect(snap.smartTrigger.totalSkipped).toBe(1);
      expect(snap.smartTrigger.skipRate).toBe(0.5);
    });

    it("should reflect session dedup stats", () => {
      const results = [
        { path: "a.md", startLine: 1, endLine: 5,
          snippet: "x", score: 0.9 },
      ];
      sessionDedup.filter("s1", results);
      sessionDedup.filter("s1", results);
      const snap = metrics.snapshot("s1", "t1");
      expect(snap.sessionDedup.totalSeen).toBe(2);
      expect(snap.sessionDedup.totalDeduped).toBe(1);
    });

    it("should handle NaN / non-finite utilization safely", () => {
      // Zero-cap budget → utilization should be 0, not NaN
      const zeroBudget = new TokenBudget({
        perRecallCap: 0,
        perTurnCap: 0,
        perSessionCap: 0,
      });
      const m = new SafeguardMetrics({
        tokenBudget: zeroBudget,
        circuitBreaker,
        smartTrigger,
        sessionDedup,
      });
      const snap = m.snapshot("s1", "t1");
      expect(snap.tokenBudget.turnUtilization).toBe(0);
      expect(snap.tokenBudget.sessionUtilization).toBe(0);
    });
  });

  /* ============================================================== */
  /*  transition-based alerts                                        */
  /* ============================================================== */

  describe("alerts", () => {
    it("should fire when session budget crosses threshold", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      // Spread across turns: 82% of session, current turn < 80%
      for (let i = 1; i <= 8; i++) {
        tokenBudget.recordUsage("s1", `t${i}`, 500);
      }
      tokenBudget.recordUsage("s1", "t9", 100);
      metrics.checkAlerts("s1", "t9");

      expect(alerts).toHaveLength(1);
      expect(alerts[0].level).toBe("warning");
      expect(alerts[0].source).toBe("tokenBudget");
      expect(alerts[0].condition).toBe("session_budget_high");
      expect(alerts[0].message).toContain("session budget");
      expect(alerts[0].message).toContain("5000");
    });

    it("should fire when turn budget crosses threshold", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      tokenBudget.recordUsage("s1", "t1", 608);
      metrics.checkAlerts("s1", "t1");

      const turnAlerts = alerts.filter(
        (a) => a.condition === "turn_budget_high",
      );
      expect(turnAlerts.length).toBeGreaterThanOrEqual(1);
      expect(turnAlerts[0].level).toBe("warning");
      expect(turnAlerts[0].message).toContain("750");
    });

    it("should fire when circuit breaker trips", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t1");

      const cb = alerts.filter(
        (a) => a.condition === "circuit_breaker_tripped",
      );
      expect(cb).toHaveLength(1);
      expect(cb[0].level).toBe("warning");
      expect(cb[0].message).toContain("circuit breaker");
    });

    it("should not fire when thresholds are not crossed", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      tokenBudget.recordUsage("s1", "t1", 100);
      metrics.checkAlerts("s1", "t1");

      expect(alerts).toHaveLength(0);
    });

    it("should include session and turn IDs in alerts", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t7");

      expect(alerts[0].sessionId).toBe("s1");
      expect(alerts[0].turnId).toBe("t7");
    });

    // --- transition-based dedup (#3, #1) ---

    it("should NOT re-fire for same condition in later turns " +
       "(transition-based, not per-turn)", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      for (let i = 1; i <= 8; i++) {
        tokenBudget.recordUsage("s1", `t${i}`, 500);
      }
      tokenBudget.recordUsage("s1", "t9", 100);
      metrics.checkAlerts("s1", "t9");
      expect(alerts).toHaveLength(1);

      // Still over 80% in next turn — should NOT fire again
      tokenBudget.recordUsage("s1", "t10", 50);
      metrics.checkAlerts("s1", "t10");

      const sessionAlerts = alerts.filter(
        (a) => a.condition === "session_budget_high",
      );
      expect(sessionAlerts).toHaveLength(1); // still just 1
    });

    it("should re-fire if condition clears then re-crosses", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      // Trip circuit breaker
      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t1");
      expect(alerts).toHaveLength(1);

      // Clear it
      circuitBreaker.reset("s1");
      metrics.checkAlerts("s1", "t2");
      expect(alerts).toHaveLength(1); // no new alert

      // Trip again
      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t3");
      expect(alerts).toHaveLength(2); // re-fired
    });

    it("should not fire duplicate in same turn", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t1");
      metrics.checkAlerts("s1", "t1");

      expect(alerts).toHaveLength(1);
    });

    // --- multiple listeners (#4 error isolation) ---

    it("should deliver to all listeners", () => {
      const a1: Alert[] = [];
      const a2: Alert[] = [];
      metrics.onAlert((a) => a1.push(a));
      metrics.onAlert((a) => a2.push(a));

      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t1");

      expect(a1).toHaveLength(1);
      expect(a2).toHaveLength(1);
    });

    it("should not break other listeners when one throws", () => {
      const delivered: Alert[] = [];
      metrics.onAlert(() => {
        throw new Error("boom");
      });
      metrics.onAlert((a) => delivered.push(a));

      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t1");

      expect(delivered).toHaveLength(1);
    });

    // --- custom thresholds ---

    it("should support custom alert thresholds", () => {
      const custom = new SafeguardMetrics({
        tokenBudget,
        circuitBreaker,
        smartTrigger,
        sessionDedup,
        thresholds: {
          sessionBudgetWarning: 0.5,
          turnBudgetWarning: 0.5,
        },
      });
      const alerts: Alert[] = [];
      custom.onAlert((a) => alerts.push(a));

      // 51% session, current turn < 50%
      for (let i = 1; i <= 7; i++) {
        tokenBudget.recordUsage("s1", `t${i}`, 350);
      }
      tokenBudget.recordUsage("s1", "t8", 100);
      custom.checkAlerts("s1", "t8");

      const sessionAlerts = alerts.filter(
        (a) => a.condition === "session_budget_high",
      );
      expect(sessionAlerts).toHaveLength(1);
    });

    // --- edge: thresholds at 0.0 and 1.0 ---

    it("should fire immediately when threshold is 0", () => {
      const m = new SafeguardMetrics({
        tokenBudget,
        circuitBreaker,
        smartTrigger,
        sessionDedup,
        thresholds: {
          sessionBudgetWarning: 0,
          turnBudgetWarning: 0,
        },
      });
      const alerts: Alert[] = [];
      m.onAlert((a) => alerts.push(a));

      // Even with 0 usage, utilization ≥ 0 is true
      m.checkAlerts("s1", "t1");
      expect(alerts.length).toBeGreaterThanOrEqual(1);
    });

    it("should never fire when threshold is 1.0 (unless at 100%)", () => {
      const m = new SafeguardMetrics({
        tokenBudget,
        circuitBreaker,
        smartTrigger,
        sessionDedup,
        thresholds: {
          sessionBudgetWarning: 1.0,
          turnBudgetWarning: 1.0,
        },
      });
      const alerts: Alert[] = [];
      m.onAlert((a) => alerts.push(a));

      tokenBudget.recordUsage("s1", "t1", 400);
      m.checkAlerts("s1", "t1");
      const budgetAlerts = alerts.filter(
        (a) => a.source === "tokenBudget",
      );
      expect(budgetAlerts).toHaveLength(0);
    });

    // --- simultaneous alerts from multiple safeguards ---

    it("should fire alerts from all safeguards simultaneously", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      // Trip circuit breaker
      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      // Exceed session + turn budget
      tokenBudget.recordUsage("s1", "t1", 700); // turn 93%
      for (let i = 2; i <= 7; i++) {
        tokenBudget.recordUsage("s1", `t${i}`, 700);
      }
      // session = 4900/5000 = 98%, current turn t1 = 93%
      metrics.checkAlerts("s1", "t1");

      const conditions = alerts.map((a) => a.condition);
      expect(conditions).toContain("session_budget_high");
      expect(conditions).toContain("turn_budget_high");
      expect(conditions).toContain("circuit_breaker_tripped");
      expect(alerts).toHaveLength(3);
    });
  });

  /* ============================================================== */
  /*  removeAlertListener                                            */
  /* ============================================================== */

  describe("removeAlertListener", () => {
    it("should remove a specific listener", () => {
      const alerts: Alert[] = [];
      const cb: AlertCallback = (a) => alerts.push(a);
      metrics.onAlert(cb);
      metrics.removeAlertListener(cb);

      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t1");

      expect(alerts).toHaveLength(0);
    });

    it("should not affect other listeners when removing one", () => {
      const a1: Alert[] = [];
      const a2: Alert[] = [];
      const cb1: AlertCallback = (a) => a1.push(a);
      const cb2: AlertCallback = (a) => a2.push(a);
      metrics.onAlert(cb1);
      metrics.onAlert(cb2);
      metrics.removeAlertListener(cb1);

      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t1");

      expect(a1).toHaveLength(0);
      expect(a2).toHaveLength(1);
    });

    it("should handle adding listener after alerts fired", () => {
      // Fire an alert before any listener is registered
      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t1");

      // Now add listener — condition already active, so no fire
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));
      metrics.checkAlerts("s1", "t2");
      expect(alerts).toHaveLength(0);

      // Clear and re-trip → new listener should see it
      circuitBreaker.reset("s1");
      metrics.checkAlerts("s1", "t3"); // clears active
      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t4");
      expect(alerts).toHaveLength(1);
    });
  });

  /* ============================================================== */
  /*  alert history (#12)                                            */
  /* ============================================================== */

  describe("alertHistory", () => {
    it("should record fired alerts in history", () => {
      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t1");

      const history = metrics.getAlertHistory();
      expect(history).toHaveLength(1);
      expect(history[0].condition)
        .toBe("circuit_breaker_tripped");
    });

    it("should return empty array when no alerts fired", () => {
      expect(metrics.getAlertHistory()).toHaveLength(0);
    });

    it("should cap history at 100 entries", () => {
      // Fire 101 distinct alerts by cycling circuit breaker
      for (let round = 0; round < 101; round++) {
        for (let i = 0; i < 5; i++) {
          circuitBreaker.recordResult("s1", 0);
        }
        metrics.checkAlerts("s1", `t-${round}`);
        // Clear so it can re-fire
        circuitBreaker.reset("s1");
        metrics.checkAlerts("s1", `t-clear-${round}`);
        metrics.resetAlertState();
      }

      expect(metrics.getAlertHistory().length)
        .toBeLessThanOrEqual(100);
    });

    it("should clear history on resetAlertState", () => {
      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t1");
      expect(metrics.getAlertHistory()).toHaveLength(1);

      metrics.resetAlertState();
      expect(metrics.getAlertHistory()).toHaveLength(0);
    });
  });

  /* ============================================================== */
  /*  formatSnapshotMarkdown (#8 rename)                             */
  /* ============================================================== */

  describe("formatSnapshotMarkdown", () => {
    it("should return markdown with all sections", () => {
      tokenBudget.recordUsage("s1", "t1", 200);
      smartTrigger.shouldSkip("hi");
      smartTrigger.shouldSkip("tell me about X");

      const text = metrics.formatSnapshotMarkdown("s1", "t1");
      expect(typeof text).toBe("string");
      expect(text).toContain("Token Budget");
      expect(text).toContain("Circuit Breaker");
      expect(text).toContain("Smart Triggers");
      expect(text).toContain("Session Dedup");
    });

    it("should show utilization percentages", () => {
      tokenBudget.recordUsage("s1", "t1", 375);
      const text = metrics.formatSnapshotMarkdown("s1", "t1");
      expect(text).toContain("50%");
    });

    it("should show caps from config, not computed", () => {
      tokenBudget.recordUsage("s1", "t1", 200);
      const text = metrics.formatSnapshotMarkdown("s1", "t1");
      expect(text).toContain("750 cap");
      expect(text).toContain("5000 cap");
    });

    it("should handle NaN utilization gracefully", () => {
      const zeroBudget = new TokenBudget({
        perRecallCap: 0,
        perTurnCap: 0,
        perSessionCap: 0,
      });
      const m = new SafeguardMetrics({
        tokenBudget: zeroBudget,
        circuitBreaker,
        smartTrigger,
        sessionDedup,
      });
      const text = m.formatSnapshotMarkdown("s1", "t1");
      expect(text).toContain("0%");
      expect(text).not.toContain("NaN");
    });
  });

  /* ============================================================== */
  /*  resetAlertState                                                */
  /* ============================================================== */

  describe("resetAlertState", () => {
    it("should allow alerts to re-fire after reset", () => {
      const alerts: Alert[] = [];
      metrics.onAlert((a) => alerts.push(a));

      for (let i = 0; i < 5; i++) {
        circuitBreaker.recordResult("s1", 0);
      }
      metrics.checkAlerts("s1", "t1");
      expect(alerts).toHaveLength(1);

      metrics.resetAlertState();
      // Condition still active, but state was reset → re-fires
      metrics.checkAlerts("s1", "t2");
      expect(alerts).toHaveLength(2);
    });
  });
});
