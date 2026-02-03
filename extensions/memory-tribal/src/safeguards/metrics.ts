/**
 * Safeguard Metrics & Alerting (Phase 4 — Issue #11)
 *
 * Central metrics collector that aggregates stats from all safeguard modules
 * and fires alerts when thresholds are crossed.
 *
 * Usage:
 *   const metrics = new SafeguardMetrics({ tokenBudget, circuitBreaker, smartTrigger, sessionDedup });
 *   metrics.onAlert(alert => api.log.warn(`[safeguard] ${alert.message}`));
 *   // After each memory_search:
 *   const snapshot = metrics.snapshot(sessionId, turnId);
 *   metrics.checkAlerts(sessionId, turnId);
 */

import { TokenBudget } from "./token-budget";
import { CircuitBreaker } from "./circuit-breaker";
import { SmartTrigger } from "./smart-triggers";
import { SessionDedup } from "./session-dedup";

export type AlertLevel = "info" | "warning" | "error";

export interface Alert {
  level: AlertLevel;
  source: "tokenBudget" | "circuitBreaker" | "smartTrigger" | "sessionDedup";
  message: string;
  sessionId: string;
  turnId: string;
  timestamp: number;
}

export type AlertCallback = (alert: Alert) => void;

export interface AlertThresholds {
  /** Fire warning when session budget utilization exceeds this (0-1). Default: 0.8 */
  sessionBudgetWarning: number;
  /** Fire warning when turn budget utilization exceeds this (0-1). Default: 0.8 */
  turnBudgetWarning: number;
}

const DEFAULT_THRESHOLDS: AlertThresholds = {
  sessionBudgetWarning: 0.8,
  turnBudgetWarning: 0.8,
};

export interface MetricsSnapshot {
  timestamp: number;
  tokenBudget: {
    turnUsed: number;
    turnRemaining: number;
    sessionUsed: number;
    sessionRemaining: number;
    turnUtilization: number;
    sessionUtilization: number;
  };
  circuitBreaker: {
    tripped: boolean;
    consecutiveEmpty: number;
    cooldownRemaining: number | null;
  };
  smartTrigger: {
    totalChecked: number;
    totalSkipped: number;
    totalPassed: number;
    skipRate: number;
  };
  sessionDedup: {
    totalSeen: number;
    totalDeduped: number;
  };
}

export interface SafeguardMetricsConfig {
  tokenBudget: TokenBudget;
  circuitBreaker: CircuitBreaker;
  smartTrigger: SmartTrigger;
  sessionDedup: SessionDedup;
  thresholds?: Partial<AlertThresholds>;
}

export class SafeguardMetrics {
  private tokenBudget: TokenBudget;
  private circuitBreaker: CircuitBreaker;
  private smartTrigger: SmartTrigger;
  private sessionDedup: SessionDedup;
  private thresholds: AlertThresholds;

  private listeners: AlertCallback[] = [];
  /** Track fired alerts to prevent duplicates within same turn: "source:condition:turnId" */
  private firedAlerts = new Set<string>();

  constructor(config: SafeguardMetricsConfig) {
    this.tokenBudget = config.tokenBudget;
    this.circuitBreaker = config.circuitBreaker;
    this.smartTrigger = config.smartTrigger;
    this.sessionDedup = config.sessionDedup;
    this.thresholds = { ...DEFAULT_THRESHOLDS, ...config.thresholds };
  }

  /**
   * Take a snapshot of all safeguard metrics for the given session/turn.
   */
  snapshot(sessionId: string, turnId: string): MetricsSnapshot {
    const budgetConfig = this.tokenBudget.getConfig();
    const usage = this.tokenBudget.getUsage(sessionId, turnId);
    const cbStatus = this.circuitBreaker.getStatus(sessionId);
    const triggerStats = this.smartTrigger.getStats();
    const dedupStats = this.sessionDedup.getStats();

    return {
      timestamp: Date.now(),
      tokenBudget: {
        turnUsed: usage.turn,
        turnRemaining: usage.turnRemaining,
        sessionUsed: usage.session,
        sessionRemaining: usage.sessionRemaining,
        turnUtilization: budgetConfig.perTurnCap > 0
          ? usage.turn / budgetConfig.perTurnCap
          : 0,
        sessionUtilization: budgetConfig.perSessionCap > 0
          ? usage.session / budgetConfig.perSessionCap
          : 0,
      },
      circuitBreaker: {
        tripped: cbStatus.tripped,
        consecutiveEmpty: cbStatus.consecutiveEmpty,
        cooldownRemaining: cbStatus.cooldownRemaining,
      },
      smartTrigger: {
        totalChecked: triggerStats.totalChecked,
        totalSkipped: triggerStats.totalSkipped,
        totalPassed: triggerStats.totalPassed,
        skipRate: triggerStats.skipRate,
      },
      sessionDedup: {
        totalSeen: dedupStats.totalSeen,
        totalDeduped: dedupStats.totalDeduped,
      },
    };
  }

  /**
   * Check all alert conditions and fire callbacks for any crossed thresholds.
   * Deduplicates alerts within the same turn to prevent spamming.
   */
  checkAlerts(sessionId: string, turnId: string): void {
    const snap = this.snapshot(sessionId, turnId);

    // Session budget warning
    if (snap.tokenBudget.sessionUtilization >= this.thresholds.sessionBudgetWarning) {
      this.fireAlert({
        level: "warning",
        source: "tokenBudget",
        message: `session budget at ${(snap.tokenBudget.sessionUtilization * 100).toFixed(0)}% (${snap.tokenBudget.sessionUsed}/${snap.tokenBudget.sessionUsed + snap.tokenBudget.sessionRemaining} tokens)`,
        sessionId,
        turnId,
        timestamp: Date.now(),
      });
    }

    // Turn budget warning
    if (snap.tokenBudget.turnUtilization >= this.thresholds.turnBudgetWarning) {
      this.fireAlert({
        level: "warning",
        source: "tokenBudget",
        message: `turn budget at ${(snap.tokenBudget.turnUtilization * 100).toFixed(0)}% (${snap.tokenBudget.turnUsed}/${snap.tokenBudget.turnUsed + snap.tokenBudget.turnRemaining} tokens)`,
        sessionId,
        turnId,
        timestamp: Date.now(),
      });
    }

    // Circuit breaker tripped
    if (snap.circuitBreaker.tripped) {
      this.fireAlert({
        level: "warning",
        source: "circuitBreaker",
        message: `circuit breaker tripped after ${snap.circuitBreaker.consecutiveEmpty} consecutive empty recalls`,
        sessionId,
        turnId,
        timestamp: Date.now(),
      });
    }
  }

  /**
   * Register an alert listener.
   */
  onAlert(callback: AlertCallback): void {
    this.listeners.push(callback);
  }

  /**
   * Remove a specific alert listener.
   */
  removeAlertListener(callback: AlertCallback): void {
    this.listeners = this.listeners.filter((cb) => cb !== callback);
  }

  /**
   * Clear the alert dedup state (e.g., on session reset).
   */
  resetAlertState(): void {
    this.firedAlerts.clear();
  }

  /**
   * Format a snapshot as a human-readable string for the memory_metrics tool.
   */
  formatSnapshot(sessionId: string, turnId: string): string {
    const snap = this.snapshot(sessionId, turnId);
    const lines: string[] = [];

    lines.push("## Safeguard Metrics");
    lines.push("");

    lines.push("### Token Budget");
    lines.push(`- Turn: ${snap.tokenBudget.turnUsed} used / ${snap.tokenBudget.turnUsed + snap.tokenBudget.turnRemaining} cap (${(snap.tokenBudget.turnUtilization * 100).toFixed(0)}%)`);
    lines.push(`- Session: ${snap.tokenBudget.sessionUsed} used / ${snap.tokenBudget.sessionUsed + snap.tokenBudget.sessionRemaining} cap (${(snap.tokenBudget.sessionUtilization * 100).toFixed(0)}%)`);
    lines.push("");

    lines.push("### Circuit Breaker");
    lines.push(`- Status: ${snap.circuitBreaker.tripped ? "TRIPPED" : "OK"}`);
    lines.push(`- Consecutive empty: ${snap.circuitBreaker.consecutiveEmpty}`);
    if (snap.circuitBreaker.cooldownRemaining != null) {
      lines.push(`- Cooldown remaining: ${(snap.circuitBreaker.cooldownRemaining / 1000).toFixed(0)}s`);
    }
    lines.push("");

    lines.push("### Smart Triggers");
    lines.push(`- Checked: ${snap.smartTrigger.totalChecked}`);
    lines.push(`- Skipped: ${snap.smartTrigger.totalSkipped} (${(snap.smartTrigger.skipRate * 100).toFixed(0)}%)`);
    lines.push(`- Passed: ${snap.smartTrigger.totalPassed}`);
    lines.push("");

    lines.push("### Session Dedup");
    lines.push(`- Total seen: ${snap.sessionDedup.totalSeen}`);
    lines.push(`- Deduplicated: ${snap.sessionDedup.totalDeduped}`);

    return lines.join("\n");
  }

  /**
   * Fire an alert to all listeners, with per-turn deduplication.
   */
  private fireAlert(alert: Alert): void {
    const dedupKey = `${alert.source}:${alert.message.split(" ")[0]}:${alert.turnId}`;
    if (this.firedAlerts.has(dedupKey)) return;
    this.firedAlerts.add(dedupKey);

    for (const listener of this.listeners) {
      listener(alert);
    }
  }
}
