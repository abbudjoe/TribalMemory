/**
 * Safeguard Metrics & Alerting (Phase 4 — Issue #11)
 *
 * Central metrics collector that aggregates stats from all safeguard
 * modules and fires alerts when thresholds are crossed.
 *
 * ## Alert lifecycle
 *
 * - Call `checkAlerts(sessionId, turnId)` once per `memory_search`
 *   call (Step 12 in the pipeline, after session dedup).
 * - Alerts fire on **threshold transitions** only:
 *   not-crossed → crossed fires once; stays active → suppressed.
 * - When the condition clears (e.g. budget drops below threshold),
 *   the next crossing will fire again.
 * - Call `resetAlertState()` on session reset to clear tracking.
 *
 * ## Integration note (Step 12 placement)
 *
 * `checkAlerts` runs after session dedup (Step 10). This means
 * deduped results are already filtered from the returned set but
 * are still counted in the global dedup stats. Token budget usage
 * is recorded in Step 9 (before dedup), so budget alerts reflect
 * all tokens that were *considered*, not just those returned.
 *
 * ## Usage
 *
 *   const m = new SafeguardMetrics({ tokenBudget, ... });
 *   m.onAlert(a => api.log.warn(`[safeguard] ${a.message}`));
 *   m.checkAlerts(sessionId, turnId);
 *   // Inspect:
 *   const history = m.getAlertHistory();
 */

import { TokenBudget } from "./token-budget";
import { CircuitBreaker } from "./circuit-breaker";
import { SmartTrigger } from "./smart-triggers";
import { SessionDedup } from "./session-dedup";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export type AlertLevel = "info" | "warning" | "error";

/** Explicit alert condition identifiers — no message-parsing. */
export type AlertCondition =
  | "session_budget_high"
  | "turn_budget_high"
  | "circuit_breaker_tripped";

export interface Alert {
  level: AlertLevel;
  source: "tokenBudget" | "circuitBreaker";
  condition: AlertCondition;
  message: string;
  sessionId: string;
  turnId: string;
  timestamp: number;
}

export type AlertCallback = (alert: Alert) => void;

export interface AlertThresholds {
  /** Warn when session budget utilization ≥ this (0–1). */
  sessionBudgetWarning: number;
  /** Warn when turn budget utilization ≥ this (0–1). */
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
    turnCap: number;
    sessionUsed: number;
    sessionRemaining: number;
    sessionCap: number;
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

/* ------------------------------------------------------------------ */
/*  Implementation                                                     */
/* ------------------------------------------------------------------ */

/** Max alerts retained in history before oldest are evicted. */
const MAX_ALERT_HISTORY = 100;

export class SafeguardMetrics {
  private tokenBudget: TokenBudget;
  private circuitBreaker: CircuitBreaker;
  private smartTrigger: SmartTrigger;
  private sessionDedup: SessionDedup;
  private thresholds: AlertThresholds;

  private listeners: AlertCallback[] = [];

  /**
   * Tracks which conditions are currently active so we only
   * fire on transitions. Key: `condition:sessionId`.
   */
  private activeConditions = new Set<string>();

  /** Rolling alert history for debugging / inspection. */
  private alertHistory: Alert[] = [];

  constructor(config: SafeguardMetricsConfig) {
    this.tokenBudget = config.tokenBudget;
    this.circuitBreaker = config.circuitBreaker;
    this.smartTrigger = config.smartTrigger;
    this.sessionDedup = config.sessionDedup;
    this.thresholds = {
      ...DEFAULT_THRESHOLDS,
      ...config.thresholds,
    };
  }

  /* -------------------------------------------------------------- */
  /*  Snapshot                                                       */
  /* -------------------------------------------------------------- */

  /**
   * Take a point-in-time snapshot of all safeguard metrics.
   *
   * @param sessionId - Session to query budgets/breaker for.
   * @param turnId    - Turn to query turn-level budget for.
   * @returns Typed snapshot; all fields are guaranteed non-null
   *          (circuit breaker `cooldownRemaining` is null when
   *          the breaker is not tripped).
   */
  snapshot(sessionId: string, turnId: string): MetricsSnapshot {
    const budgetCfg = this.tokenBudget.getConfig();
    const usage = this.tokenBudget.getUsage(sessionId, turnId);
    const cbStatus = this.circuitBreaker.getStatus(sessionId);
    const triggerStats = this.smartTrigger.getStats();
    const dedupStats = this.sessionDedup.getStats();

    // Runtime guards — defensive against upstream API changes
    const turnCap = budgetCfg.perTurnCap ?? 0;
    const sessionCap = budgetCfg.perSessionCap ?? 0;
    const turnUsed = usage.turn ?? 0;
    const sessionUsed = usage.session ?? 0;

    return {
      timestamp: Date.now(),
      tokenBudget: {
        turnUsed,
        turnRemaining: usage.turnRemaining ?? 0,
        turnCap,
        sessionUsed,
        sessionRemaining: usage.sessionRemaining ?? 0,
        sessionCap,
        turnUtilization: turnCap > 0
          ? turnUsed / turnCap
          : 0,
        sessionUtilization: sessionCap > 0
          ? sessionUsed / sessionCap
          : 0,
      },
      circuitBreaker: {
        tripped: cbStatus.tripped ?? false,
        consecutiveEmpty: cbStatus.consecutiveEmpty ?? 0,
        cooldownRemaining: cbStatus.cooldownRemaining ?? null,
      },
      smartTrigger: {
        totalChecked: triggerStats.totalChecked ?? 0,
        totalSkipped: triggerStats.totalSkipped ?? 0,
        totalPassed: triggerStats.totalPassed ?? 0,
        skipRate: triggerStats.skipRate ?? 0,
      },
      sessionDedup: {
        totalSeen: dedupStats.totalSeen ?? 0,
        totalDeduped: dedupStats.totalDeduped ?? 0,
      },
    };
  }

  /* -------------------------------------------------------------- */
  /*  Alerting                                                       */
  /* -------------------------------------------------------------- */

  /**
   * Check all alert conditions and fire on **transitions**.
   *
   * A condition fires once when it first becomes active. It will
   * not fire again until it clears and re-crosses the threshold.
   */
  checkAlerts(sessionId: string, turnId: string): void {
    const snap = this.snapshot(sessionId, turnId);

    // --- Session budget ---
    this.evaluateCondition(
      "session_budget_high",
      sessionId,
      snap.tokenBudget.sessionUtilization
        >= this.thresholds.sessionBudgetWarning,
      {
        level: "warning",
        source: "tokenBudget",
        condition: "session_budget_high",
        message:
          `session budget at ` +
          `${pct(snap.tokenBudget.sessionUtilization)}% ` +
          `(${snap.tokenBudget.sessionUsed}/` +
          `${snap.tokenBudget.sessionCap} tokens)`,
        sessionId,
        turnId,
        timestamp: Date.now(),
      },
    );

    // --- Turn budget ---
    this.evaluateCondition(
      "turn_budget_high",
      sessionId,
      snap.tokenBudget.turnUtilization
        >= this.thresholds.turnBudgetWarning,
      {
        level: "warning",
        source: "tokenBudget",
        condition: "turn_budget_high",
        message:
          `turn budget at ` +
          `${pct(snap.tokenBudget.turnUtilization)}% ` +
          `(${snap.tokenBudget.turnUsed}/` +
          `${snap.tokenBudget.turnCap} tokens)`,
        sessionId,
        turnId,
        timestamp: Date.now(),
      },
    );

    // --- Circuit breaker ---
    this.evaluateCondition(
      "circuit_breaker_tripped",
      sessionId,
      snap.circuitBreaker.tripped,
      {
        level: "warning",
        source: "circuitBreaker",
        condition: "circuit_breaker_tripped",
        message:
          `circuit breaker tripped after ` +
          `${snap.circuitBreaker.consecutiveEmpty} ` +
          `consecutive empty recalls`,
        sessionId,
        turnId,
        timestamp: Date.now(),
      },
    );
  }

  /* -------------------------------------------------------------- */
  /*  Listener management                                            */
  /* -------------------------------------------------------------- */

  /** Register an alert listener. */
  onAlert(callback: AlertCallback): void {
    this.listeners.push(callback);
  }

  /** Remove a specific alert listener. */
  removeAlertListener(callback: AlertCallback): void {
    this.listeners = this.listeners.filter((cb) => cb !== callback);
  }

  /**
   * Clear all transition-tracking state and alert history.
   * Call on session reset or when you want alerts to re-fire.
   */
  resetAlertState(): void {
    this.activeConditions.clear();
    this.alertHistory = [];
  }

  /* -------------------------------------------------------------- */
  /*  History / inspection                                           */
  /* -------------------------------------------------------------- */

  /**
   * Get a copy of the alert history (most recent last).
   * Capped at {@link MAX_ALERT_HISTORY} entries.
   */
  getAlertHistory(): Alert[] {
    return [...this.alertHistory];
  }

  /* -------------------------------------------------------------- */
  /*  Formatting                                                     */
  /* -------------------------------------------------------------- */

  /**
   * Format a snapshot as human-readable markdown for the
   * `memory_metrics` tool.
   */
  formatSnapshotMarkdown(
    sessionId: string,
    turnId: string,
  ): string {
    const snap = this.snapshot(sessionId, turnId);
    const lines: string[] = [];

    lines.push("## Safeguard Metrics");
    lines.push("");

    lines.push("### Token Budget");
    lines.push(
      `- Turn: ${snap.tokenBudget.turnUsed} used` +
      ` / ${snap.tokenBudget.turnCap} cap` +
      ` (${pct(snap.tokenBudget.turnUtilization)}%)`,
    );
    lines.push(
      `- Session: ${snap.tokenBudget.sessionUsed} used` +
      ` / ${snap.tokenBudget.sessionCap} cap` +
      ` (${pct(snap.tokenBudget.sessionUtilization)}%)`,
    );
    lines.push("");

    lines.push("### Circuit Breaker");
    lines.push(
      `- Status: ` +
      `${snap.circuitBreaker.tripped ? "TRIPPED" : "OK"}`,
    );
    lines.push(
      `- Consecutive empty: ` +
      `${snap.circuitBreaker.consecutiveEmpty}`,
    );
    if (snap.circuitBreaker.cooldownRemaining != null) {
      const secs = snap.circuitBreaker.cooldownRemaining / 1000;
      lines.push(
        `- Cooldown remaining: ${secs.toFixed(0)}s`,
      );
    }
    lines.push("");

    lines.push("### Smart Triggers");
    lines.push(
      `- Checked: ${snap.smartTrigger.totalChecked}`,
    );
    lines.push(
      `- Skipped: ${snap.smartTrigger.totalSkipped}` +
      ` (${pct(snap.smartTrigger.skipRate)}%)`,
    );
    lines.push(
      `- Passed: ${snap.smartTrigger.totalPassed}`,
    );
    lines.push("");

    lines.push("### Session Dedup");
    lines.push(
      `- Total seen: ${snap.sessionDedup.totalSeen}`,
    );
    lines.push(
      `- Deduplicated: ${snap.sessionDedup.totalDeduped}`,
    );

    return lines.join("\n");
  }

  /* -------------------------------------------------------------- */
  /*  Internals                                                      */
  /* -------------------------------------------------------------- */

  /**
   * Evaluate a single condition for transition-based alerting.
   *
   * - First time active → fire alert, mark active
   * - Already active → no-op (no spam)
   * - Was active, now cleared → mark inactive (re-arms)
   */
  private evaluateCondition(
    condition: AlertCondition,
    sessionId: string,
    isActive: boolean,
    alert: Alert,
  ): void {
    const key = `${condition}:${sessionId}`;
    const wasActive = this.activeConditions.has(key);

    if (isActive && !wasActive) {
      this.activeConditions.add(key);
      this.recordAlert(alert);
      this.emitAlert(alert);
    } else if (!isActive && wasActive) {
      this.activeConditions.delete(key);
    }
  }

  /**
   * Record an alert in the rolling history.
   */
  private recordAlert(alert: Alert): void {
    this.alertHistory.push(alert);
    if (this.alertHistory.length > MAX_ALERT_HISTORY) {
      this.alertHistory.shift();
    }
  }

  /**
   * Emit an alert to all listeners. Each listener is isolated
   * so a throwing callback cannot break others.
   */
  private emitAlert(alert: Alert): void {
    for (const listener of this.listeners) {
      try {
        listener(alert);
      } catch {
        // Swallow — one bad listener must not break others
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/** Format a 0–1 ratio as a rounded percentage string. */
function pct(ratio: number): string {
  if (!Number.isFinite(ratio)) return "0";
  return (ratio * 100).toFixed(0);
}
