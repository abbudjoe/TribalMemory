/**
 * Token Budget Tracker
 * 
 * Tracks token usage across three scopes:
 * - Per-recall: Max tokens per single memory_search call
 * - Per-turn: Max tokens per agent turn
 * - Per-session: Max tokens across entire session
 */

import { countTokens } from "./token-utils";

export interface TokenBudgetConfig {
  perRecallCap: number;    // Max tokens per single memory_search call
  perTurnCap: number;      // Max tokens per agent turn
  perSessionCap: number;   // Max tokens across entire session
}

export const DEFAULT_TOKEN_BUDGET: TokenBudgetConfig = {
  perRecallCap: 500,
  perTurnCap: 750,
  perSessionCap: 5000,
};

export class TokenBudget {
  private config: TokenBudgetConfig;
  
  // Track per-turn usage: turnId -> tokenCount
  private turnUsage = new Map<string, number>();

  // Track per-turn timestamps: turnId -> last usage epoch ms
  private turnTimestamps = new Map<string, number>();

  // Track per-session usage: sessionId -> tokenCount
  private sessionUsage = new Map<string, number>();

  constructor(config: Partial<TokenBudgetConfig> = {}) {
    this.config = { ...DEFAULT_TOKEN_BUDGET, ...config };
  }

  /**
   * Count tokens in text (approximate).
   * Delegates to shared token-utils for consistency across safeguard modules.
   */
  countTokens(text: string): number {
    return countTokens(text);
  }

  /**
   * Check if adding tokens would exceed per-recall cap.
   */
  canUseForRecall(tokens: number): boolean {
    return tokens <= this.config.perRecallCap;
  }

  /**
   * Check if adding tokens would exceed per-turn cap.
   */
  canUseForTurn(turnId: string, additionalTokens: number): boolean {
    const currentUsage = this.turnUsage.get(turnId) ?? 0;
    return currentUsage + additionalTokens <= this.config.perTurnCap;
  }

  /**
   * Check if adding tokens would exceed per-session cap.
   */
  canUseForSession(sessionId: string, additionalTokens: number): boolean {
    const currentUsage = this.sessionUsage.get(sessionId) ?? 0;
    return currentUsage + additionalTokens <= this.config.perSessionCap;
  }

  /**
   * Record token usage for a turn and session.
   */
  recordUsage(sessionId: string, turnId: string, tokens: number): void {
    // Update timestamp FIRST to prevent race with cleanupStaleTurns:
    // if cleanup runs between these lines, the fresh timestamp
    // ensures this turn won't be evicted mid-update.
    this.turnTimestamps.set(turnId, Date.now());

    // Update turn usage
    const currentTurn = this.turnUsage.get(turnId) ?? 0;
    this.turnUsage.set(turnId, currentTurn + tokens);

    // Update session usage
    const currentSession = this.sessionUsage.get(sessionId) ?? 0;
    this.sessionUsage.set(sessionId, currentSession + tokens);
  }

  /**
   * Get current usage stats.
   */
  getUsage(sessionId: string, turnId: string): {
    turn: number;
    session: number;
    turnRemaining: number;
    sessionRemaining: number;
  } {
    const turn = this.turnUsage.get(turnId) ?? 0;
    const session = this.sessionUsage.get(sessionId) ?? 0;
    
    return {
      turn,
      session,
      turnRemaining: Math.max(0, this.config.perTurnCap - turn),
      sessionRemaining: Math.max(0, this.config.perSessionCap - session),
    };
  }

  /**
   * Get the number of tracked turns (for cleanup threshold checks).
   */
  getTurnCount(): number {
    return this.turnUsage.size;
  }

  /**
   * Clean up old turn data to prevent memory leaks.
   * Count-based: keeps the most recent N turns by insertion order.
   */
  cleanupOldTurns(keepRecentCount: number = 100): void {
    if (this.turnUsage.size <= keepRecentCount) return;

    const entries = Array.from(this.turnUsage.entries());
    const toKeep = entries.slice(-keepRecentCount);
    const keepIds = new Set(toKeep.map(([id]) => id));

    // Save timestamps before clearing
    const savedTimestamps = new Map(this.turnTimestamps);

    this.turnUsage.clear();
    this.turnTimestamps.clear();

    toKeep.forEach(([turnId, usage]) => {
      this.turnUsage.set(turnId, usage);
      const ts = savedTimestamps.get(turnId);
      if (ts !== undefined) {
        this.turnTimestamps.set(turnId, ts);
      }
    });
  }

  /**
   * Remove turns whose last activity is older than maxAgeMs.
   * Time-based cleanup complements count-based cleanupOldTurns.
   * @param maxAgeMs - Maximum age in milliseconds before eviction.
   * @returns The number of turns removed.
   */
  cleanupStaleTurns(maxAgeMs: number): number {
    const cutoff = Date.now() - maxAgeMs;
    let removed = 0;

    for (const [turnId, ts] of this.turnTimestamps) {
      if (ts < cutoff) {
        this.turnUsage.delete(turnId);
        this.turnTimestamps.delete(turnId);
        removed++;
      }
    }

    return removed;
  }

  /**
   * Reset session tracking (e.g., on session restart).
   */
  resetSession(sessionId: string): void {
    this.sessionUsage.delete(sessionId);
  }

  /**
   * Get config (for inspection/testing).
   */
  getConfig(): TokenBudgetConfig {
    return { ...this.config };
  }
}
