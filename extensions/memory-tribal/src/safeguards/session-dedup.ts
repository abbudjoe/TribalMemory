/**
 * Session Deduplication
 *
 * Prevents the same memory snippet from being injected multiple times
 * within a single session. Uses a cooldown window so memories can
 * reappear after enough time passes (default: 5 minutes).
 *
 * Identity is based on path + line range (or path + snippet hash
 * when line info is unavailable).
 *
 * Plugin config mapping:
 * - `sessionDedupEnabled` → controls whether this module runs
 * - `sessionDedupCooldownMs` → maps to `SessionDedupConfig.cooldownMs`
 */

import { createHash } from "crypto";

export interface SessionDedupConfig {
  /** Cooldown in ms before a deduped result can reappear (default: 300000 = 5 min) */
  cooldownMs: number;
  /** Max tracked sessions before oldest is evicted (default: 1000) */
  maxSessions: number;
}

const DEFAULT_CONFIG: SessionDedupConfig = {
  cooldownMs: 5 * 60 * 1000,
  maxSessions: 1000,
};

import type { MemoryResult } from "../types";

/**
 * SHA-256 based hash for snippet identity. Truncated to 16 hex chars
 * for compact keys while maintaining negligible collision probability.
 */
function hashString(str: string): string {
  return createHash("sha256").update(str).digest("hex").substring(0, 16);
}

/**
 * Build an identity key for a memory result.
 * Prefers path:startLine:endLine; falls back to path:snippetHash.
 */
function resultKey(result: MemoryResult): string {
  const path = result.path ?? "unknown";
  if (result.startLine != null && result.endLine != null) {
    return `${path}:${result.startLine}:${result.endLine}`;
  }
  const snippet = result.snippet ?? result.text ?? "";
  return `${path}:${hashString(snippet)}`;
}

export class SessionDedup {
  private config: SessionDedupConfig;
  /** Map of sessionId → Map of resultKey → timestamp of last seen */
  private sessions = new Map<string, Map<string, number>>();
  /** Track insertion order for LRU eviction */
  private sessionOrder: string[] = [];

  private stats = { totalSeen: 0, totalDeduped: 0 };

  constructor(config: Partial<SessionDedupConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Filter out results already seen in this session (within cooldown).
   * Records all passed results for future dedup.
   *
   * @param sessionId - Current session identifier.
   * @param results - Array of memory search results.
   * @returns Filtered array with duplicates removed.
   */
  filter(sessionId: string, results: MemoryResult[]): MemoryResult[] {
    const now = Date.now();
    const seen = this.getOrCreateSession(sessionId);
    const filtered: MemoryResult[] = [];

    for (const result of results) {
      this.stats.totalSeen++;
      const key = resultKey(result);
      const lastSeen = seen.get(key);

      if (lastSeen != null && now - lastSeen < this.config.cooldownMs) {
        // Duplicate within cooldown — skip
        this.stats.totalDeduped++;
        continue;
      }

      // Allow and record
      seen.set(key, now);
      filtered.push(result);
    }

    return filtered;
  }

  /**
   * Clear tracking for a specific session.
   */
  resetSession(sessionId: string): void {
    this.sessions.delete(sessionId);
    this.sessionOrder = this.sessionOrder.filter(id => id !== sessionId);
  }

  /**
   * Get dedup statistics.
   */
  getStats(): { totalSeen: number; totalDeduped: number } {
    return { ...this.stats };
  }

  /**
   * Get config (for inspection/testing).
   */
  getConfig(): SessionDedupConfig {
    return { ...this.config };
  }

  /**
   * Get session state for debugging/inspection.
   * @returns Key count and tracked keys, or null if session unknown.
   */
  getSessionState(
    sessionId: string,
  ): { keyCount: number; keys: string[] } | null {
    const seen = this.sessions.get(sessionId);
    if (!seen) return null;
    return { keyCount: seen.size, keys: Array.from(seen.keys()) };
  }

  /**
   * Get or create the seen-set for a session, with LRU eviction.
   */
  private getOrCreateSession(sessionId: string): Map<string, number> {
    let seen = this.sessions.get(sessionId);
    if (seen) {
      // Touch: move to end of LRU order
      const idx = this.sessionOrder.indexOf(sessionId);
      if (idx !== -1) {
        this.sessionOrder.splice(idx, 1);
        this.sessionOrder.push(sessionId);
      }
      return seen;
    }
    // Evict oldest session if at capacity
    if (this.sessions.size >= this.config.maxSessions) {
      const oldest = this.sessionOrder.shift();
      if (oldest) this.sessions.delete(oldest);
    }
    seen = new Map();
    this.sessions.set(sessionId, seen);
    this.sessionOrder.push(sessionId);
    return seen;
  }
}
