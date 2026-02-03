/**
 * Circuit Breaker
 * 
 * Tracks consecutive empty recalls and temporarily disables auto-recall
 * to prevent wasted computation on unproductive queries.
 */

export interface CircuitBreakerConfig {
  maxConsecutiveEmpty: number;   // After N empty recalls, trip the breaker
  cooldownMs: number;             // How long to wait before auto-reset
}

export const DEFAULT_CIRCUIT_BREAKER_CONFIG: CircuitBreakerConfig = {
  maxConsecutiveEmpty: 5,
  cooldownMs: 5 * 60 * 1000, // 5 minutes
};

export class CircuitBreaker {
  private config: CircuitBreakerConfig;
  
  // Track consecutive empty results per session
  private consecutiveEmpty = new Map<string, number>();
  
  // Track when breaker tripped for each session
  private trippedAt = new Map<string, number>();

  constructor(config: Partial<CircuitBreakerConfig> = {}) {
    this.config = { ...DEFAULT_CIRCUIT_BREAKER_CONFIG, ...config };
  }

  /**
   * Check if the circuit breaker is currently tripped for a session.
   * 
   * @sideEffect Auto-resets (clears trip state) after cooldown period expires.
   * This is intentional: callers don't need to manage cooldown resets manually.
   * @param sessionId - The session to check
   * @returns true if tripped and within cooldown, false otherwise
   */
  isTripped(sessionId: string): boolean {
    const tripTime = this.trippedAt.get(sessionId);
    
    if (!tripTime) {
      return false;
    }

    const now = Date.now();
    const elapsed = now - tripTime;

    // Auto-reset after cooldown
    if (elapsed >= this.config.cooldownMs) {
      this.reset(sessionId);
      return false;
    }

    return true;
  }

  /**
   * Record a search result (empty or non-empty).
   * Updates consecutive empty count and may trip the breaker.
   */
  recordResult(sessionId: string, resultCount: number): void {
    if (resultCount > 0) {
      // Reset on successful recall
      this.consecutiveEmpty.set(sessionId, 0);
      return;
    }

    // Increment empty count
    const current = this.consecutiveEmpty.get(sessionId) ?? 0;
    const newCount = current + 1;
    this.consecutiveEmpty.set(sessionId, newCount);

    // Trip breaker if threshold exceeded
    if (newCount >= this.config.maxConsecutiveEmpty) {
      this.trip(sessionId);
    }
  }

  /**
   * Manually trip the breaker for a session.
   */
  private trip(sessionId: string): void {
    this.trippedAt.set(sessionId, Date.now());
  }

  /**
   * Manually reset the breaker for a session.
   */
  reset(sessionId: string): void {
    this.consecutiveEmpty.delete(sessionId);
    this.trippedAt.delete(sessionId);
  }

  /**
   * Get current status for a session.
   */
  getStatus(sessionId: string): {
    consecutiveEmpty: number;
    tripped: boolean;
    tripTime: number | null;
    cooldownRemaining: number | null;
  } {
    const consecutiveEmpty = this.consecutiveEmpty.get(sessionId) ?? 0;
    const tripTime = this.trippedAt.get(sessionId) ?? null;
    const tripped = this.isTripped(sessionId);
    
    let cooldownRemaining = null;
    if (tripTime && tripped) {
      const elapsed = Date.now() - tripTime;
      cooldownRemaining = Math.max(0, this.config.cooldownMs - elapsed);
    }

    return {
      consecutiveEmpty,
      tripped,
      tripTime,
      cooldownRemaining,
    };
  }

  /**
   * Get config (for inspection/testing).
   */
  getConfig(): CircuitBreakerConfig {
    return { ...this.config };
  }
}
