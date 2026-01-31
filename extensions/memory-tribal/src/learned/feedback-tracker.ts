/**
 * FeedbackTracker: Track which retrievals get used vs ignored
 */

import type { PersistenceLayer } from "../persistence";

interface RetrievalEvent {
  query: string;
  retrievedPaths: string[];
  timestamp: number;
}

interface UsageEvent {
  query: string;
  retrievedPaths: string[];
  usedPaths: string[];
  timestamp: number;
}

export class FeedbackTracker {
  // Pending retrievals awaiting usage feedback
  private pendingRetrievals: Map<string, RetrievalEvent> = new Map();
  
  // Completed feedback events
  private usageHistory: UsageEvent[] = [];
  
  // Query→path reinforcement weights
  private queryPathWeights: Map<string, Map<string, number>> = new Map();
  
  private readonly REINFORCE_DELTA = 0.1;
  private readonly PENALIZE_DELTA = 0.02;
  private readonly MAX_HISTORY = 1000;

  private persistence: PersistenceLayer | null;

  constructor(persistence: PersistenceLayer | null = null) {
    this.persistence = persistence;
  }

  /**
   * Record a retrieval event for later feedback
   */
  recordRetrieval(sessionId: string, query: string, retrievedPaths: string[]): void {
    this.pendingRetrievals.set(sessionId, {
      query,
      retrievedPaths,
      timestamp: Date.now(),
    });
    
    // Clean up old pending retrievals (>10 min)
    const cutoff = Date.now() - 10 * 60 * 1000;
    for (const [id, event] of this.pendingRetrievals.entries()) {
      if (event.timestamp < cutoff) {
        this.pendingRetrievals.delete(id);
      }
    }
  }

  /**
   * Get the last retrieval for a session
   */
  getLastRetrieval(sessionId: string): RetrievalEvent | null {
    return this.pendingRetrievals.get(sessionId) ?? null;
  }

  /**
   * Record which retrieved memories were actually used
   */
  async recordUsage(sessionId: string, usedPaths: string[]): Promise<void> {
    const retrieval = this.pendingRetrievals.get(sessionId);
    if (!retrieval) {
      return; // No pending retrieval to link
    }

    // Record usage event
    const event: UsageEvent = {
      query: retrieval.query,
      retrievedPaths: retrieval.retrievedPaths,
      usedPaths,
      timestamp: Date.now(),
    };
    this.usageHistory.push(event);
    
    // Trim history
    if (this.usageHistory.length > this.MAX_HISTORY) {
      this.usageHistory = this.usageHistory.slice(-this.MAX_HISTORY);
    }

    // Update weights
    const queryKey = this.normalizeQuery(retrieval.query);
    let pathWeights = this.queryPathWeights.get(queryKey);
    if (!pathWeights) {
      pathWeights = new Map();
      this.queryPathWeights.set(queryKey, pathWeights);
    }

    // Reinforce used paths
    for (const path of usedPaths) {
      const current = pathWeights.get(path) ?? 0;
      pathWeights.set(path, Math.min(1.0, current + this.REINFORCE_DELTA));
      
      // Persist weight update
      if (this.persistence) {
        this.persistence.updateWeight(retrieval.query, path, this.REINFORCE_DELTA);
      }
    }

    // Penalize retrieved-but-not-used paths
    for (const path of retrieval.retrievedPaths) {
      if (!usedPaths.includes(path)) {
        const current = pathWeights.get(path) ?? 0;
        pathWeights.set(path, Math.max(-1.0, current - this.PENALIZE_DELTA));
        
        // Persist weight update
        if (this.persistence) {
          this.persistence.updateWeight(retrieval.query, path, -this.PENALIZE_DELTA);
        }
      }
    }

    // Persist usage event
    if (this.persistence) {
      this.persistence.recordUsage(retrieval.query, retrieval.retrievedPaths, usedPaths);
    }

    // Clear pending retrieval
    this.pendingRetrievals.delete(sessionId);
  }

  /**
   * Get reinforcement weight for a query→path pair
   */
  getWeight(query: string, path: string): number {
    const queryKey = this.normalizeQuery(query);
    const pathWeights = this.queryPathWeights.get(queryKey);
    const memoryWeight = pathWeights?.get(path) ?? 0;
    
    // Check persistence if not in memory
    if (memoryWeight === 0 && this.persistence) {
      return this.persistence.getWeight(query, path);
    }
    return memoryWeight;
  }

  /**
   * Adjust search results based on learned weights
   */
  rerank(query: string, results: Array<{ path: string; score: number }>): Array<{ path: string; score: number }> {
    const queryKey = this.normalizeQuery(query);
    const pathWeights = this.queryPathWeights.get(queryKey);
    
    if (!pathWeights || pathWeights.size === 0) {
      return results; // No learned weights, return as-is
    }

    return results
      .map(r => ({
        ...r,
        score: r.score + (pathWeights.get(r.path) ?? 0) * 0.2, // Blend learned weight
      }))
      .sort((a, b) => b.score - a.score);
  }

  private normalizeQuery(query: string): string {
    return query.toLowerCase().replace(/[^\w\s]/g, "").trim();
  }

  /**
   * Get total feedback events recorded
   */
  totalFeedback(): number {
    return this.usageHistory.length;
  }

  /**
   * Get retrieval→usage ratio
   */
  usageRatio(): number {
    if (this.usageHistory.length === 0) return 0;
    
    let totalRetrieved = 0;
    let totalUsed = 0;
    
    for (const event of this.usageHistory) {
      totalRetrieved += event.retrievedPaths.length;
      totalUsed += event.usedPaths.length;
    }
    
    return totalRetrieved > 0 ? totalUsed / totalRetrieved : 0;
  }

  /**
   * Export state for persistence
   */
  export(): { history: UsageEvent[]; weights: Record<string, Record<string, number>> } {
    const weights: Record<string, Record<string, number>> = {};
    for (const [query, pathWeights] of this.queryPathWeights.entries()) {
      weights[query] = Object.fromEntries(pathWeights);
    }
    return { history: this.usageHistory, weights };
  }

  /**
   * Import state from persistence
   */
  import(state: { history: UsageEvent[]; weights: Record<string, Record<string, number>> }): void {
    this.usageHistory = state.history ?? [];
    this.queryPathWeights.clear();
    for (const [query, pathWeights] of Object.entries(state.weights ?? {})) {
      this.queryPathWeights.set(query, new Map(Object.entries(pathWeights)));
    }
  }
}
