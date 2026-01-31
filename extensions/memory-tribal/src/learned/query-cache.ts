/**
 * QueryCache: Store successful query→fact mappings for fast retrieval
 */

import type { PersistenceLayer } from "../persistence";

interface CacheEntry {
  queryNormalized: string;
  factPaths: string[];
  successCount: number;
  lastSuccess: number;
}

export class QueryCache {
  private cache: Map<string, CacheEntry> = new Map();
  private minSuccesses: number;
  private hits = 0;
  private misses = 0;
  private persistence: PersistenceLayer | null;

  constructor(minSuccesses = 3, persistence: PersistenceLayer | null = null) {
    this.minSuccesses = minSuccesses;
    this.persistence = persistence;
  }

  /**
   * Normalize query for fuzzy matching
   */
  private normalize(query: string): string {
    return query
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  /**
   * Look up cached fact paths for a query
   */
  async lookup(query: string): Promise<any[] | null> {
    const normalized = this.normalize(query);
    
    // Check in-memory cache first
    let entry = this.cache.get(normalized);
    
    // Fall back to persistence if not in memory
    if (!entry && this.persistence) {
      const persisted = this.persistence.getCachedQuery(query);
      if (persisted) {
        entry = {
          queryNormalized: normalized,
          factPaths: persisted.factPaths,
          successCount: persisted.successCount,
          lastSuccess: Date.now(),
        };
        // Hydrate in-memory cache
        this.cache.set(normalized, entry);
      }
    }

    if (entry && entry.successCount >= this.minSuccesses) {
      this.hits++;
      // Return cached results as pseudo-search results
      return entry.factPaths.map((path, i) => ({
        path,
        score: 1.0 - i * 0.01, // Preserve order with decreasing scores
        snippet: `[Cached result from ${entry.successCount} successful retrievals]`,
        source: "cache",
      }));
    }

    this.misses++;
    return null;
  }

  /**
   * Record a successful retrieval for caching
   */
  async recordSuccess(query: string, factPaths: string[]): Promise<void> {
    const normalized = this.normalize(query);
    const existing = this.cache.get(normalized);

    if (existing) {
      // Merge fact paths (keep unique, preserve order by frequency)
      const pathCounts = new Map<string, number>();
      for (const p of [...existing.factPaths, ...factPaths]) {
        pathCounts.set(p, (pathCounts.get(p) ?? 0) + 1);
      }
      const sortedPaths = [...pathCounts.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([p]) => p)
        .slice(0, 10); // Keep top 10

      this.cache.set(normalized, {
        queryNormalized: normalized,
        factPaths: sortedPaths,
        successCount: existing.successCount + 1,
        lastSuccess: Date.now(),
      });
    } else {
      this.cache.set(normalized, {
        queryNormalized: normalized,
        factPaths: factPaths.slice(0, 10),
        successCount: 1,
        lastSuccess: Date.now(),
      });
    }

    // Persist to SQLite
    if (this.persistence) {
      this.persistence.upsertQueryCache(query, factPaths);
    }
  }

  /**
   * Invalidate cache entries containing a specific path (on memory update)
   */
  invalidatePath(path: string): void {
    for (const [key, entry] of this.cache.entries()) {
      if (entry.factPaths.includes(path)) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Get cache statistics
   */
  size(): number {
    return this.cache.size;
  }

  hitRate(): number {
    const total = this.hits + this.misses;
    return total > 0 ? this.hits / total : 0;
  }

  /**
   * Export cache for persistence
   */
  export(): CacheEntry[] {
    return [...this.cache.values()];
  }

  /**
   * Import cache from persistence
   */
  import(entries: CacheEntry[]): void {
    for (const entry of entries) {
      this.cache.set(entry.queryNormalized, entry);
    }
  }
}
