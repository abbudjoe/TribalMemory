/**
 * Persistence layer for learned retrieval state
 * Uses SQLite for durable storage of query cache and feedback weights
 */

import Database from "better-sqlite3";
import { join } from "path";
import { homedir } from "os";

interface QueryCacheRow {
  query_hash: string;
  query_normalized: string;
  fact_paths: string;
  success_count: number;
  last_success: number;
}

interface FeedbackWeightRow {
  query_hash: string;
  path: string;
  weight: number;
  updated_at: number;
}

interface UsageHistoryRow {
  id: number;
  query: string;
  retrieved_paths: string;
  used_paths: string;
  timestamp: number;
}

export class PersistenceLayer {
  private db: Database.Database;
  private dbPath: string;

  constructor(dbPath?: string) {
    this.dbPath = dbPath ?? join(homedir(), ".openclaw", "memory-tribal.sqlite");
    this.db = new Database(this.dbPath);
    this.initSchema();
  }

  private initSchema(): void {
    this.db.exec(`
      -- Query cache: stores successful query→fact mappings
      CREATE TABLE IF NOT EXISTS query_cache (
        query_hash TEXT PRIMARY KEY,
        query_normalized TEXT NOT NULL,
        fact_paths TEXT NOT NULL,  -- JSON array
        success_count INTEGER DEFAULT 1,
        last_success INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_query_cache_success ON query_cache(success_count);

      -- Feedback weights: learned query→path reinforcement
      CREATE TABLE IF NOT EXISTS feedback_weights (
        query_hash TEXT NOT NULL,
        path TEXT NOT NULL,
        weight REAL DEFAULT 0.0,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY (query_hash, path)
      );
      CREATE INDEX IF NOT EXISTS idx_feedback_path ON feedback_weights(path);

      -- Usage history: track retrieval→usage events
      CREATE TABLE IF NOT EXISTS usage_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        retrieved_paths TEXT NOT NULL,  -- JSON array
        used_paths TEXT NOT NULL,       -- JSON array
        timestamp INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_history(timestamp);

      -- Learned expansions: successful query expansions
      CREATE TABLE IF NOT EXISTS learned_expansions (
        query_normalized TEXT NOT NULL,
        expansion TEXT NOT NULL,
        success_count INTEGER DEFAULT 1,
        PRIMARY KEY (query_normalized, expansion)
      );

      -- Fact anchors: queries that successfully retrieved a fact
      CREATE TABLE IF NOT EXISTS fact_anchors (
        path TEXT NOT NULL,
        anchor_query TEXT NOT NULL,
        confidence REAL DEFAULT 0.5,
        created_at INTEGER NOT NULL,
        PRIMARY KEY (path, anchor_query)
      );
      CREATE INDEX IF NOT EXISTS idx_anchors_path ON fact_anchors(path);
    `);
  }

  // ========== Query Cache ==========

  private hashQuery(query: string): string {
    // Simple hash for query deduplication
    let hash = 0;
    const normalized = query.toLowerCase().replace(/[^\w\s]/g, "").trim();
    for (let i = 0; i < normalized.length; i++) {
      const char = normalized.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(16);
  }

  getCachedQuery(query: string): { factPaths: string[]; successCount: number } | null {
    const hash = this.hashQuery(query);
    const row = this.db.prepare(`
      SELECT fact_paths, success_count FROM query_cache WHERE query_hash = ?
    `).get(hash) as QueryCacheRow | undefined;

    if (row) {
      return {
        factPaths: JSON.parse(row.fact_paths),
        successCount: row.success_count,
      };
    }
    return null;
  }

  upsertQueryCache(query: string, factPaths: string[]): void {
    const hash = this.hashQuery(query);
    const normalized = query.toLowerCase().replace(/[^\w\s]/g, "").trim();
    
    this.db.prepare(`
      INSERT INTO query_cache (query_hash, query_normalized, fact_paths, success_count, last_success)
      VALUES (?, ?, ?, 1, ?)
      ON CONFLICT(query_hash) DO UPDATE SET
        fact_paths = ?,
        success_count = success_count + 1,
        last_success = ?
    `).run(hash, normalized, JSON.stringify(factPaths), Date.now(), JSON.stringify(factPaths), Date.now());
  }

  invalidateCacheByPath(path: string): number {
    // Delete cache entries that reference this path
    const result = this.db.prepare(`
      DELETE FROM query_cache WHERE fact_paths LIKE ?
    `).run(`%"${path}"%`);
    return result.changes;
  }

  getQueryCacheStats(): { entries: number; totalSuccesses: number } {
    const row = this.db.prepare(`
      SELECT COUNT(*) as entries, COALESCE(SUM(success_count), 0) as total FROM query_cache
    `).get() as { entries: number; total: number };
    return { entries: row.entries, totalSuccesses: row.total };
  }

  // ========== Feedback Weights ==========

  getWeight(query: string, path: string): number {
    const hash = this.hashQuery(query);
    const row = this.db.prepare(`
      SELECT weight FROM feedback_weights WHERE query_hash = ? AND path = ?
    `).get(hash, path) as { weight: number } | undefined;
    return row?.weight ?? 0;
  }

  updateWeight(query: string, path: string, delta: number): void {
    const hash = this.hashQuery(query);
    this.db.prepare(`
      INSERT INTO feedback_weights (query_hash, path, weight, updated_at)
      VALUES (?, ?, ?, ?)
      ON CONFLICT(query_hash, path) DO UPDATE SET
        weight = MIN(1.0, MAX(-1.0, weight + ?)),
        updated_at = ?
    `).run(hash, path, delta, Date.now(), delta, Date.now());
  }

  getWeightsForQuery(query: string): Map<string, number> {
    const hash = this.hashQuery(query);
    const rows = this.db.prepare(`
      SELECT path, weight FROM feedback_weights WHERE query_hash = ?
    `).all(hash) as FeedbackWeightRow[];
    
    const weights = new Map<string, number>();
    for (const row of rows) {
      weights.set(row.path, row.weight);
    }
    return weights;
  }

  // ========== Usage History ==========

  recordUsage(query: string, retrievedPaths: string[], usedPaths: string[]): void {
    this.db.prepare(`
      INSERT INTO usage_history (query, retrieved_paths, used_paths, timestamp)
      VALUES (?, ?, ?, ?)
    `).run(query, JSON.stringify(retrievedPaths), JSON.stringify(usedPaths), Date.now());

    // Trim old history (keep last 1000)
    this.db.prepare(`
      DELETE FROM usage_history WHERE id NOT IN (
        SELECT id FROM usage_history ORDER BY timestamp DESC LIMIT 1000
      )
    `).run();
  }

  getUsageStats(): { total: number; avgUsageRatio: number } {
    const rows = this.db.prepare(`
      SELECT retrieved_paths, used_paths FROM usage_history
    `).all() as UsageHistoryRow[];

    let totalRetrieved = 0;
    let totalUsed = 0;
    for (const row of rows) {
      totalRetrieved += JSON.parse(row.retrieved_paths).length;
      totalUsed += JSON.parse(row.used_paths).length;
    }

    return {
      total: rows.length,
      avgUsageRatio: totalRetrieved > 0 ? totalUsed / totalRetrieved : 0,
    };
  }

  // ========== Learned Expansions ==========

  recordExpansion(query: string, expansion: string): void {
    const normalized = query.toLowerCase().replace(/[^\w\s]/g, "").trim();
    this.db.prepare(`
      INSERT INTO learned_expansions (query_normalized, expansion, success_count)
      VALUES (?, ?, 1)
      ON CONFLICT(query_normalized, expansion) DO UPDATE SET
        success_count = success_count + 1
    `).run(normalized, expansion);
  }

  getLearnedExpansions(query: string): string[] {
    const normalized = query.toLowerCase().replace(/[^\w\s]/g, "").trim();
    const rows = this.db.prepare(`
      SELECT expansion FROM learned_expansions 
      WHERE query_normalized = ? 
      ORDER BY success_count DESC LIMIT 5
    `).all(normalized) as { expansion: string }[];
    return rows.map(r => r.expansion);
  }

  // ========== Fact Anchors ==========

  addAnchor(path: string, query: string, confidence = 0.5): void {
    this.db.prepare(`
      INSERT INTO fact_anchors (path, anchor_query, confidence, created_at)
      VALUES (?, ?, ?, ?)
      ON CONFLICT(path, anchor_query) DO UPDATE SET
        confidence = MIN(1.0, confidence + 0.1),
        created_at = ?
    `).run(path, query, confidence, Date.now(), Date.now());
  }

  getAnchorsForPath(path: string): string[] {
    const rows = this.db.prepare(`
      SELECT anchor_query FROM fact_anchors 
      WHERE path = ? 
      ORDER BY confidence DESC LIMIT 10
    `).all(path) as { anchor_query: string }[];
    return rows.map(r => r.anchor_query);
  }

  searchByAnchor(query: string): string[] {
    // Find paths whose anchors are similar to the query
    const rows = this.db.prepare(`
      SELECT DISTINCT path FROM fact_anchors 
      WHERE anchor_query LIKE ? 
      ORDER BY confidence DESC LIMIT 10
    `).all(`%${query.toLowerCase()}%`) as { path: string }[];
    return rows.map(r => r.path);
  }

  // ========== Maintenance ==========

  vacuum(): void {
    this.db.exec("VACUUM");
  }

  close(): void {
    this.db.close();
  }

  getStats(): Record<string, any> {
    const cacheStats = this.getQueryCacheStats();
    const usageStats = this.getUsageStats();
    
    const weightCount = (this.db.prepare(`SELECT COUNT(*) as c FROM feedback_weights`).get() as { c: number }).c;
    const anchorCount = (this.db.prepare(`SELECT COUNT(*) as c FROM fact_anchors`).get() as { c: number }).c;
    const expansionCount = (this.db.prepare(`SELECT COUNT(*) as c FROM learned_expansions`).get() as { c: number }).c;

    return {
      dbPath: this.dbPath,
      queryCacheEntries: cacheStats.entries,
      queryCacheTotalSuccesses: cacheStats.totalSuccesses,
      feedbackWeights: weightCount,
      usageHistory: usageStats.total,
      usageRatio: usageStats.avgUsageRatio,
      factAnchors: anchorCount,
      learnedExpansions: expansionCount,
    };
  }
}
