/**
 * TribalClient: HTTP client for tribal-memory server
 * 
 * Connects to the tribal-memory FastAPI server on port 18790
 */

// Constants for configuration
const DEFAULT_TIMEOUT_MS = 10000;
const HEALTH_CHECK_TIMEOUT_MS = 5000;
const DEFAULT_MAX_RESULTS = 5;
const DEFAULT_MIN_SCORE = 0.1;
/**
 * Path format: "tribal-memory:{full-uuid}"
 * Used as the unique identifier in SearchResult.path for memory entries.
 * Full UUID eliminates collision risk at any corpus size.
 */
const ID_PREFIX = "tribal-memory:";

interface SearchOptions {
  maxResults?: number;
  minScore?: number;
  tags?: string[];
}

interface SearchResult {
  id: string;
  /** Unique path in format "tribal-memory:{full-uuid}". Uses the complete memory ID. */
  path: string;
  startLine?: number;
  endLine?: number;
  score: number;
  snippet: string;
  source?: string;
  tags?: string[];
}

interface RecallResult {
  memory: {
    id: string;
    content: string;
    source_instance: string;
    source_type: string;
    created_at: string;
    updated_at: string;
    tags: string[];
    context: string | null;
    confidence: number;
    supersedes: string | null;
  };
  similarity_score: number;
  retrieval_time_ms: number;
}

export class TribalClient {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl: string, timeout = DEFAULT_TIMEOUT_MS) {
    this.baseUrl = baseUrl.replace(/\/$/, ""); // Remove trailing slash
    this.timeout = timeout;
  }

  /**
   * Search memories with multiple query variants
   * Returns results even if some variants fail (partial success)
   */
  async search(queries: string[], options: SearchOptions = {}): Promise<SearchResult[]> {
    const { maxResults = DEFAULT_MAX_RESULTS, minScore = DEFAULT_MIN_SCORE } = options;

    // Search with each query variant and merge results
    const allResults: SearchResult[] = [];
    const seenIds = new Set<string>();

    for (const query of queries) {
      try {
        const results = await this.recall(query, { maxResults, minScore });
        
        for (const result of results) {
          // Validate id exists (should always be present from recall)
          if (!result.id) {
            console.warn("[tribal-client] Skipping result without id:", result);
            continue;
          }
          if (!seenIds.has(result.id)) {
            seenIds.add(result.id);
            allResults.push(result);
          }
        }
      } catch (err) {
        // Continue with other variants if one fails
        console.warn(`[tribal-client] Search failed for "${query}":`, err);
      }
    }

    // Sort by score and limit
    return allResults
      .sort((a, b) => b.score - a.score)
      .slice(0, maxResults);
  }

  /**
   * Recall memories matching a query (maps to /v1/recall)
   * Throws on error - caller should handle
   */
  async recall(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/v1/recall`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          limit: options.maxResults ?? DEFAULT_MAX_RESULTS,
          min_relevance: options.minScore ?? DEFAULT_MIN_SCORE,
          tags: options.tags,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Transform server response to SearchResult format
      return (data.results ?? []).map((r: RecallResult) => {
        if (!r.memory?.id) {
          throw new Error("Invalid response: memory missing id");
        }
        return {
          id: r.memory.id,
          path: `${ID_PREFIX}${r.memory.id}`,
          score: r.similarity_score,
          snippet: r.memory.content,
          source: r.memory.source_type,
          tags: r.memory.tags,
        };
      });
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Store a new memory (maps to /v1/remember)
   * Throws on error - caller should handle
   */
  async remember(content: string, options: {
    sourceType?: string;
    context?: string;
    tags?: string[];
    skipDedup?: boolean;
  } = {}): Promise<{ success: boolean; memoryId?: string; duplicateOf?: string }> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/v1/remember`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content,
          source_type: options.sourceType ?? "auto_capture",
          context: options.context,
          tags: options.tags,
          skip_dedup: options.skipDedup ?? false,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: data.success,
        memoryId: data.memory_id,
        duplicateOf: data.duplicate_of,
      };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Get a memory by ID (maps to /v1/memory/{id})
   * Returns null if not found (404), throws on other errors
   */
  async get(id: string): Promise<any | null> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/v1/memory/${id}`, {
        signal: controller.signal,
      });

      if (response.status === 404) {
        return null;
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Correct an existing memory (maps to /v1/correct)
   * Throws on error - caller should handle
   */
  async correct(
    originalId: string,
    correctedContent: string,
    context?: string
  ): Promise<{ success: boolean; memoryId?: string }> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/v1/correct`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          original_id: originalId,
          corrected_content: correctedContent,
          context,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: data.success,
        memoryId: data.memory_id,
      };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Forget (delete) a memory (maps to DELETE /v1/forget/{id})
   * Returns false on error - safe to ignore failures
   */
  async forget(id: string): Promise<boolean> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/v1/forget/${id}`, {
        method: "DELETE",
        signal: controller.signal,
      });

      if (!response.ok) {
        return false;
      }

      const data = await response.json();
      return data.success;
    } catch {
      return false;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Health check (maps to /v1/health)
   * Returns { ok: false } on any error (never throws)   */
  async health(): Promise<{ ok: boolean; instanceId?: string; memoryCount?: number }> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT_MS);

      const response = await fetch(`${this.baseUrl}/v1/health`, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      
      if (!response.ok) {
        return { ok: false };
      }

      const data = await response.json();
      return {
        ok: data.status === "ok",
        instanceId: data.instance_id,
        memoryCount: data.memory_count,
      };
    } catch {
      return { ok: false };
    }
  }

  /**
   * Get memory statistics (maps to /v1/stats)
   * Throws on error - caller should handle
   */
  async stats(): Promise<any> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/v1/stats`, {
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }
}
