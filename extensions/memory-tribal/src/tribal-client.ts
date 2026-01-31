/**
 * TribalClient: HTTP client for tribal-memory server
 * 
 * Connects to the tribal-memory FastAPI server on port 18790
 */

interface SearchOptions {
  maxResults?: number;
  minScore?: number;
  tags?: string[];
}

interface SearchResult {
  id?: string;
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

  constructor(baseUrl: string, timeout = 10000) {
    this.baseUrl = baseUrl.replace(/\/$/, ""); // Remove trailing slash
    this.timeout = timeout;
  }

  /**
   * Search memories with multiple query variants
   */
  async search(queries: string[], options: SearchOptions = {}): Promise<SearchResult[]> {
    const { maxResults = 5, minScore = 0.1 } = options;

    // Search with each query variant and merge results
    const allResults: SearchResult[] = [];
    const seenIds = new Set<string>();

    for (const query of queries) {
      try {
        const results = await this.recall(query, { maxResults, minScore });
        
        for (const result of results) {
          // id is always set by recall() but typed optional for external use
          const id = result.id ?? result.path;
          if (!seenIds.has(id)) {
            seenIds.add(id);
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
          limit: options.maxResults ?? 5,
          min_relevance: options.minScore ?? 0.1,
          tags: options.tags,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Transform server response to SearchResult format
      return (data.results ?? []).map((r: RecallResult) => ({
        id: r.memory.id,
        path: `tribal-memory:${r.memory.id.slice(0, 8)}`,
        score: r.similarity_score,
        snippet: r.memory.content,
        source: r.memory.source_type,
        tags: r.memory.tags,
      }));
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Store a new memory (maps to /v1/remember)
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
   */
  async get(id: string): Promise<any> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/v1/memory/${id}`, {
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (err) {
      if ((err as Error).message.includes("404")) {
        return null;
      }
      throw err;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Correct an existing memory (maps to /v1/correct)
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
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Health check (maps to /v1/health)
   */
  async health(): Promise<{ ok: boolean; instanceId?: string; memoryCount?: number }> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

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

  // ===========================================================================
  // Deprecated methods (v0.x compatibility)
  // These will be removed in v1.0. Use the new method names instead.
  // ===========================================================================

  /**
   * @deprecated Use `remember()` instead. Will be removed in v1.0.
   */
  async capture(content: string, metadata?: Record<string, any>): Promise<string> {
    console.warn("[tribal-client] capture() is deprecated, use remember() instead");
    const result = await this.remember(content, {
      tags: metadata?.tags,
      context: metadata?.context,
    });
    return result.memoryId ?? "";
  }

  /**
   * @deprecated Use `recall()` instead. Will be removed in v1.0.
   */
  async searchSingle(query: string, options: SearchOptions): Promise<SearchResult[]> {
    console.warn("[tribal-client] searchSingle() is deprecated, use recall() instead");
    return this.recall(query, options);
  }
}
