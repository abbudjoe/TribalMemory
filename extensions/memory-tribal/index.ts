/**
 * memory-tribal: Learned Retrieval Layer for Tribal Memory
 * 
 * Drop-in replacement for memory-core with:
 * - Query caching for known-good mappings
 * - Query expansion (question → keywords)
 * - Retrieval feedback loop
 * - Fact anchoring
 */

import { Type } from "@sinclair/typebox";
import { QueryCache } from "./src/learned/query-cache";
import { QueryExpander } from "./src/learned/query-expander";
import { FeedbackTracker } from "./src/learned/feedback-tracker";
import { TribalClient } from "./src/tribal-client";
import { PersistenceLayer } from "./src/persistence";
import { TokenBudget } from "./src/safeguards/token-budget";
import { SnippetTruncator } from "./src/safeguards/truncation";
import { CircuitBreaker } from "./src/safeguards/circuit-breaker";

interface PluginConfig {
  tribalServerUrl: string;
  queryCacheEnabled: boolean;
  queryExpansionEnabled: boolean;
  feedbackEnabled: boolean;
  minCacheSuccesses: number;
  /** Max tokens returned per single memory_search call (default: 500) */
  perRecallCap: number;
  /** Max tokens of memory content per agent turn (default: 750) */
  perTurnCap: number;
  /** Max tokens of memory content across entire session (default: 5000) */
  perSessionCap: number;
  /** Max tokens per individual memory snippet (default: 100) */
  maxTokensPerSnippet: number;
  /** Consecutive empty recalls before circuit breaker trips (default: 5) */
  maxConsecutiveEmpty: number;
  /** Circuit breaker cooldown in ms (default: 300000 = 5 minutes) */
  circuitBreakerCooldownMs: number;
}

export default function memoryTribal(api: any) {
  const config: PluginConfig = api.config?.plugins?.entries?.["memory-tribal"]?.config ?? {
    tribalServerUrl: "http://localhost:18790",
    queryCacheEnabled: true,
    queryExpansionEnabled: true,
    feedbackEnabled: true,
    minCacheSuccesses: 3,
    perRecallCap: 500,
    perTurnCap: 750,
    perSessionCap: 5000,
    maxTokensPerSnippet: 100,
    maxConsecutiveEmpty: 5,
    circuitBreakerCooldownMs: 5 * 60 * 1000,
  };

  // Initialize persistence layer
  let persistence: PersistenceLayer | null = null;
  try {
    persistence = new PersistenceLayer();
    api.log?.info?.("[memory-tribal] Persistence layer initialized");
  } catch (err: any) {
    api.log?.warn?.(`[memory-tribal] Persistence unavailable: ${err.message}`);
  }

  // Initialize components
  const queryCache = new QueryCache(config.minCacheSuccesses, persistence);
  const queryExpander = new QueryExpander(persistence);
  const feedbackTracker = new FeedbackTracker(persistence);
  const tribalClient = new TribalClient(config.tribalServerUrl);

  // Initialize safeguards
  const tokenBudget = new TokenBudget({
    perRecallCap: config.perRecallCap,
    perTurnCap: config.perTurnCap,
    perSessionCap: config.perSessionCap,
  });
  const snippetTruncator = new SnippetTruncator({
    maxTokensPerSnippet: config.maxTokensPerSnippet,
  });
  const circuitBreaker = new CircuitBreaker({
    maxConsecutiveEmpty: config.maxConsecutiveEmpty,
    cooldownMs: config.circuitBreakerCooldownMs,
  });

  // Fallback to built-in memory search if tribal server unavailable
  let useBuiltinFallback = false;

  /** Extract file path from a memory result ID (ID may be path or path:line) */
  function pathForId(id: string): string | null {
    if (!id) return null;
    // IDs may be "path:startLine" or just "path"
    const colonIdx = id.lastIndexOf(":");
    if (colonIdx > 0 && /^\d+$/.test(id.slice(colonIdx + 1))) {
      return id.slice(0, colonIdx);
    }
    return id;
  }

  /** Invalidate query cache entries that reference any of the given paths */
  function invalidatePaths(paths: string[]): void {
    for (const p of paths) {
      queryCache.invalidatePath?.(p);
      persistence?.invalidateCacheByPath?.(p);
    }
  }

  /**
   * memory_search - Enhanced with learned retrieval layer
   */
  api.registerTool({
    name: "memory_search",
    description: "Semantically search memory files (MEMORY.md + memory/*.md) with learned retrieval enhancements. Returns snippets with path and line ranges.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
      maxResults: Type.Optional(Type.Number({ description: "Maximum results to return" })),
      minScore: Type.Optional(Type.Number({ description: "Minimum similarity score" })),
    }),
    async execute(toolCallId: string, params: { query: string; maxResults?: number; minScore?: number }, context: any) {
      const { query, maxResults = 5, minScore = 0.1 } = params;
      const sessionId = context?.sessionId ?? "unknown";

      try {
        // Step 0: Check circuit breaker
        if (circuitBreaker.isTripped(sessionId)) {
          api.log?.debug?.(`[memory-tribal] Circuit breaker tripped for session ${sessionId}`);
          return {
            content: [{ type: "text", text: "Memory recall temporarily paused (circuit breaker active). Will auto-reset shortly." }],
          };
        }

        // Step 1: Check query cache for known-good mappings
        if (config.queryCacheEnabled) {
          const cached = await queryCache.lookup(query);
          if (cached) {
            api.log?.debug?.(`[memory-tribal] Cache hit for query: ${query}`);
            return formatResults(cached, "cache");
          }
        }

        // Step 2: Expand query for better matching
        let queries = [query];
        if (config.queryExpansionEnabled) {
          queries = queryExpander.expand(query);
          api.log?.debug?.(`[memory-tribal] Expanded queries: ${queries.join(", ")}`);
        }

        // Step 3: Search with expanded queries
        let results: any[] = [];
        
        if (!useBuiltinFallback) {
          try {
            results = await tribalClient.search(queries, { maxResults, minScore });
          } catch (err) {
            api.log?.warn?.(`[memory-tribal] Tribal server unavailable, using builtin fallback`);
            useBuiltinFallback = true;
          }
        }

        // Fallback to builtin memory search
        if (useBuiltinFallback && api.runtime?.memorySearch) {
          results = await api.runtime.memorySearch(query, { maxResults, minScore });
        }

        // Step 4: Invalidate cached paths superseded by corrections in results
        const supersededPaths = results
          .map(r => r.supersedes ? pathForId(r.supersedes) : null)
          .filter((p): p is string => !!p);
        if (supersededPaths.length > 0 && config.queryCacheEnabled) {
          invalidatePaths([...new Set(supersededPaths)]);
        }

        // Step 5: Rerank using learned feedback weights
        const canRerank = config.feedbackEnabled &&
          results.length > 0 &&
          results.every(r => typeof r.path === "string" && typeof r.score === "number");
        if (canRerank) {
          results = feedbackTracker.rerank(query, results);
        }

        // Step 6: Learn which expansion variant worked best
        if (config.queryExpansionEnabled && results.length > 0) {
          const bestVariant = results.find(r => r.sourceQuery && r.sourceQuery !== query)?.sourceQuery;
          if (bestVariant) {
            queryExpander.learnExpansion(query, bestVariant);
          }
        }

        // Step 7: Record circuit breaker result
        circuitBreaker.recordResult(sessionId, results.length);

        // Step 8: Apply snippet truncation (before budget accounting)
        results = snippetTruncator.truncateResults(results);

        // Step 9: Apply token budgets
        const turnId = context?.turnId ?? `turn-${Date.now()}`;
        let totalRecallTokens = 0;
        const budgetedResults: any[] = [];

        for (const result of results) {
          const text = result.snippet ?? result.text ?? "";
          const tokens = tokenBudget.countTokens(text);

          // Check per-recall cap — break (not continue) because results are
          // ranked by relevance; skipping to less-relevant results wastes budget
          if (totalRecallTokens + tokens > config.perRecallCap) break;
          // Check per-turn cap
          if (!tokenBudget.canUseForTurn(turnId, tokens)) break;
          // Check per-session cap
          if (!tokenBudget.canUseForSession(sessionId, tokens)) break;

          totalRecallTokens += tokens;
          budgetedResults.push(result);
        }

        // Record token usage
        if (totalRecallTokens > 0) {
          tokenBudget.recordUsage(sessionId, turnId, totalRecallTokens);
          // Periodic cleanup to prevent turn usage map from growing unbounded
          if (tokenBudget.getTurnCount() > 200) {
            tokenBudget.cleanupOldTurns(100);
          }
        }

        results = budgetedResults;

        // Step 10: Record retrieval for feedback tracking
        if (config.feedbackEnabled && results.length > 0) {
          feedbackTracker.recordRetrieval(sessionId, query, results.map(r => r.id ?? r.path));
        }

        return formatResults(results, "search");
      } catch (err: any) {
        return {
          content: [{ type: "text", text: `Memory search error: ${err.message}` }],
          isError: true,
        };
      }
    },
  });

  /**
   * memory_get - Compatible with memory-core
   */
  api.registerTool({
    name: "memory_get",
    description: "Read memory file content by path (MEMORY.md, memory/*.md, or configured extraPaths)",
    parameters: Type.Object({
      path: Type.String({ description: "Path to memory file" }),
      from: Type.Optional(Type.Number({ description: "Starting line number (1-indexed)" })),
      lines: Type.Optional(Type.Number({ description: "Number of lines to read" })),
    }),
    async execute(toolCallId: string, params: { path: string; from?: number; lines?: number }) {
      // Delegate to builtin or read file directly
      if (api.runtime?.memoryGet) {
        return await api.runtime.memoryGet(params.path, params.from, params.lines);
      }
      
      // Fallback: read file directly
      const fs = await import("fs/promises");
      const path = await import("path");
      
      try {
        const content = await fs.readFile(params.path, "utf-8");
        const allLines = content.split("\n");
        
        const startLine = (params.from ?? 1) - 1;
        const numLines = params.lines ?? allLines.length;
        const selectedLines = allLines.slice(startLine, startLine + numLines);
        
        return {
          content: [{ type: "text", text: selectedLines.join("\n") }],
        };
      } catch (err: any) {
        return {
          content: [{ type: "text", text: `Error reading ${params.path}: ${err.message}` }],
          isError: true,
        };
      }
    },
  });

  /**
   * memory_feedback - Record which memories were useful (optional)
   */
  api.registerTool(
    {
      name: "memory_feedback",
      description: "Record which retrieved memories were actually used in the response",
      parameters: Type.Object({
        usedPaths: Type.Array(Type.String(), { description: "Paths of memories that were used" }),
      }),
      async execute(toolCallId: string, params: { usedPaths: string[] }, context: any) {
        const sessionId = context?.sessionId ?? "unknown";
        
        if (config.feedbackEnabled) {
          await feedbackTracker.recordUsage(sessionId, params.usedPaths);
          
          // Update query cache with successful mappings
          const retrieval = feedbackTracker.getLastRetrieval(sessionId);
          if (retrieval && config.queryCacheEnabled) {
            await queryCache.recordSuccess(retrieval.query, params.usedPaths);
          }
        }
        
        return {
          content: [{ type: "text", text: `Recorded feedback for ${params.usedPaths.length} memories` }],
        };
      },
    },
    { optional: true }
  );

  /**
   * memory_stats - Get retrieval statistics (optional)
   */
  api.registerTool(
    {
      name: "memory_stats",
      description: "Get learned retrieval statistics and cache hit rates",
      parameters: Type.Object({}),
      async execute() {
        const stats = {
          cacheEntries: queryCache.size(),
          cacheHitRate: queryCache.hitRate(),
          feedbackRecorded: feedbackTracker.totalFeedback(),
          expansionRules: queryExpander.ruleCount(),
        };
        
        return {
          content: [{ type: "text", text: JSON.stringify(stats, null, 2) }],
        };
      },
    },
    { optional: true }
  );

  api.log?.info?.("[memory-tribal] Plugin loaded");
}

function formatResults(results: any[], source: string) {
  if (!results || results.length === 0) {
    return {
      content: [{ type: "text", text: "No matches found." }],
    };
  }

  const formatted = results.map((r, i) => {
    const path = r.path ?? "unknown";
    const lines = r.startLine && r.endLine ? ` (lines ${r.startLine}-${r.endLine})` : "";
    const score = r.score ? ` [score: ${r.score.toFixed(3)}]` : "";
    const snippet = r.snippet ?? r.text ?? "";
    return `### Result ${i + 1}: ${path}${lines}${score}\n${snippet}`;
  }).join("\n\n");

  return {
    content: [{ type: "text", text: `Found ${results.length} results (source: ${source}):\n\n${formatted}` }],
  };
}
