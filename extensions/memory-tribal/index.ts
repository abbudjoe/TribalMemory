/**
 * memory-tribal: OpenClaw plugin for Tribal Memory
 *
 * Cross-agent long-term memory with:
 * - Auto-recall (before_agent_start lifecycle hook)
 * - Auto-capture (agent_end lifecycle hook)
 * - Query caching, expansion, and feedback
 * - Safeguards: token budgets, circuit breaker, smart triggers, session dedup
 *
 * Updated to OpenClaw plugin SDK (v0.2.0).
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { Type } from "@sinclair/typebox";
import { QueryCache } from "./src/learned/query-cache";
import { QueryExpander } from "./src/learned/query-expander";
import { FeedbackTracker } from "./src/learned/feedback-tracker";
import { TribalClient } from "./src/tribal-client";
import { PersistenceLayer } from "./src/persistence";
import { TokenBudget } from "./src/safeguards/token-budget";
import { SnippetTruncator } from "./src/safeguards/truncation";
import { CircuitBreaker } from "./src/safeguards/circuit-breaker";
import { SmartTrigger } from "./src/safeguards/smart-triggers";
import { SessionDedup } from "./src/safeguards/session-dedup";
import { SafeguardMetrics } from "./src/safeguards/metrics";
import type { MemoryResult } from "./src/types";

// ============================================================================
// Config type (mirrors openclaw.plugin.json configSchema)
// ============================================================================

interface PluginConfig {
  serverUrl: string;
  autoRecall: boolean;
  autoCapture: boolean;
  queryCacheEnabled: boolean;
  queryCacheMinSuccesses: number;
  queryExpansionEnabled: boolean;
  feedbackEnabled: boolean;
  maxTokensPerRecall: number;
  maxTokensPerTurn: number;
  maxTokensPerSession: number;
  maxTokensPerSnippet: number;
  turnMaxAgeMs: number;
  circuitBreakerMaxEmpty: number;
  circuitBreakerCooldownMs: number;
  smartTriggerEnabled: boolean;
  smartTriggerMinQueryLength: number;
  smartTriggerSkipEmojiOnly: boolean;
  sessionDedupEnabled: boolean;
  sessionDedupCooldownMs: number;
}

/** Context passed to tool execute() by OpenClaw runtime. */
interface ToolContext {
  sessionId?: string;
  turnId?: string;
  [key: string]: unknown;
}

// Defaults (used when pluginConfig values are missing)
const DEFAULTS: PluginConfig = {
  serverUrl: "http://127.0.0.1:18790",
  autoRecall: true,
  autoCapture: true,
  queryCacheEnabled: true,
  queryCacheMinSuccesses: 3,
  queryExpansionEnabled: true,
  feedbackEnabled: true,
  maxTokensPerRecall: 500,
  maxTokensPerTurn: 750,
  maxTokensPerSession: 5000,
  maxTokensPerSnippet: 100,
  turnMaxAgeMs: 30 * 60 * 1000,
  circuitBreakerMaxEmpty: 5,
  circuitBreakerCooldownMs: 5 * 60 * 1000,
  smartTriggerEnabled: true,
  smartTriggerMinQueryLength: 2,
  smartTriggerSkipEmojiOnly: true,
  sessionDedupEnabled: true,
  sessionDedupCooldownMs: 5 * 60 * 1000,
};

// ============================================================================
// Auto-capture triggers (rule-based filter)
// ============================================================================

const MEMORY_TRIGGERS = [
  /remember|zapamatuj si|pamatuj/iu,
  /prefer|radši|nechci|preferuji/iu,
  /rozhodli jsme|budeme používat/iu,
  /\+\d{10,}/u,
  /[\w.-]+@[\w.-]+\.\w+/u,
  /my\s+\w+\s+is|is\s+my/iu,
  /i (like|prefer|hate|love|want|need)/iu,
  /always|never|important/iu,
];

const CAPTURE_MIN_LENGTH = 10;
const CAPTURE_MAX_LENGTH = 500;

// ============================================================================
// /remember command detection
// ============================================================================

/**
 * Regex to detect /remember command.
 * Handles optional channel prefixes like "[Telegram Joe...] /remember ..."
 */
const REMEMBER_COMMAND_RE = /^(?:\[.*?\]\s*)?\/remember\s+(.+)$/is;

/**
 * Extract content from /remember command if present.
 * Returns the text to remember, or null if not a /remember command.
 */
function extractRememberCommand(prompt: string): string | null {
  const match = prompt.match(REMEMBER_COMMAND_RE);
  if (!match) return null;
  return match[1].trim();
}

function shouldCapture(text: string): boolean {
  if (text.length < CAPTURE_MIN_LENGTH || text.length > CAPTURE_MAX_LENGTH) {
    return false;
  }
  if (text.includes("<relevant-memories>")) return false;
  if (text.startsWith("<") && text.includes("</")) return false;
  if (text.includes("**") && text.includes("\n-")) return false;
  const emojiCount = (text.match(/[\u{1F300}-\u{1F9FF}]/gu) || []).length;
  if (emojiCount > 3) return false;
  return MEMORY_TRIGGERS.some((r) => r.test(text));
}

// ============================================================================
// Plugin definition (new SDK format)
// ============================================================================

const memoryTribalPlugin = {
  id: "memory-tribal",
  name: "Tribal Memory",
  description: "Cross-agent long-term memory with auto-recall/capture",
  kind: "memory" as const,

  register(api: OpenClawPluginApi) {
    // ========================================================================
    // Config resolution
    // ========================================================================

    const raw = (api.pluginConfig ?? {}) as Partial<PluginConfig>;
    const config: PluginConfig = { ...DEFAULTS, ...raw };

    // Backward compatibility: accept old config names
    const rawAny = raw as Record<string, unknown>;
    const renames: Record<string, keyof PluginConfig> = {
      tribalServerUrl: "serverUrl",
      minCacheSuccesses: "queryCacheMinSuccesses",
      maxConsecutiveEmpty: "circuitBreakerMaxEmpty",
      smartTriggersEnabled: "smartTriggerEnabled",
      minQueryLength: "smartTriggerMinQueryLength",
      skipEmojiOnly: "smartTriggerSkipEmojiOnly",
      dedupCooldownMs: "sessionDedupCooldownMs",
    };
    const migrated: string[] = [];
    for (const [oldKey, newKey] of Object.entries(renames)) {
      if (oldKey in rawAny && !(newKey in rawAny)) {
        (config as Record<string, unknown>)[newKey] = rawAny[oldKey];
        migrated.push(oldKey);
      }
    }
    if (migrated.length > 0) {
      api.logger.warn(
        `[memory-tribal] Deprecated config names: ${migrated.join(", ")}. ` +
        `See README for migration table.`,
      );
    }

    // ========================================================================
    // Initialize components
    // ========================================================================

    let persistence: PersistenceLayer | null = null;
    try {
      persistence = new PersistenceLayer();
      api.logger.info("[memory-tribal] Persistence layer initialized");
    } catch (err: any) {
      api.logger.warn(
        `[memory-tribal] Persistence unavailable: ${err.message}`,
      );
    }

    const queryCache = new QueryCache(
      config.queryCacheMinSuccesses, persistence,
    );
    const queryExpander = new QueryExpander(persistence);
    const feedbackTracker = new FeedbackTracker(persistence);
    const tribalClient = new TribalClient(config.serverUrl);

    // Safeguards
    const tokenBudget = new TokenBudget({
      perRecallCap: config.maxTokensPerRecall,
      perTurnCap: config.maxTokensPerTurn,
      perSessionCap: config.maxTokensPerSession,
    });
    const snippetTruncator = new SnippetTruncator({
      maxTokensPerSnippet: config.maxTokensPerSnippet,
    });
    const circuitBreaker = new CircuitBreaker({
      maxConsecutiveEmpty: config.circuitBreakerMaxEmpty,
      cooldownMs: config.circuitBreakerCooldownMs,
    });
    const smartTrigger = new SmartTrigger({
      minQueryLength: config.smartTriggerMinQueryLength,
      skipEmojiOnly: config.smartTriggerSkipEmojiOnly,
    });
    const sessionDedup = new SessionDedup({
      cooldownMs: config.sessionDedupCooldownMs,
    });
    const safeguardMetrics = new SafeguardMetrics({
      tokenBudget,
      circuitBreaker,
      smartTrigger,
      sessionDedup,
    });

    let cleanupCallCount = 0;
    let useBuiltinFallback = false;

    // Wire up alerting
    safeguardMetrics.onAlert((alert) => {
      api.logger.warn(
        `[memory-tribal] ALERT [${alert.source}] ${alert.message} ` +
        `(session=${alert.sessionId})`,
      );
    });

    // ========================================================================
    // Helper functions
    // ========================================================================

    function pathForId(id: string): string | null {
      if (!id) return null;
      const colonIdx = id.lastIndexOf(":");
      if (colonIdx > 0 && /^\d+$/.test(id.slice(colonIdx + 1))) {
        return id.slice(0, colonIdx);
      }
      return id;
    }

    function invalidatePaths(paths: string[]): void {
      for (const p of paths) {
        queryCache.invalidatePath?.(p);
        persistence?.invalidateCacheByPath?.(p);
      }
    }

    function formatResults(
      results: MemoryResult[], source: string,
    ) {
      if (!results || results.length === 0) {
        return {
          content: [{ type: "text", text: "No matches found." }],
        };
      }
      const formatted = results.map((r, i) => {
        const path = r.path ?? "unknown";
        const lines = r.startLine && r.endLine
          ? ` (lines ${r.startLine}-${r.endLine})` : "";
        const score = r.score
          ? ` [score: ${r.score.toFixed(3)}]` : "";
        const snippet = r.snippet ?? r.text ?? "";
        return `### Result ${i + 1}: ${path}${lines}${score}\n${snippet}`;
      }).join("\n\n");

      return {
        content: [{
          type: "text",
          text: `Found ${results.length} results (source: ${source}):\n\n${formatted}`,
        }],
      };
    }

    /** Apply safeguards pipeline to raw results */
    function applySafeguards(
      results: MemoryResult[],
      sessionId: string,
      turnId: string,
    ): MemoryResult[] {
      // Snippet truncation
      let filtered = snippetTruncator.truncateResults(results);

      // Token budgets
      let totalRecallTokens = 0;
      const budgeted: MemoryResult[] = [];
      for (const result of filtered) {
        const text = result.snippet ?? result.text ?? "";
        const tokens = tokenBudget.countTokens(text);
        if (totalRecallTokens + tokens > config.maxTokensPerRecall) break;
        if (!tokenBudget.canUseForTurn(turnId, tokens)) break;
        if (!tokenBudget.canUseForSession(sessionId, tokens)) break;
        totalRecallTokens += tokens;
        budgeted.push(result);
      }

      if (totalRecallTokens > 0) {
        tokenBudget.recordUsage(sessionId, turnId, totalRecallTokens);
        if (tokenBudget.getTurnCount() > 200) {
          tokenBudget.cleanupOldTurns(100);
        }
        cleanupCallCount++;
        if (cleanupCallCount % 10 === 0) {
          tokenBudget.cleanupStaleTurns(config.turnMaxAgeMs);
        }
      }
      filtered = budgeted;

      // Session dedup
      if (config.sessionDedupEnabled) {
        filtered = sessionDedup.filter(sessionId, filtered);
      }

      return filtered;
    }

    // ========================================================================
    // Tools
    // ========================================================================

    api.registerTool({
      name: "memory_search",
      description:
        "Semantically search memory files (MEMORY.md + memory/*.md) " +
        "with learned retrieval enhancements.",
      parameters: Type.Object({
        query: Type.String({ description: "Search query" }),
        maxResults: Type.Optional(
          Type.Number({ description: "Maximum results to return" }),
        ),
        minScore: Type.Optional(
          Type.Number({ description: "Minimum similarity score" }),
        ),
      }),
      async execute(
        toolCallId: string,
        params: { query: string; maxResults?: number; minScore?: number },
        context: ToolContext,
      ) {
        const { query, maxResults = 5, minScore = 0.1 } = params;
        const sessionId = context?.sessionId ?? "unknown";

        try {
          // Smart trigger check
          if (config.smartTriggerEnabled) {
            const classification = smartTrigger.classify(query);
            if (classification.skip) {
              api.logger.debug(
                `[memory-tribal] Smart trigger skip: ${classification.reason}`,
              );
              return {
                content: [{
                  type: "text",
                  text: "No recall needed for this query.",
                }],
              };
            }
          }

          // Circuit breaker check
          if (circuitBreaker.isTripped(sessionId)) {
            api.logger.debug(
              `[memory-tribal] Circuit breaker tripped for ${sessionId}`,
            );
            return {
              content: [{
                type: "text",
                text: "Memory recall temporarily paused (circuit breaker). " +
                      "Will auto-reset shortly.",
              }],
            };
          }

          // Query cache check
          if (config.queryCacheEnabled) {
            const cached = await queryCache.lookup(query);
            if (cached) {
              api.logger.debug(
                `[memory-tribal] Cache hit for: ${query}`,
              );
              return formatResults(cached, "cache");
            }
          }

          // Expand query
          let queries = [query];
          if (config.queryExpansionEnabled) {
            queries = queryExpander.expand(query);
          }

          // Search
          let results: MemoryResult[] = [];
          if (!useBuiltinFallback) {
            try {
              results = await tribalClient.search(
                queries, { maxResults, minScore },
              );
            } catch {
              api.logger.warn(
                "[memory-tribal] Tribal server unavailable, " +
                "using builtin fallback",
              );
              useBuiltinFallback = true;
            }
          }
          if (useBuiltinFallback && api.runtime?.memorySearch) {
            results = await api.runtime.memorySearch(
              query, { maxResults, minScore },
            );
          }

          // Invalidate superseded cache entries
          const supersededPaths = results
            .map(r => r.supersedes ? pathForId(r.supersedes) : null)
            .filter((p): p is string => !!p);
          if (supersededPaths.length > 0 && config.queryCacheEnabled) {
            invalidatePaths([...new Set(supersededPaths)]);
          }

          // Rerank with feedback
          const canRerank = config.feedbackEnabled &&
            results.length > 0 &&
            results.every(
              r => typeof r.path === "string" &&
                   typeof r.score === "number",
            );
          if (canRerank) {
            results = feedbackTracker.rerank(query, results);
          }

          // Learn expansion
          if (config.queryExpansionEnabled && results.length > 0) {
            const bestVariant = results.find(
              r => r.sourceQuery && r.sourceQuery !== query,
            )?.sourceQuery;
            if (bestVariant) {
              queryExpander.learnExpansion(query, bestVariant);
            }
          }

          // Circuit breaker record
          circuitBreaker.recordResult(sessionId, results.length);

          // Apply safeguards pipeline
          const turnId = context?.turnId ?? `turn-${Date.now()}`;
          results = applySafeguards(results, sessionId, turnId);

          // Record retrieval for feedback
          if (config.feedbackEnabled && results.length > 0) {
            feedbackTracker.recordRetrieval(
              sessionId, query,
              results.map(r => r.id ?? r.path).filter(Boolean) as string[],
            );
          }

          // Alert check
          safeguardMetrics.checkAlerts(sessionId, turnId);

          return formatResults(results, "search");
        } catch (err: any) {
          return {
            content: [{
              type: "text",
              text: `Memory search error: ${err.message}`,
            }],
            isError: true,
          };
        }
      },
    });

    api.registerTool({
      name: "memory_get",
      description:
        "Read memory file content by path " +
        "(MEMORY.md, memory/*.md, or configured extraPaths)",
      parameters: Type.Object({
        path: Type.String({ description: "Path to memory file" }),
        from: Type.Optional(
          Type.Number({ description: "Starting line (1-indexed)" }),
        ),
        lines: Type.Optional(
          Type.Number({ description: "Number of lines to read" }),
        ),
      }),
      async execute(
        toolCallId: string,
        params: { path: string; from?: number; lines?: number },
      ) {
        if (api.runtime?.memoryGet) {
          return await api.runtime.memoryGet(
            params.path, params.from, params.lines,
          );
        }

        // Fallback: read file directly
        const fs = await import("fs/promises");
        try {
          const content = await fs.readFile(params.path, "utf-8");
          const allLines = content.split("\n");
          const startLine = (params.from ?? 1) - 1;
          const numLines = params.lines ?? allLines.length;
          const selectedLines = allLines.slice(
            startLine, startLine + numLines,
          );
          return {
            content: [{ type: "text", text: selectedLines.join("\n") }],
          };
        } catch (err: any) {
          return {
            content: [{
              type: "text",
              text: `Error reading ${params.path}: ${err.message}`,
            }],
            isError: true,
          };
        }
      },
    });

    api.registerTool(
      {
        name: "memory_feedback",
        description: "Record which retrieved memories were actually used",
        parameters: Type.Object({
          usedPaths: Type.Array(Type.String(), {
            description: "Paths of memories that were used",
          }),
        }),
        async execute(
          toolCallId: string,
          params: { usedPaths: string[] },
          context: ToolContext,
        ) {
          const sessionId = context?.sessionId ?? "unknown";
          if (config.feedbackEnabled) {
            await feedbackTracker.recordUsage(
              sessionId, params.usedPaths,
            );
            const retrieval = feedbackTracker.getLastRetrieval(sessionId);
            if (retrieval && config.queryCacheEnabled) {
              await queryCache.recordSuccess(
                retrieval.query, params.usedPaths,
              );
            }
          }
          return {
            content: [{
              type: "text",
              text: `Recorded feedback for ${params.usedPaths.length} memories`,
            }],
          };
        },
      },
      { optional: true },
    );

    api.registerTool(
      {
        name: "memory_metrics",
        description:
          "Get safeguard metrics snapshot: token budgets, " +
          "circuit breaker, smart triggers, session dedup.",
        parameters: Type.Object({}),
        async execute(
          toolCallId: string, params: Record<string, unknown>, context: ToolContext,
        ) {
          const sessionId = context?.sessionId ?? "unknown";
          const turnId = context?.turnId ?? `turn-${Date.now()}`;
          const text = safeguardMetrics.formatSnapshotMarkdown(
            sessionId, turnId,
          );
          return { content: [{ type: "text", text }] };
        },
      },
      { optional: true },
    );

    // ========================================================================
    // Explicit store/recall tools (direct TribalMemory server access)
    // ========================================================================

    api.registerTool({
      name: "tribal_store",
      description:
        "Deliberately store a memory in TribalMemory. Use for " +
        "high-signal information: architecture decisions, lessons " +
        "learned, user preferences, key facts. More intentional " +
        "than auto-capture.",
      parameters: Type.Object({
        content: Type.String({
          description: "The memory content to store",
        }),
        tags: Type.Optional(
          Type.Array(Type.String(), {
            description:
              "Categorization tags (e.g. 'decision', 'preference', " +
              "'lesson', 'architecture')",
          }),
        ),
        context: Type.Optional(
          Type.String({
            description:
              "Additional context about when/why this was stored",
          }),
        ),
      }),
      async execute(
        toolCallId: string,
        params: { content: string; tags?: string[]; context?: string },
        context: ToolContext,
      ) {
        try {
          const sessionId = context?.sessionId ?? "unknown";
          const result = await tribalClient.remember(params.content, {
            sourceType: "deliberate",
            context:
              params.context ??
              `Deliberately stored (session: ${sessionId})`,
            tags: params.tags,
          });

          if (result.duplicateOf) {
            return {
              content: [{
                type: "text",
                text:
                  `Memory already exists (duplicate of ` +
                  `${result.duplicateOf}). Not stored again.`,
              }],
            };
          }

          const tagStr = params.tags?.length
            ? ` [tags: ${params.tags.join(", ")}]`
            : "";
          return {
            content: [{
              type: "text",
              text:
                `Stored memory ${result.memoryId}${tagStr}: ` +
                `"${params.content.slice(0, 80)}..."`,
            }],
          };
        } catch (err: any) {
          return {
            content: [{
              type: "text",
              text: `Failed to store memory: ${err.message}`,
            }],
            isError: true,
          };
        }
      },
    });

    api.registerTool({
      name: "tribal_recall",
      description:
        "Query TribalMemory with full control over retrieval " +
        "parameters. Use for targeted recall with specific tags, " +
        "temporal filters, or custom relevance thresholds.",
      parameters: Type.Object({
        query: Type.String({ description: "Search query" }),
        limit: Type.Optional(
          Type.Number({
            description: "Maximum results (1-50, default 5)",
          }),
        ),
        min_relevance: Type.Optional(
          Type.Number({
            description:
              "Minimum similarity score (0.0-1.0, default 0.3)",
          }),
        ),
        tags: Type.Optional(
          Type.Array(Type.String(), {
            description: "Filter by tags",
          }),
        ),
        after: Type.Optional(
          Type.String({
            description:
              "Only memories with events on/after this date " +
              "(ISO or natural language, e.g. 'last week')",
          }),
        ),
        before: Type.Optional(
          Type.String({
            description:
              "Only memories with events on/before this date " +
              "(ISO or natural language)",
          }),
        ),
      }),
      async execute(
        toolCallId: string,
        params: {
          query: string;
          limit?: number;
          min_relevance?: number;
          tags?: string[];
          after?: string;
          before?: string;
        },
        context: ToolContext,
      ) {
        try {
          const results = await tribalClient.search(
            [params.query],
            {
              maxResults: params.limit ?? 5,
              minScore: params.min_relevance ?? 0.3,
              tags: params.tags,
              after: params.after,
              before: params.before,
            },
          );

          if (results.length === 0) {
            return {
              content: [{
                type: "text",
                text: "No memories found matching query.",
              }],
            };
          }

          const formatted = results.map((r, i) => {
            const score = r.score?.toFixed(3) ?? "N/A";
            const tags = r.tags?.length
              ? ` [${r.tags.join(", ")}]`
              : "";
            return (
              `### ${i + 1}. [${score}]${tags}\n${r.snippet}`
            );
          }).join("\n\n");

          return {
            content: [{
              type: "text",
              text:
                `Found ${results.length} memories:\n\n${formatted}`,
            }],
          };
        } catch (err: any) {
          return {
            content: [{
              type: "text",
              text: `Failed to recall memories: ${err.message}`,
            }],
            isError: true,
          };
        }
      },
    });

    // ========================================================================
    // CLI Commands
    // ========================================================================

    api.registerCli(
      ({ program }) => {
        const tm = program
          .command("tribal-memory")
          .description("Tribal Memory plugin commands");

        tm.command("status")
          .description("Check connection to tribal-memory server")
          .action(async () => {
            const health = await tribalClient.health();
            if (health.ok) {
              console.log(
                `✅ Connected (instance: ${health.instanceId}, ` +
                `memories: ${health.memoryCount})`,
              );
            } else {
              console.log(
                `❌ Cannot reach server at ${config.serverUrl}`,
              );
            }
          });

        tm.command("stats")
          .description("Show memory statistics")
          .action(async () => {
            try {
              const stats = await tribalClient.stats();
              console.log(JSON.stringify(stats, null, 2));
            } catch (err: any) {
              console.error(`Error: ${err.message}`);
            }
          });

        tm.command("search")
          .description("Search memories")
          .argument("<query>", "Search query")
          .option("--limit <n>", "Max results", "5")
          .action(async (query: string, opts: { limit: string }) => {
            try {
              const limit = parseInt(opts.limit, 10);
              const maxResults = isNaN(limit) ? 5 : limit;
              const results = await tribalClient.search(
                [query], { maxResults },
              );
              for (const r of results) {
                const score = r.score?.toFixed(3) ?? "N/A";
                const snippet = r.snippet?.slice(0, 80) ?? "";
                console.log(`[${score}] ${snippet}`);
              }
              if (results.length === 0) console.log("No results.");
            } catch (err: any) {
              console.error(`Error: ${err.message}`);
            }
          });
      },
      { commands: ["tribal-memory"] },
    );

    // ========================================================================
    // Lifecycle Hooks
    // ========================================================================

    /**
     * Handle /remember command and auto-recall.
     *
     * /remember command: If the prompt starts with /remember (with optional
     * channel prefix), store the content immediately and inform the agent.
     *
     * Auto-recall: inject relevant memories before agent responds.
     * Triggered on every `before_agent_start` event. Searches tribal
     * memory for context relevant to the user's prompt and injects
     * matching memories via `prependContext`. Gated by smart triggers
     * (skips low-value prompts) and circuit breaker (backs off after
     * repeated empty recalls).
     *
     * Returns `{ prependContext }` with XML-wrapped memory snippets,
     * or void if no relevant memories are found.
     */
    api.on("before_agent_start", async (event, ctx) => {
      if (!event.prompt || event.prompt.length < 5) return;

      const sessionId = ctx?.sessionKey ?? `hook-${Date.now()}`;

      // ======================================================================
      // /remember command handler
      // ======================================================================
      const rememberContent = extractRememberCommand(event.prompt);
      if (rememberContent) {
        try {
          const result = await tribalClient.remember(rememberContent, {
            sourceType: "user_explicit",
            context: `User /remember command (session: ${sessionId})`,
          });

          if (result.duplicateOf) {
            api.logger.info(
              `[memory-tribal] /remember: duplicate of ${result.duplicateOf}`,
            );
            return {
              prependContext:
                `<system-note>\n` +
                `User requested to remember: "${rememberContent.slice(0, 100)}..."\n` +
                `This memory already exists (duplicate). No action needed.\n` +
                `</system-note>`,
            };
          }

          api.logger.info(
            `[memory-tribal] /remember: stored ${result.memoryId}: ` +
            `"${rememberContent.slice(0, 60)}..."`,
          );
          return {
            prependContext:
              `<system-note>\n` +
              `User requested to remember: "${rememberContent.slice(0, 100)}..."\n` +
              `Memory stored successfully (id: ${result.memoryId}).\n` +
              `Acknowledge this briefly to the user.\n` +
              `</system-note>`,
          };
        } catch (err: any) {
          api.logger.warn(
            `[memory-tribal] /remember failed: ${err.message}`,
          );
          return {
            prependContext:
              `<system-note>\n` +
              `User tried to remember: "${rememberContent.slice(0, 100)}..."\n` +
              `Storage failed: ${err.message}\n` +
              `Apologize and suggest trying again.\n` +
              `</system-note>`,
          };
        }
      }

      // ======================================================================
      // Auto-recall (only if autoRecall is enabled)
      // ======================================================================
      if (!config.autoRecall) return;

      // Extract search query from prompt (filter system messages, extract
      // from channel prefixes)
      let searchQuery = event.prompt;
      const lines = event.prompt.split("\n");
      const userLines = lines
        .filter(
          (line) => !line.startsWith("System:") && line.trim().length > 0,
        )
        .map((line) => {
          // Extract message content from channel prefixes like:
          // [Telegram Joe (@abbudjoe) id:123 ...] actual message
          const match = line.match(
            /^\[(?:Telegram|Matrix|Discord|Signal)[^\]]+\]\s*(.+)$/i,
          );
          return match ? match[1] : line;
        });
      if (userLines.length > 0) {
        // Use last few non-system lines, max 500 chars
        searchQuery = userLines.slice(-3).join(" ").slice(0, 500);
      } else {
        // Fallback: truncate to 500 chars
        searchQuery = event.prompt.slice(0, 500);
      }

      if (searchQuery.length < 5) return;

      try {
        // Smart trigger gate
        if (config.smartTriggerEnabled) {
          const classification = smartTrigger.classify(searchQuery);
          if (classification.skip) return;
        }

        // Circuit breaker gate
        if (circuitBreaker.isTripped(sessionId)) return;

        // Search tribal memory
        let results: MemoryResult[] = [];
        if (!useBuiltinFallback) {
          try {
            results = await tribalClient.search(
              [searchQuery],
              { maxResults: 3, minScore: 0.3 },
            );
          } catch {
            api.logger.warn(
              "[memory-tribal] Auto-recall: server unavailable",
            );
            useBuiltinFallback = true;
          }
        }

        if (results.length === 0) {
          circuitBreaker.recordResult(sessionId, 0);
          return;
        }
        circuitBreaker.recordResult(sessionId, results.length);

        // Apply safeguards
        const turnId = `auto-recall-${Date.now()}`;
        results = applySafeguards(results, sessionId, turnId);

        if (results.length === 0) return;

        const memoryContext = results
          .map(r => `- ${r.snippet ?? r.text ?? ""}`)
          .join("\n");

        api.logger.info(
          `[memory-tribal] Injecting ${results.length} memories ` +
          `into context`,
        );

        return {
          prependContext:
            `<relevant-memories>\n` +
            `The following memories may be relevant:\n` +
            `${memoryContext}\n` +
            `</relevant-memories>`,
        };
      } catch (err) {
        api.logger.warn(
          `[memory-tribal] Auto-recall failed: ${String(err)}`,
        );
      }
    });

    /**
     * Auto-capture: store learnings after agent turns.
     *
     * Triggered on every `agent_end` event (only when `event.success`
     * is true). Scans user and assistant messages for memorable content
     * using rule-based triggers (preferences, decisions, entities).
     * Stores up to 3 memories per turn via the tribal-memory HTTP API.
     * Deduplication is handled server-side.
     */
    if (config.autoCapture) {
      api.on("agent_end", async (event, ctx) => {
        if (!event.success || !event.messages?.length) return;

        try {
          const texts: string[] = [];
          for (const msg of event.messages) {
            // Defensive: skip anything that isn't an object
            if (!msg || typeof msg !== "object") continue;

            try {
              const msgObj = msg as Record<string, unknown>;
              const role = msgObj.role;
              if (role !== "user" && role !== "assistant") continue;

              const content = msgObj.content;
              if (typeof content === "string") {
                texts.push(content);
                continue;
              }
              if (Array.isArray(content)) {
                for (const block of content) {
                  if (
                    block &&
                    typeof block === "object" &&
                    "type" in block &&
                    (block as Record<string, unknown>).type === "text" &&
                    "text" in block &&
                    typeof (block as Record<string, unknown>).text ===
                      "string"
                  ) {
                    texts.push(
                      (block as Record<string, unknown>).text as string,
                    );
                  }
                }
              }
              // Other content types (images, etc.) are silently skipped
            } catch {
              // Skip malformed messages — don't let one bad message
              // prevent capturing from the rest of the conversation
              continue;
            }
          }

          const toCapture = texts.filter(shouldCapture);
          if (toCapture.length === 0) return;

          let stored = 0;
          const captured: string[] = [];
          for (const text of toCapture.slice(0, 3)) {
            try {
              const result = await tribalClient.remember(text, {
                sourceType: "auto_capture",
                context: `OpenClaw auto-capture (session: ${ctx?.sessionKey ?? "unknown"})`,
              });
              if (result.success) {
                stored++;
                captured.push(text.slice(0, 60));
              }
            } catch {
              // Best-effort capture — skip on failure
            }
          }

          if (stored > 0) {
            api.logger.info(
              `[memory-tribal] Auto-captured ${stored} memories: ` +
              captured.map(t => `"${t}..."`).join(", "),
            );
          }
        } catch (err) {
          api.logger.warn(
            `[memory-tribal] Auto-capture failed: ${String(err)}`,
          );
        }
      });
    }

    // ========================================================================
    // Service (health monitoring)
    // ========================================================================

    api.registerService({
      id: "memory-tribal",
      start: async () => {
        const health = await tribalClient.health();
        if (health.ok) {
          api.logger.info(
            `[memory-tribal] Connected to server ` +
            `(instance: ${health.instanceId}, ` +
            `memories: ${health.memoryCount})`,
          );
        } else {
          api.logger.warn(
            `[memory-tribal] Server not reachable at ${config.serverUrl}. ` +
            `Will retry on first search.`,
          );
        }
      },
      stop: () => {
        api.logger.info("[memory-tribal] Service stopped");
      },
    });

    api.logger.info(
      `[memory-tribal] Plugin registered ` +
      `(server: ${config.serverUrl}, ` +
      `autoRecall: ${config.autoRecall}, ` +
      `autoCapture: ${config.autoCapture})`,
    );
  },
};

export default memoryTribalPlugin;
