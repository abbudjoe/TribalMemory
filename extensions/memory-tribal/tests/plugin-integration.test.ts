/**
 * Integration tests for the full memory-tribal plugin lifecycle.
 *
 * Issue #111: Tests the plugin register() → hooks → tools flow
 * with a mocked TribalClient (mock fetch). Covers:
 *
 * - Auto-recall on before_agent_start hook
 * - Auto-capture on agent_end hook
 * - /remember command handling
 * - Error handling (server down, timeout)
 * - Query extraction (filtering system messages)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import plugin from "../index";

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Types
interface RegisteredTool {
  name: string;
  description: string;
  parameters: unknown;
  execute: (
    toolCallId: string,
    params: Record<string, unknown>,
    context: Record<string, unknown>,
  ) => Promise<{
    content: Array<{ type: string; text: string }>;
    isError?: boolean;
  }>;
}

type HookHandler = (
  event: Record<string, unknown>,
  ctx: Record<string, unknown>,
) => Promise<{ prependContext?: string } | void>;

interface RegisteredService {
  id: string;
  start: () => Promise<void>;
  stop: () => void;
}

// Mock API factory
function createMockApi(configOverrides: Record<string, unknown> = {}) {
  const tools: RegisteredTool[] = [];
  const hooks: Record<string, HookHandler[]> = {};
  const services: RegisteredService[] = [];

  return {
    tools,
    hooks,
    services,
    pluginConfig: {
      serverUrl: "http://localhost:18790",
      autoRecall: true,
      autoCapture: true,
      smartTriggerEnabled: false,
      circuitBreakerMaxEmpty: 5,
      circuitBreakerCooldownMs: 5 * 60 * 1000,
      sessionDedupEnabled: false,
      ...configOverrides,
    },
    logger: {
      info: vi.fn(),
      warn: vi.fn(),
      debug: vi.fn(),
      error: vi.fn(),
    },
    runtime: {},
    on(event: string, handler: HookHandler) {
      if (!hooks[event]) hooks[event] = [];
      hooks[event].push(handler);
    },
    registerTool(toolDef: RegisteredTool, _opts?: unknown) {
      tools.push(toolDef);
    },
    registerCli: vi.fn(),
    registerService(service: RegisteredService) {
      services.push(service);
    },
  };
}

// Helper to fire a hook and return the result
async function fireHook(
  api: ReturnType<typeof createMockApi>,
  hookName: string,
  event: Record<string, unknown>,
  ctx: Record<string, unknown> = {},
): Promise<{ prependContext?: string } | void> {
  const handlers = api.hooks[hookName] ?? [];
  let result: { prependContext?: string } | void;
  for (const handler of handlers) {
    result = await handler(event, ctx);
  }
  return result;
}

// Helper: mock a successful /v1/recall response
function mockRecallResponse(
  results: Array<{
    id: string;
    content: string;
    score: number;
    tags?: string[];
    sourceType?: string;
  }>,
) {
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: async () => ({
      results: results.map((r) => ({
        memory: {
          id: r.id,
          content: r.content,
          source_type: r.sourceType ?? "auto_capture",
          tags: r.tags ?? [],
          source_instance: "test",
          created_at: "2026-02-07T00:00:00Z",
          updated_at: "2026-02-07T00:00:00Z",
          context: null,
          confidence: 1.0,
          supersedes: null,
        },
        similarity_score: r.score,
        retrieval_time_ms: 5,
      })),
    }),
  });
}

// Helper: mock a successful /v1/remember response
function mockRememberResponse(memoryId: string, duplicate = false) {
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: async () =>
      duplicate
        ? { success: false, duplicate_of: memoryId }
        : { success: true, memory_id: memoryId },
  });
}

// Helper: mock a network error (server down / timeout)
function mockNetworkError(message = "fetch failed") {
  mockFetch.mockRejectedValueOnce(new Error(message));
}

// Helper: mock a failed server response
function mockServerError(status = 500, statusText = "Internal Server Error") {
  mockFetch.mockResolvedValueOnce({
    ok: false,
    status,
    statusText,
  });
}

describe("Plugin Integration: Full Lifecycle", () => {
  let api: ReturnType<typeof createMockApi>;

  beforeEach(() => {
    mockFetch.mockReset();
    api = createMockApi();
    plugin.register(api as any);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("plugin registration", () => {
    it("registers all expected tools", () => {
      const toolNames = api.tools.map((t) => t.name);
      expect(toolNames).toContain("memory_search");
      expect(toolNames).toContain("memory_get");
      expect(toolNames).toContain("memory_feedback");
      expect(toolNames).toContain("memory_metrics");
      expect(toolNames).toContain("tribal_store");
      expect(toolNames).toContain("tribal_recall");
    });

    it("registers before_agent_start hook", () => {
      expect(api.hooks["before_agent_start"]).toBeDefined();
      expect(api.hooks["before_agent_start"].length).toBeGreaterThan(0);
    });

    it("registers agent_end hook when autoCapture is enabled", () => {
      expect(api.hooks["agent_end"]).toBeDefined();
      expect(api.hooks["agent_end"].length).toBeGreaterThan(0);
    });

    it("does NOT register agent_end hook when autoCapture is disabled", () => {
      const noCapApi = createMockApi({ autoCapture: false });
      plugin.register(noCapApi as any);
      expect(noCapApi.hooks["agent_end"]).toBeUndefined();
    });

    it("registers a service", () => {
      expect(api.services.length).toBe(1);
      expect(api.services[0].id).toBe("memory-tribal");
    });
  });

  describe("auto-recall (before_agent_start)", () => {
    it("injects memories as prependContext for relevant prompts", async () => {
      mockRecallResponse([
        { id: "m1", content: "Joe prefers TypeScript", score: 0.85 },
        { id: "m2", content: "Project uses React", score: 0.72 },
      ]);

      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "What tech stack do we use?" },
        { sessionKey: "session-1" },
      );

      expect(result).toBeDefined();
      expect(result?.prependContext).toContain("<relevant-memories>");
      expect(result?.prependContext).toContain("Joe prefers TypeScript");
    });

    it("returns void when no memories match", async () => {
      mockRecallResponse([]);

      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "What is the meaning of life?" },
        { sessionKey: "session-2" },
      );

      expect(result).toBeUndefined();
    });

    it("skips short prompts (< 5 chars)", async () => {
      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "hi" },
        { sessionKey: "session-3" },
      );

      expect(result).toBeUndefined();
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("does not recall when autoRecall is disabled", async () => {
      const noRecallApi = createMockApi({ autoRecall: false });
      plugin.register(noRecallApi as any);

      const result = await fireHook(
        noRecallApi,
        "before_agent_start",
        { prompt: "What is our tech stack?" },
        { sessionKey: "session-5" },
      );

      expect(mockFetch).not.toHaveBeenCalled();
      expect(result).toBeUndefined();
    });

    it("handles server errors gracefully (returns void)", async () => {
      mockNetworkError("Connection refused");

      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "Tell me about the architecture"