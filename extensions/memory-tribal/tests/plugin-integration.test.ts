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

// ============================================================================
// Mock fetch globally
// ============================================================================

const mockFetch = vi.fn();
global.fetch = mockFetch;

// ============================================================================
// Types
// ============================================================================

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

// ============================================================================
// Mock API factory
// ============================================================================

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
      smartTriggerEnabled: false,  // Disable smart triggers for cleaner tests
      circuitBreakerMaxEmpty: 5,
      circuitBreakerCooldownMs: 5 * 60 * 1000,
      sessionDedupEnabled: false,  // Disable dedup for cleaner tests
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

/** Helper to fire a hook and return the result */
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

/** Helper: mock a successful /v1/recall response */
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

/** Helper: mock a successful /v1/remember response */
function mockRememberResponse(memoryId: string, duplicate = false) {
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: async () =>
      duplicate
        ? { success: false, duplicate_of: memoryId }
        : { success: true, memory_id: memoryId },
  });
}

/** Helper: mock a failed server response */
function mockServerError(status = 500, statusText = "Internal Server Error") {
  mockFetch.mockResolvedValueOnce({
    ok: false,
    status,
    statusText,
  });
}

/** Helper: mock a network error (server down / timeout) */
function mockNetworkError(message = "fetch failed") {
  mockFetch.mockRejectedValueOnce(new Error(message));
}

// ============================================================================
// Tests
// ============================================================================

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

  // ==========================================================================
  // Registration
  // ==========================================================================

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

    it("logs registration info", () => {
      expect(api.logger.info).toHaveBeenCalledWith(
        expect.stringContaining("[memory-tribal] Plugin registered"),
      );
    });
  });

  // ==========================================================================
  // Auto-recall (before_agent_start)
  // ==========================================================================

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
      expect(result?.prependContext).toContain("Project uses React");
      expect(result?.prependContext).toContain("</relevant-memories>");
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

    it("skips prompts shorter than 5 characters", async () => {
      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "hi" },
        { sessionKey: "session-3" },
      );

      expect(result).toBeUndefined();
      // No fetch call should have been made
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("skips when prompt is missing", async () => {
      const result = await fireHook(
        api,
        "before_agent_start",
        {},
        { sessionKey: "session-4" },
      );

      expect(result).toBeUndefined();
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("does not recall when autoRecall is disabled", async () => {
      const noRecallApi = createMockApi({ autoRecall: false });
      plugin.register(noRecallApi as any);

      // Should not make a search call (only /remember handler is active)
      const result = await fireHook(
        noRecallApi,
        "before_agent_start",
        { prompt: "What is our tech stack?" },
        { sessionKey: "session-5" },
      );

      // No recall fetch calls should have been made
      expect(mockFetch).not.toHaveBeenCalled();
      expect(result).toBeUndefined();
    });

    it("handles server errors gracefully (returns void)", async () => {
      mockNetworkError("Connection refused");

      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "Tell me about the architecture" },
        { sessionKey: "session-6" },
      );

      // TribalClient.search() catches per-query errors internally
      // and returns []. The hook sees 0 results and returns void.
      expect(result).toBeUndefined();
    });

    it("calls /v1/recall with minScore 0.3 and maxResults 3", async () => {
      mockRecallResponse([
        { id: "m1", content: "Some memory", score: 0.5 },
      ]);

      await fireHook(
        api,
        "before_agent_start",
        { prompt: "How does auth work in our system?" },
        { sessionKey: "session-7" },
      );

      const recallCall = mockFetch.mock.calls.find(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/recall"),
      );
      expect(recallCall).toBeDefined();
      const body = JSON.parse(recallCall![1].body);
      expect(body.min_relevance).toBe(0.3);
      expect(body.limit).toBe(3);
    });

    it("logs injection count when memories found", async () => {
      mockRecallResponse([
        { id: "m1", content: "Memory A", score: 0.9 },
        { id: "m2", content: "Memory B", score: 0.8 },
      ]);

      await fireHook(
        api,
        "before_agent_start",
        { prompt: "What do we know about the project?" },
        { sessionKey: "session-8" },
      );

      expect(api.logger.info).toHaveBeenCalledWith(
        expect.stringContaining("Injecting 2 memories"),
      );
    });
  });

  // ==========================================================================
  // Query extraction (filtering system messages)
  // ==========================================================================

  describe("query extraction", () => {
    it("filters out System: lines from prompt", async () => {
      mockRecallResponse([
        { id: "m1", content: "Relevant memory", score: 0.8 },
      ]);

      await fireHook(
        api,
        "before_agent_start",
        {
          prompt:
            "System: exec output here\nSystem: more output\nWhat is the weather?",
        },
        { sessionKey: "session-q1" },
      );

      const recallCall = mockFetch.mock.calls.find(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/recall"),
      );
      expect(recallCall).toBeDefined();
      const body = JSON.parse(recallCall![1].body);
      // Query should only contain the user message, not System: lines
      expect(body.query).toBe("What is the weather?");
    });

    it("extracts message from channel prefix (Telegram)", async () => {
      mockRecallResponse([
        { id: "m1", content: "Relevant memory", score: 0.8 },
      ]);

      await fireHook(
        api,
        "before_agent_start",
        {
          prompt:
            "[Telegram Joe (@abbudjoe) id:123 +5s] Tell me about the project",
        },
        { sessionKey: "session-q2" },
      );

      const recallCall = mockFetch.mock.calls.find(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/recall"),
      );
      expect(recallCall).toBeDefined();
      const body = JSON.parse(recallCall![1].body);
      expect(body.query).toBe("Tell me about the project");
    });

    it("handles combined System + channel prefix prompts", async () => {
      mockRecallResponse([
        { id: "m1", content: "Relevant memory", score: 0.8 },
      ]);

      await fireHook(
        api,
        "before_agent_start",
        {
          prompt:
            "System: Running exec...\n" +
            "System: Output captured\n" +
            "[Telegram Joe (@abbudjoe) id:3219 +1m] What is TribalMemory?",
        },
        { sessionKey: "session-q3" },
      );

      const recallCall = mockFetch.mock.calls.find(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/recall"),
      );
      expect(recallCall).toBeDefined();
      const body = JSON.parse(recallCall![1].body);
      expect(body.query).toBe("What is TribalMemory?");
    });

    it("uses last 3 non-system lines for multi-line prompts", async () => {
      mockRecallResponse([]);

      await fireHook(
        api,
        "before_agent_start",
        {
          prompt: "line1\nline2\nline3\nline4\nline5",
        },
        { sessionKey: "session-q4" },
      );

      const recallCall = mockFetch.mock.calls.find(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/recall"),
      );
      expect(recallCall).toBeDefined();
      const body = JSON.parse(recallCall![1].body);
      expect(body.query).toBe("line3 line4 line5");
    });

    it("truncates query to 500 chars max", async () => {
      mockRecallResponse([]);

      const longPrompt = "A".repeat(600);
      await fireHook(
        api,
        "before_agent_start",
        { prompt: longPrompt },
        { sessionKey: "session-q5" },
      );

      const recallCall = mockFetch.mock.calls.find(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/recall"),
      );
      expect(recallCall).toBeDefined();
      const body = JSON.parse(recallCall![1].body);
      expect(body.query.length).toBeLessThanOrEqual(500);
    });
  });

  // ==========================================================================
  // /remember command handling
  // ==========================================================================

  describe("/remember command (before_agent_start)", () => {
    it("stores memory and returns success prependContext", async () => {
      mockRememberResponse("mem-new-1");

      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "/remember Joe prefers dark mode" },
        { sessionKey: "session-r1" },
      );

      expect(result).toBeDefined();
      expect(result?.prependContext).toContain("<system-note>");
      expect(result?.prependContext).toContain("Joe prefers dark mode");
      expect(result?.prependContext).toContain("mem-new-1");
      expect(result?.prependContext).toContain("stored successfully");
    });

    it("handles /remember with channel prefix", async () => {
      mockRememberResponse("mem-new-2");

      const result = await fireHook(
        api,
        "before_agent_start",
        {
          prompt:
            "[Telegram Joe (@abbudjoe)] /remember My birthday is March 15",
        },
        { sessionKey: "session-r2" },
      );

      expect(result).toBeDefined();
      expect(result?.prependContext).toContain("My birthday is March 15");
      expect(result?.prependContext).toContain("stored successfully");
    });

    it("handles duplicate /remember gracefully", async () => {
      mockRememberResponse("existing-dup", true);

      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "/remember Already stored fact" },
        { sessionKey: "session-r3" },
      );

      expect(result).toBeDefined();
      expect(result?.prependContext).toContain("already exists");
      expect(result?.prependContext).toContain("duplicate");
    });

    it("handles server error on /remember", async () => {
      mockServerError(500, "Internal Server Error");

      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "/remember This will fail" },
        { sessionKey: "session-r4" },
      );

      expect(result).toBeDefined();
      expect(result?.prependContext).toContain("Storage failed");
      expect(result?.prependContext).toContain("Apologize");
    });

    it("handles network error on /remember", async () => {
      mockNetworkError("Connection refused");

      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "/remember This will also fail" },
        { sessionKey: "session-r5" },
      );

      expect(result).toBeDefined();
      expect(result?.prependContext).toContain("Storage failed");
    });

    it("sends source_type as user_explicit for /remember", async () => {
      mockRememberResponse("mem-explicit");

      await fireHook(
        api,
        "before_agent_start",
        { prompt: "/remember Test explicit" },
        { sessionKey: "session-r6" },
      );

      const rememberCall = mockFetch.mock.calls.find(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/remember"),
      );
      expect(rememberCall).toBeDefined();
      const body = JSON.parse(rememberCall![1].body);
      expect(body.source_type).toBe("user_explicit");
    });

    it("does NOT trigger auto-recall after /remember command", async () => {
      mockRememberResponse("mem-no-recall");

      await fireHook(
        api,
        "before_agent_start",
        { prompt: "/remember Something important" },
        { sessionKey: "session-r7" },
      );

      // Only one fetch call should have been made (the remember call)
      // No /v1/recall call should happen
      const recallCalls = mockFetch.mock.calls.filter(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/recall"),
      );
      expect(recallCalls.length).toBe(0);
    });

    it("handles case-insensitive /REMEMBER command", async () => {
      mockRememberResponse("mem-upper");

      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "/REMEMBER case insensitive fact" },
        { sessionKey: "session-r8" },
      );

      expect(result).toBeDefined();
      expect(result?.prependContext).toContain("stored successfully");
    });
  });

  // ==========================================================================
  // Auto-capture (agent_end)
  // ==========================================================================

  describe("auto-capture (agent_end)", () => {
    it("captures memorable messages after successful turn", async () => {
      // Expect up to 1 remember call for the capturable message
      mockRememberResponse("cap-1");

      await fireHook(
        api,
        "agent_end",
        {
          success: true,
          messages: [
            { role: "user", content: "I prefer dark mode always" },
            { role: "assistant", content: "Noted, dark mode preference." },
          ],
        },
        { sessionKey: "session-c1" },
      );

      // At least one /v1/remember call should have been made
      const rememberCalls = mockFetch.mock.calls.filter(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/remember"),
      );
      expect(rememberCalls.length).toBeGreaterThan(0);

      // Verify it was sent as auto_capture
      const body = JSON.parse(rememberCalls[0][1].body);
      expect(body.source_type).toBe("auto_capture");
    });

    it("skips capture when event.success is false", async () => {
      await fireHook(
        api,
        "agent_end",
        {
          success: false,
          messages: [
            { role: "user", content: "I prefer TypeScript always" },
          ],
        },
        { sessionKey: "session-c2" },
      );

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("skips capture when messages are empty", async () => {
      await fireHook(
        api,
        "agent_end",
        { success: true, messages: [] },
        { sessionKey: "session-c3" },
      );

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("skips capture when messages are missing", async () => {
      await fireHook(
        api,
        "agent_end",
        { success: true },
        { sessionKey: "session-c4" },
      );

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("skips non-triggering messages", async () => {
      await fireHook(
        api,
        "agent_end",
        {
          success: true,
          messages: [
            { role: "user", content: "What is the weather today?" },
            { role: "assistant", content: "It's 25°C and sunny." },
          ],
        },
        { sessionKey: "session-c5" },
      );

      // No trigger-matching messages → no remember calls
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("skips system and tool role messages", async () => {
      mockRememberResponse("cap-sys");

      await fireHook(
        api,
        "agent_end",
        {
          success: true,
          messages: [
            { role: "system", content: "I always prefer Python" },
            { role: "tool", content: "Remember this output" },
            { role: "user", content: "I always prefer TypeScript" },
          ],
        },
        { sessionKey: "session-c6" },
      );

      // Only the user message should be captured
      const rememberCalls = mockFetch.mock.calls.filter(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/remember"),
      );
      expect(rememberCalls.length).toBe(1);
      const body = JSON.parse(rememberCalls[0][1].body);
      expect(body.content).toBe("I always prefer TypeScript");
    });

    it("handles content block arrays (multi-part messages)", async () => {
      mockRememberResponse("cap-block");

      await fireHook(
        api,
        "agent_end",
        {
          success: true,
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "I always use vim for editing" },
                { type: "image", url: "http://example.com/img.png" },
              ],
            },
          ],
        },
        { sessionKey: "session-c7" },
      );

      const rememberCalls = mockFetch.mock.calls.filter(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/remember"),
      );
      expect(rememberCalls.length).toBe(1);
      const body = JSON.parse(rememberCalls[0][1].body);
      expect(body.content).toBe("I always use vim for editing");
    });

    it("limits captures to 3 per turn", async () => {
      // Mock 5 remember responses (but only 3 should be used)
      for (let i = 0; i < 5; i++) {
        mockRememberResponse(`cap-limit-${i}`);
      }

      await fireHook(
        api,
        "agent_end",
        {
          success: true,
          messages: [
            { role: "user", content: "I prefer dark mode always" },
            { role: "user", content: "Remember my timezone is MST" },
            { role: "user", content: "I love using vim for editing" },
            { role: "user", content: "My email is joe@test.com" },
            { role: "user", content: "I need more coffee always" },
          ],
        },
        { sessionKey: "session-c8" },
      );

      const rememberCalls = mockFetch.mock.calls.filter(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/remember"),
      );
      expect(rememberCalls.length).toBeLessThanOrEqual(3);
    });

    it("handles server failure during capture gracefully (best-effort)", async () => {
      // First capture fails, second succeeds
      mockNetworkError("Connection refused");
      mockRememberResponse("cap-after-fail");

      await fireHook(
        api,
        "agent_end",
        {
          success: true,
          messages: [
            { role: "user", content: "Remember this important thing" },
            { role: "user", content: "I prefer Python always" },
          ],
        },
        { sessionKey: "session-c9" },
      );

      // Should not throw — best-effort capture
      // At least the second one should have been attempted
      expect(mockFetch).toHaveBeenCalled();
    });

    it("handles malformed messages without crashing", async () => {
      mockRememberResponse("cap-malformed");

      await fireHook(
        api,
        "agent_end",
        {
          success: true,
          messages: [
            null,
            undefined,
            42,
            { role: "user", content: "I always use TypeScript" },
          ],
        },
        { sessionKey: "session-c10" },
      );

      // Should process the valid message without crashing
      const rememberCalls = mockFetch.mock.calls.filter(
        (c: any[]) =>
          typeof c[0] === "string" && c[0].includes("/v1/remember"),
      );
      expect(rememberCalls.length).toBe(1);
    });

    it("logs count of captured memories", async () => {
      mockRememberResponse("cap-log-1");

      await fireHook(
        api,
        "agent_end",
        {
          success: true,
          messages: [
            { role: "user", content: "I prefer dark mode always" },
          ],
        },
        { sessionKey: "session-c11" },
      );

      expect(api.logger.info).toHaveBeenCalledWith(
        expect.stringContaining("Auto-captured 1 memories"),
      );
    });
  });

  // ==========================================================================
  // Error handling (server down, timeout)
  // ==========================================================================

  describe("error handling", () => {
    it("auto-recall survives server being completely down", async () => {
      mockNetworkError("ECONNREFUSED");

      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "Tell me about the architecture" },
        { sessionKey: "session-e1" },
      );

      // TribalClient.search() catches errors per-query and returns [].
      // The hook sees 0 results and returns undefined — no crash.
      expect(result).toBeUndefined();
    });

    it("auto-recall survives HTTP 500 responses", async () => {
      mockServerError(500, "Internal Server Error");

      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "What is the deployment process?" },
        { sessionKey: "session-e2" },
      );

      expect(result).toBeUndefined();
    });

    it("auto-capture survives server errors without crashing", async () => {
      mockNetworkError("ECONNREFUSED");

      // Should not throw
      await fireHook(
        api,
        "agent_end",
        {
          success: true,
          messages: [
            { role: "user", content: "I always use dark mode" },
          ],
        },
        { sessionKey: "session-e3" },
      );

      // Best-effort — doesn't crash, may log a warning
    });

    it("tribal_store tool returns error on server failure", async () => {
      mockServerError(503, "Service Unavailable");

      const storeTool = api.tools.find((t) => t.name === "tribal_store")!;
      const result = await storeTool.execute(
        "call-err-1",
        { content: "Test content" },
        { sessionId: "session-e4" },
      );

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Failed to store");
    });

    it("tribal_recall tool returns empty on network failure", async () => {
      mockNetworkError("Connection refused");

      const recallTool = api.tools.find(
        (t) => t.name === "tribal_recall",
      )!;
      const result = await recallTool.execute(
        "call-err-2",
        { query: "test query" },
        { sessionId: "session-e5" },
      );

      // TribalClient.search() catches per-query errors and returns []
      // tribal_recall formats [] as "No memories found"
      expect(result.content[0].text).toContain("No memories found");
    });

    it("memory_search tool returns error on unexpected exception", async () => {
      // Simulate a truly unexpected error by having fetch throw after
      // the server-unavailability flag is already set
      mockNetworkError("unexpected");
      // The second search also fails
      mockNetworkError("unexpected again");

      const searchTool = api.tools.find(
        (t) => t.name === "memory_search",
      )!;

      // First call sets useBuiltinFallback = true
      await searchTool.execute(
        "call-err-3a",
        { query: "first search test" },
        { sessionId: "session-e6" },
      );

      // Second call uses builtin fallback (which doesn't exist) — should handle gracefully
      const result2 = await searchTool.execute(
        "call-err-3b",
        { query: "second search test" },
        { sessionId: "session-e6" },
      );

      // Should not crash
      expect(result2.content).toBeDefined();
    });
  });

  // ==========================================================================
  // Service lifecycle
  // ==========================================================================

  describe("service lifecycle", () => {
    it("service start logs connection status on success", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: "ok",
          instance_id: "test-inst",
          memory_count: 42,
        }),
      });

      await api.services[0].start();

      expect(api.logger.info).toHaveBeenCalledWith(
        expect.stringContaining("Connected to server"),
      );
    });

    it("service start warns when server is unreachable", async () => {
      mockNetworkError("ECONNREFUSED");

      await api.services[0].start();

      expect(api.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining("not reachable"),
      );
    });

    it("service stop logs shutdown", () => {
      api.services[0].stop();

      expect(api.logger.info).toHaveBeenCalledWith(
        expect.stringContaining("Service stopped"),
      );
    });
  });

  // ==========================================================================
  // Smart trigger gating (when enabled)
  // ==========================================================================

  describe("smart trigger gating", () => {
    it("skips auto-recall for emoji-only prompts when enabled", async () => {
      const smartApi = createMockApi({
        smartTriggerEnabled: true,
        smartTriggerSkipEmojiOnly: true,
      });
      plugin.register(smartApi as any);

      const result = await fireHook(
        smartApi,
        "before_agent_start",
        { prompt: "👍👍👍👍👍" },
        { sessionKey: "session-st1" },
      );

      expect(result).toBeUndefined();
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("skips auto-recall for short queries when enabled", async () => {
      const smartApi = createMockApi({
        smartTriggerEnabled: true,
        smartTriggerMinQueryLength: 10,
      });
      plugin.register(smartApi as any);

      const result = await fireHook(
        smartApi,
        "before_agent_start",
        { prompt: "hello" },
        { sessionKey: "session-st2" },
      );

      expect(result).toBeUndefined();
      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  // ==========================================================================
  // Circuit breaker integration
  // ==========================================================================

  describe("circuit breaker integration", () => {
    it("stops auto-recall after consecutive empty results", async () => {
      // 5 consecutive empty recalls to trip the circuit breaker
      for (let i = 0; i < 5; i++) {
        mockRecallResponse([]);
        await fireHook(
          api,
          "before_agent_start",
          { prompt: `Test query number ${i + 1} to trip breaker` },
          { sessionKey: "session-cb" },
        );
      }

      // Reset mock to track next call
      mockFetch.mockReset();

      // 6th call should be blocked by circuit breaker — no fetch call
      const result = await fireHook(
        api,
        "before_agent_start",
        { prompt: "This should be blocked by circuit breaker" },
        { sessionKey: "session-cb" },
      );

      expect(result).toBeUndefined();
      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  // ==========================================================================
  // Config migration (backward compat)
  // ==========================================================================

  describe("config migration", () => {
    it("warns about deprecated config names", () => {
      // Must NOT include `serverUrl` in the config so the rename
      // from `tribalServerUrl` → `serverUrl` actually triggers.
      const oldConfigApi = createMockApi();
      // Overwrite pluginConfig directly to omit serverUrl
      oldConfigApi.pluginConfig = {
        tribalServerUrl: "http://custom:9999",
        autoRecall: false,
        autoCapture: false,
      } as any;
      plugin.register(oldConfigApi as any);

      expect(oldConfigApi.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining("Deprecated config names"),
      );
    });
  });
});
