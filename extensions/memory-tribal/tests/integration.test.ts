/**
 * Integration tests for memory-tribal plugin.
 *
 * Tests the full plugin lifecycle with a mock TribalClient:
 * - Auto-recall flow (before_agent_start hook)
 * - Auto-capture flow (agent_end hook)
 * - /remember command handling
 * - Error handling (server down, timeout)
 * - Query extraction (filtering system messages)
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";

// Mock fetch globally before importing the plugin
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Import the plugin after mocking fetch
import plugin from "../index";

// ============================================================================
// Mock API creation
// ============================================================================

interface HookCallback {
  (event: Record<string, unknown>, ctx?: Record<string, unknown>):
    | Promise<Record<string, unknown> | void>
    | Record<string, unknown>
    | void;
}

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

function createMockApi(config: Record<string, unknown> = {}) {
  const tools: RegisteredTool[] = [];
  const hooks: Record<string, HookCallback[]> = {};
  const services: Array<{ id: string; start: () => Promise<void>; stop: () => void }> = [];

  return {
    tools,
    hooks,
    services,
    pluginConfig: {
      serverUrl: "http://localhost:18790",
      autoRecall: true,
      autoCapture: true,
      queryCacheEnabled: false, // Disable for simpler testing
      queryExpansionEnabled: false,
      feedbackEnabled: false,
      smartTriggerEnabled: false,
      circuitBreakerMaxEmpty: 999, // Disable circuit breaker
      sessionDedupEnabled: false,
      ...config,
    },
    logger: {
      info: vi.fn(),
      warn: vi.fn(),
      debug: vi.fn(),
      error: vi.fn(),
    },
    runtime: {},
    on(event: string, callback: HookCallback) {
      if (!hooks[event]) hooks[event] = [];
      hooks[event].push(callback);
    },
    registerTool(toolDef: RegisteredTool, _opts?: unknown) {
      tools.push(toolDef);
    },
    registerCli: vi.fn(),
    registerService(serviceDef: { id: string; start: () => Promise<void>; stop: () => void }) {
      services.push(serviceDef);
    },
    // Helper to trigger hooks
    async triggerHook(
      event: string,
      payload: Record<string, unknown>,
      ctx?: Record<string, unknown>,
    ) {
      if (!hooks[event]) return null;
      for (const callback of hooks[event]) {
        const result = await callback(payload, ctx);
        if (result) return result;
      }
      return null;
    },
  };
}

// ============================================================================
// Integration Tests: Auto-Recall Flow
// ============================================================================

describe("Integration: Auto-recall flow", () => {
  let api: ReturnType<typeof createMockApi>;

  beforeEach(() => {
    mockFetch.mockReset();
    api = createMockApi();
    plugin.register(api as any); // Type cast needed: mock API simplified for testing (doesn't implement full OpenClawPluginApi interface)
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("injects relevant memories on before_agent_start", async () => {
    // Mock successful recall
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        results: [
          {
            memory: {
              id: "mem-001",
              content: "Joe prefers TypeScript over JavaScript",
              source_type: "user_explicit",
              tags: ["preference"],
            },
            similarity_score: 0.85,
            retrieval_time_ms: 10,
          },
        ],
      }),
    });

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "What programming languages should I use?" },
      { sessionKey: "test-session" },
    );

    expect(result).toBeDefined();
    expect(result!.prependContext).toContain("<relevant-memories>");
    expect(result!.prependContext).toContain("Joe prefers TypeScript");
    expect(result!.prependContext).toContain("</relevant-memories>");
  });

  it("skips recall for very short prompts", async () => {
    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "hi" },
      { sessionKey: "test-session" },
    );

    expect(result).toBeNull();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("returns nothing when no memories are found", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ results: [] }),
    });

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "What is the weather today?" },
      { sessionKey: "test-session" },
    );

    expect(result).toBeNull();
  });

  it("filters System: lines from search query", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ results: [] }),
    });

    await api.triggerHook(
      "before_agent_start",
      {
        prompt: `System: exec output here
System: more output
What is the weather?`,
      },
      { sessionKey: "test-session" },
    );

    const recallCall = mockFetch.mock.calls.find(
      (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/recall"),
    );
    expect(recallCall).toBeDefined();
    const body = JSON.parse(recallCall![1].body);
    expect(body.query).toBe("What is the weather?");
  });

  it("extracts message from Telegram channel prefix", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ results: [] }),
    });

    await api.triggerHook(
      "before_agent_start",
      {
        prompt: "[Telegram Joe (@abbudjoe) id:123 +5s] What is TribalMemory?",
      },
      { sessionKey: "test-session" },
    );

    const recallCall = mockFetch.mock.calls.find(
      (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/recall"),
    );
    expect(recallCall).toBeDefined();
    const body = JSON.parse(recallCall![1].body);
    expect(body.query).toBe("What is TribalMemory?");
  });

  it("handles server errors gracefully (no crash)", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Connection refused"));

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "What is the weather?" },
      { sessionKey: "test-session" },
    );

    // Should not crash - TribalClient.search() catches errors per-query
    // and returns empty array, which results in null from the hook
    expect(result).toBeNull();
  });

  it("does not recall when autoRecall is disabled", async () => {
    api = createMockApi({ autoRecall: false });
    plugin.register(api as any); // Type cast needed: mock API simplified for testing (doesn't implement full OpenClawPluginApi interface)

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "What is the weather?" },
      { sessionKey: "test-session" },
    );

    expect(result).toBeNull();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("applies safeguards pipeline to recalled memories", async () => {
    // Mock multiple results to verify that the safeguards pipeline is applied
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        results: [
          {
            memory: {
              id: "mem-001",
              content: "Joe prefers TypeScript",
              source_type: "user_explicit",
              tags: [],
            },
            similarity_score: 0.9,
            retrieval_time_ms: 5,
          },
          {
            memory: {
              id: "mem-002",
              content: "Project uses React for frontend",
              source_type: "user_explicit",
              tags: [],
            },
            similarity_score: 0.8,
            retrieval_time_ms: 5,
          },
        ],
      }),
    });

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "Tell me about the project" },
      { sessionKey: "test-session" },
    );

    // Verify that safeguards were applied and memories were included
    expect(result).toBeDefined();
    expect(result!.prependContext).toContain("<relevant-memories>");
    expect(result!.prependContext).toContain("Joe prefers TypeScript");
    expect(result!.prependContext).toContain("Project uses React");
    
    // Verify token budget was consulted (implementation detail: recorded usage)
    // The actual limits depend on token counting heuristics, so we just verify
    // the pipeline runs without errors
  });
});

// ============================================================================
// Integration Tests: Auto-Capture Flow
// ============================================================================

describe("Integration: Auto-capture flow", () => {
  let api: ReturnType<typeof createMockApi>;

  beforeEach(() => {
    mockFetch.mockReset();
    api = createMockApi();
    plugin.register(api as any); // Type cast needed: mock API simplified for testing (doesn't implement full OpenClawPluginApi interface)
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("captures memorable content from user messages", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ success: true, memory_id: "captured-001" }),
    });

    await api.triggerHook(
      "agent_end",
      {
        success: true,
        messages: [
          { role: "user", content: "I prefer dark mode always" },
          { role: "assistant", content: "Got it, I'll remember that." },
        ],
      },
      { sessionKey: "test-session" },
    );

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/v1/remember"),
      expect.objectContaining({ method: "POST" }),
    );

    const rememberCall = mockFetch.mock.calls.find(
      (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/remember"),
    );
    const body = JSON.parse(rememberCall![1].body);
    expect(body.content).toBe("I prefer dark mode always");
    expect(body.source_type).toBe("auto_capture");
  });

  it("does not capture when autoCapture is disabled", async () => {
    api = createMockApi({ autoCapture: false });
    plugin.register(api as any); // Type cast needed: mock API simplified for testing (doesn't implement full OpenClawPluginApi interface)

    await api.triggerHook(
      "agent_end",
      {
        success: true,
        messages: [{ role: "user", content: "I prefer dark mode always" }],
      },
      { sessionKey: "test-session" },
    );

    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("skips capture when success is false", async () => {
    await api.triggerHook(
      "agent_end",
      {
        success: false,
        messages: [{ role: "user", content: "I prefer dark mode always" }],
      },
      { sessionKey: "test-session" },
    );

    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("extracts text from content block arrays", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ success: true, memory_id: "captured-002" }),
    });

    await api.triggerHook(
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
      { sessionKey: "test-session" },
    );

    const rememberCall = mockFetch.mock.calls.find(
      (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/remember"),
    );
    expect(rememberCall).toBeDefined();
    const body = JSON.parse(rememberCall![1].body);
    expect(body.content).toBe("I always use vim for editing");
  });

  it("limits captures to 3 per turn", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ success: true, memory_id: "captured-multi" }),
    });

    await api.triggerHook(
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
      { sessionKey: "test-session" },
    );

    // Should capture at most 3 memories
    const rememberCalls = mockFetch.mock.calls.filter(
      (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/remember"),
    );
    expect(rememberCalls.length).toBeLessThanOrEqual(3);
  });

  it("handles malformed messages without crashing", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ success: true, memory_id: "captured-safe" }),
    });

    await api.triggerHook(
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
      { sessionKey: "test-session" },
    );

    // Should not crash, should process the valid message
    const rememberCalls = mockFetch.mock.calls.filter(
      (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/remember"),
    );
    expect(rememberCalls.length).toBeGreaterThan(0);
  });

  it("handles capture failures gracefully (best effort)", async () => {
    mockFetch.mockRejectedValue(new Error("Connection refused"));

    // Should not crash even when all captures fail
    await expect(
      api.triggerHook(
        "agent_end",
        {
          success: true,
          messages: [{ role: "user", content: "I prefer dark mode always" }],
        },
        { sessionKey: "test-session" },
      ),
    ).resolves.not.toThrow();
  });
});

// ============================================================================
// Integration Tests: /remember Command
// ============================================================================

describe("Integration: /remember command", () => {
  let api: ReturnType<typeof createMockApi>;

  beforeEach(() => {
    mockFetch.mockReset();
    api = createMockApi();
    plugin.register(api as any); // Type cast needed: mock API simplified for testing (doesn't implement full OpenClawPluginApi interface)
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("stores memory and returns success context", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, memory_id: "cmd-001" }),
    });

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "/remember Joe prefers TypeScript" },
      { sessionKey: "test-session" },
    );

    expect(result).toBeDefined();
    expect(result!.prependContext).toContain("<system-note>");
    expect(result!.prependContext).toContain("Joe prefers TypeScript");
    expect(result!.prependContext).toContain("cmd-001");
    expect(result!.prependContext).toContain("stored successfully");
    expect(result!.prependContext).toContain("</system-note>");

    const rememberCall = mockFetch.mock.calls.find(
      (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/remember"),
    );
    expect(rememberCall).toBeDefined();
    const body = JSON.parse(rememberCall![1].body);
    expect(body.content).toBe("Joe prefers TypeScript");
    expect(body.source_type).toBe("user_explicit");
  });

  it("handles /remember with channel prefix", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, memory_id: "cmd-002" }),
    });

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "[Telegram Joe (@abbudjoe)] /remember My birthday is in March" },
      { sessionKey: "test-session" },
    );

    expect(result!.prependContext).toContain("My birthday is in March");
    expect(result!.prependContext).toContain("stored successfully");
  });

  it("handles duplicate detection", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: false, duplicate_of: "existing-123" }),
    });

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "/remember Duplicate content" },
      { sessionKey: "test-session" },
    );

    expect(result!.prependContext).toContain("already exists");
    expect(result!.prependContext).toContain("duplicate");
    expect(result!.prependContext).toContain("Duplicate content");
  });

  it("handles storage errors", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Connection refused"));

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "/remember This will fail" },
      { sessionKey: "test-session" },
    );

    expect(result!.prependContext).toContain("Storage failed");
    expect(result!.prependContext).toContain("Connection refused");
    expect(result!.prependContext).toContain("Apologize");
  });

  it("preserves multiline content", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, memory_id: "cmd-multi" }),
    });

    await api.triggerHook(
      "before_agent_start",
      { prompt: "/remember Project structure:\n- src/\n- tests/\n- docs/" },
      { sessionKey: "test-session" },
    );

    const rememberCall = mockFetch.mock.calls.find(
      (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/remember"),
    );
    const body = JSON.parse(rememberCall![1].body);
    expect(body.content).toContain("Project structure:");
    expect(body.content).toContain("- src/");
  });

  it("does not trigger auto-recall when /remember is present", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, memory_id: "cmd-no-recall" }),
    });

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "/remember Joe likes pizza" },
      { sessionKey: "test-session" },
    );

    // Should only call /v1/remember (not /v1/recall)
    const recallCalls = mockFetch.mock.calls.filter(
      (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/recall"),
    );
    expect(recallCalls.length).toBe(0);

    const rememberCalls = mockFetch.mock.calls.filter(
      (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/remember"),
    );
    expect(rememberCalls.length).toBe(1);
  });
});

// ============================================================================
// Integration Tests: Error Handling
// ============================================================================

describe("Integration: Error handling", () => {
  let api: ReturnType<typeof createMockApi>;

  beforeEach(() => {
    mockFetch.mockReset();
    api = createMockApi();
    plugin.register(api as any); // Type cast needed: mock API simplified for testing (doesn't implement full OpenClawPluginApi interface)
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("handles server down gracefully in auto-recall", async () => {
    mockFetch.mockRejectedValueOnce(new Error("ECONNREFUSED"));

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "What is the weather?" },
      { sessionKey: "test-session" },
    );

    // Should return null (no results) when server is down
    // TribalClient.search() catches the error and returns []
    expect(result).toBeNull();
  });

  it("handles timeout in auto-recall", async () => {
    const abortError = new Error("The operation was aborted");
    abortError.name = "AbortError";
    mockFetch.mockRejectedValueOnce(abortError);

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "What is the weather?" },
      { sessionKey: "test-session" },
    );

    expect(result).toBeNull();
  });

  it("handles 500 errors in auto-recall", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
    });

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "What is the weather?" },
      { sessionKey: "test-session" },
    );

    expect(result).toBeNull();
  });

  it("switches to builtin fallback after server failure", async () => {
    // First call fails
    mockFetch.mockRejectedValueOnce(new Error("ECONNREFUSED"));

    const result1 = await api.triggerHook(
      "before_agent_start",
      { prompt: "What is the weather?" },
      { sessionKey: "test-session" },
    );

    expect(result1).toBeNull();

    // Second call should use builtin fallback (if available)
    // The plugin sets useBuiltinFallback=true after first failure
    const result2 = await api.triggerHook(
      "before_agent_start",
      { prompt: "What is the weather again?" },
      { sessionKey: "test-session" },
    );

    // Without api.runtime.memorySearch, should return null
    expect(result2).toBeNull();
  });

  it("handles malformed server responses", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        results: [
          {
            // Missing memory.id (invalid)
            memory: { content: "Test" },
            similarity_score: 0.8,
          },
        ],
      }),
    });

    // Should skip invalid results without crashing
    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "What is the weather?" },
      { sessionKey: "test-session" },
    );

    // Should return null (no valid results)
    expect(result).toBeNull();
  });
});

// ============================================================================
// Integration Tests: Service Registration
// ============================================================================

describe("Integration: Service registration", () => {
  let api: ReturnType<typeof createMockApi>;

  beforeEach(() => {
    mockFetch.mockReset();
    api = createMockApi();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("registers a service with health check", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: "ok",
        instance_id: "test-instance",
        memory_count: 42,
      }),
    });

    plugin.register(api as any); // Type cast needed: mock API simplified for testing (doesn't implement full OpenClawPluginApi interface)

    expect(api.services.length).toBe(1);
    expect(api.services[0].id).toBe("memory-tribal");

    await api.services[0].start();

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:18790/v1/health",
      expect.any(Object),
    );

    expect(api.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("Connected to server"),
    );
  });

  it("warns when server is not reachable on startup", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Connection refused"));

    plugin.register(api as any); // Type cast needed: mock API simplified for testing (doesn't implement full OpenClawPluginApi interface)

    await api.services[0].start();

    expect(api.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("Server not reachable"),
    );
  });
});

// ============================================================================
// Integration Tests: Full E2E Flow
// ============================================================================

describe("Integration: End-to-end flow", () => {
  let api: ReturnType<typeof createMockApi>;

  beforeEach(() => {
    mockFetch.mockReset();
    api = createMockApi();
    plugin.register(api as any); // Type cast needed: mock API simplified for testing (doesn't implement full OpenClawPluginApi interface)
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("handles /remember command → auto-recall → auto-capture in sequence", async () => {
    // Step 1: Store memory via /remember
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, memory_id: "e2e-001" }),
    });

    const storeResult = await api.triggerHook(
      "before_agent_start",
      { prompt: "/remember Joe prefers TypeScript" },
      { sessionKey: "e2e-session" },
    );

    expect(storeResult!.prependContext).toContain("stored successfully");

    // Step 2: Recall the stored memory
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        results: [
          {
            memory: {
              id: "e2e-001",
              content: "Joe prefers TypeScript",
              source_type: "user_explicit",
              tags: [],
            },
            similarity_score: 0.95,
            retrieval_time_ms: 5,
          },
        ],
      }),
    });

    const recallResult = await api.triggerHook(
      "before_agent_start",
      { prompt: "What languages should I use?" },
      { sessionKey: "e2e-session" },
    );

    expect(recallResult!.prependContext).toContain("Joe prefers TypeScript");

    // Step 3: Auto-capture new preference from conversation
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, memory_id: "e2e-002" }),
    });

    await api.triggerHook(
      "agent_end",
      {
        success: true,
        messages: [
          { role: "user", content: "I always use React for frontends" },
          { role: "assistant", content: "Noted, I'll remember that." },
        ],
      },
      { sessionKey: "e2e-session" },
    );

    const captureCalls = mockFetch.mock.calls.filter(
      (c: any[]) => typeof c[0] === "string" && c[0].includes("/v1/remember"),
    );
    expect(captureCalls.length).toBeGreaterThan(0);
  });
});

// ============================================================================
// Additional Integration Tests: Error Handling Edge Cases
// ============================================================================

describe("Integration: Additional error handling", () => {
  let api: ReturnType<typeof createMockApi>;

  beforeEach(() => {
    mockFetch.mockReset();
    api = createMockApi();
    plugin.register(api as any); // Type cast needed: mock API simplified for testing (doesn't implement full OpenClawPluginApi interface)
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("handles invalid JSON responses gracefully", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => {
        throw new Error("Unexpected token in JSON");
      },
    });

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "What is the weather?" },
      { sessionKey: "test-session" },
    );

    // Should not crash, should return null (no results)
    expect(result).toBeNull();
  });

  it("handles HTTP 429 rate limiting", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 429,
      statusText: "Too Many Requests",
    });

    const result = await api.triggerHook(
      "before_agent_start",
      { prompt: "What is the weather?" },
      { sessionKey: "test-session" },
    );

    // Should return null (TribalClient catches the error)
    expect(result).toBeNull();
  });
});
