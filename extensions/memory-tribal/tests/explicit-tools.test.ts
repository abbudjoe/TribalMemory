/**
 * Tests for tribal_store and tribal_recall tools.
 *
 * Verifies that the explicit store/recall tools correctly call the
 * TribalClient methods and format responses.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";

// Mock fetch globally before importing anything that uses it
const mockFetch = vi.fn();
global.fetch = mockFetch;

// We test the tools through the plugin's register() by providing a
// minimal mock of OpenClawPluginApi.
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

function createMockApi() {
  const tools: RegisteredTool[] = [];
  return {
    tools,
    pluginConfig: {
      serverUrl: "http://localhost:18790",
      autoRecall: false,
      autoCapture: false,
    },
    logger: {
      info: vi.fn(),
      warn: vi.fn(),
      debug: vi.fn(),
      error: vi.fn(),
    },
    runtime: {},
    on: vi.fn(),
    registerTool(toolDef: RegisteredTool, _opts?: unknown) {
      tools.push(toolDef);
    },
    registerCli: vi.fn(),
    registerService: vi.fn(),
  };
}

// Import the plugin
import plugin from "../index";

describe("tribal_store tool", () => {
  let api: ReturnType<typeof createMockApi>;
  let storeTool: RegisteredTool;

  beforeEach(() => {
    mockFetch.mockReset();
    api = createMockApi();
    plugin.register(api as any);
    storeTool = api.tools.find((t) => t.name === "tribal_store")!;
    expect(storeTool).toBeDefined();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("is registered with correct name and description", () => {
    expect(storeTool.name).toBe("tribal_store");
    expect(storeTool.description).toContain("Deliberately store");
  });

  it("stores a memory via /v1/remember", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        memory_id: "mem-001",
      }),
    });

    const result = await storeTool.execute(
      "call-1",
      { content: "Auth uses JWT with RS256 signing" },
      { sessionId: "test-session" },
    );

    expect(result.content[0].text).toContain("Stored memory mem-001");
    expect(result.isError).toBeUndefined();
  });

  it("includes tags in the response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        memory_id: "mem-002",
      }),
    });

    const result = await storeTool.execute(
      "call-2",
      {
        content: "We decided to use FastEmbed as default",
        tags: ["decision", "architecture"],
      },
      { sessionId: "test-session" },
    );

    expect(result.content[0].text).toContain(
      "tags: decision, architecture",
    );
  });

  it("handles duplicate detection gracefully", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: false,
        duplicate_of: "existing-123",
      }),
    });

    const result = await storeTool.execute(
      "call-3",
      { content: "Already stored content" },
      { sessionId: "test-session" },
    );

    expect(result.content[0].text).toContain("already exists");
    expect(result.content[0].text).toContain("existing-123");
    expect(result.isError).toBeUndefined();
  });

  it("sends source_type as 'deliberate'", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        memory_id: "mem-003",
      }),
    });

    await storeTool.execute(
      "call-4",
      { content: "Test content" },
      { sessionId: "test-session" },
    );

    const rememberCall = mockFetch.mock.calls.find(
      (c: any[]) =>
        typeof c[0] === "string" && c[0].includes("/v1/remember"),
    );
    expect(rememberCall).toBeDefined();
    const body = JSON.parse(rememberCall![1].body);
    expect(body.source_type).toBe("deliberate");
  });

  it("returns error on server failure", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
    });

    const result = await storeTool.execute(
      "call-5",
      { content: "This will fail" },
      { sessionId: "test-session" },
    );

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("Failed to store");
  });
});

describe("tribal_recall tool", () => {
  let api: ReturnType<typeof createMockApi>;
  let recallTool: RegisteredTool;

  beforeEach(() => {
    mockFetch.mockReset();
    api = createMockApi();
    plugin.register(api as any);
    recallTool = api.tools.find((t) => t.name === "tribal_recall")!;
    expect(recallTool).toBeDefined();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("is registered with correct name and description", () => {
    expect(recallTool.name).toBe("tribal_recall");
    expect(recallTool.description).toContain("full control");
  });

  it("returns formatted results", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        results: [
          {
            memory: {
              id: "mem-100",
              content: "Auth uses JWT RS256",
              source_type: "deliberate",
              tags: ["architecture"],
            },
            similarity_score: 0.92,
            retrieval_time_ms: 5,
          },
        ],
      }),
    });

    const result = await recallTool.execute(
      "call-r1",
      { query: "how does auth work" },
      { sessionId: "test-session" },
    );

    expect(result.content[0].text).toContain("Found 1 memories");
    expect(result.content[0].text).toContain("0.920");
    expect(result.content[0].text).toContain("Auth uses JWT RS256");
    expect(result.content[0].text).toContain("architecture");
  });

  it("returns 'no memories found' on empty results", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ results: [] }),
    });

    const result = await recallTool.execute(
      "call-r2",
      { query: "something obscure" },
      { sessionId: "test-session" },
    );

    expect(result.content[0].text).toContain("No memories found");
  });

  it("passes tags filter to the client", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ results: [] }),
    });

    await recallTool.execute(
      "call-r3",
      { query: "decisions", tags: ["decision"] },
      { sessionId: "test-session" },
    );

    const recallCall = mockFetch.mock.calls.find(
      (c: any[]) =>
        typeof c[0] === "string" && c[0].includes("/v1/recall"),
    );
    expect(recallCall).toBeDefined();
    const body = JSON.parse(recallCall![1].body);
    expect(body.tags).toEqual(["decision"]);
  });

  it("passes temporal filters to the client", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ results: [] }),
    });

    await recallTool.execute(
      "call-r4",
      {
        query: "recent events",
        after: "2026-02-01",
        before: "2026-02-05",
      },
      { sessionId: "test-session" },
    );

    const recallCall = mockFetch.mock.calls.find(
      (c: any[]) =>
        typeof c[0] === "string" && c[0].includes("/v1/recall"),
    );
    expect(recallCall).toBeDefined();
    const body = JSON.parse(recallCall![1].body);
    expect(body.after).toBe("2026-02-01");
    expect(body.before).toBe("2026-02-05");
  });

  it("respects custom limit and min_relevance", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ results: [] }),
    });

    await recallTool.execute(
      "call-r5",
      { query: "test", limit: 10, min_relevance: 0.5 },
      { sessionId: "test-session" },
    );

    const recallCall = mockFetch.mock.calls.find(
      (c: any[]) =>
        typeof c[0] === "string" && c[0].includes("/v1/recall"),
    );
    expect(recallCall).toBeDefined();
    const body = JSON.parse(recallCall![1].body);
    expect(body.limit).toBe(10);
    expect(body.min_relevance).toBe(0.5);
  });

  it("returns empty results on server failure (search is resilient)", async () => {
    mockFetch.mockRejectedValueOnce(
      new Error("Connection refused"),
    );

    const result = await recallTool.execute(
      "call-r6",
      { query: "test" },
      { sessionId: "test-session" },
    );

    // search() catches per-query errors and returns empty results
    // rather than propagating — this is by design for resilience
    expect(result.content[0].text).toContain("No memories found");
  });
});
