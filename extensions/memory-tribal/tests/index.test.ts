import { describe, it, expect, vi, beforeEach } from "vitest";

const state = vi.hoisted(() => ({
  queryCacheInstance: null as any,
  queryExpanderInstance: null as any,
  feedbackTrackerInstance: null as any,
  tribalClientInstance: null as any,
  persistenceInstance: null as any,
}));

vi.mock("../src/learned/query-cache", () => {
  class QueryCache {
    lookup: any;
    invalidatePath: any;
    recordSuccess: any;
    size: any;
    hitRate: any;

    constructor(_minSuccesses: number, _persistence: any) {
      state.queryCacheInstance = this;
      this.lookup = vi.fn(async () => null);
      this.invalidatePath = vi.fn();
      this.recordSuccess = vi.fn(async () => {});
      this.size = vi.fn(() => 0);
      this.hitRate = vi.fn(() => 0);
    }
  }

  return { QueryCache };
});

vi.mock("../src/learned/query-expander", () => {
  class QueryExpander {
    expand: any;
    learnExpansion: any;
    ruleCount: any;

    constructor(_persistence: any) {
      state.queryExpanderInstance = this;
      this.expand = vi.fn((query: string) => [query, "variant query"]);
      this.learnExpansion = vi.fn();
      this.ruleCount = vi.fn(() => 1);
    }
  }

  return { QueryExpander };
});

vi.mock("../src/learned/feedback-tracker", () => {
  class FeedbackTracker {
    recordRetrieval: any;
    recordUsage: any;
    rerank: any;
    getLastRetrieval: any;
    totalFeedback: any;

    constructor(_persistence: any) {
      state.feedbackTrackerInstance = this;
      this.recordRetrieval = vi.fn();
      this.recordUsage = vi.fn(async () => {});
      this.rerank = vi.fn((query: string, results: any[]) => results);
      this.getLastRetrieval = vi.fn(() => null);
      this.totalFeedback = vi.fn(() => 0);
    }
  }

  return { FeedbackTracker };
});

vi.mock("../src/tribal-client", () => {
  class TribalClient {
    search: any;
    correct: any;
    forget: any;

    constructor(_baseUrl: string) {
      state.tribalClientInstance = this;
      this.search = vi.fn(async () => []);
      this.correct = vi.fn(async () => ({ success: true, memoryId: "new" }));
      this.forget = vi.fn(async () => true);
    }
  }

  return { TribalClient };
});

vi.mock("../src/persistence", () => {
  class PersistenceLayer {
    invalidateCacheByPath: any;

    constructor() {
      state.persistenceInstance = this;
      this.invalidateCacheByPath = vi.fn();
    }
  }

  return { PersistenceLayer };
});

import memoryTribal from "../index";

function setupApi(configOverrides: Record<string, any> = {}) {
  const tools = new Map<string, any>();
  const api = {
    config: {
      plugins: {
        entries: {
          "memory-tribal": {
            config: {
              tribalServerUrl: "http://localhost:18790",
              queryCacheEnabled: true,
              queryExpansionEnabled: true,
              feedbackEnabled: true,
              minCacheSuccesses: 3,
              ...configOverrides,
            },
          },
        },
      },
    },
    log: {
      info: vi.fn(),
      warn: vi.fn(),
      debug: vi.fn(),
    },
    runtime: {
      memorySearch: vi.fn(),
      memoryGet: vi.fn(),
    },
    registerTool: vi.fn((tool: any) => {
      tools.set(tool.name, tool);
    }),
  };

  memoryTribal(api);
  return { api, tools };
}

beforeEach(() => {
  state.queryCacheInstance = null;
  state.queryExpanderInstance = null;
  state.feedbackTrackerInstance = null;
  state.tribalClientInstance = null;
  state.persistenceInstance = null;
  vi.clearAllMocks();
});


describe("memory-tribal plugin", () => {
  it("applies rerank, learns expansions, and invalidates superseded cache paths", async () => {
    const { tools } = setupApi();

    const memorySearch = tools.get("memory_search");
    expect(memorySearch).toBeDefined();

    const supersededId = "deadbeefcafebabe";
    const results = [
      {
        path: "tribal-memory:aaaa1111",
        score: 0.4,
        snippet: "Result A",
        sourceQuery: "variant query",
        supersedes: supersededId,
      },
      {
        path: "tribal-memory:bbbb2222",
        score: 0.9,
        snippet: "Result B",
        sourceQuery: "original query",
      },
    ];

    state.tribalClientInstance.search.mockResolvedValue(results);
    state.feedbackTrackerInstance.rerank.mockReturnValue([results[1], results[0]]);

    const response = await memorySearch.execute(
      "call-1",
      { query: "original query", maxResults: 5, minScore: 0.1 },
      { sessionId: "session-1" }
    );

    expect(state.feedbackTrackerInstance.rerank).toHaveBeenCalled();
    expect(state.queryExpanderInstance.learnExpansion).toHaveBeenCalledWith(
      "original query",
      "variant query"
    );

    const expectedPath = `tribal-memory:${supersededId.slice(0, 8)}`;
    expect(state.queryCacheInstance.invalidatePath).toHaveBeenCalledWith(expectedPath);
    expect(state.persistenceInstance.invalidateCacheByPath).toHaveBeenCalledWith(expectedPath);

    expect(state.feedbackTrackerInstance.recordRetrieval).toHaveBeenCalledWith(
      "session-1",
      "original query",
      ["tribal-memory:bbbb2222", "tribal-memory:aaaa1111"]
    );

    const outputText = response.content[0].text as string;
    expect(outputText).toContain("Result 1");
    expect(outputText).toContain("Result B");
  });

  it("invalidates cache on memory_correct and memory_forget", async () => {
    const { tools } = setupApi();

    const memoryCorrect = tools.get("memory_correct");
    const memoryForget = tools.get("memory_forget");

    await memoryCorrect.execute("call-2", {
      originalId: "abc12345deadbeef",
      correctedContent: "Updated",
      context: "Fix",
    });

    await memoryForget.execute("call-3", { memoryId: "deadbeefcafebabe" });

    expect(state.tribalClientInstance.correct).toHaveBeenCalled();
    expect(state.tribalClientInstance.forget).toHaveBeenCalled();

    expect(state.queryCacheInstance.invalidatePath).toHaveBeenCalledWith("tribal-memory:abc12345");
    expect(state.queryCacheInstance.invalidatePath).toHaveBeenCalledWith("tribal-memory:deadbeef");
    expect(state.persistenceInstance.invalidateCacheByPath).toHaveBeenCalledWith("tribal-memory:abc12345");
    expect(state.persistenceInstance.invalidateCacheByPath).toHaveBeenCalledWith("tribal-memory:deadbeef");
  });
});
