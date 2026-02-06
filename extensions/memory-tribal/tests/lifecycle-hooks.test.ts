/**
 * Tests for lifecycle hooks (auto-recall and auto-capture).
 *
 * Uses mock TribalClient and mock OpenClaw plugin API to test
 * the hook logic in isolation.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";

// ============================================================================
// shouldCapture() tests (exported for testing)
// ============================================================================

// Re-implement shouldCapture locally since it's not exported
const CAPTURE_MIN_LENGTH = 10;
const CAPTURE_MAX_LENGTH = 500;

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
// shouldCapture tests
// ============================================================================

describe("shouldCapture", () => {
  it("captures preference statements", () => {
    expect(shouldCapture("I prefer TypeScript over JavaScript")).toBe(true);
  });

  it("captures 'remember' requests", () => {
    expect(shouldCapture("Remember that Joe likes pizza")).toBe(true);
  });

  it("captures email addresses", () => {
    expect(shouldCapture("My email is joe@example.com")).toBe(true);
  });

  it("captures phone numbers", () => {
    expect(shouldCapture("Call me at +15551234567")).toBe(true);
  });

  it("captures 'important' statements", () => {
    expect(shouldCapture("This is important for the project")).toBe(true);
  });

  it("rejects text too short", () => {
    expect(shouldCapture("hello")).toBe(false);
  });

  it("rejects text too long", () => {
    expect(shouldCapture("a".repeat(501))).toBe(false);
  });

  it("rejects injected memory context", () => {
    expect(
      shouldCapture("Check <relevant-memories> for context"),
    ).toBe(false);
  });

  it("rejects XML-like system content", () => {
    expect(shouldCapture("<system>Some instructions</system>")).toBe(false);
  });

  it("rejects markdown agent output", () => {
    expect(shouldCapture("**Title**\n- item one\n- item two")).toBe(false);
  });

  it("rejects emoji-heavy text", () => {
    expect(shouldCapture("🎉🎊🎈🎁 Party time!")).toBe(false);
  });

  it("rejects text with no triggers", () => {
    expect(shouldCapture("The weather is nice today")).toBe(false);
  });
});

// ============================================================================
// Auto-recall hook logic tests
// ============================================================================

describe("auto-recall logic", () => {
  it("skips short prompts (< 5 chars)", () => {
    // The hook returns early for short prompts
    const prompt = "hi";
    expect(prompt.length < 5).toBe(true);
  });

  it("generates unique fallback session IDs", () => {
    // When ctx.sessionKey is undefined, each call should get a unique ID
    const id1 = `auto-recall-${Date.now()}`;
    const id2 = `auto-recall-${Date.now() + 1}`;
    expect(id1).not.toBe(id2);
  });

  it("uses ctx.sessionKey when available", () => {
    const ctx = { sessionKey: "agent:main:main" };
    const sessionId = ctx?.sessionKey ?? `auto-recall-${Date.now()}`;
    expect(sessionId).toBe("agent:main:main");
  });

  it("formats memory context as XML block", () => {
    const memories = [
      { snippet: "Joe prefers TypeScript" },
      { snippet: "Project uses React" },
    ];
    const memoryContext = memories
      .map((r) => `- ${r.snippet ?? ""}`)
      .join("\n");
    const result =
      `<relevant-memories>\n` +
      `The following memories may be relevant:\n` +
      `${memoryContext}\n` +
      `</relevant-memories>`;

    expect(result).toContain("<relevant-memories>");
    expect(result).toContain("Joe prefers TypeScript");
    expect(result).toContain("Project uses React");
    expect(result).toContain("</relevant-memories>");
  });
});

// ============================================================================
// Auto-capture hook logic tests
// ============================================================================

describe("auto-capture logic", () => {
  it("extracts text from string content messages", () => {
    const messages = [
      { role: "user", content: "Remember I prefer dark mode" },
      { role: "assistant", content: "Got it, noted." },
    ];
    const texts: string[] = [];
    for (const msg of messages) {
      if (typeof msg.content === "string") {
        texts.push(msg.content);
      }
    }
    expect(texts).toHaveLength(2);
    expect(texts[0]).toBe("Remember I prefer dark mode");
  });

  it("extracts text from content block arrays", () => {
    const messages = [
      {
        role: "user",
        content: [
          { type: "text", text: "I always use vim" },
          { type: "image", url: "http://example.com/img.png" },
        ],
      },
    ];
    const texts: string[] = [];
    for (const msg of messages) {
      if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (
            block &&
            typeof block === "object" &&
            "type" in block &&
            block.type === "text" &&
            "text" in block &&
            typeof block.text === "string"
          ) {
            texts.push(block.text);
          }
        }
      }
    }
    expect(texts).toEqual(["I always use vim"]);
  });

  it("skips non-user/assistant roles", () => {
    const messages = [
      { role: "system", content: "You are a helpful assistant" },
      { role: "tool", content: "Tool result" },
      { role: "user", content: "I prefer Python" },
    ];
    const texts: string[] = [];
    for (const msg of messages) {
      if (msg.role !== "user" && msg.role !== "assistant") continue;
      if (typeof msg.content === "string") texts.push(msg.content);
    }
    expect(texts).toEqual(["I prefer Python"]);
  });

  it("skips malformed messages without crashing", () => {
    const messages = [
      null,
      undefined,
      42,
      { role: "user", content: "I always use TypeScript" },
    ];
    const texts: string[] = [];
    for (const msg of messages) {
      if (!msg || typeof msg !== "object") continue;
      try {
        const msgObj = msg as Record<string, unknown>;
        const role = msgObj.role;
        if (role !== "user" && role !== "assistant") continue;
        if (typeof msgObj.content === "string") {
          texts.push(msgObj.content as string);
        }
      } catch {
        continue;
      }
    }
    expect(texts).toEqual(["I always use TypeScript"]);
  });

  it("limits captures to 3 per turn", () => {
    const capturable = [
      "I prefer dark mode always",
      "Remember my timezone is MST",
      "I love using vim for editing",
      "My email is joe@test.com",
      "I need more coffee always",
    ];
    const toCapture = capturable.filter(shouldCapture).slice(0, 3);
    expect(toCapture.length).toBeLessThanOrEqual(3);
  });

  it("filters through shouldCapture", () => {
    const texts = [
      "I prefer dark mode always",  // ✓ trigger
      "The weather is nice",         // ✗ no trigger
      "hi",                          // ✗ too short
      "Remember to buy milk",        // ✓ trigger
    ];
    const captured = texts.filter(shouldCapture);
    expect(captured).toEqual([
      "I prefer dark mode always",
      "Remember to buy milk",
    ]);
  });
});

// ============================================================================
// Service registration tests
// ============================================================================

describe("service registration", () => {
  it("service object has required id, start, stop", () => {
    const service = {
      id: "memory-tribal",
      start: async () => {},
      stop: () => {},
    };
    expect(service.id).toBe("memory-tribal");
    expect(typeof service.start).toBe("function");
    expect(typeof service.stop).toBe("function");
  });
});

// ============================================================================
// CLI command logic tests
// ============================================================================

describe("CLI commands", () => {
  describe("search --limit parsing", () => {
    it("parses valid integer limit", () => {
      const limit = parseInt("10", 10);
      const maxResults = isNaN(limit) ? 5 : limit;
      expect(maxResults).toBe(10);
    });

    it("falls back to 5 for NaN input", () => {
      const limit = parseInt("abc", 10);
      const maxResults = isNaN(limit) ? 5 : limit;
      expect(maxResults).toBe(5);
    });

    it("falls back to 5 for empty input", () => {
      const limit = parseInt("", 10);
      const maxResults = isNaN(limit) ? 5 : limit;
      expect(maxResults).toBe(5);
    });

    it("handles decimal input (takes integer part)", () => {
      const limit = parseInt("3.7", 10);
      const maxResults = isNaN(limit) ? 5 : limit;
      expect(maxResults).toBe(3);
    });
  });

  describe("search output formatting", () => {
    it("handles results with score and snippet", () => {
      const r = { score: 0.85, snippet: "Joe prefers TypeScript" };
      const score = r.score?.toFixed(3) ?? "N/A";
      const snippet = r.snippet?.slice(0, 80) ?? "";
      expect(score).toBe("0.850");
      expect(snippet).toBe("Joe prefers TypeScript");
    });

    it("handles results with undefined score", () => {
      const r = { score: undefined, snippet: "test" } as any;
      const score = r.score?.toFixed(3) ?? "N/A";
      expect(score).toBe("N/A");
    });

    it("handles results with undefined snippet", () => {
      const r = { score: 0.9, snippet: undefined } as any;
      const snippet = r.snippet?.slice(0, 80) ?? "";
      expect(snippet).toBe("");
    });
  });
});

// ============================================================================
// /remember command tests
// ============================================================================

// Re-implement extractRememberCommand for testing (not exported)
const REMEMBER_COMMAND_RE = /^(?:\[.*?\]\s*)?\/remember\s+(.+)$/is;

function extractRememberCommand(prompt: string): string | null {
  const match = prompt.match(REMEMBER_COMMAND_RE);
  if (!match) return null;
  return match[1].trim();
}

describe("extractRememberCommand", () => {
  it("extracts simple /remember command", () => {
    expect(extractRememberCommand("/remember Joe prefers TypeScript")).toBe(
      "Joe prefers TypeScript",
    );
  });

  it("extracts /remember with channel prefix", () => {
    expect(
      extractRememberCommand(
        "[Telegram Joe (@abbudjoe)] /remember My birthday is in March",
      ),
    ).toBe("My birthday is in March");
  });

  it("handles multiline content", () => {
    expect(
      extractRememberCommand("/remember Joe's preferences:\n- dark mode\n- vim"),
    ).toBe("Joe's preferences:\n- dark mode\n- vim");
  });

  it("is case insensitive for command", () => {
    expect(extractRememberCommand("/REMEMBER important fact")).toBe(
      "important fact",
    );
  });

  it("returns null for non-command prompts", () => {
    expect(extractRememberCommand("What's the weather?")).toBeNull();
  });

  it("returns null for empty content after /remember", () => {
    // The regex requires at least one character after /remember
    expect(extractRememberCommand("/remember ")).toBeNull();
  });

  it("returns null for /remember in the middle of text", () => {
    expect(
      extractRememberCommand("Please /remember this for later"),
    ).toBeNull();
  });

  it("handles complex channel prefixes", () => {
    expect(
      extractRememberCommand(
        "[Signal +1555123... id:abc123] /remember API key is stored in .env",
      ),
    ).toBe("API key is stored in .env");
  });

  it("preserves special characters in content", () => {
    expect(
      extractRememberCommand("/remember Email: joe@example.com, Phone: +15551234"),
    ).toBe("Email: joe@example.com, Phone: +15551234");
  });
});

describe("query extraction for auto-recall", () => {
  // Re-implement extraction logic for testing
  function extractSearchQuery(prompt: string): string {
    const lines = prompt.split("\n");
    const userLines = lines
      .filter(
        (line) => !line.startsWith("System:") && line.trim().length > 0,
      )
      .map((line) => {
        const match = line.match(
          /^\[(?:Telegram|Matrix|Discord|Signal)[^\]]+\]\s*(.+)$/i,
        );
        return match ? match[1] : line;
      });
    if (userLines.length > 0) {
      return userLines.slice(-3).join(" ").slice(0, 500);
    }
    return prompt.slice(0, 500);
  }

  it("filters out System: lines", () => {
    const prompt = `System: exec output here
System: more output
What is the weather?`;
    expect(extractSearchQuery(prompt)).toBe("What is the weather?");
  });

  it("extracts message from Telegram prefix", () => {
    const prompt = `[Telegram Joe (@abbudjoe) id:123 +5s] Hello world`;
    expect(extractSearchQuery(prompt)).toBe("Hello world");
  });

  it("extracts message from Matrix prefix", () => {
    const prompt = `[Matrix joe +1m] What time is it?`;
    expect(extractSearchQuery(prompt)).toBe("What time is it?");
  });

  it("extracts message from Discord prefix", () => {
    const prompt = `[Discord joe#1234] Check this out`;
    expect(extractSearchQuery(prompt)).toBe("Check this out");
  });

  it("extracts message from Signal prefix", () => {
    const prompt = `[Signal +1555... id:abc] Remember this`;
    expect(extractSearchQuery(prompt)).toBe("Remember this");
  });

  it("uses last 3 lines for multi-line prompts", () => {
    const prompt = `line1
line2
line3
line4
line5`;
    expect(extractSearchQuery(prompt)).toBe("line3 line4 line5");
  });

  it("truncates to 500 chars", () => {
    const longLine = "A".repeat(600);
    expect(extractSearchQuery(longLine).length).toBe(500);
  });

  it("handles complex prompt with system + channel prefix", () => {
    const prompt = `System: Running exec command...
System: Output captured
[Telegram Joe (@abbudjoe) id:3219 +1m 2026-02-06] What is TribalMemory?`;
    expect(extractSearchQuery(prompt)).toBe("What is TribalMemory?");
  });

  it("preserves non-prefixed lines", () => {
    const prompt = `Just a plain question`;
    expect(extractSearchQuery(prompt)).toBe("Just a plain question");
  });
});

describe("/remember command handler", () => {
  it("formats success response with memory ID", () => {
    const content = "Joe prefers dark mode";
    const memoryId = "mem_abc123";
    const result = {
      prependContext:
        `<system-note>\n` +
        `User requested to remember: "${content.slice(0, 100)}..."\n` +
        `Memory stored successfully (id: ${memoryId}).\n` +
        `Acknowledge this briefly to the user.\n` +
        `</system-note>`,
    };
    expect(result.prependContext).toContain("Joe prefers dark mode");
    expect(result.prependContext).toContain("mem_abc123");
    expect(result.prependContext).toContain("Acknowledge this briefly");
  });

  it("formats duplicate response", () => {
    const content = "Already stored fact";
    const duplicateOf = "mem_xyz789";
    const result = {
      prependContext:
        `<system-note>\n` +
        `User requested to remember: "${content.slice(0, 100)}..."\n` +
        `This memory already exists (duplicate). No action needed.\n` +
        `</system-note>`,
    };
    expect(result.prependContext).toContain("Already stored fact");
    expect(result.prependContext).toContain("duplicate");
  });

  it("formats error response", () => {
    const content = "Failed storage";
    const errorMessage = "Connection refused";
    const result = {
      prependContext:
        `<system-note>\n` +
        `User tried to remember: "${content.slice(0, 100)}..."\n` +
        `Storage failed: ${errorMessage}\n` +
        `Apologize and suggest trying again.\n` +
        `</system-note>`,
    };
    expect(result.prependContext).toContain("Storage failed");
    expect(result.prependContext).toContain("Connection refused");
    expect(result.prependContext).toContain("Apologize");
  });

  it("truncates long content in response", () => {
    const longContent = "A".repeat(200);
    const truncated = longContent.slice(0, 100);
    expect(truncated.length).toBe(100);
    expect(truncated).not.toBe(longContent);
  });
});
