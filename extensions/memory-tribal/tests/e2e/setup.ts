/**
 * E2E test infrastructure for memory-tribal plugin.
 *
 * Spawns a real TribalMemory server on a random port with a fresh
 * temporary database. Provides a TribalClient pointed at the test server.
 *
 * Usage:
 *   const env = await startTestServer();
 *   // ... run tests against env.client / env.baseUrl ...
 *   await env.cleanup();
 */

import { TribalClient } from "../../src/tribal-client";
import { spawn, type ChildProcess } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

/** Valid SourceType values accepted by the TribalMemory server. */
export const VALID_SOURCE_TYPES = [
  "user_explicit",
  "auto_capture",
  "correction",
  "cross_instance",
  "legacy",
  "unknown",
] as const;
export type SourceType = (typeof VALID_SOURCE_TYPES)[number];

export interface TestEnvironment {
  /** TribalClient connected to the test server */
  client: TribalClient;
  /** Base URL of the test server (e.g., http://127.0.0.1:18799) */
  baseUrl: string;
  /** Port the server is listening on */
  port: number;
  /** Path to the temporary database directory */
  dbPath: string;
  /** Clean up: kill server + remove temp files */
  cleanup: () => Promise<void>;
}

/**
 * Find a free port by binding to port 0 and reading the assigned port.
 * 
 * ⚠️ Known limitation: Race condition between srv.close() and server startup.
 * Another process could claim the port in the window after we close the test
 * server but before tribalmemory binds to it. This is acceptable for E2E tests
 * running in isolation, but parallel test runs may experience flakes.
 */
async function findFreePort(): Promise<number> {
  const { createServer } = await import("net");
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address();
      if (addr && typeof addr === "object") {
        const port = addr.port;
        srv.close(() => resolve(port));
      } else {
        srv.close(() => reject(new Error("Could not get port")));
      }
    });
    srv.on("error", reject);
  });
}

/**
 * Wait for the server health endpoint to respond.
 */
async function waitForHealth(
  baseUrl: string,
  timeoutMs: number = parseInt(process.env.E2E_HEALTH_TIMEOUT_MS || "30000", 10),
): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`${baseUrl}/v1/health`);
      if (res.ok) {
        // Validate health response structure
        const data = (await res.json()) as { status?: string };
        if (data.status === "ok") {
          return;
        }
      }
    } catch {
      // Server not ready yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(
    `Server at ${baseUrl} did not become healthy within ${timeoutMs}ms`,
  );
}

/**
 * Start a real TribalMemory server for E2E testing.
 *
 * Creates a temporary directory for the database, writes a config file,
 * spawns `tribalmemory serve`, waits for health, and returns a
 * TestEnvironment with a connected TribalClient.
 */
export async function startTestServer(): Promise<TestEnvironment> {
  const port = await findFreePort();
  const dbPath = mkdtempSync(join(tmpdir(), "tribal-e2e-"));
  const configPath = join(dbPath, "config.yaml");

  writeFileSync(
    configPath,
    `db:
  path: ${dbPath}/data
server:
  host: 127.0.0.1
  port: ${port}
instance_id: e2e-test
search:
  lazy_spacy: true
`,
  );

  const baseUrl = `http://127.0.0.1:${port}`;

  // Spawn tribalmemory serve
  const serverProcess: ChildProcess = spawn(
    "tribalmemory",
    ["serve", "--config", configPath],
    {
      env: {
        ...process.env,
        // Override PYTHONPATH to ensure the test uses the local source tree
        // rather than any system-installed version of TribalMemory
        PYTHONPATH: join(__dirname, "../../../../src"),
      },
      stdio: "pipe",
    },
  );

  // Collect stderr for debugging
  const debugMode = process.env.DEBUG === "1" || process.env.DEBUG === "true";
  let stderr = "";
  serverProcess.stderr?.on("data", (chunk: Buffer) => {
    const text = chunk.toString();
    stderr += text;
    if (debugMode) {
      console.error("[tribalmemory stderr]", text);
    }
  });

  try {
    await waitForHealth(baseUrl);
  } catch (err) {
    serverProcess.kill("SIGKILL");
    throw new Error(
      `Failed to start test server on port ${port}.\nStderr: ${stderr}\n${err}`,
    );
  }

  const client = new TribalClient(baseUrl);

  const cleanup = async () => {
    serverProcess.kill("SIGTERM");
    // Wait for graceful shutdown
    await new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        serverProcess.kill("SIGKILL");
        resolve();
      }, 5000);
      serverProcess.on("exit", () => {
        clearTimeout(timeout);
        resolve();
      });
    });
    // Remove temp directory
    try {
      rmSync(dbPath, { recursive: true, force: true });
    } catch {
      // Best effort cleanup
    }
  };

  return { client, baseUrl, port, dbPath, cleanup };
}

/**
 * Make a raw HTTP request to the server (bypasses TribalClient).
 * Useful for testing invalid payloads that the client would reject.
 * 
 * @param T - Expected response body type (defaults to Record<string, unknown>)
 */
export async function rawPost<T = Record<string, unknown>>(
  baseUrl: string,
  path: string,
  body: Record<string, unknown>,
): Promise<{ status: number; body: T }> {
  const res = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = (await res.json()) as T;
  return { status: res.status, body: data };
}
