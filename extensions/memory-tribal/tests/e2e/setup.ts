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
  timeoutMs: number = 30000,
): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`${baseUrl}/v1/health`);
      if (res.ok) return;
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
        PYTHONPATH: join(__dirname, "../../../../src"),
      },
      stdio: "pipe",
    },
  );

  // Collect stderr for debugging
  let stderr = "";
  serverProcess.stderr?.on("data", (chunk: Buffer) => {
    stderr += chunk.toString();
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
 */
export async function rawPost(
  baseUrl: string,
  path: string,
  body: Record<string, unknown>,
): Promise<{ status: number; body: Record<string, unknown> }> {
  const res = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = (await res.json()) as Record<string, unknown>;
  return { status: res.status, body: data };
}
