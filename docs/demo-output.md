# Demo Output: "One Brain, Two Agents"

```
🧠 Tribal Memory — One Brain, Two Agents
Your AI tools don't share a brain. Until now.

────────────────────────────────────────────

▶ Agent A (Claude Code)  instance: claude-code

  Storing: "The auth service uses JWT with RS256 signing"
  ✅ Stored (id: 6e1bc5c6...)

  Storing: "Database is Postgres 16 with pgvector for embeddings"
  ✅ Stored (id: 428c6f3d...)

  Storing: "Frontend uses Next.js 15 with App Router"
  ✅ Stored (id: 73a9aadd...)

────────────────────────────────────────────

▶ Agent B (Codex CLI)  instance: codex — different agent, same brain

  Asking: "How does authentication work?"

  █████████░░░░░░░░░░░ 48%  The auth service uses JWT with RS256 signing
    └─ from: claude-code  tags: ['architecture', 'auth']

  Asking: "What database and frontend stack are we using?"

  ███████░░░░░░░░░░░░░ 37%  Frontend uses Next.js 15 with App Router
    └─ from: claude-code  tags: ['architecture', 'frontend']

  ████████░░░░░░░░░░░░ 35%  Database is Postgres 16 with pgvector for embeddings
    └─ from: claude-code  tags: ['architecture', 'database']

────────────────────────────────────────────

⚡ Claude Code stored it. Codex recalled it.
   Same server. Shared memory. Zero config.

   pip install tribalmemory
   tribalmemory init --local
   tribalmemory serve

   https://github.com/abbudjoe/TribalMemory
```
