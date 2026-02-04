# Demo Output: "One Brain, Two Agents"

This is captured output from running `demo.sh`, which demonstrates cross-agent memory sharing.

**Setup:**
- Server running on `http://localhost:18790`
- Agent A (Claude Code) stores architecture memories
- Agent B (Codex CLI) retrieves them via semantic search

**To run the demo yourself:**
```bash
tribalmemory serve
./demo.sh
```

> **Note:** Similarity scores depend on your embedding model. Local embeddings (Ollama) will produce
> different scores than OpenAI. The scores below are from a local Ollama setup with `nomic-embed-text`.

## Output

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
