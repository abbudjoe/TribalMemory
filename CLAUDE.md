# CLAUDE.md - TribalMemory Project

## Project Overview
Shared memory infrastructure for multi-instance AI agents. Enables accumulated learning across instances and sessions.

## Tech Stack
- Python 3.10+
- pytest + pytest-asyncio for testing
- FastEmbed embeddings (BAAI/bge-small-en-v1.5, 384 dims)
- MCP server for Claude Code integration
- TypeScript extensions for learned retrieval

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_mcp_server.py -v

# Run MCP server
python -m tribalmemory.mcp
```

## TDD Requirements (MANDATORY)

1. **RED** — Write failing test first
2. **GREEN** — Minimal code to pass
3. **REFACTOR** — Clean up while green

Every feature needs tests. Every bug fix needs a failing regression test first.

## Code Style

- Line length: 100 chars
- Use type hints
- Async functions for all I/O
- Dataclasses for data structures
- Abstract base classes for interfaces

## PR Review Loop (MANDATORY)

**All changes must follow this flow. No exceptions.**

1. Create feature branch from main
2. Make changes, commit
3. Push branch, open PR
4. **Post PR comment: `@claude review this PR`** ← REQUIRED, DO NOT SKIP  
   ⚠️ **Note:** This is a comment on the PR (not the commit message)
5. **Wait 5 minutes, then check PR for review comments**
6. **Address ALL comments from the review** (every single item)
7. Commit fixes, **push**
8. **Post PR comment: `@claude review this PR`** ← REQUIRED AFTER EVERY PUSH
9. **Wait 5 minutes, check for new review comments**
10. Issues remaining? → Go to step 6 (address all comments again)
11. When clean: Comment `@abbudjoe ready for merge`
12. Joe reviews and merges

**CRITICAL:**
- ❌ No direct commits to main (except hotfixes approved by Joe)
- ❌ No merges without Claude Code review
- ❌ No skipping the `@claude review this PR` comment (required after EVERY push)
- ❌ Do NOT put `@claude review this PR` in commit messages
- ❌ Do NOT rely on automatic GitHub Action triggers
- ✅ The comment must be **standalone** on the PR (not combined with other text)
- ✅ Comment after **every push** to trigger re-review
- ✅ **Check back after 5 minutes** to view and address all review comments
- ✅ **Address EVERY comment** — partial fixes are not acceptable

## Project Structure

```
TribalMemory/
├── src/tribalmemory/
│   ├── interfaces.py      # Core interfaces
│   ├── utils.py           # Shared utilities
│   ├── services/          # Service implementations
│   │   ├── fastembed_service.py  # FastEmbed embedding service
│   │   ├── memory.py      # Memory service
│   │   ├── graph_store.py # Entity/relationship graph
│   │   └── vector_store.py # LanceDB vector store
│   ├── mcp/               # MCP server
│   │   └── server.py      # FastMCP server implementation
│   └── server/            # HTTP server configuration
├── extensions/
│   └── memory-tribal/     # TypeScript learned retrieval layer
├── tests/
│   └── test_*.py          # Test files
├── eval/
│   └── memory-test/       # Evaluation harness
└── docs/
    └── portable-memory.md # Portable memory specification
```

## Checklist Before PR

1. ✅ All tests pass (`pytest`)
2. ✅ No type errors
3. ✅ Code formatted
4. ✅ New tests for new features
5. ✅ Documentation updated if needed
6. ✅ `@claude review this PR` comment posted
