# CLAUDE.md - TribalMemory Project

## Project Overview
Shared memory infrastructure for multi-instance AI agents. Enables accumulated learning across instances and sessions.

## Tech Stack
- Python 3.10+
- pytest + pytest-asyncio for testing
- OpenAI embeddings (text-embedding-3-small)
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
4. **Comment: `@claude review this PR`** ← REQUIRED, DO NOT SKIP
5. Check Claude Code's review comments
6. Fix all identified issues
7. Commit fixes, push
8. Issues remaining? → Go to step 4
9. When clean: Comment `@abbudjoe ready for merge`
10. Joe reviews and merges

**CRITICAL:**
- ❌ No direct commits to main (except hotfixes approved by Joe)
- ❌ No merges without Claude Code review
- ❌ No skipping step 4 (the explicit comment triggers review)
- ❌ Do NOT rely on automatic GitHub Action triggers
- ✅ Step 4 comment must be **standalone** (not combined with other text)

## Project Structure

```
TribalMemory/
├── src/tribalmemory/
│   ├── interfaces.py      # Core interfaces
│   ├── utils.py           # Shared utilities
│   ├── services/          # Service implementations
│   │   ├── embeddings.py  # OpenAI embedding service
│   │   └── memory.py      # Memory service
│   ├── mcp/               # MCP server
│   │   └── server.py      # FastMCP server implementation
│   ├── a21/               # A21 system integration
│   │   └── providers/     # Embedding providers
│   └── server/            # Server configuration
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
