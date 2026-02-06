# Contributing to TribalMemory

Thank you for your interest in contributing to TribalMemory! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [Community](#community)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers.

---

## Getting Started

TribalMemory is a shared memory infrastructure for multi-instance AI agents. Before contributing, we recommend:

1. **Read the [README](README.md)** — Understand what the project does
2. **Try it out** — Install and experiment with the basic features
3. **Review open issues** — See what needs work or where you can help
4. **Check discussions** — Join conversations about the project's direction

### Good First Issues

Look for issues labeled [`good first issue`](https://github.com/abbudjoe/TribalMemory/labels/good%20first%20issue) — these are specifically curated for new contributors.

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/TribalMemory.git
   cd TribalMemory
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install in development mode:**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify the installation:**

   ```bash
   pytest tests/test_tier1_functional.py -v
   ```

### Optional: Benchmark Dependencies

For running standard benchmarks (RAGAS, BEIR, BABILong):

```bash
pip install -e ".[benchmarks]"
```

---

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/short-description` — New features
- `fix/short-description` — Bug fixes
- `docs/short-description` — Documentation updates
- `refactor/short-description` — Code refactoring
- `test/short-description` — Test additions or fixes

### Commit Messages

Write clear, concise commit messages:

```
<type>: <short summary>

[optional body with more details]

[optional footer with issue references]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples:**

```
feat: add memory expiration support

fix: handle empty query in recall

docs: update API reference for v0.2
```

### TDD Requirement (Mandatory)

We follow strict Test-Driven Development:

1. **RED** — Write a failing test first
2. **GREEN** — Write minimal code to make it pass
3. **REFACTOR** — Clean up while keeping tests green

Every new feature needs tests. Every bug fix needs a failing regression test first.

---

## Testing

### Test Tiers

| Tier | Purpose | Command | Requirement |
|------|---------|---------|-------------|
| **Tier 1** | Functional tests | `pytest -m tier1` | Must pass (100%) |
| **Tier 2** | Capability tests | `pytest -m tier2` | Target >30% improvement |
| **Tier 3** | Emergence tests | `pytest -m tier3` | Stretch goals |
| **Security** | Security tests | `pytest -m security` | Must pass |
| **Benchmark** | Standard benchmarks | `pytest -m benchmark` | Reference metrics |

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific tier
pytest -m tier1 -v

# Run a specific test file
pytest tests/test_tier1_functional.py -v

# Run integration tests (requires API keys)
pytest -m integration
```

### Test Requirements

- All Tier 1 tests must pass before submitting a PR
- New features must include tests
- Bug fixes must include a regression test
- Tests should be deterministic and not flaky

---

## Pull Request Process

### Before Submitting

1. ✅ All tests pass (`pytest`)
2. ✅ Code is formatted (`black .`)
3. ✅ Linting passes (`ruff check .`)
4. ✅ No type errors
5. ✅ Documentation updated if needed
6. ✅ Commit messages are clear
7. ✅ `@claude review this PR` comment posted (see PR Workflow below)

### PR Workflow

**All changes must follow this flow. No exceptions.**

1. **Create your feature branch:**

   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make your changes and commit:**

   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

3. **Push to your fork:**

   ```bash
   git push origin feature/your-feature
   ```

4. **Open a Pull Request** on GitHub

5. **Post the mandatory review trigger:**
   
   Comment on your PR: **`@claude review this PR`**
   
   ⚠️ **CRITICAL**: This comment must be standalone — do not combine with other text.

6. **Check Claude Code's review comments**

7. **Fix all identified issues**

8. **Commit fixes and push**

9. **Issues remaining?** → Go back to step 5 (re-trigger Claude review)

10. **When clean**: Comment `@abbudjoe ready for merge`

11. **Joe reviews and merges**

**Important Notes:**
- ❌ No merges without Claude Code review
- ❌ No skipping the `@claude review this PR` comment
- ❌ Do NOT rely on automatic GitHub Action triggers
- ✅ The `@claude` comment must be standalone

### PR Review Guidelines

- Be patient — Reviews may take a few days
- Be responsive — Address feedback promptly
- Be open — Accept constructive criticism graciously
- Keep PRs focused — One feature/fix per PR

---

## Code Style

### Python

- **Line length:** 100 characters
- **Formatter:** [Black](https://github.com/psf/black)
- **Linter:** [Ruff](https://github.com/astral-sh/ruff)
- **Type hints:** Required on all public APIs

```bash
# Format code
black .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Style Guidelines

- Use type hints everywhere
- Use async functions for all I/O operations
- Use dataclasses for data structures
- Use abstract base classes for interfaces
- Write docstrings for all public functions

**Example:**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class MemoryResult:
    """Result from a memory recall operation.
    
    Attributes:
        content: The memory content.
        similarity: Cosine similarity score (0-1).
        source_id: Optional ID of the source memory.
    """
    content: str
    similarity: float
    source_id: Optional[str] = None


async def recall(query: str, top_k: int = 5) -> list[MemoryResult]:
    """Retrieve memories semantically similar to the query.
    
    Args:
        query: The search query text.
        top_k: Maximum number of results to return.
    
    Returns:
        List of MemoryResult objects, sorted by similarity (descending).
    
    Raises:
        ValueError: If query is empty.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    # Implementation...
```

### TypeScript (Extensions)

For the OpenClaw extension (`extensions/memory-tribal/`):

- Use TypeScript strict mode
- Follow ESLint rules
- Use async/await over raw promises

---

## Reporting Bugs

### Before Reporting

1. **Search existing issues** — It may already be reported
2. **Try the latest version** — It may already be fixed
3. **Gather information** — Collect error messages, logs, steps to reproduce

### Bug Report Template

When opening an issue, include:

- **Summary:** Brief description of the bug
- **Environment:** OS, Python version, TribalMemory version
- **Steps to reproduce:** Detailed steps to trigger the bug
- **Expected behavior:** What should happen
- **Actual behavior:** What actually happens
- **Logs/errors:** Any relevant error messages
- **Additional context:** Screenshots, related issues, etc.

---

## Requesting Features

### Before Requesting

1. **Search existing issues** — It may already be requested
2. **Check the roadmap** — It may be planned
3. **Consider scope** — Does it fit the project's goals?

### Feature Request Template

When proposing a feature:

- **Summary:** Brief description of the feature
- **Motivation:** Why is this needed? What problem does it solve?
- **Proposed solution:** How might it work?
- **Alternatives considered:** Other approaches you've thought about
- **Additional context:** Examples, mockups, related projects

---

## Community

### Getting Help

- **GitHub Discussions:** Ask questions, share ideas
- **Issues:** Report bugs, request features

### Recognition

Contributors are recognized in:

- The project's CHANGELOG
- GitHub's contributor graph
- Release notes for significant contributions

### Maintainers

Current maintainers:

- [@abbudjoe](https://github.com/abbudjoe) — Project lead

---

## License

By contributing to TribalMemory, you agree that your contributions will be licensed under the [Business Source License 1.1](LICENSE).

---

Thank you for contributing to TribalMemory! 🧠✨
