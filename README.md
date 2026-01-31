# Tribal Memory

Shared memory infrastructure for multi-instance AI agents. Enables accumulated learning across instances and sessions.

## Features

- **Vector-based memory storage** — Fast semantic search using LanceDB
- **OpenAI embeddings** — High-quality text embeddings for similarity matching
- **Memory correction** — Track and learn from mistakes
- **Multi-instance support** — Share knowledge across agent instances
- **FastAPI server** — HTTP API for easy integration

## Installation

```bash
pip install tribalmemory
```

## Quick Start

```python
from tribalmemory.services import create_memory_service, MemorySource

# Create a memory service
service = create_memory_service(
    instance_id="my-agent",
    db_path="./tribal_memory_db"
)

# Store a memory
result = await service.remember(
    "User prefers TypeScript for web projects",
    source_type=MemorySource.USER_EXPLICIT,
    tags=["preference", "coding"]
)

# Recall relevant memories
results = await service.recall("What language should I use for the web app?")
for r in results:
    print(f"{r.similarity_score:.2f}: {r.memory.content}")

# Correct a memory
await service.correct(
    original_id=result.memory_id,
    corrected_content="User prefers TypeScript for web, Python for scripts"
)
```

## Running the Server

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# Start the server
tribalmemory
# Server runs on http://localhost:18790
```

## Configuration

Create `~/.tribal-memory/config.yaml`:

```yaml
embedding:
  model: text-embedding-3-small
  
storage:
  uri: ~/.tribal-memory/lancedb

server:
  host: 127.0.0.1
  port: 18790
```

## OpenClaw Integration

Tribal Memory includes a plugin for [OpenClaw](https://github.com/openclaw/openclaw):

```bash
# Install the plugin
openclaw plugins install ./extensions/memory-tribal

# Enable in config
openclaw config set plugins.slots.memory=memory-tribal
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
black --check .
```

## License

Apache 2.0 — see [LICENSE](LICENSE)
