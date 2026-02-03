# Local Embeddings with Ollama

Run Tribal Memory with **zero cloud dependencies**. No API keys, no external services, complete privacy.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

## Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com/download
```

## Pull an Embedding Model

```bash
# Recommended: nomic-embed-text (768 dimensions, fast, good quality)
ollama pull nomic-embed-text

# Alternative: all-minilm (384 dimensions, very fast, smaller)
ollama pull all-minilm

# Alternative: mxbai-embed-large (1024 dimensions, higher quality)
ollama pull mxbai-embed-large
```

## Set Up Tribal Memory

```bash
pip install tribalmemory

# Initialize with local mode
tribalmemory init --local

# Start the server
tribalmemory serve
```

That's it. Server is running at `http://localhost:18790` with:
- **Embeddings**: Ollama (`nomic-embed-text`, 768 dimensions)
- **Storage**: LanceDB at `~/.tribal-memory/lancedb`
- **Privacy**: Zero data leaves your machine

## With Claude Code (MCP)

```bash
# Auto-configure everything
tribalmemory init --local --claude-code
```

This adds Tribal Memory to Claude Code's MCP config. Next time you start Claude Code, it'll have persistent memory.

## Manual Config

If you prefer to customize, edit `~/.tribal-memory/config.yaml`:

```yaml
instance_id: my-agent

embedding:
  provider: openai              # OpenAI-compatible API
  model: nomic-embed-text       # Ollama model name
  api_base: http://localhost:11434/v1
  dimensions: 768
  # api_key not needed for local Ollama

db:
  provider: lancedb
  path: ~/.tribal-memory/lancedb

server:
  host: 127.0.0.1
  port: 18790
```

## Embedding Model Comparison

| Model | Dimensions | Speed | Quality | Size |
|-------|-----------|-------|---------|------|
| `nomic-embed-text` | 768 | Fast | Good | 274MB |
| `all-minilm` | 384 | Very fast | OK | 45MB |
| `mxbai-embed-large` | 1024 | Medium | Better | 670MB |
| `snowflake-arctic-embed` | 1024 | Medium | Great | 670MB |

**Recommendation**: Start with `nomic-embed-text`. Good balance of quality and speed.

## Troubleshooting

### "Connection refused" on port 11434

Ollama isn't running. Start it:

```bash
ollama serve
```

### "Model not found"

Pull the model first:

```bash
ollama pull nomic-embed-text
```

### Switching embedding models

⚠️ **If you change models, existing embeddings won't match the new dimensions.** You have two options:

1. **Start fresh**: Delete `~/.tribal-memory/lancedb` and re-store memories
2. **Export and re-import**: 
   ```bash
   # Export with the old model running
   curl http://localhost:18790/export > backup.json
   
   # Delete old data, update config, restart
   rm -rf ~/.tribal-memory/lancedb
   # Edit config.yaml with new model/dimensions
   tribalmemory serve
   
   # Re-import (will re-embed with new model)
   curl -X POST http://localhost:18790/import \
     -H "Content-Type: application/json" \
     -d @backup.json
   ```

## Docker (Local Mode)

```bash
# Start Tribal Memory + Ollama together
docker compose --profile local up

# Pull the embedding model inside the Ollama container
docker compose exec ollama ollama pull nomic-embed-text
```
