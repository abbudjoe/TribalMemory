# Local Embeddings with FastEmbed

Run Tribal Memory with **zero cloud dependencies** and **no Ollama
required**. FastEmbed uses ONNX runtime for CPU-optimized embeddings
— pure Python, fast, lightweight.

## Install

```bash
pip install "tribalmemory[fastembed]"
```

That's it. No GPU, no external services, no API keys.

## Configure

Edit `~/.tribal-memory/config.yaml`:

```yaml
instance_id: my-agent

embedding:
  provider: fastembed
  model: BAAI/bge-small-en-v1.5   # 384 dims, ~130MB
  dimensions: 384

db:
  provider: lancedb
  path: ~/.tribal-memory/lancedb
```

## Start

```bash
tribalmemory serve
```

First run downloads the model (~130MB). Subsequent starts are instant.

## With Claude Code (MCP)

```bash
tribalmemory init --local --claude-code
```

Then edit `~/.tribal-memory/config.yaml` to set
`embedding.provider: fastembed`.

## Model Comparison

| Model | Dims | Size | Quality | Speed |
|-------|------|------|---------|-------|
| `BAAI/bge-small-en-v1.5` | 384 | ~130MB | Good | ⚡ Fast |
| `BAAI/bge-base-en-v1.5` | 768 | ~440MB | Better | Fast |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | ~560MB | Great | Fast |
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.3GB | Best | Medium |

**Recommendation**: Start with `bge-small-en-v1.5`. Upgrade to
`bge-base` or `nomic-embed` if you need higher recall quality.

## FastEmbed vs Ollama vs OpenAI

| Feature | FastEmbed | Ollama | OpenAI |
|---------|-----------|--------|--------|
| API key needed | ❌ | ❌ | ✅ |
| External service | ❌ | ✅ (ollama serve) | ✅ (cloud) |
| Install | `pip install` | separate binary | `pip install` |
| GPU required | ❌ | ❌ (optional) | N/A |
| Startup time | ~1s | ~5s | instant |
| Embedding speed | ~10ms/text | ~50ms/text | ~200ms/text |
| Privacy | 100% local | 100% local | Cloud |

## Switching Providers

⚠️ **Different providers produce different embedding vectors.**
Embeddings from different providers are incompatible even if they
have the same dimensions (e.g., `bge-base-en-v1.5` at 768 dims
vs Ollama's `nomic-embed-text` at 768 dims). Always re-embed when
switching providers. Options:

1. **Start fresh**: Delete `~/.tribal-memory/lancedb`
2. **Export → switch → re-import**:
   ```bash
   curl http://localhost:18790/export > backup.json
   rm -rf ~/.tribal-memory/lancedb
   # Update config.yaml to use fastembed
   tribalmemory serve
   curl -X POST http://localhost:18790/import \
     -H "Content-Type: application/json" -d @backup.json
   ```
