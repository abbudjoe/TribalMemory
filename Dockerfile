# Tribal Memory — Multi-stage Docker build
# Supports both OpenAI and local (Ollama) embeddings

FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python package
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir .

# Create default config directory
RUN mkdir -p /data/lancedb

# Default config
ENV TRIBAL_MEMORY_CONFIG=/app/config.yaml
ENV TRIBAL_MEMORY_INSTANCE_ID=tribal-memory-docker

# Expose server port
EXPOSE 18790

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:18790/').raise_for_status()" || exit 1

# Default: run the HTTP server
CMD ["tribalmemory", "serve", "--host", "0.0.0.0"]
