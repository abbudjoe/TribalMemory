"""OpenAI Embedding Service.

Production embedding service using OpenAI's text-embedding-3-small model.
"""

import asyncio
import math
import os
from typing import Optional

import httpx

from ..interfaces import IEmbeddingService
from ..utils import normalize_embedding


class OpenAIEmbeddingService(IEmbeddingService):
    """OpenAI-compatible embedding service.
    
    Supports OpenAI, Ollama, and any OpenAI-compatible embedding API.
    
    Features:
    - Async HTTP calls with retry logic
    - Batch embedding support
    - Rate limiting awareness
    - Configurable model, dimensions, and API base URL
    - Local-only mode (Ollama) — no API key needed
    
    Usage:
        # OpenAI
        service = OpenAIEmbeddingService(api_key="sk-...")
        
        # Ollama (local)
        service = OpenAIEmbeddingService(
            api_base="http://localhost:11434/v1",
            model="nomic-embed-text",
            dimensions=768,
        )
    """
    
    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSIONS = 1536
    DEFAULT_API_BASE = "https://api.openai.com/v1"
    LOCAL_API_KEY_PLACEHOLDER = "local-no-key-needed"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
        api_base: Optional[str] = None,
    ):
        """Initialize OpenAI-compatible embedding service.
        
        Supports OpenAI, Ollama, and any OpenAI-compatible embedding API.
        
        Args:
            api_key: API key. Falls back to OPENAI_API_KEY env var.
                     Not required when api_base points to a local service (e.g., Ollama).
            model: Embedding model to use.
            dimensions: Output embedding dimensions.
            max_retries: Max retry attempts on transient failures.
            timeout_seconds: Request timeout.
            backoff_base: Base for exponential backoff (default 2.0).
            backoff_max: Maximum backoff delay in seconds (default 60.0).
            api_base: Base URL for the embedding API. Defaults to OpenAI.
                      For Ollama: "http://localhost:11434/v1"
                      For any OpenAI-compatible API: "http://host:port/v1"
        
        Security Note:
            API keys are stored in memory and used in HTTP headers. Never log
            the _client object or include it in error reports. For production,
            consider using a secrets manager rather than environment variables.
        """
        # Validate dimensions
        if dimensions < 1 or dimensions > 8192:
            raise ValueError(
                f"Dimensions must be between 1 and 8192, got {dimensions}"
            )
        
        # Build the API URL from api_base
        self.api_url = self._resolve_api_url(api_base)
        
        # Determine if this is a local (non-OpenAI) service
        is_local = (
            api_base is not None
            and api_base.strip() != ""
            and "api.openai.com" not in api_base.lower()
        )
        
        # For local services (non-OpenAI), api_key is optional
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        if not self.api_key:
            if not is_local:
                raise ValueError(
                    "OpenAI API key required. Pass api_key "
                    "or set OPENAI_API_KEY env var."
                )
            # Use a placeholder for local services (e.g., Ollama)
            self.api_key = self.LOCAL_API_KEY_PLACEHOLDER
        
        self.model = model
        self.dimensions = dimensions
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
        self._client: Optional[httpx.AsyncClient] = None
    
    @staticmethod
    def _resolve_api_url(api_base: Optional[str] = None) -> str:
        """Resolve the full embeddings API URL from an optional base.
        
        Args:
            api_base: Base URL (e.g., "http://localhost:11434/v1").
                      If None, uses OpenAI default.
        
        Returns:
            Full URL ending in /embeddings.
        
        Raises:
            ValueError: If api_base is not a valid HTTP(S) URL.
        """
        if api_base is None or api_base.strip() == "":
            return (
                f"{OpenAIEmbeddingService.DEFAULT_API_BASE}"
                "/embeddings"
            )
        
        base = api_base.strip().rstrip("/")
        
        # Basic URL validation
        if base and not base.startswith(("http://", "https://")):
            raise ValueError(
                f"api_base must be an HTTP(S) URL, got: {base}"
            )
        
        # If already ends with /embeddings, use as-is
        if base.endswith("/embeddings"):
            return base
        
        return f"{base}/embeddings"
    
    @property
    def provider_name(self) -> str:
        """Return the provider identifier for metadata."""
        return "openai"

    def __repr__(self) -> str:
        """Safe repr that masks API key to prevent accidental logging."""
        return f"OpenAIEmbeddingService(model={self.model!r}, api_key=***)"
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._client
    
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        results = await self.embed_batch([text])
        return results[0]
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        cleaned_texts = [self._clean_text(t) for t in texts]
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "input": cleaned_texts,
            "dimensions": self.dimensions,
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await client.post(self.api_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    embeddings = sorted(data["data"], key=lambda x: x["index"])
                    return [normalize_embedding(e["embedding"]) for e in embeddings]
                
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    await asyncio.sleep(retry_after)
                    continue
                
                elif response.status_code >= 500:
                    backoff = min(self.backoff_base ** attempt, self.backoff_max)
                    await asyncio.sleep(backoff)
                    continue
                
                else:
                    try:
                        error_detail = (
                            response.json()
                            .get("error", {})
                            .get("message", response.text)
                        )
                    except Exception:
                        error_detail = response.text[:200]
                    raise RuntimeError(
                        f"Embedding API error "
                        f"({response.status_code}): "
                        f"{error_detail}"
                    )
                    
            except httpx.TimeoutException as e:
                last_error = e
                backoff = min(self.backoff_base ** attempt, self.backoff_max)
                await asyncio.sleep(backoff)
            except httpx.RequestError as e:
                last_error = e
                backoff = min(self.backoff_base ** attempt, self.backoff_max)
                await asyncio.sleep(backoff)
        
        raise RuntimeError(f"OpenAI API failed after {self.max_retries} retries: {last_error}")
    
    def similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if len(a) != len(b):
            raise ValueError(f"Embedding dimensions don't match: {len(a)} vs {len(b)}")
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _clean_text(self, text: str) -> str:
        """Clean text for embedding.
        
        Normalizes whitespace and truncates to fit within token limits.
        Uses encoding-aware truncation to avoid splitting UTF-8 characters.
        """
        cleaned = " ".join(text.split())
        
        # OpenAI has token limits; truncate very long texts
        # text-embedding-3-small supports 8191 tokens
        # Rough estimate: 4 bytes per token (worst case UTF-8)
        max_bytes = 8191 * 4
        
        encoded = cleaned.encode('utf-8')
        if len(encoded) > max_bytes:
            # Truncate bytes and decode safely (ignore partial chars)
            cleaned = encoded[:max_bytes].decode('utf-8', errors='ignore')
        
        return cleaned

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
