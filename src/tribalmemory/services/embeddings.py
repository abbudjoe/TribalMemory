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
    """OpenAI embedding service using text-embedding-3-small.
    
    Features:
    - Async HTTP calls with retry logic
    - Batch embedding support
    - Rate limiting awareness
    - Configurable model and dimensions
    
    Usage:
        service = OpenAIEmbeddingService(api_key="sk-...")
        embedding = await service.embed("Hello world")
    """
    
    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSIONS = 1536
    API_URL = "https://api.openai.com/v1/embeddings"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
    ):
        """Initialize OpenAI embedding service.
        
        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Embedding model to use.
            dimensions: Output embedding dimensions.
            max_retries: Max retry attempts on transient failures.
            timeout_seconds: Request timeout.
            backoff_base: Base for exponential backoff (default 2.0).
            backoff_max: Maximum backoff delay in seconds (default 60.0).
        
        Security Note:
            API keys are stored in memory and used in HTTP headers. Never log
            the _client object or include it in error reports. For production,
            consider using a secrets manager rather than environment variables.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key or set OPENAI_API_KEY env var."
            )
        
        self.model = model
        self.dimensions = dimensions
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
        self._client: Optional[httpx.AsyncClient] = None
    
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
                response = await client.post(self.API_URL, json=payload)
                
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
                    error_detail = response.json().get("error", {}).get("message", response.text)
                    raise RuntimeError(f"OpenAI API error ({response.status_code}): {error_detail}")
                    
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
