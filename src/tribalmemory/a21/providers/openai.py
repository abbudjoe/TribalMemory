"""OpenAI Embedding Provider."""

import asyncio
import math
from datetime import datetime
from typing import Optional
import httpx

from .base import EmbeddingProvider, ProviderHealth, ProviderStatus
from ..config.providers import EmbeddingConfig


class OpenAIEmbeddingProvider(EmbeddingProvider[EmbeddingConfig]):
    """OpenAI embedding provider implementation."""
    
    API_URL = "https://api.openai.com/v1/embeddings"
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def dimensions(self) -> int:
        return self.config.dimensions
    
    @property
    def model_name(self) -> str:
        return self.config.model
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client.
        
        Creates an async HTTP client for API requests.
        Ensures cleanup if initialization fails partway through.
        
        Raises:
            ValueError: If API key is not configured
        """
        if not self.config.api_key:
            raise ValueError("OpenAI API key required")
        
        client = None
        try:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                }
            )
            self._client = client
            self._initialized = True
        except Exception:
            # Ensure cleanup on partial initialization failure
            if client:
                await client.aclose()
            raise
    
    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False
    
    async def health_check(self) -> ProviderHealth:
        if not self._client:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                message="Client not initialized"
            )
        
        try:
            start = datetime.utcnow()
            await self.embed("health check")
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return ProviderHealth(
                status=ProviderStatus.HEALTHY,
                latency_ms=latency
            )
        except Exception as e:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                message=str(e)
            )
    
    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        if not self._client:
            raise RuntimeError("Provider not initialized")
        
        # Clean texts
        cleaned = [self._clean_text(t) for t in texts]
        
        payload = {
            "model": self.config.model,
            "input": cleaned,
            "dimensions": self.config.dimensions,
        }
        
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.post(self.API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    embeddings = sorted(data["data"], key=lambda x: x["index"])
                    return [self._normalize_embedding(e["embedding"]) for e in embeddings]
                
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    await asyncio.sleep(retry_after)
                    continue
                
                elif response.status_code >= 500:
                    backoff = min(self.config.backoff_base ** attempt, self.config.backoff_max)
                    await asyncio.sleep(backoff)
                    continue
                
                else:
                    error = response.json().get("error", {}).get("message", response.text)
                    raise RuntimeError(f"OpenAI API error ({response.status_code}): {error}")
                    
            except httpx.TimeoutException as e:
                last_error = e
                backoff = min(self.config.backoff_base ** attempt, self.config.backoff_max)
                await asyncio.sleep(backoff)
            except httpx.RequestError as e:
                last_error = e
                backoff = min(self.config.backoff_base ** attempt, self.config.backoff_max)
                await asyncio.sleep(backoff)
        
        raise RuntimeError(f"OpenAI API failed after {self.config.max_retries} retries: {last_error}")
    
    def _clean_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        max_bytes = 8191 * 4
        encoded = cleaned.encode('utf-8')
        if len(encoded) > max_bytes:
            cleaned = encoded[:max_bytes].decode('utf-8', errors='ignore')
        return cleaned

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        """Normalize embedding to unit length for consistent similarity math."""
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]
