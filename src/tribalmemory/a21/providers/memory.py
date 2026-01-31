"""In-memory storage provider."""

import time
from datetime import datetime
from typing import Optional, Any, Callable

from .base import StorageProvider, EmbeddingProvider, ProviderHealth, ProviderStatus
from ..config.providers import StorageConfig
from ...interfaces import MemoryEntry, RecallResult, StoreResult


class InMemoryStorageProvider(StorageProvider[StorageConfig]):
    """In-memory storage for testing and development."""
    
    def __init__(
        self,
        config: StorageConfig,
        embedding_provider: EmbeddingProvider,
    ):
        super().__init__(config)
        self._embedding = embedding_provider
        self._store: dict[str, MemoryEntry] = {}
        self._deleted: set[str] = set()
    
    async def initialize(self) -> None:
        self._initialized = True
    
    async def shutdown(self) -> None:
        self._store.clear()
        self._deleted.clear()
        self._initialized = False
    
    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            status=ProviderStatus.HEALTHY,
            latency_ms=0.1,
            message=f"In-memory store with {len(self._store)} entries"
        )
    
    async def store(self, entry: MemoryEntry) -> StoreResult:
        if entry.embedding is None:
            entry.embedding = await self._embedding.embed(entry.content)
        
        # Validate embedding dimensions
        if len(entry.embedding) != self.config.embedding_dimensions:
            return StoreResult(
                success=False,
                error=f"Invalid embedding dimension: expected {self.config.embedding_dimensions}, got {len(entry.embedding)}"
            )
        
        self._store[entry.id] = entry
        return StoreResult(success=True, memory_id=entry.id)
    
    async def recall(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RecallResult]:
        start = time.perf_counter()
        
        results = []
        for entry in self._store.values():
            if entry.id in self._deleted:
                continue
            if entry.embedding is None:
                continue
            
            # Apply filters
            if filters and not self._matches_filters(entry, filters):
                continue
            
            sim = self._embedding.similarity(query_embedding, entry.embedding)
            if sim >= min_similarity:
                results.append((entry, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return [
            RecallResult(memory=e, similarity_score=s, retrieval_time_ms=elapsed_ms)
            for e, s in results[:limit]
        ]
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        if memory_id in self._deleted:
            return None
        return self._store.get(memory_id)
    
    async def delete(self, memory_id: str) -> bool:
        if memory_id in self._store:
            self._deleted.add(memory_id)
            return True
        return False
    
    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[MemoryEntry]:
        entries = [
            e for e in self._store.values()
            if e.id not in self._deleted
        ]
        
        if filters:
            entries = [e for e in entries if self._matches_filters(e, filters)]
        
        return entries[offset:offset + limit]
    
    async def count(self, filters: Optional[dict[str, Any]] = None) -> int:
        entries = await self.list(limit=100000, filters=filters)
        return len(entries)
    
    def _matches_filters(self, entry: MemoryEntry, filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key == "tags":
                if not any(t in entry.tags for t in value):
                    return False
            elif key == "source_instance":
                if entry.source_instance != value:
                    return False
            elif key == "source_type":
                if entry.source_type.value != value:
                    return False
        return True
