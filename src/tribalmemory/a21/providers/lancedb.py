"""LanceDB storage provider."""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from .base import StorageProvider, EmbeddingProvider, ProviderHealth, ProviderStatus
from ..config.providers import StorageConfig
from ...interfaces import MemoryEntry, MemorySource, RecallResult, StoreResult

logger = logging.getLogger(__name__)


class LanceDBStorageProvider(StorageProvider[StorageConfig]):
    """LanceDB-backed storage provider."""
    
    def __init__(
        self,
        config: StorageConfig,
        embedding_provider: EmbeddingProvider,
    ):
        super().__init__(config)
        self._embedding = embedding_provider
        self._db = None
        self._table = None
    
    async def initialize(self) -> None:
        try:
            import lancedb
        except ImportError:
            raise ImportError("LanceDB not installed. Run: pip install lancedb")
        
        if self.config.uri:
            self._db = lancedb.connect(self.config.uri, api_key=self.config.api_key)
        elif self.config.path:
            Path(self.config.path).mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(self.config.path)
        else:
            raise ValueError("LanceDB requires path or uri")
        
        if self.config.table_name in self._db.table_names():
            self._table = self._db.open_table(self.config.table_name)
        else:
            self._table = self._create_table()
        
        self._initialized = True
    
    async def shutdown(self) -> None:
        self._db = None
        self._table = None
        self._initialized = False
    
    async def health_check(self) -> ProviderHealth:
        if not self._table:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                message="Table not initialized"
            )
        
        try:
            start = datetime.utcnow()
            count = await self.count()
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return ProviderHealth(
                status=ProviderStatus.HEALTHY,
                latency_ms=latency,
                message=f"LanceDB with {count} entries"
            )
        except Exception as e:
            return ProviderHealth(
                status=ProviderStatus.DEGRADED,
                message=str(e)
            )
    
    def _create_table(self):
        import pyarrow as pa
        
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.config.embedding_dimensions)),
            pa.field("source_instance", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),
            pa.field("tags", pa.string()),
            pa.field("context", pa.string()),
            pa.field("confidence", pa.float32()),
            pa.field("supersedes", pa.string()),
            pa.field("related_to", pa.string()),
            pa.field("deleted", pa.bool_()),
        ])
        
        return self._db.create_table(self.config.table_name, schema=schema)
    
    async def store(self, entry: MemoryEntry) -> StoreResult:
        if entry.embedding is None:
            entry.embedding = await self._embedding.embed(entry.content)
        
        # Validate dimensions
        if len(entry.embedding) != self.config.embedding_dimensions:
            return StoreResult(
                success=False,
                error=f"Invalid embedding dimension: {len(entry.embedding)}"
            )
        
        row = {
            "id": entry.id,
            "content": entry.content,
            "vector": entry.embedding,
            "source_instance": entry.source_instance,
            "source_type": entry.source_type.value,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "tags": json.dumps(entry.tags),
            "context": entry.context or "",
            "confidence": entry.confidence,
            "supersedes": entry.supersedes or "",
            "related_to": json.dumps(entry.related_to),
            "deleted": False,
        }
        
        try:
            self._table.add([row])
            return StoreResult(success=True, memory_id=entry.id)
        except Exception as e:
            logger.error(f"Failed to store memory {entry.id}: {e}")
            return StoreResult(success=False, error=str(e))
    
    async def recall(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RecallResult]:
        start = time.perf_counter()
        
        query = self._table.search(query_embedding).where("deleted = false")
        
        if filters:
            for key, value in filters.items():
                if key == "source_instance":
                    safe_val = self._sanitize(value)
                    query = query.where(f"source_instance = '{safe_val}'")
        
        results = query.limit(limit * 2).to_list()
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        recall_results = []
        for row in results:
            distance = row.get("_distance", 0)
            similarity = max(0, 1 - (distance * distance / 2))
            
            if similarity < min_similarity:
                continue
            
            entry = self._row_to_entry(row)
            recall_results.append(RecallResult(
                memory=entry,
                similarity_score=similarity,
                retrieval_time_ms=elapsed_ms
            ))
        
        recall_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return recall_results[:limit]
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        safe_id = self._sanitize(memory_id)
        
        results = (
            self._table.search()
            .where(f"id = '{safe_id}' AND deleted = false")
            .limit(1)
            .to_list()
        )
        
        if not results:
            return None
        return self._row_to_entry(results[0])
    
    async def delete(self, memory_id: str) -> bool:
        safe_id = self._sanitize(memory_id)
        
        try:
            self._table.update(
                where=f"id = '{safe_id}'",
                values={"deleted": True, "updated_at": datetime.utcnow().isoformat()}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[MemoryEntry]:
        query = self._table.search().where("deleted = false")
        results = query.limit(limit + offset).to_list()
        return [self._row_to_entry(r) for r in results[offset:offset + limit]]
    
    async def count(self, filters: Optional[dict[str, Any]] = None) -> int:
        results = self._table.search().where("deleted = false").to_list()
        return len(results)
    
    def _sanitize(self, value: str) -> str:
        if not re.match(r'^[a-zA-Z0-9\-_]+$', value):
            raise ValueError(f"Invalid value format: {value[:20]}...")
        return value
    
    def _row_to_entry(self, row: dict) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            embedding=row.get("vector"),
            source_instance=row.get("source_instance", "unknown"),
            source_type=MemorySource(row.get("source_type", "unknown")),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else datetime.utcnow(),
            tags=json.loads(row.get("tags", "[]")),
            context=row.get("context") or None,
            confidence=row.get("confidence", 1.0),
            supersedes=row.get("supersedes") or None,
            related_to=json.loads(row.get("related_to", "[]")),
        )
