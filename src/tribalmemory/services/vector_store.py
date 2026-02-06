"""Vector Store implementations.

Provides both LanceDB (persistent) and in-memory storage options.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from ..interfaces import (
    IVectorStore,
    IEmbeddingService,
    MemoryEntry,
    MemorySource,
    RecallResult,
    StoreResult,
)


class LanceDBVectorStore(IVectorStore):
    """LanceDB-backed vector store for persistent storage.
    
    Supports both local file storage and LanceDB Cloud.
    """
    
    TABLE_NAME = "memories"
    
    def __init__(
        self,
        embedding_service: IEmbeddingService,
        db_path: Optional[Union[str, Path]] = None,
        db_uri: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.embedding_service = embedding_service
        self.db_path = Path(db_path) if db_path else None
        self.db_uri = db_uri
        self.api_key = api_key or os.environ.get("LANCEDB_API_KEY")
        
        self._db = None
        self._table = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazily initialize database connection."""
        if self._initialized:
            return
        
        try:
            import lancedb
        except ImportError:
            raise ImportError("LanceDB not installed. Run: pip install lancedb")
        
        if self.db_uri:
            self._db = lancedb.connect(self.db_uri, api_key=self.api_key)
        elif self.db_path:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.db_path))
        else:
            raise ValueError("Either db_path or db_uri must be provided")
        
        if self.TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(self.TABLE_NAME)
        else:
            self._table = self._create_table()
        
        self._initialized = True
    
    def _create_table(self) -> "lancedb.table.Table":
        """Create the memories table with the defined schema."""
        import pyarrow as pa
        
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self._get_embedding_dim())),
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
        
        return self._db.create_table(self.TABLE_NAME, schema=schema)
    
    def _get_embedding_dim(self) -> int:
        """Get the expected embedding dimension from the embedding service."""
        if hasattr(self.embedding_service, 'dimensions'):
            return self.embedding_service.dimensions
        return 1536  # Default for text-embedding-3-small
    
    async def store(self, entry: MemoryEntry) -> StoreResult:
        await self._ensure_initialized()
        
        try:
            if entry.embedding is None:
                entry.embedding = await self.embedding_service.embed(entry.content)
            
            # Validate embedding dimensions
            expected_dim = self._get_embedding_dim()
            if len(entry.embedding) != expected_dim:
                return StoreResult(
                    success=False,
                    error=f"Invalid embedding dimension: got {len(entry.embedding)}, expected {expected_dim}"
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
            
            self._table.add([row])
            return StoreResult(success=True, memory_id=entry.id)
            
        except Exception as e:
            return StoreResult(success=False, error=str(e))
    
    async def recall(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        filters: Optional[dict] = None,
    ) -> list[RecallResult]:
        await self._ensure_initialized()
        
        start = time.perf_counter()
        
        results = (
            self._table.search(query_embedding)
            .where("deleted = false")
            .limit(limit * 2)
            .to_list()
        )
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        recall_results = []
        for row in results:
            # LanceDB returns L2 distance. Convert to cosine similarity.
            # For normalized vectors (which FastEmbed embeddings are):
            # L2_distance² = 2 * (1 - cosine_similarity)
            # Therefore: cosine_similarity = 1 - (L2_distance² / 2)
            distance = row.get("_distance", 0)
            similarity = max(0, 1 - (distance * distance / 2))
            
            if similarity < min_similarity:
                continue
            
            entry = self._row_to_entry(row)
            
            # Apply filters
            if filters:
                if "tags" in filters and filters["tags"]:
                    if not any(t in entry.tags for t in filters["tags"]):
                        continue
            
            recall_results.append(RecallResult(
                memory=entry,
                similarity_score=similarity,
                retrieval_time_ms=elapsed_ms
            ))
        
        recall_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return recall_results[:limit]
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        await self._ensure_initialized()
        
        # Sanitize memory_id to prevent SQL injection
        safe_id = self._sanitize_id(memory_id)
        
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
        await self._ensure_initialized()
        
        # Sanitize memory_id to prevent SQL injection
        safe_id = self._sanitize_id(memory_id)
        
        try:
            self._table.update(
                where=f"id = '{safe_id}'",
                values={"deleted": True, "updated_at": datetime.utcnow().isoformat()}
            )
            return True
        except Exception:
            return False
    
    def _sanitize_id(self, memory_id: str) -> str:
        """Sanitize memory_id to prevent SQL injection.
        
        UUIDs should only contain alphanumeric characters and hyphens.
        Rejects any input containing quotes, semicolons, or other SQL metacharacters.
        """
        import re
        # UUID pattern: only allow alphanumeric and hyphens
        if not re.match(r'^[a-zA-Z0-9\-]+$', memory_id):
            raise ValueError(f"Invalid memory_id format: {memory_id[:20]}...")
        return memory_id
    
    async def list(
        self,
        limit: int = 1000,
        offset: int = 0,
        filters: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        await self._ensure_initialized()
        
        results = (
            self._table.search()
            .where("deleted = false")
            .limit(limit + offset)
            .to_list()
        )
        
        entries = [self._row_to_entry(row) for row in results[offset:offset + limit]]
        
        if filters and "tags" in filters and filters["tags"]:
            entries = [e for e in entries if any(t in e.tags for t in filters["tags"])]
        
        return entries
    
    async def count(self, filters: Optional[dict] = None) -> int:
        entries = await self.list(limit=100000, filters=filters)
        return len(entries)

    async def get_stats(self) -> dict:
        """Compute stats natively over LanceDB rows.

        Iterates rows in pages to avoid loading all embeddings into
        RAM. Only the metadata columns are read.
        """
        await self._ensure_initialized()

        by_source: dict[str, int] = {}
        by_instance: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        total = 0
        corrections = 0

        page_size = 1000
        offset = 0
        while True:
            rows = (
                self._table.search()
                .where("deleted = false")
                .select(["source_type", "source_instance", "tags",
                         "supersedes"])
                .limit(page_size + offset)
                .to_list()
            )
            page = rows[offset:]
            if not page:
                break

            for row in page:
                total += 1
                src = row.get("source_type", "unknown")
                by_source[src] = by_source.get(src, 0) + 1

                inst = row.get("source_instance", "unknown")
                by_instance[inst] = by_instance.get(inst, 0) + 1

                tags = json.loads(row.get("tags", "[]"))
                for tag in tags:
                    by_tag[tag] = by_tag.get(tag, 0) + 1

                if row.get("supersedes"):
                    corrections += 1

            if len(page) < page_size:
                break
            offset += page_size

        return {
            "total_memories": total,
            "by_source_type": by_source,
            "by_tag": by_tag,
            "by_instance": by_instance,
            "corrections": corrections,
        }

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


class InMemoryVectorStore(IVectorStore):
    """Simple in-memory vector store for testing."""
    
    def __init__(self, embedding_service: IEmbeddingService):
        self.embedding_service = embedding_service
        self._store: dict[str, MemoryEntry] = {}
        self._deleted: set[str] = set()
    
    async def store(self, entry: MemoryEntry) -> StoreResult:
        if entry.embedding is None:
            entry.embedding = await self.embedding_service.embed(entry.content)
        
        self._store[entry.id] = entry
        return StoreResult(success=True, memory_id=entry.id)
    
    async def recall(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        filters: Optional[dict] = None,
    ) -> list[RecallResult]:
        start = time.perf_counter()
        
        results = []
        for entry in self._store.values():
            if entry.id in self._deleted:
                continue
            if entry.embedding is None:
                continue
            
            # Apply filters
            if filters and "tags" in filters and filters["tags"]:
                if not any(t in entry.tags for t in filters["tags"]):
                    continue
            
            sim = self.embedding_service.similarity(query_embedding, entry.embedding)
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
        limit: int = 1000,
        offset: int = 0,
        filters: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        entries = [
            e for e in list(self._store.values())
            if e.id not in self._deleted
        ]
        
        if filters and "tags" in filters and filters["tags"]:
            entries = [e for e in entries if any(t in e.tags for t in filters["tags"])]
        
        return entries[offset:offset + limit]

    async def upsert(self, entry: MemoryEntry) -> StoreResult:
        """Insert or replace, clearing any soft-delete tombstone."""
        self._deleted.discard(entry.id)
        if entry.embedding is None:
            entry.embedding = (
                await self.embedding_service.embed(entry.content)
            )
        self._store[entry.id] = entry
        return StoreResult(success=True, memory_id=entry.id)

    async def count(self, filters: Optional[dict] = None) -> int:
        entries = await self.list(limit=100000, filters=filters)
        return len(entries)

    async def get_stats(self) -> dict:
        """Compute stats in a single pass over in-memory entries."""
        by_source: dict[str, int] = {}
        by_instance: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        total = 0
        corrections = 0

        for entry in self._store.values():
            if entry.id in self._deleted:
                continue
            total += 1
            src = entry.source_type.value
            by_source[src] = by_source.get(src, 0) + 1
            inst = entry.source_instance
            by_instance[inst] = by_instance.get(inst, 0) + 1
            for tag in entry.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1
            if entry.supersedes:
                corrections += 1

        return {
            "total_memories": total,
            "by_source_type": by_source,
            "by_tag": by_tag,
            "by_instance": by_instance,
            "corrections": corrections,
        }
