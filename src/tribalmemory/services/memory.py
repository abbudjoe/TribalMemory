"""Tribal Memory Service - Main API for agents."""

import os
from datetime import datetime
from typing import Optional
import uuid

from ..interfaces import (
    IMemoryService,
    IEmbeddingService,
    IVectorStore,
    MemoryEntry,
    MemorySource,
    RecallResult,
    StoreResult,
)
from .deduplication import SemanticDeduplicationService


class TribalMemoryService(IMemoryService):
    """Production tribal memory service.
    
    Usage:
        service = TribalMemoryService(
            instance_id="clawdio-1",
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        await service.remember("Joe prefers TypeScript")
        results = await service.recall("What language for Wally?")
    """
    
    def __init__(
        self,
        instance_id: str,
        embedding_service: IEmbeddingService,
        vector_store: IVectorStore,
        dedup_exact_threshold: float = 0.98,
        dedup_near_threshold: float = 0.90,
        auto_reject_duplicates: bool = True,
    ):
        self.instance_id = instance_id
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.auto_reject_duplicates = auto_reject_duplicates
        
        self.dedup_service = SemanticDeduplicationService(
            vector_store=vector_store,
            embedding_service=embedding_service,
            exact_threshold=dedup_exact_threshold,
            near_threshold=dedup_near_threshold,
        )
    
    async def remember(
        self,
        content: str,
        source_type: MemorySource = MemorySource.AUTO_CAPTURE,
        context: Optional[str] = None,
        tags: Optional[list[str]] = None,
        skip_dedup: bool = False,
    ) -> StoreResult:
        """Store a new memory."""
        if not content or not content.strip():
            return StoreResult(success=False, error="TribalMemory: Empty content not allowed")
        
        content = content.strip()
        
        try:
            embedding = await self.embedding_service.embed(content)
        except Exception as e:
            return StoreResult(success=False, error=f"Embedding generation failed: {e}")
        
        if not skip_dedup and self.auto_reject_duplicates:
            is_dup, dup_id = await self.dedup_service.is_duplicate(content, embedding)
            if is_dup:
                return StoreResult(success=False, duplicate_of=dup_id)
        
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            source_instance=self.instance_id,
            source_type=source_type,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=tags or [],
            context=context,
            confidence=1.0,
        )
        
        return await self.vector_store.store(entry)
    
    async def recall(
        self,
        query: str,
        limit: int = 5,
        min_relevance: float = 0.7,
        tags: Optional[list[str]] = None,
    ) -> list[RecallResult]:
        """Recall relevant memories.
        
        Args:
            query: Natural language query
            limit: Maximum results
            min_relevance: Minimum similarity score
            tags: Filter by tags (e.g., ["work", "preferences"])
        """
        try:
            query_embedding = await self.embedding_service.embed(query)
        except Exception:
            return []
        
        filters = {"tags": tags} if tags else None
        
        results = await self.vector_store.recall(
            query_embedding,
            limit=limit,
            min_similarity=min_relevance,
            filters=filters,
        )
        
        return results
    
    async def correct(
        self,
        original_id: str,
        corrected_content: str,
        context: Optional[str] = None,
    ) -> StoreResult:
        """Store a correction to an existing memory."""
        original = await self.vector_store.get(original_id)
        if not original:
            return StoreResult(success=False, error=f"Original memory {original_id} not found")
        
        try:
            embedding = await self.embedding_service.embed(corrected_content)
        except Exception as e:
            return StoreResult(success=False, error=f"Embedding generation failed: {e}")
        
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=corrected_content,
            embedding=embedding,
            source_instance=self.instance_id,
            source_type=MemorySource.CORRECTION,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=original.tags,
            context=context or f"Correction of memory {original_id}",
            confidence=1.0,
            supersedes=original_id,
            related_to=[original_id],
        )
        
        return await self.vector_store.store(entry)
    
    async def forget(self, memory_id: str) -> bool:
        """Forget (soft delete) a memory."""
        return await self.vector_store.delete(memory_id)
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID with full provenance."""
        return await self.vector_store.get(memory_id)
    
    async def get_stats(self) -> dict:
        """Get memory statistics.
        
        Note: Stats are computed over up to 10,000 most recent memories.
        For systems with >10k memories, consider using count() with filters.
        """
        all_memories = await self.vector_store.list(limit=10000)
        
        by_source: dict[str, int] = {}
        by_instance: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        
        for m in all_memories:
            source = m.source_type.value
            by_source[source] = by_source.get(source, 0) + 1
            
            instance = m.source_instance
            by_instance[instance] = by_instance.get(instance, 0) + 1
            
            for tag in m.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1
        
        corrections = sum(1 for m in all_memories if m.supersedes)
        
        return {
            "total_memories": len(all_memories),
            "by_source_type": by_source,
            "by_tag": by_tag,
            "by_instance": by_instance,
            "corrections": corrections,
        }


def create_memory_service(
    instance_id: Optional[str] = None,
    db_path: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> TribalMemoryService:
    """Factory function to create a memory service with sensible defaults.
    
    Args:
        instance_id: Unique identifier for this agent instance.
        db_path: Path for LanceDB persistent storage. If None, uses in-memory.
        openai_api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
    
    Returns:
        Configured TribalMemoryService ready for use.
    
    Warning:
        If db_path is provided but LanceDB is not installed, falls back to
        in-memory storage. This means data will NOT persist across restarts.
    """
    import logging
    
    from .embeddings import OpenAIEmbeddingService
    from .vector_store import InMemoryVectorStore, LanceDBVectorStore
    
    logger = logging.getLogger(__name__)
    
    if not instance_id:
        instance_id = os.environ.get("TRIBAL_MEMORY_INSTANCE_ID", "default")
    
    embedding_service = OpenAIEmbeddingService(api_key=openai_api_key)
    
    if db_path:
        try:
            vector_store = LanceDBVectorStore(
                embedding_service=embedding_service,
                db_path=db_path
            )
        except ImportError:
            logger.warning(
                "LanceDB not installed. Falling back to in-memory storage. "
                "Data will NOT persist across restarts. Install with: pip install lancedb"
            )
            vector_store = InMemoryVectorStore(embedding_service)
    else:
        vector_store = InMemoryVectorStore(embedding_service)
    
    return TribalMemoryService(
        instance_id=instance_id,
        embedding_service=embedding_service,
        vector_store=vector_store
    )
