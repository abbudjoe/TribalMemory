"""Memory System - High-level API for A2.1.

This is the main entry point for interacting with tribal memory.
It provides a clean, high-level interface while delegating to
the underlying providers through the container.
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from .config import SystemConfig
from .container import Container
from ..interfaces import MemoryEntry, MemorySource, RecallResult, StoreResult


class MemorySystem:
    """High-level memory system API.
    
    This class provides a simple, clean interface for memory operations
    while managing all the underlying complexity through the container.
    
    Usage:
        config = SystemConfig.from_env()
        system = MemorySystem(config)
        
        async with system:
            await system.remember("Important fact")
            results = await system.recall("What was that fact?")
    
    Or manually:
        system = MemorySystem(config)
        await system.start()
        try:
            await system.remember("Important fact")
        finally:
            await system.stop()
    """
    
    def __init__(self, config: SystemConfig):
        """Initialize memory system.
        
        Args:
            config: System configuration
        """
        self.config = config
        self._container = Container(config)
        self._started = False
    
    async def start(self) -> None:
        """Start the memory system."""
        if self._started:
            return
        
        # Validate config
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")
        
        await self._container.initialize()
        self._started = True
    
    async def stop(self) -> None:
        """Stop the memory system."""
        if not self._started:
            return
        await self._container.shutdown()
        self._started = False
    
    async def remember(
        self,
        content: str,
        source_type: MemorySource = MemorySource.AUTO_CAPTURE,
        context: Optional[str] = None,
        tags: Optional[list[str]] = None,
        skip_dedup: bool = False,
    ) -> StoreResult:
        """Store a new memory.
        
        Args:
            content: The memory content
            source_type: How this memory was captured
            context: Additional context about capture
            tags: Tags for categorization
            skip_dedup: Skip duplicate checking
            
        Returns:
            StoreResult with success status
        """
        self._ensure_started()
        
        # Validate
        if not content or not content.strip():
            return StoreResult(success=False, error="Empty content not allowed")
        
        content = content.strip()
        
        # Generate embedding
        try:
            embedding = await self._container.embedding.embed(content)
        except Exception as e:
            return StoreResult(success=False, error=f"Embedding failed: {e}")
        
        # Check for duplicates
        if not skip_dedup and self._container.deduplication:
            is_dup, dup_id = await self._container.deduplication.is_duplicate(
                content, embedding
            )
            if is_dup:
                return StoreResult(success=False, duplicate_of=dup_id)
        
        # Create entry
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            source_instance=self.config.instance_id,
            source_type=source_type,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=tags or [],
            context=context,
            confidence=1.0,
        )
        
        return await self._container.storage.store(entry)
    
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
            tags: Filter by tags
            
        Returns:
            List of RecallResults sorted by relevance
        """
        self._ensure_started()
        
        try:
            query_embedding = await self._container.embedding.embed(query)
        except Exception:
            return []
        
        filters = {"tags": tags} if tags else None
        
        return await self._container.storage.recall(
            query_embedding,
            limit=limit,
            min_similarity=min_relevance,
            filters=filters,
        )
    
    async def correct(
        self,
        original_id: str,
        corrected_content: str,
        context: Optional[str] = None,
    ) -> StoreResult:
        """Store a correction to an existing memory.
        
        Args:
            original_id: ID of memory being corrected
            corrected_content: The corrected information
            context: Why this correction was made
            
        Returns:
            StoreResult for the correction entry
        """
        self._ensure_started()
        
        # Verify original exists
        original = await self._container.storage.get(original_id)
        if not original:
            return StoreResult(success=False, error=f"Original memory {original_id} not found")
        
        # Generate embedding
        try:
            embedding = await self._container.embedding.embed(corrected_content)
        except Exception as e:
            return StoreResult(success=False, error=f"Embedding failed: {e}")
        
        # Create correction entry
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=corrected_content,
            embedding=embedding,
            source_instance=self.config.instance_id,
            source_type=MemorySource.CORRECTION,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=original.tags,
            context=context or f"Correction of {original_id}",
            confidence=1.0,
            supersedes=original_id,
            related_to=[original_id],
        )
        
        return await self._container.storage.store(entry)
    
    async def forget(self, memory_id: str) -> bool:
        """Forget (soft delete) a memory.
        
        Args:
            memory_id: ID of memory to forget
            
        Returns:
            True if forgotten successfully
        """
        self._ensure_started()
        return await self._container.storage.delete(memory_id)
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            MemoryEntry or None if not found
        """
        self._ensure_started()
        return await self._container.storage.get(memory_id)
    
    async def health(self) -> dict[str, Any]:
        """Check system health.
        
        Returns:
            Dict with provider health statuses including:
            - status: "running" or "stopped"
            - instance_id: This instance's ID
            - providers: Dict of provider name to health info
        """
        if not self._started:
            return {"status": "stopped"}
        
        health = await self._container.health_check()
        return {
            "status": "running",
            "instance_id": self.config.instance_id,
            "providers": {
                name: {"status": h.status.value, "latency_ms": h.latency_ms}
                for name, h in health.items()
            }
        }
    
    async def stats(self) -> dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dict with memory counts and breakdowns including:
            - total_memories: Total count of active memories
            - instance_id: This instance's ID
        """
        self._ensure_started()
        
        total = await self._container.storage.count()
        
        return {
            "total_memories": total,
            "instance_id": self.config.instance_id,
        }
    
    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("MemorySystem not started. Call start() first.")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
