"""Core interfaces for Tribal Memory system.

These interfaces define the contract that both A2.1 and A2.2 implementations must satisfy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal, Optional
import uuid

# Valid retrieval methods for RecallResult
RetrievalMethod = Literal["vector", "graph", "hybrid", "entity"]


class MemorySource(Enum):
    """Source of a memory entry."""
    USER_EXPLICIT = "user_explicit"  # User said "remember X"
    AUTO_CAPTURE = "auto_capture"    # System detected important info
    CORRECTION = "correction"        # Correction to existing memory
    CROSS_INSTANCE = "cross_instance"  # Propagated from another instance
    LEGACY = "legacy"                # Pre-tribal-memory import
    UNKNOWN = "unknown"


@dataclass
class MemoryEntry:
    """A single memory entry with full provenance.
    
    Attributes:
        id: Unique identifier (UUID by default)
        content: The actual memory content
        embedding: Vector embedding (None until generated)
        source_instance: Which agent instance created this memory
        source_type: How this memory was captured (user_explicit, auto_capture, etc.)
        created_at: When the memory was first created
        updated_at: When the memory was last modified
        tags: Categorization tags for filtering
        context: What triggered this memory (conversation context, etc.)
        confidence: How confident we are in this memory (0.0-1.0).
                   Currently always 1.0; reserved for future use with
                   uncertain inferences or low-confidence auto-captures.
        supersedes: ID of memory this corrects (for correction chains)
        related_to: IDs of related memories. Reserved for future use with
                   knowledge graphs and memory clustering.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[list[float]] = None
    
    # Provenance
    source_instance: str = "unknown"
    source_type: MemorySource = MemorySource.UNKNOWN
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    tags: list[str] = field(default_factory=list)
    context: Optional[str] = None
    confidence: float = 1.0
    
    # Relationships
    supersedes: Optional[str] = None
    related_to: list[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        """Concise repr for debugging."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"MemoryEntry(id={self.id[:8]}..., content='{content_preview}', source={self.source_type.value})"


@dataclass
class RecallResult:
    """Result of a memory recall query.
    
    Attributes:
        memory: The recalled memory entry.
        similarity_score: Relevance score (0.0-1.0 for vector, 1.0 for exact entity match).
        retrieval_time_ms: Time taken for retrieval.
        retrieval_method: How this result was found (see RetrievalMethod type).
    """
    memory: MemoryEntry
    similarity_score: float
    retrieval_time_ms: float
    retrieval_method: RetrievalMethod = "vector"
    
    def __repr__(self) -> str:
        return f"RecallResult(score={self.similarity_score:.3f}, method={self.retrieval_method}, memory_id={self.memory.id[:8]}...)"


@dataclass
class StoreResult:
    """Result of storing a memory."""
    success: bool
    memory_id: Optional[str] = None
    duplicate_of: Optional[str] = None  # If rejected as duplicate
    error: Optional[str] = None
    
    def __repr__(self) -> str:
        if self.success:
            return f"StoreResult(success=True, id={self.memory_id[:8] if self.memory_id else None}...)"
        elif self.duplicate_of:
            return f"StoreResult(success=False, duplicate_of={self.duplicate_of[:8]}...)"
        else:
            return f"StoreResult(success=False, error='{self.error}')"


class IEmbeddingService(ABC):
    """Interface for embedding generation."""
    
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        pass


class IVectorStore(ABC):
    """Interface for vector storage and retrieval."""
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> StoreResult:
        """Store a memory entry."""
        pass
    
    @abstractmethod
    async def recall(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        filters: Optional[dict] = None,
    ) -> list[RecallResult]:
        """Recall memories similar to query.
        
        Args:
            query_embedding: Vector to search for
            limit: Maximum results
            min_similarity: Minimum similarity threshold
            filters: Optional metadata filters (e.g., tags, source_instance)
        """
        pass
    
    @abstractmethod
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        pass
    
    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory (soft delete with tombstone)."""
        pass

    async def upsert(self, entry: MemoryEntry) -> StoreResult:
        """Insert or replace a memory entry by ID.

        Default implementation: delete existing + store new.
        Subclasses may override for atomic upsert support.
        """
        await self.delete(entry.id)
        return await self.store(entry)
    
    @abstractmethod
    async def list(
        self,
        limit: int = 1000,
        offset: int = 0,
        filters: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """List memories with pagination and optional filtering."""
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[dict] = None) -> int:
        """Count memories matching filters."""
        pass

    async def get_stats(self) -> dict:
        """Compute aggregate statistics over all memories.

        Returns dict with keys:
            total_memories, by_source_type, by_tag, by_instance, corrections

        Default implementation iterates in pages of 500. Subclasses
        should override with native queries (SQL GROUP BY, etc.) for
        stores with >10k entries.
        """
        page_size = 500
        total = 0
        corrections = 0
        by_source: dict[str, int] = {}
        by_instance: dict[str, int] = {}
        by_tag: dict[str, int] = {}

        offset = 0
        while True:
            page = await self.list(limit=page_size, offset=offset)
            if not page:
                break
            total += len(page)
            for m in page:
                src = m.source_type.value
                by_source[src] = by_source.get(src, 0) + 1
                inst = m.source_instance
                by_instance[inst] = by_instance.get(inst, 0) + 1
                for tag in m.tags:
                    by_tag[tag] = by_tag.get(tag, 0) + 1
                if m.supersedes:
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


class IDeduplicationService(ABC):
    """Interface for detecting duplicate memories."""
    
    @abstractmethod
    async def is_duplicate(
        self,
        content: str,
        embedding: list[float],
        threshold: float = 0.95
    ) -> tuple[bool, Optional[str]]:
        """Check if content is duplicate.
        
        Returns:
            Tuple of (is_duplicate, duplicate_of_id)
            - is_duplicate: True if content exceeds threshold similarity
            - duplicate_of_id: ID of the matching memory (or None)
        
        .. versionchanged:: 0.2.0
            Return type changed from 3-tuple to 2-tuple. The similarity_score
            was removed from the return value. If you need the similarity score,
            use :meth:`find_similar` to get scored results, or use
            :meth:`get_duplicate_report` for detailed duplicate analysis.
        
        Migration guide:
            Old: ``is_dup, dup_id, score = await dedup.is_duplicate(...)``
            New: ``is_dup, dup_id = await dedup.is_duplicate(...)``
            
            To get score: ``results = await dedup.find_similar(content, embedding, threshold)``
            The first result's score is the duplicate's similarity.
        """
        pass
    
    @abstractmethod
    async def find_similar(
        self,
        content: str,
        embedding: list[float],
        threshold: float = 0.85,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar memories.
        
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        pass


class ITimestampService(ABC):
    """Interface for RFC 3161 timestamping."""
    
    @abstractmethod
    async def timestamp(self, data: bytes) -> bytes:
        """Get RFC 3161 timestamp token for data."""
        pass
    
    @abstractmethod
    async def verify(self, data: bytes, token: bytes) -> tuple[bool, Optional[datetime]]:
        """Verify timestamp token. Returns (valid, timestamp)."""
        pass


class IMemoryService(ABC):
    """High-level interface for memory operations.
    
    This is the main interface that agents (LLMs) interact with.
    Designed for intuitive use with simple, verb-based methods.
    """
    
    @abstractmethod
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
            tags: Tags for categorization and filtering
            skip_dedup: If True, store even if similar memory exists
        """
        pass
    
    @abstractmethod
    async def recall(
        self,
        query: str,
        limit: int = 5,
        min_relevance: float = 0.7,
        tags: Optional[list[str]] = None,
        graph_expansion: bool = True,
    ) -> list[RecallResult]:
        """Recall relevant memories for a query.
        
        Args:
            query: Natural language query
            limit: Maximum results
            min_relevance: Minimum similarity score
            tags: Filter by tags (e.g., ["work", "preferences"])
            graph_expansion: Expand candidates via entity graph (default True)
        
        Returns:
            List of RecallResult objects with retrieval_method indicating source:
            - "vector": Vector similarity search
            - "hybrid": Vector + BM25 merge
            - "graph": Entity graph traversal
        """
        pass
    
    @abstractmethod
    async def correct(
        self,
        original_id: str,
        corrected_content: str,
        context: Optional[str] = None
    ) -> StoreResult:
        """Store a correction to an existing memory.
        
        Creates a correction chain - the new memory supersedes the original.
        """
        pass
    
    @abstractmethod
    async def forget(self, memory_id: str) -> bool:
        """Forget (soft delete) a memory."""
        pass
    
    @abstractmethod
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID with full provenance."""
        pass
