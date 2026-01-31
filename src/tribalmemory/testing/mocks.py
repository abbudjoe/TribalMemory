"""Mock implementations for testing."""

import asyncio
import hashlib
import random
import re
import time
from datetime import datetime
from typing import Optional

from ..interfaces import (
    IEmbeddingService,
    IVectorStore,
    IMemoryService,
    IDeduplicationService,
    ITimestampService,
    MemoryEntry,
    MemorySource,
    RecallResult,
    StoreResult,
)
from .embedding_utils import hash_to_embedding_extended
from .semantic_expansions import (
    SHORT_IMPORTANT_WORDS,
    get_expanded_terms,
    get_word_variants,
)


# Scoring constants for text matching in recall
CANDIDATE_MULTIPLIER = 3  # Fetch N times more candidates than limit for re-ranking
MIN_CANDIDATE_THRESHOLD = 0.1  # Minimum similarity for candidate consideration
BASE_TEXT_MATCH_SCORE = 0.7  # Base score when meaningful words overlap
OVERLAP_BOOST_PER_WORD = 0.05  # Additional score per overlapping word


class MockEmbeddingService(IEmbeddingService):
    """Mock embedding service for testing.
    
    Uses deterministic hashing for reproducible tests.
    Can be configured to simulate failures and latency.
    """
    
    def __init__(
        self,
        embedding_dim: int = 1536,
        latency_ms: float = 0,
        failure_rate: float = 0,
        timeout_after_n: Optional[int] = None,
        skip_latency: bool = False,
    ):
        """Initialize mock embedding service.
        
        Args:
            embedding_dim: Dimension of generated embeddings.
            latency_ms: Simulated latency per call.
            failure_rate: Probability of failure (0.0-1.0).
            timeout_after_n: Simulate timeout after N calls.
            skip_latency: If True, skip all latency simulation (fast mode for dev).
        """
        self.embedding_dim = embedding_dim
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.timeout_after_n = timeout_after_n
        self.skip_latency = skip_latency
        self._call_count = 0
    
    async def embed(self, text: str) -> list[float]:
        """Generate deterministic embedding from text hash."""
        self._call_count += 1
        
        # Simulate timeout
        if self.timeout_after_n and self._call_count > self.timeout_after_n:
            if not self.skip_latency:
                await asyncio.sleep(30)  # Will trigger timeout
        
        # Simulate latency
        if self.latency_ms > 0 and not self.skip_latency:
            await asyncio.sleep(self.latency_ms / 1000)
        
        # Simulate failures
        if self.failure_rate > 0 and random.random() < self.failure_rate:
            raise RuntimeError("Mock embedding API failure")
        
        return self._hash_to_embedding(text)
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch."""
        return [await self.embed(t) for t in texts]
    
    def similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity.
        
        Note: Zero vectors (all zeros) return 0.0 similarity with any other vector.
        This is intentional - zero vectors indicate corrupted/missing embeddings
        and should not match anything. Tests in test_negative_security.py verify
        that corrupted embeddings are excluded from results.
        """
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def _hash_to_embedding(self, text: str) -> list[float]:
        """Convert text to deterministic embedding that preserves semantic similarity.
        
        Delegates to shared utility for consistent behavior across mock implementations.
        Uses extended version with sliding window hashes for substring matching.
        """
        return hash_to_embedding_extended(text, self.embedding_dim)


class MockVectorStore(IVectorStore):
    """In-memory vector store for testing."""
    
    def __init__(
        self,
        embedding_service: IEmbeddingService,
        latency_ms: float = 0,
        max_capacity: Optional[int] = None
    ):
        self.embedding_service = embedding_service
        self.latency_ms = latency_ms
        self.max_capacity = max_capacity
        self._store: dict[str, MemoryEntry] = {}
        self._deleted: set[str] = set()
    
    async def store(self, entry: MemoryEntry) -> StoreResult:
        """Store a memory entry."""
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
        
        if self.max_capacity and len(self._store) >= self.max_capacity:
            return StoreResult(
                success=False,
                error="Storage capacity reached"
            )
        
        self._store[entry.id] = entry
        return StoreResult(success=True, memory_id=entry.id)
    
    async def recall(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        filters: Optional[dict] = None,
    ) -> list[RecallResult]:
        """Recall memories similar to query."""
        start = time.perf_counter()
        
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
        
        results = []
        for memory in self._store.values():
            if memory.id in self._deleted:
                continue
            if memory.embedding is None:
                continue
            
            # Apply filters
            if filters and "tags" in filters and filters["tags"]:
                if not any(t in memory.tags for t in filters["tags"]):
                    continue
            
            sim = self.embedding_service.similarity(query_embedding, memory.embedding)
            if sim >= min_similarity:
                results.append((memory, sim))
        
        # Sort by similarity, take top limit
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:limit]
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return [
            RecallResult(
                memory=mem,
                similarity_score=sim,
                retrieval_time_ms=elapsed_ms
            )
            for mem, sim in results
        ]
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        if memory_id in self._deleted:
            return None
        return self._store.get(memory_id)
    
    async def delete(self, memory_id: str) -> bool:
        """Soft delete a memory."""
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
        """List memories with optional filtering."""
        entries = [
            m for m in list(self._store.values())
            if m.id not in self._deleted
        ]
        
        if filters and "tags" in filters and filters["tags"]:
            entries = [e for e in entries if any(t in e.tags for t in filters["tags"])]
        
        return entries[offset:offset + limit]
    
    async def count(self, filters: Optional[dict] = None) -> int:
        """Count memories matching filters."""
        entries = await self.list(limit=100000, filters=filters)
        return len(entries)
    
    def clear(self):
        """Clear all data (for test cleanup)."""
        self._store.clear()
        self._deleted.clear()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - clears store to prevent test pollution."""
        self.clear()


class MockDeduplicationService(IDeduplicationService):
    """Mock deduplication service."""
    
    def __init__(
        self,
        vector_store: MockVectorStore,
        embedding_service: IEmbeddingService
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    async def is_duplicate(
        self,
        content: str,
        embedding: list[float],
        threshold: float = 0.90
    ) -> tuple[bool, Optional[str]]:
        """Check if content is duplicate.
        
        Default threshold lowered to 0.90 to catch near-duplicates like:
        - "Joe prefers concise responses" vs "Joe likes concise answers"
        - Typo corrections and minor paraphrases
        
        Returns:
            Tuple of (is_duplicate, duplicate_of_id)
        """
        similar = await self.find_similar(content, embedding, threshold)
        if similar:
            return True, similar[0][0]
        return False, None
    
    async def find_similar(
        self,
        content: str,
        embedding: list[float],
        threshold: float = 0.85,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar memories."""
        results = await self.vector_store.recall(
            embedding,
            limit=limit,
            min_similarity=threshold
        )
        return [(r.memory.id, r.similarity_score) for r in results]


class MockMemoryService(IMemoryService):
    """High-level mock memory service for testing."""
    
    def __init__(
        self,
        instance_id: str = "test-instance",
        embedding_service: Optional[IEmbeddingService] = None,
        vector_store: Optional[IVectorStore] = None
    ):
        self.instance_id = instance_id
        self.embedding_service = embedding_service or MockEmbeddingService()
        self.vector_store = vector_store or MockVectorStore(self.embedding_service)
        self.dedup_service = MockDeduplicationService(
            self.vector_store,
            self.embedding_service
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
        # Validate
        if not content or not content.strip():
            return StoreResult(success=False, error="Empty content not allowed")
        
        # Generate embedding
        embedding = await self.embedding_service.embed(content)
        
        # Check for duplicates
        if not skip_dedup:
            is_dup, dup_id = await self.dedup_service.is_duplicate(content, embedding)
            if is_dup:
                return StoreResult(success=False, duplicate_of=dup_id)
        
        # Create entry
        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            source_instance=self.instance_id,
            source_type=source_type,
            context=context,
            tags=tags or []
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
        
        Uses both embedding similarity and text matching to better simulate
        real semantic search behavior in the mock.
        """
        query_embedding = await self.embedding_service.embed(query)
        filters = {"tags": tags} if tags else None
        
        # Get results from vector store with lowered threshold
        # We'll re-filter based on combined score
        results = await self.vector_store.recall(
            query_embedding, 
            limit=limit * CANDIDATE_MULTIPLIER,
            min_similarity=min(MIN_CANDIDATE_THRESHOLD, min_relevance / 2),
            filters=filters
        )
        
        # Boost scores based on text matching (simulates semantic similarity better)
        query_lower = query.lower()
        # Filter out very short words for matching (stopwords-ish)
        query_words = {w for w in re.findall(r'\b\w+\b', query_lower) if len(w) > 2}
        
        # Add common short words that matter
        query_words.update(
            w for w in re.findall(r'\b\w+\b', query_lower) 
            if w in SHORT_IMPORTANT_WORDS
        )
        
        # Expand query words with variants (pseudo-stemming)
        expanded_query = set()
        for w in query_words:
            expanded_query.update(get_word_variants(w))
        query_words = expanded_query
        
        # Apply semantic expansions for common concepts
        query_words = get_expanded_terms(query_words, query_lower)
        
        def is_corrupted_embedding(emb: list[float] | None) -> bool:
            """Check if embedding is corrupted (zero vector, NaN, etc.)."""
            if emb is None:
                return True
            if all(x == 0.0 for x in emb):
                return True
            if any(x != x for x in emb):  # NaN check
                return True
            return False
        
        boosted_results = []
        for r in results:
            # Skip memories with corrupted embeddings (security consideration)
            if is_corrupted_embedding(r.memory.embedding):
                continue
            
            content_lower = r.memory.content.lower()
            content_words = {w for w in re.findall(r'\b\w+\b', content_lower) if len(w) > 2}
            
            # Calculate text match boost
            text_boost = 0.0
            
            # Exact substring match is strong signal
            if query_lower in content_lower:
                text_boost = max(text_boost, 0.9)
            
            # Word overlap scoring
            if query_words and content_words:
                overlap = query_words & content_words
                # If any meaningful (>=3 chars) words overlap, it's relevant
                meaningful_overlap = [w for w in overlap if len(w) >= 3]
                if meaningful_overlap:
                    # More overlap = higher score, base score is 0.7 (meets default threshold)
                    score = BASE_TEXT_MATCH_SCORE + OVERLAP_BOOST_PER_WORD * len(meaningful_overlap)
                    text_boost = max(text_boost, score)
                elif overlap:
                    text_boost = max(text_boost, 0.5)
            
            # Combined score: max of embedding sim and text boost
            combined_score = max(r.similarity_score, text_boost)
            
            if combined_score >= min_relevance:
                boosted_results.append(RecallResult(
                    memory=r.memory,
                    similarity_score=combined_score,
                    retrieval_time_ms=r.retrieval_time_ms
                ))
        
        # Also check memories not returned by vector search (text-only matches)
        returned_ids = {r.memory.id for r in results}
        all_memories = await self.vector_store.list(limit=1000, filters=filters)
        
        for memory in all_memories:
            if memory.id in returned_ids or memory.id in self.vector_store._deleted:
                continue
            
            # Skip memories with corrupted embeddings (security consideration)
            if is_corrupted_embedding(memory.embedding):
                continue
                
            content_lower = memory.content.lower()
            content_words = {w for w in re.findall(r'\b\w+\b', content_lower) if len(w) > 2}
            
            text_boost = 0.0
            if query_lower in content_lower:
                text_boost = 0.9
            elif query_words and content_words:
                overlap = query_words & content_words
                meaningful_overlap = [w for w in overlap if len(w) >= 3]
                if meaningful_overlap:
                    text_boost = (
                        BASE_TEXT_MATCH_SCORE + OVERLAP_BOOST_PER_WORD * len(meaningful_overlap)
                    )
                elif overlap:
                    text_boost = 0.5
            
            if text_boost >= min_relevance:
                boosted_results.append(RecallResult(
                    memory=memory,
                    similarity_score=text_boost,
                    retrieval_time_ms=0.0
                ))
        
        # Sort by score and limit
        boosted_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return boosted_results[:limit]
    
    async def correct(
        self,
        original_id: str,
        corrected_content: str,
        context: Optional[str] = None
    ) -> StoreResult:
        """Store a correction to an existing memory."""
        embedding = await self.embedding_service.embed(corrected_content)
        
        entry = MemoryEntry(
            content=corrected_content,
            embedding=embedding,
            source_instance=self.instance_id,
            source_type=MemorySource.CORRECTION,
            context=context,
            supersedes=original_id
        )
        
        return await self.vector_store.store(entry)
    
    async def forget(self, memory_id: str) -> bool:
        """Forget a memory."""
        return await self.vector_store.delete(memory_id)
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID with full provenance."""
        return await self.vector_store.get(memory_id)


class MockTimestampService(ITimestampService):
    """Mock RFC 3161 timestamp service for testing.
    
    Generates deterministic timestamps for reproducible tests.
    Does NOT provide cryptographic guarantees - use real TSA in production.
    """
    
    def __init__(self, fail_verify: bool = False):
        """Initialize mock timestamp service.
        
        Args:
            fail_verify: If True, verify() always returns False (for testing failures)
        """
        self.fail_verify = fail_verify
        self._timestamps: dict[bytes, datetime] = {}
    
    async def timestamp(self, data: bytes) -> bytes:
        """Generate a mock timestamp token.
        
        Token format: "MOCK_TSA|{iso_timestamp}|{data_hash}"
        This is NOT RFC 3161 compliant - use for testing only.
        """
        import hashlib
        
        now = datetime.utcnow()
        data_hash = hashlib.sha256(data).hexdigest()[:16]
        token = f"MOCK_TSA|{now.isoformat()}|{data_hash}".encode()
        
        # Store for verification
        self._timestamps[token] = now
        
        return token
    
    async def verify(self, data: bytes, token: bytes) -> tuple[bool, Optional[datetime]]:
        """Verify a mock timestamp token.
        
        Returns:
            Tuple of (is_valid, timestamp_datetime)
        """
        if self.fail_verify:
            return False, None
        
        try:
            decoded = token.decode()
            if not decoded.startswith("MOCK_TSA|"):
                return False, None
            
            parts = decoded.split("|")
            if len(parts) != 3:
                return False, None
            
            timestamp_str = parts[1]
            stored_hash = parts[2]
            
            # Verify data hash matches
            import hashlib
            actual_hash = hashlib.sha256(data).hexdigest()[:16]
            if actual_hash != stored_hash:
                return False, None
            
            timestamp = datetime.fromisoformat(timestamp_str)
            return True, timestamp
            
        except Exception:
            return False, None
