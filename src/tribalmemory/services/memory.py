"""Tribal Memory Service - Main API for agents."""

import logging
import os
from datetime import datetime
from pathlib import Path
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
from .fts_store import FTSStore, hybrid_merge
from .reranker import IReranker, NoopReranker, create_reranker

logger = logging.getLogger(__name__)


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
        fts_store: Optional[FTSStore] = None,
        hybrid_search: bool = True,
        hybrid_vector_weight: float = 0.7,
        hybrid_text_weight: float = 0.3,
        hybrid_candidate_multiplier: int = 4,
        reranker: Optional[IReranker] = None,
        rerank_pool_multiplier: int = 2,
    ):
        self.instance_id = instance_id
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.auto_reject_duplicates = auto_reject_duplicates
        self.fts_store = fts_store
        self.hybrid_search = hybrid_search and fts_store is not None
        self.hybrid_vector_weight = hybrid_vector_weight
        self.hybrid_text_weight = hybrid_text_weight
        self.hybrid_candidate_multiplier = hybrid_candidate_multiplier
        self.reranker = reranker or NoopReranker()
        self.rerank_pool_multiplier = rerank_pool_multiplier
        
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
        
        result = await self.vector_store.store(entry)
        
        # Index in FTS for hybrid search (best-effort; vector store is primary)
        if result.success and self.fts_store:
            try:
                self.fts_store.index(entry.id, content, tags or [])
            except Exception as e:
                logger.warning("FTS indexing failed for %s: %s", entry.id, e)
        
        return result
    
    async def recall(
        self,
        query: str,
        limit: int = 5,
        min_relevance: float = 0.7,
        tags: Optional[list[str]] = None,
    ) -> list[RecallResult]:
        """Recall relevant memories using hybrid search.
        
        When hybrid search is enabled (FTS store available), combines
        vector similarity with BM25 keyword matching for better results.
        Falls back to vector-only search when FTS is unavailable.
        
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

        if self.hybrid_search and self.fts_store:
            return await self._hybrid_recall(
                query, query_embedding, limit, min_relevance, filters
            )
        
        # Vector-only fallback
        results = await self.vector_store.recall(
            query_embedding,
            limit=limit,
            min_similarity=min_relevance,
            filters=filters,
        )
        
        return self._filter_superseded(results)

    async def _hybrid_recall(
        self,
        query: str,
        query_embedding: list[float],
        limit: int,
        min_relevance: float,
        filters: Optional[dict],
    ) -> list[RecallResult]:
        """Hybrid recall: vector + BM25 combined, then reranked."""
        candidate_limit = limit * self.hybrid_candidate_multiplier

        # 1. Vector search — get wide candidate pool
        vector_results = await self.vector_store.recall(
            query_embedding,
            limit=candidate_limit,
            min_similarity=min_relevance * 0.5,  # Lower threshold for candidates
            filters=filters,
        )

        # 2. BM25 search
        bm25_results = self.fts_store.search(query, limit=candidate_limit)

        # 3. Build lookup for vector results
        vector_for_merge = [
            {"id": r.memory.id, "score": r.similarity_score}
            for r in vector_results
        ]

        # 4. Hybrid merge
        merged = hybrid_merge(
            vector_for_merge,
            bm25_results,
            self.hybrid_vector_weight,
            self.hybrid_text_weight,
        )

        # 5. Build candidate results for reranking — need full MemoryEntry for each
        # Create lookup from vector results
        entry_map = {r.memory.id: r for r in vector_results}

        # Get rerank_pool_multiplier * limit candidates before reranking
        rerank_pool_size = min(limit * self.rerank_pool_multiplier, len(merged))
        
        # Separate cached (vector) hits from BM25-only hits that need fetching
        cached_hits: list[tuple[dict, RecallResult]] = []
        bm25_only_ids: list[dict] = []
        
        for m in merged[:rerank_pool_size]:
            if m["id"] in entry_map:
                cached_hits.append((m, entry_map[m["id"]]))
            else:
                bm25_only_ids.append(m)
        
        # Batch-fetch BM25-only hits concurrently
        import asyncio
        fetched_entries = await asyncio.gather(
            *(self.vector_store.get(m["id"]) for m in bm25_only_ids)
        ) if bm25_only_ids else []
        
        # Build candidate list
        candidates: list[RecallResult] = []
        
        # Add cached vector hits
        for m, recall_result in cached_hits:
            candidates.append(RecallResult(
                memory=recall_result.memory,
                similarity_score=m["final_score"],
                retrieval_time_ms=recall_result.retrieval_time_ms,
            ))
        
        # Add fetched BM25-only hits
        for m, entry in zip(bm25_only_ids, fetched_entries):
            if entry and m["final_score"] >= min_relevance * 0.5:
                candidates.append(RecallResult(
                    memory=entry,
                    similarity_score=m["final_score"],
                    retrieval_time_ms=0,
                ))

        # 6. Rerank candidates
        reranked = self.reranker.rerank(query, candidates, top_k=limit)

        return self._filter_superseded(reranked)
    
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
        result = await self.vector_store.delete(memory_id)
        if result and self.fts_store:
            try:
                self.fts_store.delete(memory_id)
            except Exception as e:
                logger.warning("FTS cleanup failed for %s: %s", memory_id, e)
        return result
    
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID with full provenance."""
        return await self.vector_store.get(memory_id)
    
    async def get_stats(self) -> dict:
        """Get memory statistics.

        Delegates to vector_store.get_stats() which computes aggregates
        efficiently (paginated by default, native queries for SQL-backed
        stores).
        """
        return await self.vector_store.get_stats()

    @staticmethod
    def _filter_superseded(results: list[RecallResult]) -> list[RecallResult]:
        """Remove memories that are superseded by corrections in the result set."""
        superseded_ids: set[str] = {
            r.memory.supersedes for r in results if r.memory.supersedes
        }
        if not superseded_ids:
            return results
        return [r for r in results if r.memory.id not in superseded_ids]


def create_memory_service(
    instance_id: Optional[str] = None,
    db_path: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_dimensions: Optional[int] = None,
    hybrid_search: bool = True,
    hybrid_vector_weight: float = 0.7,
    hybrid_text_weight: float = 0.3,
    hybrid_candidate_multiplier: int = 4,
    reranking: str = "heuristic",
    recency_decay_days: float = 30.0,
    tag_boost_weight: float = 0.1,
    rerank_pool_multiplier: int = 2,
) -> TribalMemoryService:
    """Factory function to create a memory service with sensible defaults.
    
    Args:
        instance_id: Unique identifier for this agent instance.
        db_path: Path for LanceDB persistent storage. If None, uses in-memory.
        openai_api_key: API key. Falls back to OPENAI_API_KEY env var.
            Not required for local models (when api_base is set).
        api_base: Base URL for the embedding API.
            For Ollama: "http://localhost:11434/v1"
        embedding_model: Embedding model name. Default: "text-embedding-3-small".
        embedding_dimensions: Embedding output dimensions. Default: 1536.
        hybrid_search: Enable BM25 hybrid search (default: True).
        hybrid_vector_weight: Weight for vector similarity (default: 0.7).
        hybrid_text_weight: Weight for BM25 text score (default: 0.3).
        hybrid_candidate_multiplier: Multiplier for candidate pool size
            (default: 4). Retrieves 4× limit from each source before
            merging.
        reranking: Reranking mode: "auto", "cross-encoder", "heuristic", "none"
            (default: "heuristic").
        recency_decay_days: Half-life for recency boost (default: 30.0).
        tag_boost_weight: Weight for tag match boost (default: 0.1).
        rerank_pool_multiplier: How many candidates to give the reranker
            (N × limit). Default: 2.
    
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
    
    kwargs: dict = {"api_key": openai_api_key}
    if api_base is not None:
        kwargs["api_base"] = api_base
    if embedding_model is not None:
        kwargs["model"] = embedding_model
    if embedding_dimensions is not None:
        kwargs["dimensions"] = embedding_dimensions
    
    embedding_service = OpenAIEmbeddingService(**kwargs)
    
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

    # Create FTS store for hybrid search (co-located with LanceDB)
    fts_store = None
    if hybrid_search and db_path:
        try:
            fts_db_path = str(Path(db_path) / "fts_index.db")
            fts_store = FTSStore(fts_db_path)
            if fts_store.is_available():
                logger.info("Hybrid search enabled (SQLite FTS5)")
            else:
                logger.warning(
                    "FTS5 not available in SQLite build. "
                    "Hybrid search disabled, using vector-only."
                )
                fts_store = None
        except Exception as e:
            logger.warning(f"FTS store init failed: {e}. Using vector-only.")
            fts_store = None
    
    # Create reranker
    from ..server.config import SearchConfig
    search_config = SearchConfig(
        reranking=reranking,
        recency_decay_days=recency_decay_days,
        tag_boost_weight=tag_boost_weight,
    )
    reranker = create_reranker(search_config)
    
    return TribalMemoryService(
        instance_id=instance_id,
        embedding_service=embedding_service,
        vector_store=vector_store,
        fts_store=fts_store,
        hybrid_search=hybrid_search,
        hybrid_vector_weight=hybrid_vector_weight,
        hybrid_text_weight=hybrid_text_weight,
        hybrid_candidate_multiplier=hybrid_candidate_multiplier,
        reranker=reranker,
        rerank_pool_multiplier=rerank_pool_multiplier,
    )
