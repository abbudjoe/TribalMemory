"""Tribal Memory Service - Main API for agents."""

import asyncio
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
from .graph_store import GraphStore, EntityExtractor
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
    
    # Graph expansion scoring constants
    GRAPH_1HOP_SCORE = 0.85  # Score for direct entity mentions
    GRAPH_2HOP_SCORE = 0.70  # Score for connected entity mentions
    GRAPH_EXPANSION_BUFFER = 2  # Multiplier for candidate pool before fetching
    
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
        graph_store: Optional[GraphStore] = None,
        graph_enabled: bool = True,
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
        self.graph_store = graph_store
        self.graph_enabled = graph_enabled and graph_store is not None
        self.entity_extractor = EntityExtractor() if self.graph_enabled else None
        
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
        
        # Extract and store entities for graph-enriched search
        if result.success and self.graph_enabled and self.entity_extractor:
            try:
                entities, relationships = self.entity_extractor.extract_with_relationships(
                    content
                )
                for entity in entities:
                    self.graph_store.add_entity(entity, memory_id=entry.id)
                for rel in relationships:
                    self.graph_store.add_relationship(rel, memory_id=entry.id)
                if entities:
                    logger.debug(
                        "Extracted entities: %s, relationships: %s from %s",
                        [e.name for e in entities],
                        [(r.source, r.relation_type, r.target) for r in relationships],
                        entry.id
                    )
            except Exception as e:
                logger.warning("Graph indexing failed for %s: %s", entry.id, e)
        
        return result
    
    async def recall(
        self,
        query: str,
        limit: int = 5,
        min_relevance: float = 0.7,
        tags: Optional[list[str]] = None,
        graph_expansion: bool = True,
    ) -> list[RecallResult]:
        """Recall relevant memories using hybrid search with optional graph expansion.
        
        When hybrid search is enabled (FTS store available), combines
        vector similarity with BM25 keyword matching for better results.
        Falls back to vector-only search when FTS is unavailable.
        
        When graph expansion is enabled, entities are extracted from the query
        and the candidate pool is expanded via entity graph traversal.
        
        Args:
            query: Natural language query
            limit: Maximum results
            min_relevance: Minimum similarity score
            tags: Filter by tags (e.g., ["work", "preferences"])
            graph_expansion: Expand candidates via entity graph (default True)
        
        Returns:
            List of RecallResult objects with retrieval_method indicating source:
            - "vector": Pure vector similarity search
            - "hybrid": Vector + BM25 merge
            - "graph": Entity graph traversal (1-hop or 2-hop)
        """
        try:
            query_embedding = await self.embedding_service.embed(query)
        except Exception:
            return []
        
        filters = {"tags": tags} if tags else None

        if self.hybrid_search and self.fts_store:
            results = await self._hybrid_recall(
                query, query_embedding, limit, min_relevance, filters
            )
        else:
            # Vector-only fallback
            vector_results = await self.vector_store.recall(
                query_embedding,
                limit=limit,
                min_similarity=min_relevance,
                filters=filters,
            )
            # Mark as vector retrieval
            results = [
                RecallResult(
                    memory=r.memory,
                    similarity_score=r.similarity_score,
                    retrieval_time_ms=r.retrieval_time_ms,
                    retrieval_method="vector",
                )
                for r in vector_results
            ]
        
        # Graph expansion: find additional memories via entity connections
        if graph_expansion and self.graph_enabled and self.entity_extractor:
            results = await self._expand_via_graph(
                query, results, limit, min_relevance
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
        
        # Add cached vector hits (mark as hybrid since we used BM25 merge)
        for m, recall_result in cached_hits:
            candidates.append(RecallResult(
                memory=recall_result.memory,
                similarity_score=m["final_score"],
                retrieval_time_ms=recall_result.retrieval_time_ms,
                retrieval_method="hybrid",
            ))
        
        # Add fetched BM25-only hits
        for m, entry in zip(bm25_only_ids, fetched_entries):
            if entry and m["final_score"] >= min_relevance * 0.5:
                candidates.append(RecallResult(
                    memory=entry,
                    similarity_score=m["final_score"],
                    retrieval_time_ms=0,
                    retrieval_method="hybrid",
                ))

        # 6. Rerank candidates
        reranked = self.reranker.rerank(query, candidates, top_k=limit)

        return self._filter_superseded(reranked)

    async def _expand_via_graph(
        self,
        query: str,
        existing_results: list[RecallResult],
        limit: int,
        min_relevance: float,
    ) -> list[RecallResult]:
        """Expand recall candidates via entity graph traversal.
        
        Extracts entities from the query, finds memories connected to those
        entities via the graph, and merges them with existing results.
        
        Args:
            query: The original query string.
            existing_results: Results from vector/hybrid search.
            limit: Maximum total results.
            min_relevance: Minimum relevance threshold (filters graph results too).
            
        Returns:
            Combined results with graph-expanded memories, sorted by score.
        """
        # Extract entities from query
        query_entities = self.entity_extractor.extract(query)
        if not query_entities:
            return existing_results
        
        # Collect memory IDs from existing results to avoid duplicates
        existing_ids = {r.memory.id for r in existing_results}
        
        # Find memories connected to query entities via graph
        graph_memory_ids: set[str] = set()
        entity_to_hops: dict[str, int] = {}  # Track hop distance for scoring
        
        for entity in query_entities:
            # Direct mentions (1 hop)
            direct_ids = self.graph_store.get_memories_for_entity(entity.name)
            for mid in direct_ids:
                if mid not in existing_ids:
                    graph_memory_ids.add(mid)
                    # Use setdefault to preserve shortest path (1-hop takes precedence)
                    entity_to_hops.setdefault(mid, 1)
            
            # Connected entities (2 hops)
            connected = self.graph_store.find_connected(entity.name, hops=1)
            for connected_entity in connected:
                connected_ids = self.graph_store.get_memories_for_entity(
                    connected_entity.name
                )
                for mid in connected_ids:
                    if mid not in existing_ids:
                        graph_memory_ids.add(mid)
                        # Use setdefault to preserve shortest path
                        entity_to_hops.setdefault(mid, 2)
        
        if not graph_memory_ids:
            return existing_results
        
        # Cap graph candidates to prevent memory leak (#2)
        max_graph_candidates = limit * self.GRAPH_EXPANSION_BUFFER
        if len(graph_memory_ids) > max_graph_candidates:
            # Prioritize 1-hop over 2-hop when capping
            one_hop_ids = [mid for mid in graph_memory_ids if entity_to_hops[mid] == 1]
            two_hop_ids = [mid for mid in graph_memory_ids if entity_to_hops[mid] == 2]
            
            capped_ids: list[str] = []
            capped_ids.extend(one_hop_ids[:max_graph_candidates])
            remaining = max_graph_candidates - len(capped_ids)
            if remaining > 0:
                capped_ids.extend(two_hop_ids[:remaining])
            
            graph_memory_ids = set(capped_ids)
        
        # Fetch graph-connected memories
        graph_results: list[RecallResult] = []
        for memory_id in graph_memory_ids:
            entry = await self.vector_store.get(memory_id)
            if entry:
                # Score based on hop distance using class constants
                hops = entity_to_hops[memory_id]  # Fail fast if logic is wrong (#3)
                graph_score = (
                    self.GRAPH_1HOP_SCORE if hops == 1 
                    else self.GRAPH_2HOP_SCORE
                )
                
                # Filter by min_relevance (#4)
                if graph_score >= min_relevance:
                    graph_results.append(RecallResult(
                        memory=entry,
                        similarity_score=graph_score,
                        retrieval_time_ms=0,
                        retrieval_method="graph",
                    ))
        
        # Combine existing + graph results (#10: single sort, no redundant pre-sort)
        combined = existing_results + graph_results
        combined.sort(key=lambda r: r.similarity_score, reverse=True)
        
        return combined[:limit]
    
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
        if result and self.graph_store:
            try:
                self.graph_store.delete_memory(memory_id)
            except Exception as e:
                logger.warning("Graph cleanup failed for %s: %s", memory_id, e)
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

    async def recall_entity(
        self,
        entity_name: str,
        hops: int = 1,
        limit: int = 10,
    ) -> list[RecallResult]:
        """Recall all memories associated with an entity and its connections.
        
        This enables entity-centric queries like:
        - "Tell me everything about auth-service"
        - "What do we know about PostgreSQL?"
        
        Args:
            entity_name: Name of the entity to query
            hops: Number of relationship hops to traverse (1 = direct only)
            limit: Maximum results to return
        
        Returns:
            List of recall results for memories mentioning the entity or connected entities
        """
        if not self.graph_enabled:
            logger.warning("Graph search not enabled, returning empty results")
            return []
        
        # Get memories directly mentioning the entity
        direct_memories = set(self.graph_store.get_memories_for_entity(entity_name))
        
        # Get memories for connected entities (if hops > 0)
        if hops > 0:
            connected = self.graph_store.find_connected(entity_name, hops=hops)
            for entity in connected:
                direct_memories.update(
                    self.graph_store.get_memories_for_entity(entity.name)
                )
        
        if not direct_memories:
            return []
        
        # Fetch full memory entries
        results: list[RecallResult] = []
        for memory_id in list(direct_memories)[:limit]:
            entry = await self.vector_store.get(memory_id)
            if entry:
                results.append(RecallResult(
                    memory=entry,
                    similarity_score=1.0,  # Entity match confidence (exact)
                    retrieval_time_ms=0,
                    retrieval_method="entity",
                ))
        
        return results

    def get_entity_graph(
        self,
        entity_name: str,
        hops: int = 2,
    ) -> dict:
        """Get the relationship graph around an entity.
        
        Returns a dict with:
        - entities: list of connected entities with types
        - relationships: list of relationships
        
        Useful for visualization and debugging.
        """
        if not self.graph_enabled:
            return {"entities": [], "relationships": []}
        
        connected = self.graph_store.find_connected(
            entity_name, hops=hops, include_source=True
        )
        
        entities = [
            {"name": e.name, "type": e.entity_type}
            for e in connected
        ]
        
        # Get relationships for all entities
        relationships = []
        seen_rels = set()
        for entity in connected:
            for rel in self.graph_store.get_relationships_for_entity(entity.name):
                rel_key = (rel.source, rel.target, rel.relation_type)
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    relationships.append({
                        "source": rel.source,
                        "target": rel.target,
                        "type": rel.relation_type,
                    })
        
        return {"entities": entities, "relationships": relationships}


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
    
    # Create graph store for entity-enriched search (co-located with LanceDB)
    graph_store = None
    if db_path:
        try:
            graph_db_path = str(Path(db_path) / "graph.db")
            graph_store = GraphStore(graph_db_path)
            logger.info("Graph store enabled (SQLite)")
        except Exception as e:
            logger.warning(f"Graph store init failed: {e}. Graph search disabled.")
            graph_store = None
    
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
        graph_store=graph_store,
        graph_enabled=graph_store is not None,
    )
