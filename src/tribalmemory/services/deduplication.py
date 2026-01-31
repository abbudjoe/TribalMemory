"""Semantic Deduplication Service."""

from typing import Optional

from ..interfaces import IDeduplicationService, IVectorStore, IEmbeddingService


class SemanticDeduplicationService(IDeduplicationService):
    """Semantic deduplication using embedding similarity."""
    
    def __init__(
        self,
        vector_store: IVectorStore,
        embedding_service: IEmbeddingService,
        exact_threshold: float = 0.98,
        near_threshold: float = 0.90,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.exact_threshold = exact_threshold
        self.near_threshold = near_threshold
    
    async def is_duplicate(
        self,
        content: str,
        embedding: list[float],
        threshold: Optional[float] = None
    ) -> tuple[bool, Optional[str]]:
        """Check if content is a duplicate.
        
        Returns:
            Tuple of (is_duplicate, duplicate_of_id)
        """
        threshold = threshold or self.exact_threshold
        
        results = await self.vector_store.recall(
            embedding,
            limit=1,
            min_similarity=threshold
        )
        
        if results and results[0].similarity_score >= threshold:
            return True, results[0].memory.id
        
        return False, None
    
    async def find_similar(
        self,
        content: str,
        embedding: list[float],
        threshold: Optional[float] = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Find similar memories.
        
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        threshold = threshold or self.near_threshold
        
        results = await self.vector_store.recall(
            embedding,
            limit=limit,
            min_similarity=threshold
        )
        
        return [(r.memory.id, r.similarity_score) for r in results]
    
    async def get_duplicate_report(
        self,
        content: str,
        embedding: list[float]
    ) -> dict:
        """Get detailed duplicate analysis report.
        
        Args:
            content: Text content to analyze.
            embedding: Pre-computed embedding vector for the content.
        
        Returns:
            Dict with keys:
                - is_duplicate (bool): True if above exact threshold
                - is_near_duplicate (bool): True if above near threshold
                - top_match (dict|None): Best matching memory with id, content preview, similarity
                - candidates (list): Top 5 similar memories with id, similarity, content_preview
        """
        results = await self.vector_store.recall(
            embedding,
            limit=5,
            min_similarity=0.7
        )
        
        if not results:
            return {
                "is_duplicate": False,
                "is_near_duplicate": False,
                "top_match": None,
                "candidates": []
            }
        
        top = results[0]
        
        return {
            "is_duplicate": top.similarity_score >= self.exact_threshold,
            "is_near_duplicate": top.similarity_score >= self.near_threshold,
            "top_match": {
                "id": top.memory.id,
                "content": top.memory.content[:200] + "..." if len(top.memory.content) > 200 else top.memory.content,
                "similarity": top.similarity_score,
            },
            "candidates": [
                {"id": r.memory.id, "similarity": r.similarity_score, "content_preview": r.memory.content[:100]}
                for r in results
            ],
        }
