"""Tests for similarity score calculation and min_relevance filtering.

Addresses Issue #153: min_relevance filter bug and scores exceeding 1.0.
"""
import pytest
import tempfile
from pathlib import Path
import numpy as np

from tribalmemory.services.vector_store import LanceDBVectorStore, InMemoryVectorStore
from tribalmemory.testing.mocks import MockEmbeddingService
from tribalmemory.interfaces import MemoryEntry, MemorySource


class TestSimilarityCalculation:
    """Test that similarity scores are correctly calculated and bounded."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service with known normalized vectors."""
        service = MockEmbeddingService(embedding_dim=1536)
        # Add dimensions property for compatibility with LanceDBVectorStore
        service.dimensions = 1536
        
        # Override with known normalized vectors for predictable results
        async def embed_mock(text: str) -> list[float]:
            # Create 1536-dim normalized vectors with predictable patterns
            dim = 1536
            base_vector = np.zeros(dim)
            
            if "vector1" in text:
                # Put all weight in first dimension
                base_vector[0] = 1.0
            elif "vector2" in text:
                # Very similar to vector1 (99% in first dim, 1% in second)
                base_vector[0] = 0.99
                base_vector[1] = 0.1
            elif "vector3" in text:
                # Orthogonal to vector1 (all weight in second dimension)
                base_vector[1] = 1.0
            else:
                # Random but reproducible based on text content
                import hashlib
                seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
                np.random.seed(seed)
                base_vector = np.random.rand(dim)
            
            # Normalize to unit length
            norm = np.linalg.norm(base_vector)
            if norm > 0:
                base_vector = base_vector / norm
            
            return base_vector.tolist()
        
        service.embed = embed_mock
        return service

    @pytest.mark.asyncio
    async def test_lancedb_similarity_scores_bounded_to_unit_range(self, mock_embedding_service):
        """Verify LanceDB similarity scores are always in [0, 1] range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceDBVectorStore(
                embedding_service=mock_embedding_service,
                db_path=Path(tmpdir),
            )

            # Store test entries with known embeddings
            entry1 = MemoryEntry(
                content="This is vector1 for testing",
                source_type=MemorySource.USER_EXPLICIT,
                source_instance="test",
            )
            await store.store(entry1)

            entry2 = MemoryEntry(
                content="This is vector2 for testing",
                source_type=MemorySource.USER_EXPLICIT,
                source_instance="test",
            )
            await store.store(entry2)

            entry3 = MemoryEntry(
                content="This is vector3 for testing",
                source_type=MemorySource.USER_EXPLICIT,
                source_instance="test",
            )
            await store.store(entry3)

            # Query with vector1 - should match itself perfectly
            query_embedding = await mock_embedding_service.embed("vector1")
            results = await store.recall(
                query_embedding=query_embedding,
                limit=10,
                min_similarity=0.0,  # Get all results
            )

            # All similarity scores must be in [0, 1] range
            for result in results:
                assert 0.0 <= result.similarity_score <= 1.0, (
                    f"Similarity score {result.similarity_score} out of [0, 1] range"
                )

            # Top result should be near-perfect match (vector1 with itself)
            assert results[0].similarity_score >= 0.99, (
                f"Self-similarity should be ~1.0, got {results[0].similarity_score}"
            )

    @pytest.mark.asyncio
    async def test_lancedb_min_relevance_filter_at_0_9(self, mock_embedding_service):
        """Verify min_relevance=0.9 strictly filters results (Issue #153)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceDBVectorStore(
                embedding_service=mock_embedding_service,
                db_path=Path(tmpdir),
            )

            # Store entries with varying similarity to query
            await store.store(MemoryEntry(
                content="This is vector1 exact match",
                source_type=MemorySource.USER_EXPLICIT,
                source_instance="test",
            ))
            await store.store(MemoryEntry(
                content="This is vector2 similar",
                source_type=MemorySource.USER_EXPLICIT,
                source_instance="test",
            ))
            await store.store(MemoryEntry(
                content="This is vector3 orthogonal",
                source_type=MemorySource.USER_EXPLICIT,
                source_instance="test",
            ))

            # Query with min_relevance=0.9
            query_embedding = await mock_embedding_service.embed("vector1")
            results = await store.recall(
                query_embedding=query_embedding,
                limit=10,
                min_similarity=0.9,
            )

            # ALL returned results must have similarity >= 0.9
            for result in results:
                assert result.similarity_score >= 0.9, (
                    f"Result with similarity {result.similarity_score} "
                    f"returned despite min_relevance=0.9"
                )

    @pytest.mark.asyncio
    async def test_lancedb_min_relevance_filter_at_0_7(self, mock_embedding_service):
        """Verify min_relevance=0.7 strictly filters results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceDBVectorStore(
                embedding_service=mock_embedding_service,
                db_path=Path(tmpdir),
            )

            # Store test entries
            await store.store(MemoryEntry(
                content="This is vector1 test",
                source_type=MemorySource.USER_EXPLICIT,
                source_instance="test",
            ))
            await store.store(MemoryEntry(
                content="This is vector2 test",
                source_type=MemorySource.USER_EXPLICIT,
                source_instance="test",
            ))
            await store.store(MemoryEntry(
                content="This is vector3 test",
                source_type=MemorySource.USER_EXPLICIT,
                source_instance="test",
            ))

            # Query with min_relevance=0.7
            query_embedding = await mock_embedding_service.embed("vector1")
            results = await store.recall(
                query_embedding=query_embedding,
                limit=10,
                min_similarity=0.7,
            )

            # ALL returned results must have similarity >= 0.7
            for result in results:
                assert result.similarity_score >= 0.7, (
                    f"Result with similarity {result.similarity_score} "
                    f"returned despite min_relevance=0.7"
                )

    @pytest.mark.asyncio
    async def test_inmemory_similarity_scores_bounded(self, mock_embedding_service):
        """Verify in-memory store also has bounded similarity scores."""
        store = InMemoryVectorStore(embedding_service=mock_embedding_service)

        # Store test entries
        await store.store(MemoryEntry(
            content="This is vector1 for testing",
            source_type=MemorySource.USER_EXPLICIT,
            source_instance="test",
        ))
        await store.store(MemoryEntry(
            content="This is vector2 for testing",
            source_type=MemorySource.USER_EXPLICIT,
            source_instance="test",
        ))

        # Query
        query_embedding = await mock_embedding_service.embed("vector1")
        results = await store.recall(
            query_embedding=query_embedding,
            limit=10,
            min_similarity=0.0,
        )

        # All similarity scores must be in [0, 1] range
        for result in results:
            assert 0.0 <= result.similarity_score <= 1.0, (
                f"Similarity score {result.similarity_score} out of [0, 1] range"
            )

    @pytest.mark.asyncio
    async def test_inmemory_min_relevance_filter(self, mock_embedding_service):
        """Verify in-memory store respects min_relevance filter."""
        store = InMemoryVectorStore(embedding_service=mock_embedding_service)

        # Store entries
        await store.store(MemoryEntry(
            content="This is vector1 test",
            source_type=MemorySource.USER_EXPLICIT,
            source_instance="test",
        ))
        await store.store(MemoryEntry(
            content="This is vector3 test",
            source_type=MemorySource.USER_EXPLICIT,
            source_instance="test",
        ))

        # Query with high threshold
        query_embedding = await mock_embedding_service.embed("vector1")
        results = await store.recall(
            query_embedding=query_embedding,
            limit=10,
            min_similarity=0.9,
        )

        # All results must meet threshold
        for result in results:
            assert result.similarity_score >= 0.9

    @pytest.mark.asyncio
    async def test_similarity_ordering(self, mock_embedding_service):
        """Verify results are ordered by similarity score (descending)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceDBVectorStore(
                embedding_service=mock_embedding_service,
                db_path=Path(tmpdir),
            )

            # Store entries with varying similarity
            await store.store(MemoryEntry(
                content="This is vector1 identical",
                source_type=MemorySource.USER_EXPLICIT,
                source_instance="test",
            ))
            await store.store(MemoryEntry(
                content="This is vector2 similar",
                source_type=MemorySource.USER_EXPLICIT,
                source_instance="test",
            ))
            await store.store(MemoryEntry(
                content="This is vector3 different",
                source_type=MemorySource.USER_EXPLICIT,
                source_instance="test",
            ))

            # Query
            query_embedding = await mock_embedding_service.embed("vector1")
            results = await store.recall(
                query_embedding=query_embedding,
                limit=10,
                min_similarity=0.0,
            )

            # Verify descending order
            for i in range(len(results) - 1):
                assert results[i].similarity_score >= results[i + 1].similarity_score, (
                    f"Results not properly ordered: {results[i].similarity_score} < "
                    f"{results[i + 1].similarity_score}"
                )
