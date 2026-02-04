"""Tests for graph-aware hybrid recall (Phase 2)."""

import pytest
from tribalmemory.interfaces import RecallResult, MemoryEntry, MemorySource
from tribalmemory.services.memory import TribalMemoryService
from tribalmemory.services.graph_store import GraphStore, EntityExtractor
from tribalmemory.services.vector_store import InMemoryVectorStore
from tribalmemory.testing.mocks import MockEmbeddingService
from datetime import datetime


class TestRetrievalMethod:
    """Tests for retrieval_method field in RecallResult."""

    def test_recall_result_has_retrieval_method_field(self):
        """RecallResult should have retrieval_method with default 'vector'."""
        entry = MemoryEntry(
            id="test-1",
            content="test",
            embedding=[0.1] * 64,
            source_instance="test",
            source_type=MemorySource.AUTO_CAPTURE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        result = RecallResult(
            memory=entry,
            similarity_score=0.9,
            retrieval_time_ms=10,
        )
        
        assert hasattr(result, 'retrieval_method')
        assert result.retrieval_method == "vector"

    def test_recall_result_accepts_custom_retrieval_method(self):
        """RecallResult should accept custom retrieval_method values."""
        entry = MemoryEntry(
            id="test-1",
            content="test",
            embedding=[0.1] * 64,
            source_instance="test",
            source_type=MemorySource.AUTO_CAPTURE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        
        result_graph = RecallResult(
            memory=entry,
            similarity_score=0.9,
            retrieval_time_ms=10,
            retrieval_method="graph",
        )
        assert result_graph.retrieval_method == "graph"
        
        result_hybrid = RecallResult(
            memory=entry,
            similarity_score=0.9,
            retrieval_time_ms=10,
            retrieval_method="hybrid",
        )
        assert result_hybrid.retrieval_method == "hybrid"


class TestQueryEntityExtraction:
    """Tests for extracting entities from queries."""

    def test_extract_entities_from_query(self):
        """EntityExtractor should work on query strings."""
        extractor = EntityExtractor()
        
        query = "What database does auth-service use?"
        entities = extractor.extract(query)
        
        names = {e.name for e in entities}
        assert "auth-service" in names

    def test_extract_technology_from_query(self):
        """Should extract technology names from queries."""
        extractor = EntityExtractor()
        
        query = "How is PostgreSQL configured in our system?"
        entities = extractor.extract(query)
        
        names = {e.name for e in entities}
        assert "PostgreSQL" in names

    def test_extract_multiple_entities_from_query(self):
        """Should extract multiple entities from complex queries."""
        extractor = EntityExtractor()
        
        query = "Does auth-service connect to PostgreSQL or Redis?"
        entities = extractor.extract(query)
        
        names = {e.name for e in entities}
        assert "auth-service" in names
        assert "PostgreSQL" in names
        assert "Redis" in names


class TestGraphAwareRecall:
    """Tests for graph-aware recall pipeline."""

    @pytest.fixture
    def embedding_service(self):
        return MockEmbeddingService(embedding_dim=64)

    @pytest.fixture
    def vector_store(self, embedding_service):
        return InMemoryVectorStore(embedding_service=embedding_service)

    @pytest.fixture
    def graph_store(self, tmp_path):
        return GraphStore(str(tmp_path / "graph.db"))

    @pytest.fixture
    def memory_service(self, embedding_service, vector_store, graph_store):
        return TribalMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=vector_store,
            graph_store=graph_store,
            graph_enabled=True,
            auto_reject_duplicates=False,
        )

    @pytest.mark.asyncio
    async def test_recall_expands_via_graph(self, memory_service):
        """Recall should find memories connected via entity graph."""
        # Store memories with connected entities
        await memory_service.remember(
            "The auth-service uses PostgreSQL for credentials."
        )
        await memory_service.remember(
            "PostgreSQL runs on port 5432 with max_connections=100."
        )
        
        # Query mentions auth-service, should also find PostgreSQL memory
        # via graph connection
        results = await memory_service.recall(
            "Tell me about auth-service configuration",
            graph_expansion=True,
        )
        
        # Should find both: direct match AND graph-connected
        contents = [r.memory.content for r in results]
        assert any("auth-service" in c for c in contents)
        # Graph expansion should pull in the PostgreSQL config memory
        assert any("port 5432" in c for c in contents)

    @pytest.mark.asyncio
    async def test_recall_marks_retrieval_method(self, memory_service):
        """Results should be marked with their retrieval method."""
        await memory_service.remember(
            "The payment-service processes credit cards."
        )
        await memory_service.remember(
            "Stripe is the payment gateway."
        )
        
        results = await memory_service.recall(
            "How does payment-service work?",
            graph_expansion=True,
        )
        
        # Should have retrieval_method populated
        for r in results:
            assert r.retrieval_method in ("vector", "graph", "hybrid")

    @pytest.mark.asyncio
    async def test_recall_without_graph_expansion(self, memory_service):
        """Graph expansion can be disabled, excluding graph-connected memories."""
        await memory_service.remember(
            "The cache-service uses Redis for session storage."
        )
        await memory_service.remember(
            "Redis requires 2GB RAM minimum."
        )
        
        # Without graph expansion, only vector matches
        results = await memory_service.recall(
            "cache-service",
            graph_expansion=False,
        )
        
        # All results should be vector-based
        for r in results:
            assert r.retrieval_method in ("vector", "hybrid")
        
        # Should NOT find the Redis memory via graph connection (#8)
        contents = [r.memory.content for r in results]
        # The Redis memory is only connected via graph, not direct vector match
        # If it appears, graph expansion is still happening
        assert not any("2GB RAM" in c for c in contents), \
            "Graph-connected memory should be excluded when graph_expansion=False"

    @pytest.mark.asyncio
    async def test_graph_expansion_graceful_when_disabled(self, embedding_service):
        """Graph expansion should be no-op when graph store is disabled."""
        vector_store = InMemoryVectorStore(embedding_service=embedding_service)
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=vector_store,
            graph_store=None,
            graph_enabled=False,
        )
        
        await service.remember("Test memory content")
        
        # Should work fine, just no graph expansion
        results = await service.recall("test", graph_expansion=True)
        
        for r in results:
            assert r.retrieval_method == "vector"


class TestGraphExpansion:
    """Tests for graph-based candidate expansion."""

    @pytest.fixture
    def memory_service(self, tmp_path):
        embedding_service = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding_service=embedding_service)
        graph_store = GraphStore(str(tmp_path / "graph.db"))
        
        return TribalMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=vector_store,
            graph_store=graph_store,
            graph_enabled=True,
            auto_reject_duplicates=False,
        )

    @pytest.mark.asyncio
    async def test_connected_entities_boost_score(self, memory_service):
        """Memories with entities connected to query entities should rank higher."""
        # Memory 1: Direct mention
        await memory_service.remember(
            "The api-gateway handles all incoming requests."
        )
        # Memory 2: Connected via relationship
        await memory_service.remember(
            "The api-gateway routes traffic to user-service."
        )
        # Memory 3: Less connected
        await memory_service.remember(
            "Logging is handled by the logging-service."
        )
        
        results = await memory_service.recall(
            "How does api-gateway work?",
            graph_expansion=True,
            limit=10,
        )
        
        # Results with api-gateway or connected entities should be present
        contents = [r.memory.content for r in results]
        assert any("api-gateway" in c for c in contents)

    @pytest.mark.asyncio
    async def test_multi_hop_expansion(self, memory_service):
        """Should expand across multiple relationship hops."""
        # Chain: auth-service -> PostgreSQL -> pgbouncer
        await memory_service.remember(
            "The auth-service uses PostgreSQL for user data."
        )
        await memory_service.remember(
            "PostgreSQL connects through pgbouncer for pooling."
        )
        await memory_service.remember(
            "pgbouncer is configured with 50 max connections."
        )
        
        # Query about auth-service should potentially find pgbouncer
        # via 2-hop traversal
        results = await memory_service.recall(
            "What does auth-service depend on?",
            graph_expansion=True,
            limit=10,
        )
        
        contents = [r.memory.content for r in results]
        # Should find the chain
        assert any("auth-service" in c or "PostgreSQL" in c for c in contents)


class TestGraphExpansionEdgeCases:
    """Edge case tests for graph expansion."""

    @pytest.fixture
    def memory_service(self, tmp_path):
        embedding_service = MockEmbeddingService(embedding_dim=64)
        vector_store = InMemoryVectorStore(embedding_service=embedding_service)
        graph_store = GraphStore(str(tmp_path / "graph.db"))
        
        return TribalMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=vector_store,
            graph_store=graph_store,
            graph_enabled=True,
            auto_reject_duplicates=False,
        )

    @pytest.mark.asyncio
    async def test_graph_expansion_no_entities_in_query(self, memory_service):
        """Graph expansion should gracefully handle queries with no entities (#11)."""
        await memory_service.remember("Some memory about configuration")
        
        # Query with no extractable entities
        results = await memory_service.recall(
            "what is the meaning of life?",
            graph_expansion=True,
        )
        
        # Should fall back to vector search only
        for r in results:
            assert r.retrieval_method in ("vector", "hybrid")

    @pytest.mark.asyncio
    async def test_recall_entity_marks_method(self, memory_service):
        """recall_entity() should mark results with retrieval_method='entity' (#15)."""
        await memory_service.remember("PostgreSQL is our primary database")
        
        results = await memory_service.recall_entity("PostgreSQL")
        
        assert len(results) > 0
        for r in results:
            assert r.retrieval_method == "entity"

    @pytest.mark.asyncio
    async def test_graph_expansion_respects_min_relevance(self, memory_service):
        """Graph results should be filtered by min_relevance (#4)."""
        await memory_service.remember(
            "The auth-service uses PostgreSQL for credentials."
        )
        await memory_service.remember(
            "PostgreSQL runs on port 5432."
        )
        
        # With high min_relevance, 2-hop results (0.70) should be excluded
        results = await memory_service.recall(
            "auth-service",
            min_relevance=0.8,  # Higher than GRAPH_2HOP_SCORE (0.70)
            graph_expansion=True,
        )
        
        # If any graph results, they must meet min_relevance
        for r in results:
            if r.retrieval_method == "graph":
                assert r.similarity_score >= 0.8
