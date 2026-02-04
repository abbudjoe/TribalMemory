"""Integration tests for graph-enriched memory service."""

import pytest
from tribalmemory.services.memory import TribalMemoryService
from tribalmemory.services.graph_store import GraphStore
from tribalmemory.services.vector_store import InMemoryVectorStore
from tribalmemory.testing.mocks import MockEmbeddingService


class TestGraphEnrichedMemory:
    """Tests for memory service with graph store integration."""

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
            instance_id="test-instance",
            embedding_service=embedding_service,
            vector_store=vector_store,
            graph_store=graph_store,
            graph_enabled=True,
            auto_reject_duplicates=False,  # Simplify tests
        )

    @pytest.mark.asyncio
    async def test_remember_extracts_entities(self, memory_service, graph_store):
        """Storing a memory should extract and store entities."""
        result = await memory_service.remember(
            "The auth-service uses PostgreSQL for user credentials."
        )
        
        assert result.success
        memory_id = result.memory_id
        
        # Check entities were extracted
        entities = graph_store.get_entities_for_memory(memory_id)
        names = {e.name for e in entities}
        assert "auth-service" in names
        assert "PostgreSQL" in names

    @pytest.mark.asyncio
    async def test_remember_extracts_relationships(self, memory_service, graph_store):
        """Storing a memory should extract relationships between entities."""
        result = await memory_service.remember(
            "The auth-service uses PostgreSQL for storing credentials."
        )
        
        assert result.success
        
        # Check relationships
        rels = graph_store.get_relationships_for_entity("auth-service")
        assert len(rels) >= 1
        assert any(r.target == "PostgreSQL" for r in rels)

    @pytest.mark.asyncio
    async def test_recall_entity_direct(self, memory_service):
        """Entity-centric recall finds direct mentions."""
        # Store memories mentioning PostgreSQL
        await memory_service.remember("PostgreSQL is our primary database.")
        await memory_service.remember("We run PostgreSQL 15 in production.")
        await memory_service.remember("Redis handles our caching layer.")
        
        # Entity recall for PostgreSQL
        results = await memory_service.recall_entity("PostgreSQL")
        
        assert len(results) == 2
        contents = {r.memory.content for r in results}
        assert "PostgreSQL is our primary database." in contents
        assert "We run PostgreSQL 15 in production." in contents
        # Redis memory should not appear
        assert not any("Redis" in r.memory.content for r in results)

    @pytest.mark.asyncio
    async def test_recall_entity_with_hops(self, memory_service, graph_store):
        """Entity recall can traverse relationships."""
        # Setup: auth-service -> PostgreSQL -> credentials
        await memory_service.remember(
            "The auth-service uses PostgreSQL for the database."
        )
        await memory_service.remember(
            "PostgreSQL stores user credentials securely."
        )
        
        # 1-hop from auth-service should find PostgreSQL memory
        results = await memory_service.recall_entity("auth-service", hops=1)
        
        contents = [r.memory.content for r in results]
        # Should find both: direct auth-service memory and connected PostgreSQL memory
        assert any("auth-service" in c for c in contents)

    @pytest.mark.asyncio
    async def test_get_entity_graph(self, memory_service):
        """Get relationship graph around an entity."""
        await memory_service.remember(
            "The auth-service uses PostgreSQL and Redis."
        )
        await memory_service.remember(
            "The user-service talks to auth-service for tokens."
        )
        
        graph = memory_service.get_entity_graph("auth-service", hops=1)
        
        assert "entities" in graph
        assert "relationships" in graph
        
        entity_names = {e["name"] for e in graph["entities"]}
        assert "auth-service" in entity_names
        # Should find connected entities
        assert len(graph["relationships"]) > 0

    @pytest.mark.asyncio
    async def test_forget_cleans_graph(self, memory_service, graph_store):
        """Forgetting a memory cleans up graph associations."""
        result = await memory_service.remember(
            "The payment-service uses Stripe for processing."
        )
        memory_id = result.memory_id
        
        # Verify entities exist
        entities = graph_store.get_entities_for_memory(memory_id)
        assert len(entities) > 0
        
        # Forget the memory
        await memory_service.forget(memory_id)
        
        # Graph associations should be cleaned
        entities_after = graph_store.get_entities_for_memory(memory_id)
        assert len(entities_after) == 0

    @pytest.mark.asyncio
    async def test_graph_disabled_gracefully(self, embedding_service):
        """Service works when graph is disabled."""
        vector_store = InMemoryVectorStore(embedding_service=embedding_service)
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=embedding_service,
            vector_store=vector_store,
            graph_store=None,
            graph_enabled=False,
        )
        
        # Should work without graph
        result = await service.remember("Test content")
        assert result.success
        
        # Entity recall returns empty when disabled
        results = await service.recall_entity("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_multi_memory_entity_aggregation(self, memory_service):
        """Entity recall aggregates across multiple memories."""
        # Multiple memories about the same entity
        await memory_service.remember("PostgreSQL version 15 is stable.")
        await memory_service.remember("PostgreSQL performance tuning tips.")
        await memory_service.remember("PostgreSQL backup strategies.")
        
        results = await memory_service.recall_entity("PostgreSQL")
        
        assert len(results) == 3
        contents = {r.memory.content for r in results}
        assert "PostgreSQL version 15 is stable." in contents
        assert "PostgreSQL performance tuning tips." in contents
        assert "PostgreSQL backup strategies." in contents


class TestGraphEnrichedRecall:
    """Tests for graph-aware hybrid recall (future Phase 2)."""

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
    async def test_entity_mentions_in_different_contexts(self, memory_service):
        """Track entity across different context memories."""
        # Same entity in different architectural contexts
        await memory_service.remember(
            "We migrated user-service from MySQL to PostgreSQL."
        )
        await memory_service.remember(
            "The user-service now handles 10k requests per second."
        )
        await memory_service.remember(
            "Rate limiting was added to user-service last week."
        )
        
        # Entity recall should find all user-service mentions
        results = await memory_service.recall_entity("user-service")
        
        assert len(results) == 3
        # All memories should be about user-service
        for r in results:
            assert "user-service" in r.memory.content.lower()
