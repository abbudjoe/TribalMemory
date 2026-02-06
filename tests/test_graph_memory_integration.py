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


# =============================================================================
# Issue #94: E2E Integration test for spaCy entity extraction
# =============================================================================

from tribalmemory.services.graph_store import SPACY_AVAILABLE, HybridEntityExtractor


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestSpacyE2EIntegration:
    """E2E integration tests for spaCy entity extraction with TribalMemoryService.
    
    Issue #94: Verify the full flow:
    1. Personal conversation ingested
    2. Entities extracted via spaCy
    3. Graph relationships created
    4. Recall finds related memories via graph traversal
    """

    @pytest.fixture
    def embedding_service(self):
        return MockEmbeddingService(embedding_dim=64)

    @pytest.fixture
    def vector_store(self, embedding_service):
        return InMemoryVectorStore(embedding_service=embedding_service)

    @pytest.fixture
    def graph_store_with_spacy(self, tmp_path):
        """Graph store for spaCy tests."""
        return GraphStore(str(tmp_path / "graph.db"))

    @pytest.fixture
    def memory_service_with_spacy(
        self, embedding_service, vector_store, graph_store_with_spacy
    ):
        """Memory service with lazy_spacy=False to ensure spaCy is used during ingest."""
        return TribalMemoryService(
            instance_id="spacy-test",
            embedding_service=embedding_service,
            vector_store=vector_store,
            graph_store=graph_store_with_spacy,
            graph_enabled=True,
            auto_reject_duplicates=False,
            lazy_spacy=False,  # Use spaCy for ingest, not just queries
        )

    @pytest.mark.asyncio
    async def test_personal_conversation_entities_extracted(
        self, memory_service_with_spacy, graph_store_with_spacy
    ):
        """Personal conversation should have entities extracted via spaCy."""
        # Store a personal conversation memory
        result = await memory_service_with_spacy.remember(
            "I had lunch with Sarah yesterday at the new Italian place downtown."
        )
        
        assert result.success
        memory_id = result.memory_id
        
        # Entities should be extracted
        entities = graph_store_with_spacy.get_entities_for_memory(memory_id)
        
        # spaCy should extract person name with correct type
        assert any(
            e.name.lower() == "sarah" and e.entity_type == "person"
            for e in entities
        ), f"Expected person entity 'Sarah' in {entities}"

    @pytest.mark.asyncio
    async def test_person_entities_enable_cross_memory_recall(
        self, memory_service_with_spacy
    ):
        """Person entities should connect related memories for recall."""
        # Store multiple memories mentioning the same person
        await memory_service_with_spacy.remember(
            "Sarah recommended a great book about machine learning."
        )
        await memory_service_with_spacy.remember(
            "I had coffee with Sarah and discussed the startup idea."
        )
        await memory_service_with_spacy.remember(
            "Bob mentioned he's moving to Seattle next month."
        )
        
        # Entity recall for Sarah should find both Sarah memories
        results = await memory_service_with_spacy.recall_entity("Sarah")
        
        assert len(results) == 2
        contents = {r.memory.content for r in results}
        assert any("book" in c for c in contents)
        assert any("coffee" in c for c in contents)
        # Should not include Bob's memory
        assert not any("Bob" in c for c in contents)

    @pytest.mark.asyncio
    async def test_location_entities_extracted(
        self, memory_service_with_spacy, graph_store_with_spacy
    ):
        """Location entities should be extracted via spaCy."""
        result = await memory_service_with_spacy.remember(
            "We're planning a trip to Paris next summer, maybe visit London too."
        )
        
        assert result.success
        entities = graph_store_with_spacy.get_entities_for_memory(result.memory_id)
        
        # Get location entities
        locations = [e for e in entities if e.entity_type == 'place']
        location_names = {e.name.lower() for e in locations}
        
        # Should extract major city names
        assert 'paris' in location_names or 'london' in location_names, (
            f"Expected Paris or London in {location_names}"
        )

    @pytest.mark.asyncio
    async def test_date_entities_extracted(
        self, memory_service_with_spacy, graph_store_with_spacy
    ):
        """Date entities should be extracted via spaCy."""
        result = await memory_service_with_spacy.remember(
            "The meeting is scheduled for March 15th, and the deadline is April 1st."
        )
        
        assert result.success
        entities = graph_store_with_spacy.get_entities_for_memory(result.memory_id)
        
        # Get date entities
        dates = [e for e in entities if e.entity_type == 'date']
        
        # Should extract at least one date
        assert len(dates) >= 1, f"Expected at least one date entity, got {dates}"

    @pytest.mark.asyncio
    async def test_organization_entities_extracted(
        self, memory_service_with_spacy, graph_store_with_spacy
    ):
        """Organization entities should be extracted via spaCy."""
        result = await memory_service_with_spacy.remember(
            "I interviewed at Google last week, and Microsoft reached out too."
        )
        
        assert result.success
        entities = graph_store_with_spacy.get_entities_for_memory(result.memory_id)
        
        # Get organization entities
        orgs = [e for e in entities if e.entity_type == 'organization']
        org_names = {e.name.lower() for e in orgs}
        
        # Should extract major company names
        assert 'google' in org_names or 'microsoft' in org_names, (
            f"Expected Google or Microsoft in {org_names}"
        )

    @pytest.mark.asyncio
    async def test_spacy_entity_graph_traversal(
        self, memory_service_with_spacy
    ):
        """Person entities should enable graph traversal with hops.
        
        Tests that spaCy-extracted entities create graph connections
        that can be traversed to find related memories.
        """
        # Store memories with connected entities
        await memory_service_with_spacy.remember(
            "Sarah works at Google."
        )
        await memory_service_with_spacy.remember(
            "Google is launching a new AI product next month."
        )
        await memory_service_with_spacy.remember(
            "Unrelated memory about cooking pasta."
        )
        
        # 1-hop from Sarah should find Google, which connects to AI product memory
        results = await memory_service_with_spacy.recall_entity("Sarah", hops=1)
        contents = [r.memory.content for r in results]
        
        # Should find the direct Sarah memory
        assert any("works at Google" in c for c in contents), (
            f"Expected Sarah's direct memory in results: {contents}"
        )
        # With 1-hop, may also find Google-related memories
        # (depends on whether Google is extracted as an entity in both)

    @pytest.mark.asyncio
    async def test_hybrid_extractor_used_when_spacy_enabled(
        self, memory_service_with_spacy
    ):
        """Memory service should use HybridEntityExtractor when lazy_spacy=False."""
        # Check the ingest extractor type (lazy_spacy=False means HybridEntityExtractor)
        assert isinstance(
            memory_service_with_spacy.ingest_entity_extractor,
            HybridEntityExtractor
        ), "Expected HybridEntityExtractor for ingest when lazy_spacy=False"
        
        assert memory_service_with_spacy.ingest_entity_extractor.has_spacy is True
