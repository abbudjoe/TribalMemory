"""Tests for graph-enriched memory storage."""

import pytest
from tribalmemory.services.graph_store import (
    GraphStore,
    Entity,
    Relationship,
    EntityExtractor,
)


class TestEntityExtractor:
    """Tests for entity extraction from text."""

    def test_extract_service_names(self):
        """Extract service names from architecture text."""
        extractor = EntityExtractor()
        text = "The auth-service handles authentication and talks to user-db."
        
        entities = extractor.extract(text)
        
        names = {e.name for e in entities}
        assert "auth-service" in names
        assert "user-db" in names

    def test_extract_technology_names(self):
        """Extract technology/framework names."""
        extractor = EntityExtractor()
        text = "We use PostgreSQL for the database and Redis for caching."
        
        entities = extractor.extract(text)
        
        names = {e.name for e in entities}
        assert "PostgreSQL" in names
        assert "Redis" in names

    def test_extract_relationships(self):
        """Extract relationships between entities."""
        extractor = EntityExtractor()
        text = "The auth-service uses PostgreSQL for storing credentials."
        
        entities, relationships = extractor.extract_with_relationships(text)
        
        assert len(relationships) >= 1
        rel = relationships[0]
        assert rel.source == "auth-service"
        assert rel.target == "PostgreSQL"
        assert rel.relation_type in ("uses", "stores_in", "connects_to")

    def test_extract_empty_text(self):
        """Handle empty or whitespace-only text."""
        extractor = EntityExtractor()
        
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []

    def test_extract_no_entities(self):
        """Handle text with no extractable entities."""
        extractor = EntityExtractor()
        text = "The quick brown fox jumps over the lazy dog."
        
        entities = extractor.extract(text)
        # May return empty or generic entities, should not crash
        assert isinstance(entities, list)


class TestGraphStore:
    """Tests for graph storage and querying."""

    @pytest.fixture
    def graph_store(self, tmp_path):
        """Create a temporary graph store."""
        db_path = tmp_path / "test_graph.db"
        return GraphStore(str(db_path))

    def test_store_entity(self, graph_store):
        """Store and retrieve a single entity."""
        entity = Entity(name="auth-service", entity_type="service")
        
        graph_store.add_entity(entity, memory_id="mem-123")
        
        entities = graph_store.get_entities_for_memory("mem-123")
        assert len(entities) == 1
        assert entities[0].name == "auth-service"
        assert entities[0].entity_type == "service"

    def test_store_relationship(self, graph_store):
        """Store and retrieve a relationship."""
        e1 = Entity(name="auth-service", entity_type="service")
        e2 = Entity(name="PostgreSQL", entity_type="technology")
        rel = Relationship(source="auth-service", target="PostgreSQL", relation_type="uses")
        
        graph_store.add_entity(e1, memory_id="mem-123")
        graph_store.add_entity(e2, memory_id="mem-123")
        graph_store.add_relationship(rel, memory_id="mem-123")
        
        rels = graph_store.get_relationships_for_entity("auth-service")
        assert len(rels) == 1
        assert rels[0].target == "PostgreSQL"

    def test_find_connected_entities(self, graph_store):
        """Find entities connected to a given entity (1-hop)."""
        # Setup: auth-service -> PostgreSQL -> user-data
        graph_store.add_entity(Entity(name="auth-service", entity_type="service"), "mem-1")
        graph_store.add_entity(Entity(name="PostgreSQL", entity_type="technology"), "mem-2")
        graph_store.add_entity(Entity(name="user-data", entity_type="data"), "mem-3")
        
        graph_store.add_relationship(
            Relationship(source="auth-service", target="PostgreSQL", relation_type="uses"),
            memory_id="mem-1"
        )
        graph_store.add_relationship(
            Relationship(source="PostgreSQL", target="user-data", relation_type="stores"),
            memory_id="mem-2"
        )
        
        # 1-hop from auth-service
        connected = graph_store.find_connected(entity_name="auth-service", hops=1)
        names = {e.name for e in connected}
        assert "PostgreSQL" in names
        assert "user-data" not in names  # 2 hops away

    def test_find_connected_entities_multi_hop(self, graph_store):
        """Find entities within 2 hops."""
        graph_store.add_entity(Entity(name="auth-service", entity_type="service"), "mem-1")
        graph_store.add_entity(Entity(name="PostgreSQL", entity_type="technology"), "mem-2")
        graph_store.add_entity(Entity(name="user-data", entity_type="data"), "mem-3")
        
        graph_store.add_relationship(
            Relationship(source="auth-service", target="PostgreSQL", relation_type="uses"),
            memory_id="mem-1"
        )
        graph_store.add_relationship(
            Relationship(source="PostgreSQL", target="user-data", relation_type="stores"),
            memory_id="mem-2"
        )
        
        # 2-hops from auth-service
        connected = graph_store.find_connected(entity_name="auth-service", hops=2)
        names = {e.name for e in connected}
        assert "PostgreSQL" in names
        assert "user-data" in names

    def test_get_memories_for_entity(self, graph_store):
        """Get all memory IDs associated with an entity."""
        graph_store.add_entity(Entity(name="PostgreSQL", entity_type="technology"), "mem-1")
        graph_store.add_entity(Entity(name="PostgreSQL", entity_type="technology"), "mem-2")
        graph_store.add_entity(Entity(name="PostgreSQL", entity_type="technology"), "mem-3")
        
        memory_ids = graph_store.get_memories_for_entity("PostgreSQL")
        assert set(memory_ids) == {"mem-1", "mem-2", "mem-3"}

    def test_delete_memory_cascades(self, graph_store):
        """Deleting a memory removes its entities and relationships."""
        graph_store.add_entity(Entity(name="auth-service", entity_type="service"), "mem-123")
        graph_store.add_relationship(
            Relationship(source="auth-service", target="PostgreSQL", relation_type="uses"),
            memory_id="mem-123"
        )
        
        graph_store.delete_memory("mem-123")
        
        assert graph_store.get_entities_for_memory("mem-123") == []
        assert graph_store.get_relationships_for_entity("auth-service") == []


class TestGraphStoreIntegration:
    """Integration tests for graph + memory pipeline."""

    @pytest.fixture
    def graph_store(self, tmp_path):
        db_path = tmp_path / "test_graph.db"
        return GraphStore(str(db_path))

    def test_entity_centric_query(self, graph_store):
        """Query memories through entity name."""
        # Store several memories about PostgreSQL
        graph_store.add_entity(
            Entity(name="PostgreSQL", entity_type="technology"),
            memory_id="mem-pg-setup"
        )
        graph_store.add_entity(
            Entity(name="PostgreSQL", entity_type="technology"),
            memory_id="mem-pg-perf"
        )
        graph_store.add_entity(
            Entity(name="PostgreSQL", entity_type="technology"),
            memory_id="mem-pg-backup"
        )
        
        # Entity-centric recall
        memories = graph_store.get_memories_for_entity("PostgreSQL")
        assert len(memories) == 3
        assert "mem-pg-setup" in memories
        assert "mem-pg-perf" in memories
        assert "mem-pg-backup" in memories

    def test_relationship_traversal_for_recall(self, graph_store):
        """Use relationships to expand recall candidates."""
        # Setup: auth-service uses PostgreSQL uses pgbouncer
        graph_store.add_entity(Entity(name="auth-service", entity_type="service"), "mem-auth")
        graph_store.add_entity(Entity(name="PostgreSQL", entity_type="technology"), "mem-pg")
        graph_store.add_entity(Entity(name="pgbouncer", entity_type="technology"), "mem-pgb")
        
        graph_store.add_relationship(
            Relationship(source="auth-service", target="PostgreSQL", relation_type="uses"),
            memory_id="mem-auth"
        )
        graph_store.add_relationship(
            Relationship(source="PostgreSQL", target="pgbouncer", relation_type="uses"),
            memory_id="mem-pg"
        )
        
        # Query: "What does auth-service connect to?"
        # Should find PostgreSQL (1-hop) and optionally pgbouncer (2-hop)
        connected = graph_store.find_connected("auth-service", hops=2)
        connected_names = {e.name for e in connected}
        
        assert "PostgreSQL" in connected_names
        assert "pgbouncer" in connected_names
        
        # Get all memories related to these entities
        all_memories = set()
        for entity in connected:
            all_memories.update(graph_store.get_memories_for_entity(entity.name))
        
        assert "mem-pg" in all_memories
        assert "mem-pgb" in all_memories
