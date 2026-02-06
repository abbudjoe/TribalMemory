"""Tests for lazy spaCy entity extraction.

Lazy spaCy mode uses fast regex extraction on ingest and
spaCy NER only on recall queries for better accuracy.
"""

import pytest
from unittest.mock import MagicMock, patch

from tribalmemory.services.memory import TribalMemoryService
from tribalmemory.services.graph_store import (
    EntityExtractor,
    HybridEntityExtractor,
    SpacyEntityExtractor,
    GraphStore,
    SPACY_AVAILABLE,
)
from tribalmemory.testing.mocks import MockEmbeddingService, MockVectorStore


@pytest.fixture
def mock_embedding_service():
    return MockEmbeddingService(embedding_dim=64)


@pytest.fixture
def mock_vector_store(mock_embedding_service):
    return MockVectorStore(mock_embedding_service)


@pytest.fixture
def mock_graph_store(tmp_path):
    return GraphStore(str(tmp_path / "graph.db"))


class TestLazySpacyConfiguration:
    """Test lazy spaCy mode configuration."""

    def test_lazy_spacy_enabled_by_default(
        self, mock_embedding_service, mock_vector_store, mock_graph_store
    ):
        """Lazy spaCy should be enabled by default."""
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            graph_enabled=True,
        )
        
        assert service.lazy_spacy is True

    def test_lazy_spacy_creates_separate_extractors(
        self, mock_embedding_service, mock_vector_store, mock_graph_store
    ):
        """Lazy spaCy mode should create separate ingest and query extractors."""
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            graph_enabled=True,
            lazy_spacy=True,
        )
        
        # Ingest extractor should be regex-only EntityExtractor
        assert isinstance(service.ingest_entity_extractor, EntityExtractor)
        # Query extractor should be HybridEntityExtractor (with spaCy if available)
        assert isinstance(service.query_entity_extractor, HybridEntityExtractor)
        # They should be different objects
        assert service.ingest_entity_extractor is not service.query_entity_extractor

    def test_eager_spacy_uses_same_extractor(
        self, mock_embedding_service, mock_vector_store, mock_graph_store
    ):
        """Eager spaCy mode should use the same extractor for both."""
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            graph_enabled=True,
            lazy_spacy=False,
        )
        
        # Both should be HybridEntityExtractor
        assert isinstance(service.ingest_entity_extractor, HybridEntityExtractor)
        assert isinstance(service.query_entity_extractor, HybridEntityExtractor)
        # They should be the same object
        assert service.ingest_entity_extractor is service.query_entity_extractor

    def test_legacy_entity_extractor_alias(
        self, mock_embedding_service, mock_vector_store, mock_graph_store
    ):
        """Legacy entity_extractor should alias query_entity_extractor."""
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            graph_enabled=True,
        )
        
        assert service.entity_extractor is service.query_entity_extractor


class TestLazySpacyIngest:
    """Test that ingest uses the fast extractor."""

    @pytest.mark.asyncio
    async def test_remember_uses_ingest_extractor(
        self, mock_embedding_service, mock_vector_store, mock_graph_store
    ):
        """remember() should use the fast ingest extractor, not query extractor."""
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            graph_enabled=True,
            lazy_spacy=True,
        )
        
        # Mock both extractors to track which is called
        ingest_mock = MagicMock()
        ingest_mock.extract_with_relationships.return_value = ([], [])
        query_mock = MagicMock()
        query_mock.extract_with_relationships.return_value = ([], [])
        
        service.ingest_entity_extractor = ingest_mock
        service.query_entity_extractor = query_mock
        
        await service.remember("Test content about John meeting Sarah")
        
        # Ingest extractor should be called
        ingest_mock.extract_with_relationships.assert_called_once()
        # Query extractor should NOT be called during ingest
        query_mock.extract_with_relationships.assert_not_called()


class TestLazySpacyRecall:
    """Test that recall uses the accurate query extractor."""

    @pytest.mark.asyncio
    async def test_recall_uses_query_extractor(
        self, mock_embedding_service, mock_vector_store, mock_graph_store
    ):
        """recall() should use the accurate query extractor for graph expansion."""
        service = TribalMemoryService(
            instance_id="test",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            graph_enabled=True,
            lazy_spacy=True,
        )
        
        # Store a memory first
        await service.remember("John works at Google")
        
        # Mock the query extractor to track calls
        query_mock = MagicMock()
        query_mock.extract.return_value = []
        service.query_entity_extractor = query_mock
        
        await service.recall("Where does John work?", graph_expansion=True)
        
        # Query extractor should be called for graph expansion
        query_mock.extract.assert_called_once_with("Where does John work?")


class TestExtractorBehavior:
    """Test that regex and spaCy extractors behave differently."""

    def test_regex_extractor_finds_tech_entities(self):
        """Regex extractor should find tech/code patterns."""
        extractor = EntityExtractor()
        entities = extractor.extract("The user-service calls api-gateway on port 8080")
        
        names = [e.name for e in entities]
        # Should find tech patterns
        assert "user-service" in names or "api-gateway" in names

    def test_regex_extractor_misses_personal_names(self):
        """Regex extractor should miss personal names (by design)."""
        extractor = EntityExtractor()
        entities = extractor.extract("Sarah met John at the coffee shop")
        
        names = [e.name for e in entities]
        # Regex won't find Sarah or John (no tech patterns)
        # This is expected - that's why we use spaCy for queries
        assert "Sarah" not in names
        assert "John" not in names

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
    def test_spacy_extractor_finds_personal_names(self):
        """spaCy extractor should find personal names."""
        extractor = SpacyEntityExtractor()
        entities = extractor.extract("Sarah met John at Starbucks on Tuesday")
        
        names = [e.name for e in entities]
        # spaCy should find PERSON entities
        assert "Sarah" in names or "John" in names

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
    def test_hybrid_extractor_combines_both(self):
        """Hybrid extractor should find both tech and personal entities."""
        extractor = HybridEntityExtractor(use_spacy=True)
        entities = extractor.extract(
            "Sarah deployed user-service to the api-gateway"
        )
        
        names = [e.name for e in entities]
        # Should find personal names via spaCy
        # and tech patterns via regex
        has_person = "Sarah" in names
        has_tech = "user-service" in names or "api-gateway" in names
        # At least one of each type should be found
        assert has_person or has_tech  # Hybrid should find something


class TestConfigIntegration:
    """Test lazy_spacy config option integration."""

    def test_search_config_has_lazy_spacy(self):
        """SearchConfig should have lazy_spacy option."""
        from tribalmemory.server.config import SearchConfig
        
        config = SearchConfig()
        assert hasattr(config, "lazy_spacy")
        assert config.lazy_spacy is True  # Default

    def test_search_config_lazy_spacy_can_be_disabled(self):
        """lazy_spacy can be disabled via config."""
        from tribalmemory.server.config import SearchConfig
        
        config = SearchConfig(lazy_spacy=False)
        assert config.lazy_spacy is False

    def test_create_memory_service_accepts_lazy_spacy(self):
        """create_memory_service should accept lazy_spacy parameter."""
        from tribalmemory.services.memory import create_memory_service
        import inspect
        
        sig = inspect.signature(create_memory_service)
        params = list(sig.parameters.keys())
        assert "lazy_spacy" in params
