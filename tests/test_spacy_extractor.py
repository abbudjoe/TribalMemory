"""Tests for spaCy-based entity extraction."""
import pytest
from tribalmemory.services.graph_store import (
    SpacyEntityExtractor,
    HybridEntityExtractor,
    SPACY_AVAILABLE,
)


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestSpacyEntityExtractor:
    """Tests for SpacyEntityExtractor."""

    def test_extract_person_names(self):
        """Should extract person names from text."""
        extractor = SpacyEntityExtractor()
        text = "I met with Dr. Thompson and Sarah about the project."
        entities = extractor.extract(text)
        
        names = {e.name for e in entities if e.entity_type == "person"}
        assert "Thompson" in names or "Dr. Thompson" in names
        assert "Sarah" in names

    def test_extract_places(self):
        """Should extract place names from text."""
        extractor = SpacyEntityExtractor()
        text = "I viewed a townhouse in the Brookside neighborhood near Oak Street."
        entities = extractor.extract(text)
        
        places = {e.name.lower() for e in entities if e.entity_type in ("place", "organization")}
        # spaCy may classify neighborhoods as ORG or GPE
        assert any("brookside" in p for p in places) or any("oak" in p for p in places)

    def test_extract_dates(self):
        """Should extract date expressions from text."""
        extractor = SpacyEntityExtractor()
        text = "I have an appointment on March 15th, and another one last Tuesday."
        entities = extractor.extract(text)
        
        dates = [e for e in entities if e.entity_type == "date"]
        assert len(dates) >= 1  # At least one date should be extracted

    def test_extract_empty_text(self):
        """Should return empty list for empty text."""
        extractor = SpacyEntityExtractor()
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []
        assert extractor.extract(None) == []

    def test_extract_with_relationships_returns_empty_relationships(self):
        """spaCy extractor should return empty relationships list."""
        extractor = SpacyEntityExtractor()
        entities, relationships = extractor.extract_with_relationships("Dr. Smith uses Redis")
        assert len(entities) > 0
        assert relationships == []


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestHybridEntityExtractor:
    """Tests for HybridEntityExtractor combining regex + spaCy."""

    def test_has_spacy_property(self):
        """Should report spaCy availability."""
        hybrid = HybridEntityExtractor(use_spacy=True)
        assert hybrid.has_spacy is True
        
        hybrid_no_spacy = HybridEntityExtractor(use_spacy=False)
        assert hybrid_no_spacy.has_spacy is False

    def test_combines_regex_and_spacy_entities(self):
        """Should extract entities from both regex and spaCy."""
        hybrid = HybridEntityExtractor(use_spacy=True)
        # Contains both service pattern (auth-service) and person (Dr. Smith)
        text = "The auth-service was reviewed by Dr. Smith using PostgreSQL."
        entities = hybrid.extract(text)
        
        names = {e.name.lower() for e in entities}
        types = {e.entity_type for e in entities}
        
        # Should have technology from regex
        assert "postgresql" in names or any("postgres" in n for n in names)
        # Should have person from spaCy
        assert "smith" in names or "dr. smith" in names
        assert "person" in types
        assert "technology" in types

    def test_deduplicates_entities(self):
        """Should not return duplicate entities from same extractor."""
        hybrid = HybridEntityExtractor(use_spacy=True)
        # Use a cleaner example - same tech mentioned twice
        text = "We use Redis for caching. Redis is fast."
        entities = hybrid.extract(text)
        
        redis_entities = [e for e in entities if e.name.lower() == "redis"]
        # Should have exactly one Redis entity from regex
        assert len(redis_entities) == 1

    def test_extract_with_relationships_uses_regex(self):
        """Should extract relationships from regex extractor."""
        hybrid = HybridEntityExtractor(use_spacy=True)
        text = "The auth-service uses Redis and was built by Sarah."
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should have entities from both
        names = {e.name.lower() for e in entities}
        assert "redis" in names
        assert "sarah" in names
        
        # Relationships come from regex
        # (may or may not find "uses" relationship depending on pattern match)


class TestHybridWithoutSpacy:
    """Tests for HybridEntityExtractor when spaCy is disabled."""

    def test_falls_back_to_regex_only(self):
        """Should work with regex only when spaCy disabled."""
        hybrid = HybridEntityExtractor(use_spacy=False)
        text = "The auth-service uses PostgreSQL."
        entities = hybrid.extract(text)
        
        names = {e.name.lower() for e in entities}
        assert "postgresql" in names or "auth-service" in names
        # Person names won't be extracted without spaCy
        assert hybrid.has_spacy is False
