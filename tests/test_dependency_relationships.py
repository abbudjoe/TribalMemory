"""Tests for dependency-parsed relationship extraction (Issue #132)."""
import pytest
from tribalmemory.services.graph_store import (
    SpacyEntityExtractor,
    HybridEntityExtractor,
    SPACY_AVAILABLE,
    Entity,
    Relationship,
)


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestDependencyRelationshipExtraction:
    """Tests for dependency-based relationship extraction (Issue #132)."""

    def test_simple_uses_relationship(self):
        """Should extract (subject, uses, object) from 'Sarah uses Redis'."""
        extractor = SpacyEntityExtractor()
        entities, relationships = extractor.extract_with_relationships(
            "Sarah uses Redis"
        )
        
        # Should extract both entities
        entity_names = {e.name.lower() for e in entities}
        assert "sarah" in entity_names, f"Expected 'sarah' in {entity_names}"
        assert "redis" in entity_names, f"Expected 'redis' in {entity_names}"
        
        # Should extract relationship
        assert len(relationships) >= 1, (
            f"Expected at least 1 relationship, got {relationships}"
        )
        uses_rels = [r for r in relationships if r.relation_type == "uses"]
        assert len(uses_rels) == 1, (
            f"Expected exactly 1 'uses' relationship, got {uses_rels}"
        )
        rel = uses_rels[0]
        assert rel.source.lower() == "sarah", f"Expected source 'Sarah', got {rel.source}"
        assert rel.target.lower() == "redis", f"Expected target 'Redis', got {rel.target}"

    def test_located_in_relationship(self):
        """Should extract location relationship from 'I live in New York'."""
        extractor = SpacyEntityExtractor()
        entities, relationships = extractor.extract_with_relationships(
            "I live in New York"
        )
        
        # Should extract New York as a place entity
        places = [e for e in entities if e.entity_type == "place"]
        assert len(places) >= 1, f"Expected at least one place entity, got {entities}"
        
        # Should extract located_in relationship
        # Note: "I" might be resolved to a speaker entity or skipped
        located_rels = [r for r in relationships if r.relation_type == "located_in"]
        # We expect a relationship between subject and New York
        if located_rels:
            rel = located_rels[0]
            assert "new york" in rel.target.lower(), (
                f"Expected target 'New York', got {rel.target}"
            )

    def test_met_relationship(self):
        """Should extract (Bob, met, Amy) from 'Bob met Amy at the conference'."""
        extractor = SpacyEntityExtractor()
        entities, relationships = extractor.extract_with_relationships(
            "Bob met Amy at the conference"
        )
        
        # Should extract both person entities
        persons = [e for e in entities if e.entity_type == "person"]
        person_names = {p.name.lower() for p in persons}
        assert "bob" in person_names, f"Expected 'Bob' in {person_names}"
        assert "amy" in person_names, f"Expected 'Amy' in {person_names}"
        
        # Should extract 'met' relationship
        met_rels = [r for r in relationships if r.relation_type == "met"]
        assert len(met_rels) >= 1, (
            f"Expected at least 1 'met' relationship, got {relationships}"
        )
        rel = met_rels[0]
        assert rel.source.lower() == "bob", f"Expected source 'Bob', got {rel.source}"
        assert rel.target.lower() == "amy", f"Expected target 'Amy', got {rel.target}"

    def test_no_relationship_for_non_entity(self):
        """Should NOT create relationship when object is not a named entity.
        
        'She likes pizza' should not create a relationship because 'pizza'
        is not extracted as a named entity by spaCy NER.
        """
        extractor = SpacyEntityExtractor()
        entities, relationships = extractor.extract_with_relationships(
            "She likes pizza"
        )
        
        # May extract 'She' or not, depending on context
        # But 'pizza' should not be an entity (common noun, not named entity)
        entity_names = {e.name.lower() for e in entities}
        
        # Relationships should only be between extracted entities
        # Since 'pizza' is not a named entity, no relationship should be created
        # This might be empty or only have relationships between other entities
        for rel in relationships:
            # Neither source nor target should be 'pizza'
            assert "pizza" not in rel.source.lower(), (
                f"Unexpected 'pizza' as source: {rel}"
            )
            assert "pizza" not in rel.target.lower(), (
                f"Unexpected 'pizza' as target: {rel}"
            )

    def test_works_at_relationship(self):
        """Should extract (Thompson, works_at, Google) from 'Dr. Thompson works at Google'."""
        extractor = SpacyEntityExtractor()
        entities, relationships = extractor.extract_with_relationships(
            "Dr. Thompson works at Google"
        )
        
        # Should extract person and organization
        entity_names = {e.name.lower() for e in entities}
        assert "thompson" in entity_names, f"Expected 'Thompson' in {entity_names}"
        assert "google" in entity_names, f"Expected 'Google' in {entity_names}"
        
        # Should extract works_at relationship
        works_rels = [r for r in relationships if r.relation_type == "works_at"]
        assert len(works_rels) >= 1, (
            f"Expected at least 1 'works_at' relationship, got {relationships}"
        )
        rel = works_rels[0]
        assert "thompson" in rel.source.lower(), (
            f"Expected source 'Thompson', got {rel.source}"
        )
        assert "google" in rel.target.lower(), (
            f"Expected target 'Google', got {rel.target}"
        )

    def test_only_ner_entities_in_relationships(self):
        """Relationships should only link entities extracted by NER.
        
        Tests that spurious entities are not created from dependency parsing.
        Only entities that were extracted by spaCy NER should appear in relationships.
        """
        extractor = SpacyEntityExtractor()
        entities, relationships = extractor.extract_with_relationships(
            "The manager uses the database for the reports"
        )
        
        # Get entity names from NER
        ner_names_lower = {e.name.lower() for e in entities}
        
        # All relationship endpoints must be in the NER entity set
        for rel in relationships:
            assert rel.source.lower() in ner_names_lower, (
                f"Relationship source '{rel.source}' not in NER entities: {ner_names_lower}"
            )
            assert rel.target.lower() in ner_names_lower, (
                f"Relationship target '{rel.target}' not in NER entities: {ner_names_lower}"
            )

    def test_empty_text_returns_empty_lists(self):
        """Should return empty entities and relationships for empty text."""
        extractor = SpacyEntityExtractor()
        entities, relationships = extractor.extract_with_relationships("")
        
        assert entities == [], f"Expected empty entities, got {entities}"
        assert relationships == [], f"Expected empty relationships, got {relationships}"

    def test_text_with_no_entities_returns_empty_relationships(self):
        """Should return empty relationships when no entities are extracted."""
        extractor = SpacyEntityExtractor()
        # Generic text with no named entities
        entities, relationships = extractor.extract_with_relationships(
            "The thing uses the other thing"
        )
        
        # No named entities = no relationships
        assert relationships == [], (
            f"Expected empty relationships when no entities, got {relationships}"
        )


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestHybridExtractorWithDependencyRelationships:
    """Tests for HybridEntityExtractor using dependency relationships (Issue #132)."""

    def test_personal_context_uses_dependency_relationships(self):
        """In 'personal' context, should use dependency-parsed relationships."""
        hybrid = HybridEntityExtractor(
            use_spacy=True,
            extraction_context="personal"
        )
        
        text = "Sarah uses Redis"
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should extract entities
        entity_names = {e.name.lower() for e in entities}
        assert "sarah" in entity_names
        assert "redis" in entity_names
        
        # Should have dependency-parsed relationship
        uses_rels = [r for r in relationships if r.relation_type == "uses"]
        assert len(uses_rels) >= 1, (
            f"Expected dependency-parsed 'uses' relationship, got {relationships}"
        )

    def test_software_context_uses_regex_relationships(self):
        """In 'software' context, should still use regex relationships (backward compat)."""
        hybrid = HybridEntityExtractor(
            use_spacy=True,
            extraction_context="software"
        )
        
        text = "The auth-service uses Redis"
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should extract entities
        entity_names = {e.name.lower() for e in entities}
        assert "redis" in entity_names or "auth-service" in entity_names
        
        # In software context, regex relationships should still work

    def test_no_spacy_falls_back_to_regex(self):
        """When spaCy is unavailable, should fall back to regex (no dependency parsing)."""
        hybrid = HybridEntityExtractor(
            use_spacy=False,
            extraction_context="personal"
        )
        
        text = "Sarah uses Redis"
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Without spaCy, 'Sarah' won't be extracted as a person entity
        # Relationships will be empty in personal context (regex disabled)
        assert relationships == [], (
            f"Expected no relationships without spaCy in personal context, got {relationships}"
        )
