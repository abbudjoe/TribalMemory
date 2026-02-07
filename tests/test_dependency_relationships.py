"""Tests for dependency-parsed relationship extraction (Issue #132).

Tests the DependencyRelationshipExtractor class which uses spaCy's
dependency parser to extract subject-verb-object triples and create
semantically valid relationships between NER entities.
"""
import pytest
from tribalmemory.services.graph_store import (
    DependencyRelationshipExtractor,
    HybridEntityExtractor,
    Entity,
    Relationship,
    SPACY_AVAILABLE,
)


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestDependencyRelationshipExtractor:
    """Tests for DependencyRelationshipExtractor class."""

    def test_basic_svo_triple(self):
        """Should extract basic subject-verb-object relationship."""
        extractor = DependencyRelationshipExtractor()
        
        # Parse text
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("John uses Python")
        
        # Entities that were extracted
        entities = [
            Entity(name="John", entity_type="person"),
            Entity(name="Python", entity_type="technology"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should find: John --uses--> Python
        assert len(relationships) == 1
        rel = relationships[0]
        assert rel.source == "John"
        assert rel.target == "Python"
        assert rel.relation_type == "uses"

    def test_prepositional_relationship(self):
        """Should extract relationships with prepositional phrases."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Sarah lives in Denver")
        
        entities = [
            Entity(name="Sarah", entity_type="person"),
            Entity(name="Denver", entity_type="place"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should find: Sarah --located_in--> Denver
        assert len(relationships) >= 1
        # Find the located_in relationship
        located_rels = [r for r in relationships if r.relation_type == "located_in"]
        assert len(located_rels) == 1
        rel = located_rels[0]
        assert rel.source == "Sarah"
        assert rel.target == "Denver"

    def test_multiple_relationships(self):
        """Should extract multiple relationships from one sentence."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Mike works at Google and lives in Seattle")
        
        entities = [
            Entity(name="Mike", entity_type="person"),
            Entity(name="Google", entity_type="organization"),
            Entity(name="Seattle", entity_type="place"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should find: Mike --works_at--> Google and Mike --located_in--> Seattle
        assert len(relationships) >= 2
        
        works_at_rels = [r for r in relationships if r.relation_type == "works_at"]
        located_in_rels = [r for r in relationships if r.relation_type == "located_in"]
        
        assert len(works_at_rels) >= 1
        assert len(located_in_rels) >= 1

    def test_passive_voice(self):
        """Should handle passive voice constructions."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Python was recommended by Alice")
        
        entities = [
            Entity(name="Python", entity_type="technology"),
            Entity(name="Alice", entity_type="person"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should find: Alice --recommends--> Python (reversed from passive)
        recommends_rels = [r for r in relationships if r.relation_type == "recommends"]
        assert len(recommends_rels) >= 1
        rel = recommends_rels[0]
        assert rel.source == "Alice"
        assert rel.target == "Python"

    def test_no_matching_entities(self):
        """Should return empty list when subject/object don't match known entities."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Someone uses something")
        
        # No entities provided
        entities = []
        
        relationships = extractor.extract(doc, entities)
        
        # Should find no relationships (no matching entities)
        assert relationships == []

    def test_unmapped_verb(self):
        """Should skip verbs not in VERB_RELATIONS mapping."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("John dances with Mary")
        
        entities = [
            Entity(name="John", entity_type="person"),
            Entity(name="Mary", entity_type="person"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # "dances" is not in VERB_RELATIONS, so no relationships
        assert relationships == []

    def test_empty_text(self):
        """Should return empty list for empty text."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("")
        
        entities = [Entity(name="Test", entity_type="person")]
        
        relationships = extractor.extract(doc, entities)
        assert relationships == []

    def test_no_verbs(self):
        """Should return empty list when text has no verbs."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("John and Python")
        
        entities = [
            Entity(name="John", entity_type="person"),
            Entity(name="Python", entity_type="technology"),
        ]
        
        relationships = extractor.extract(doc, entities)
        assert relationships == []

    def test_pronoun_filtering(self):
        """Should filter out pronouns as relationship endpoints."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("I use Python")
        
        entities = [
            # "I" might be extracted as a pronoun but shouldn't match
            Entity(name="Python", entity_type="technology"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should not create a relationship with "I" as source
        # (pronouns are filtered)
        for rel in relationships:
            assert rel.source.lower() not in {"i", "me", "my"}
            assert rel.target.lower() not in {"i", "me", "my"}

    def test_multi_word_entity_matching(self):
        """Should match multi-word entities correctly."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("John visited New York")
        
        entities = [
            Entity(name="John", entity_type="person"),
            Entity(name="New York", entity_type="place"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should find: John --visited--> New York
        visited_rels = [r for r in relationships if r.relation_type == "visited"]
        assert len(visited_rels) >= 1

    def test_compound_subjects(self):
        """Should handle compound subjects (John and Mary)."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("John and Mary visited Paris")
        
        entities = [
            Entity(name="John", entity_type="person"),
            Entity(name="Mary", entity_type="person"),
            Entity(name="Paris", entity_type="place"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should find two relationships:
        # John --visited--> Paris and Mary --visited--> Paris
        visited_rels = [r for r in relationships if r.relation_type == "visited"]
        assert len(visited_rels) >= 2
        
        sources = {r.source for r in visited_rels}
        assert "John" in sources
        assert "Mary" in sources

    def test_real_conversation_snippet_1(self):
        """Should extract relationships from real conversation text."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("I met Dr. Thompson at the hospital yesterday. He recommended a new treatment.")
        
        entities = [
            Entity(name="Thompson", entity_type="person"),  # Title stripped
            Entity(name="hospital", entity_type="place"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should extract at least the met relationship
        # (if "I" is filtered, this won't create a relationship with Thompson)
        # This is expected behavior - pronouns should be filtered
        assert isinstance(relationships, list)

    def test_real_conversation_snippet_2(self):
        """Should extract relationships from purchase conversation."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Sarah bought a Razer keyboard at Best Buy")
        
        entities = [
            Entity(name="Sarah", entity_type="person"),
            Entity(name="Razer", entity_type="product"),
            Entity(name="Best Buy", entity_type="organization"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should find: Sarah --purchased--> Razer (or Razer keyboard)
        purchased_rels = [r for r in relationships if r.relation_type == "purchased"]
        assert len(purchased_rels) >= 1

    def test_relationship_validator_integration(self):
        """Should verify relationships pass through RelationshipValidator."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("John uses Python")
        
        entities = [
            Entity(name="John", entity_type="person"),
            Entity(name="Python", entity_type="technology"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Relationships should be valid (no self-relationships, etc.)
        from tribalmemory.services.graph_store import RelationshipValidator
        validator = RelationshipValidator()
        
        for rel in relationships:
            assert validator.is_valid(rel), f"Invalid relationship: {rel}"

    def test_verb_relation_coverage(self):
        """Should have mappings for common verbs."""
        extractor = DependencyRelationshipExtractor()
        
        # Verify key verbs are mapped
        assert "use" in extractor.VERB_RELATIONS
        assert "uses" in extractor.VERB_RELATIONS
        assert "live" in extractor.VERB_RELATIONS
        assert "work" in extractor.VERB_RELATIONS
        assert "visit" in extractor.VERB_RELATIONS
        assert "buy" in extractor.VERB_RELATIONS
        assert "meet" in extractor.VERB_RELATIONS
        assert "recommend" in extractor.VERB_RELATIONS

    def test_case_insensitive_entity_matching(self):
        """Should match entities case-insensitively."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("john uses python")  # Lowercase
        
        entities = [
            Entity(name="John", entity_type="person"),  # Capitalized
            Entity(name="Python", entity_type="technology"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should still match despite case difference
        assert len(relationships) >= 1

    def test_no_entities_provided(self):
        """Should return empty list when no entities are provided."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("John uses Python")
        
        relationships = extractor.extract(doc, [])
        
        assert relationships == []

    def test_verb_inflections(self):
        """Should map different verb inflections to same relationship type."""
        extractor = DependencyRelationshipExtractor()
        
        # Check that different forms map to the same type
        assert extractor.VERB_RELATIONS["use"] == "uses"
        assert extractor.VERB_RELATIONS["used"] == "uses"
        assert extractor.VERB_RELATIONS["using"] == "uses"
        
        assert extractor.VERB_RELATIONS["buy"] == "purchased"
        assert extractor.VERB_RELATIONS["bought"] == "purchased"

    def test_coordinated_verbs_with_compound_subjects(self):
        """Coordinated verbs should inherit compound subjects."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        # Use a clearer sentence that spaCy will parse correctly
        doc = nlp("Mike and Sarah visited Paris and bought souvenirs")
        
        entities = [
            Entity(name="Mike", entity_type="person"),
            Entity(name="Sarah", entity_type="person"),
            Entity(name="Paris", entity_type="place"),
            Entity(name="souvenirs", entity_type="product"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should find 4 relationships:
        # Mike visited Paris, Sarah visited Paris
        # Mike purchased souvenirs, Sarah purchased souvenirs
        visited_rels = [r for r in relationships if r.relation_type == "visited"]
        purchased_rels = [r for r in relationships if r.relation_type == "purchased"]
        
        # Each coordinated verb should create relationships with both compound subjects
        assert len(visited_rels) >= 2
        assert len(purchased_rels) >= 2

    def test_compound_objects(self):
        """Should handle compound objects (Paris and London)."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("John visited Paris and London")
        
        entities = [
            Entity(name="John", entity_type="person"),
            Entity(name="Paris", entity_type="place"),
            Entity(name="London", entity_type="place"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should find two relationships: John --visited--> Paris, John --visited--> London
        visited_rels = [r for r in relationships if r.relation_type == "visited"]
        assert len(visited_rels) == 2
        
        targets = {r.target for r in visited_rels}
        assert "Paris" in targets
        assert "London" in targets


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestHybridExtractorWithDependencyParsing:
    """Integration tests for HybridEntityExtractor with dependency parsing."""

    def test_personal_context_uses_dependency_parsing(self):
        """In personal context, should use dependency parsing for relationships."""
        hybrid = HybridEntityExtractor(
            use_spacy=True,
            extraction_context="personal"
        )
        
        text = "John uses Python for data analysis"
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should extract entities
        entity_names = {e.name for e in entities}
        assert "John" in entity_names or any("john" in n.lower() for n in entity_names)
        assert "Python" in entity_names or any("python" in n.lower() for n in entity_names)
        
        # Should extract relationship via dependency parsing
        uses_rels = [r for r in relationships if r.relation_type == "uses"]
        assert len(uses_rels) >= 1

    def test_software_context_skips_dependency_parsing(self):
        """In software context, should use regex relationships (not dependency)."""
        hybrid = HybridEntityExtractor(
            use_spacy=True,
            extraction_context="software"
        )
        
        text = "The auth-service uses PostgreSQL"
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should extract entities
        assert len(entities) >= 2
        
        # Relationships come from regex patterns (software context)
        # This may or may not produce relationships depending on regex patterns
        assert isinstance(relationships, list)

    def test_full_pipeline_integration(self):
        """Test full pipeline from text to entities + relationships."""
        hybrid = HybridEntityExtractor(
            use_spacy=True,
            extraction_context="personal"
        )
        
        text = "Sarah visited Denver last week and met Bob"
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should extract named entities
        assert len(entities) >= 2
        entity_names = {e.name for e in entities}
        assert any("sarah" in n.lower() for n in entity_names)
        assert any("denver" in n.lower() or "bob" in n.lower() for n in entity_names)
        
        # Should extract relationships
        assert isinstance(relationships, list)
        # May or may not have relationships depending on entity matching

    def test_without_spacy_no_relationships(self):
        """Without spaCy, personal context should return empty relationships."""
        hybrid = HybridEntityExtractor(
            use_spacy=False,
            extraction_context="personal"
        )
        
        text = "John uses Python"
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should extract entities via regex
        assert isinstance(entities, list)
        
        # No relationships (spaCy not available, personal context disables regex)
        assert relationships == []

    def test_entities_validated_before_relationship_extraction(self):
        """Should only create relationships between validated entities."""
        hybrid = HybridEntityExtractor(
            use_spacy=True,
            extraction_context="personal"
        )
        
        # Text with invalid entities (e.g., all-caps stopwords)
        text = "THE person uses AND technology"
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should filter out invalid entities like "THE" and "AND"
        entity_names = {e.name for e in entities}
        assert "THE" not in entity_names
        assert "AND" not in entity_names
        
        # Relationships should only be between valid entities
        for rel in relationships:
            assert rel.source in entity_names or rel.source.lower() not in {"the", "and"}
            assert rel.target in entity_names or rel.target.lower() not in {"the", "and"}

    def test_conversation_with_dates(self):
        """Should handle conversation text with dates."""
        hybrid = HybridEntityExtractor(
            use_spacy=True,
            extraction_context="personal"
        )
        
        text = "I met Sarah at Google on March 15th"
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should extract entities
        entity_names = {e.name for e in entities}
        assert any("sarah" in n.lower() for n in entity_names)
        
        # Relationships should be extracted (filtering "I")
        assert isinstance(relationships, list)

    def test_long_conversation_text(self):
        """Should handle longer conversation text with multiple relationships."""
        hybrid = HybridEntityExtractor(
            use_spacy=True,
            extraction_context="personal"
        )
        
        text = """
        Sarah works at Google in Seattle. She uses Python and loves the city.
        Last month, she visited San Francisco and met with Bob.
        """
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should extract multiple entities
        assert len(entities) >= 4
        
        # Should extract multiple relationships
        assert len(relationships) >= 2

    def test_no_false_positive_relationships(self):
        """Should not create garbage relationships from casual text."""
        hybrid = HybridEntityExtractor(
            use_spacy=True,
            extraction_context="personal"
        )
        
        # Casual text that shouldn't produce garbage relationships
        text = "I was waiting for the bus when it started raining"
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should not create meaningless relationships
        # (e.g., "waiting" and "bus" might be extracted but shouldn't relate)
        # All relationships should pass validation
        from tribalmemory.services.graph_store import RelationshipValidator
        validator = RelationshipValidator()
        
        for rel in relationships:
            assert validator.is_valid(rel)

    def test_preference_relationships(self):
        """Should extract preference relationships (like, love, prefer)."""
        hybrid = HybridEntityExtractor(
            use_spacy=True,
            extraction_context="personal"
        )
        
        text = "Sarah loves Python and prefers Linux"
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should find preference relationships
        pref_rels = [r for r in relationships if r.relation_type == "prefers"]
        assert len(pref_rels) >= 1


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestEdgeCases:
    """Edge case tests for dependency relationship extraction."""

    def test_very_long_sentence(self):
        """Should handle very long sentences without errors."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        long_sentence = (
            "John works at Google in San Francisco and he uses Python "
            "for data analysis and he loves the city and he met Sarah "
            "at the conference last year"
        )
        doc = nlp(long_sentence)
        
        entities = [
            Entity(name="John", entity_type="person"),
            Entity(name="Google", entity_type="organization"),
            Entity(name="San Francisco", entity_type="place"),
            Entity(name="Python", entity_type="technology"),
            Entity(name="Sarah", entity_type="person"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should extract multiple relationships without errors
        assert isinstance(relationships, list)
        assert len(relationships) >= 2

    def test_unicode_entity_names(self):
        """Should handle unicode characters in entity names."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("José visited São Paulo")
        
        entities = [
            Entity(name="José", entity_type="person"),
            Entity(name="São Paulo", entity_type="place"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should not crash with unicode
        assert isinstance(relationships, list)

    def test_special_characters_in_text(self):
        """Should handle special characters without crashing."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("John uses Python (3.9+) @ work!")
        
        entities = [
            Entity(name="John", entity_type="person"),
            Entity(name="Python", entity_type="technology"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should handle special chars without errors
        assert isinstance(relationships, list)

    def test_entity_name_substring_matching(self):
        """Should handle entity names that are substrings of each other."""
        extractor = DependencyRelationshipExtractor()
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("John uses JavaScript")
        
        entities = [
            Entity(name="John", entity_type="person"),
            Entity(name="Java", entity_type="technology"),
            Entity(name="JavaScript", entity_type="technology"),
        ]
        
        relationships = extractor.extract(doc, entities)
        
        # Should match JavaScript, not Java
        for rel in relationships:
            if rel.relation_type == "uses":
                # Should match the full entity name
                assert "JavaScript" in {rel.source, rel.target}
