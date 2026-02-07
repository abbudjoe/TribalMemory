"""Tests for entity and relationship validation (Issue #129).

Validates entities and relationships before they enter the graph store
to prevent garbage data from polluting the knowledge graph.
"""

import pytest
from tribalmemory.services.graph_store import (
    Entity,
    Relationship,
    EntityValidator,
    RelationshipValidator,
    HybridEntityExtractor,
    MIN_ENTITY_NAME_LENGTH,
)


class TestEntityValidator:
    """Tests for entity validation rules."""

    @pytest.fixture
    def validator(self):
        """Create an EntityValidator instance."""
        return EntityValidator()

    def test_valid_entity_passes(self, validator):
        """Valid entities should pass validation."""
        entity = Entity(name="auth-service", entity_type="service")
        assert validator.is_valid(entity), "Valid service entity should pass"

    def test_valid_person_passes(self, validator):
        """Valid person names should pass."""
        entity = Entity(name="Dr. Smith", entity_type="person")
        assert validator.is_valid(entity), "Valid person name should pass"

    def test_valid_technology_passes(self, validator):
        """Valid technology names should pass."""
        entity = Entity(name="PostgreSQL", entity_type="technology")
        assert validator.is_valid(entity), "Valid technology should pass"

    def test_max_name_length_valid(self, validator):
        """Names at or below max length (100 chars) should pass."""
        # Exactly 100 characters
        entity = Entity(name="a" * 100, entity_type="concept")
        assert validator.is_valid(entity), "100-char name should pass"

    def test_max_name_length_invalid(self, validator):
        """Names over max length (100 chars) should fail."""
        # 101 characters
        entity = Entity(name="a" * 101, entity_type="concept")
        assert not validator.is_valid(entity), "101-char name should fail"

    def test_all_caps_stopword_rejected_the(self, validator):
        """All-caps stopword 'THE' should be rejected."""
        entity = Entity(name="THE", entity_type="concept")
        assert not validator.is_valid(entity), "All-caps 'THE' should be rejected"

    def test_all_caps_stopword_rejected_and(self, validator):
        """All-caps stopword 'AND' should be rejected."""
        entity = Entity(name="AND", entity_type="concept")
        assert not validator.is_valid(entity), "All-caps 'AND' should be rejected"

    def test_all_caps_stopword_rejected_from(self, validator):
        """All-caps stopword 'FROM' should be rejected."""
        entity = Entity(name="FROM", entity_type="concept")
        assert not validator.is_valid(entity), "All-caps 'FROM' should be rejected"

    def test_all_caps_stopword_rejected_would(self, validator):
        """All-caps stopword 'WOULD' should be rejected."""
        entity = Entity(name="WOULD", entity_type="concept")
        assert not validator.is_valid(entity), "All-caps 'WOULD' should be rejected"

    def test_all_caps_stopword_rejected_before(self, validator):
        """All-caps stopword 'BEFORE' should be rejected."""
        entity = Entity(name="BEFORE", entity_type="concept")
        assert not validator.is_valid(entity), "All-caps 'BEFORE' should be rejected"

    def test_mixed_case_stopword_rejected_as_concept(self, validator):
        """Mixed-case common words should be rejected as concepts."""
        entity = Entity(name="The", entity_type="concept")
        assert not validator.is_valid(entity), "Common word 'The' as concept should be rejected"
    
    def test_mixed_case_stopword_passes_as_non_concept(self, validator):
        """Mixed-case words should pass as non-concept types (e.g., person name)."""
        entity = Entity(name="The", entity_type="person")
        assert validator.is_valid(entity), "Mixed-case 'The' as person should pass"

    def test_numeric_only_rejected(self, validator):
        """Numeric-only entities should be rejected."""
        entity = Entity(name="12345", entity_type="concept")
        assert not validator.is_valid(entity), "Numeric-only '12345' should be rejected"

    def test_numeric_only_with_leading_zero_rejected(self, validator):
        """Numeric entities with leading zeros should be rejected."""
        entity = Entity(name="00123", entity_type="concept")
        assert not validator.is_valid(entity), "Numeric '00123' should be rejected"

    def test_no_alphabetic_characters_rejected_dashes(self, validator):
        """Entities with no alphabetic characters should be rejected (dashes)."""
        entity = Entity(name="---", entity_type="concept")
        assert not validator.is_valid(entity), "No alphabetic chars '---' should be rejected"

    def test_no_alphabetic_characters_rejected_dots(self, validator):
        """Entities with no alphabetic characters should be rejected (dots)."""
        entity = Entity(name="...", entity_type="concept")
        assert not validator.is_valid(entity), "No alphabetic chars '...' should be rejected"

    def test_no_alphabetic_characters_rejected_symbols(self, validator):
        """Entities with no alphabetic characters should be rejected (symbols)."""
        entity = Entity(name="@#$%", entity_type="concept")
        assert not validator.is_valid(entity), "No alphabetic chars '@#$%' should be rejected"

    def test_alphanumeric_passes(self, validator):
        """Alphanumeric entities with letters should pass."""
        entity = Entity(name="auth123", entity_type="service")
        assert validator.is_valid(entity), "Alphanumeric 'auth123' should pass"

    def test_single_word_common_concept_rejected_the(self, validator):
        """Single-word common English 'the' as concept should be rejected."""
        entity = Entity(name="the", entity_type="concept")
        assert not validator.is_valid(entity), "Common word 'the' as concept should be rejected"

    def test_single_word_common_concept_rejected_and(self, validator):
        """Single-word common English 'and' as concept should be rejected."""
        entity = Entity(name="and", entity_type="concept")
        assert not validator.is_valid(entity), "Common word 'and' as concept should be rejected"

    def test_single_word_common_concept_rejected_useful(self, validator):
        """Single-word common English 'useful' as concept should be rejected."""
        entity = Entity(name="useful", entity_type="concept")
        assert not validator.is_valid(entity), "Common word 'useful' as concept should be rejected"

    def test_single_word_common_concept_rejected_check(self, validator):
        """Single-word common English 'check' as concept should be rejected."""
        entity = Entity(name="check", entity_type="concept")
        assert not validator.is_valid(entity), "Common word 'check' as concept should be rejected"

    def test_single_word_common_concept_rejected_good(self, validator):
        """Single-word common English 'good' as concept should be rejected."""
        entity = Entity(name="good", entity_type="concept")
        assert not validator.is_valid(entity), "Common word 'good' as concept should be rejected"

    def test_single_word_common_concept_rejected_nice(self, validator):
        """Single-word common English 'nice' as concept should be rejected."""
        entity = Entity(name="nice", entity_type="concept")
        assert not validator.is_valid(entity), "Common word 'nice' as concept should be rejected"

    def test_single_word_common_concept_rejected_great(self, validator):
        """Single-word common English 'great' as concept should be rejected."""
        entity = Entity(name="great", entity_type="concept")
        assert not validator.is_valid(entity), "Common word 'great' as concept should be rejected"

    def test_single_word_uncommon_concept_passes(self, validator):
        """Single-word uncommon concepts should pass."""
        entity = Entity(name="microservices", entity_type="concept")
        assert validator.is_valid(entity), "Uncommon concept 'microservices' should pass"

    def test_multi_word_concept_passes(self, validator):
        """Multi-word concepts should pass even if they contain common words."""
        entity = Entity(name="the authentication system", entity_type="concept")
        assert validator.is_valid(entity), "Multi-word concept should pass"

    def test_common_word_as_non_concept_passes(self, validator):
        """Common words as non-concept types should pass."""
        # Edge case: "Check" might be a service or organization name
        entity = Entity(name="Check", entity_type="organization")
        assert validator.is_valid(entity), "Common word as non-concept type should pass"

    def test_empty_name_rejected(self, validator):
        """Empty entity names should be rejected."""
        entity = Entity(name="", entity_type="concept")
        assert not validator.is_valid(entity), "Empty name should be rejected"

    def test_whitespace_only_name_rejected(self, validator):
        """Whitespace-only names should be rejected."""
        entity = Entity(name="   ", entity_type="concept")
        assert not validator.is_valid(entity), "Whitespace-only name should be rejected"

    def test_name_below_min_length_rejected(self, validator):
        """Names below MIN_ENTITY_NAME_LENGTH should be rejected."""
        # MIN_ENTITY_NAME_LENGTH is 3
        entity = Entity(name="ab", entity_type="concept")
        assert not validator.is_valid(entity), "2-char name should be rejected (min is 3)"

    def test_name_at_min_length_passes(self, validator):
        """Names at MIN_ENTITY_NAME_LENGTH should pass."""
        # MIN_ENTITY_NAME_LENGTH is 3
        entity = Entity(name="abc", entity_type="concept")
        assert validator.is_valid(entity), "3-char name should pass (min is 3)"


class TestRelationshipValidator:
    """Tests for relationship validation rules."""

    @pytest.fixture
    def validator(self):
        """Create a RelationshipValidator instance."""
        return RelationshipValidator()

    @pytest.fixture
    def entity_validator(self):
        """Create an EntityValidator for use in relationship validation."""
        return EntityValidator()

    def test_valid_relationship_passes(self, validator):
        """Valid relationships should pass."""
        rel = Relationship(source="auth-service", target="PostgreSQL", relation_type="uses")
        assert validator.is_valid(rel), "Valid relationship should pass"

    def test_valid_relationship_with_multi_word_entities(self, validator):
        """Relationships with multi-word entities should pass."""
        rel = Relationship(
            source="authentication service",
            target="user database",
            relation_type="connects_to"
        )
        assert validator.is_valid(rel), "Valid multi-word relationship should pass"

    def test_self_relationship_rejected_exact_match(self, validator):
        """Self-relationships (exact match) should be rejected."""
        rel = Relationship(source="auth-service", target="auth-service", relation_type="depends_on")
        assert not validator.is_valid(rel), "Self-relationship (exact) should be rejected"

    def test_self_relationship_rejected_case_insensitive(self, validator):
        """Self-relationships (case-insensitive) should be rejected."""
        rel = Relationship(source="Auth-Service", target="auth-service", relation_type="depends_on")
        assert not validator.is_valid(rel), "Self-relationship (case-insensitive) should be rejected"

    def test_self_relationship_rejected_uppercase(self, validator):
        """Self-relationships (all uppercase) should be rejected."""
        rel = Relationship(source="SERVICE", target="service", relation_type="uses")
        assert not validator.is_valid(rel), "Self-relationship (case variant) should be rejected"

    def test_source_invalid_entity_rejected_numeric(self, validator):
        """Relationships with numeric-only source should be rejected."""
        rel = Relationship(source="12345", target="PostgreSQL", relation_type="uses")
        assert not validator.is_valid(rel), "Numeric source should be rejected"

    def test_target_invalid_entity_rejected_numeric(self, validator):
        """Relationships with numeric-only target should be rejected."""
        rel = Relationship(source="auth-service", target="67890", relation_type="uses")
        assert not validator.is_valid(rel), "Numeric target should be rejected"

    def test_source_below_min_length_rejected(self, validator):
        """Relationships with source below MIN_ENTITY_NAME_LENGTH should be rejected."""
        rel = Relationship(source="ab", target="PostgreSQL", relation_type="uses")
        assert not validator.is_valid(rel), "Source below min length should be rejected"

    def test_target_below_min_length_rejected(self, validator):
        """Relationships with target below MIN_ENTITY_NAME_LENGTH should be rejected."""
        rel = Relationship(source="auth-service", target="db", relation_type="uses")
        assert not validator.is_valid(rel), "Target below min length should be rejected"

    def test_source_no_alphabetic_rejected(self, validator):
        """Relationships with source having no alphabetic chars should be rejected."""
        rel = Relationship(source="---", target="PostgreSQL", relation_type="connects_to")
        assert not validator.is_valid(rel), "Source with no letters should be rejected"

    def test_target_no_alphabetic_rejected(self, validator):
        """Relationships with target having no alphabetic chars should be rejected."""
        rel = Relationship(source="auth-service", target="...", relation_type="connects_to")
        assert not validator.is_valid(rel), "Target with no letters should be rejected"

    def test_source_all_caps_stopword_rejected(self, validator):
        """Relationships with all-caps stopword source should be rejected."""
        rel = Relationship(source="THE", target="database", relation_type="uses")
        assert not validator.is_valid(rel), "All-caps stopword source should be rejected"

    def test_target_all_caps_stopword_rejected(self, validator):
        """Relationships with all-caps stopword target should be rejected."""
        rel = Relationship(source="service", target="AND", relation_type="depends_on")
        assert not validator.is_valid(rel), "All-caps stopword target should be rejected"

    def test_both_entities_at_min_length_passes(self, validator):
        """Relationships with both entities at min length should pass."""
        rel = Relationship(source="api", target="db2", relation_type="uses")
        assert validator.is_valid(rel), "Both at min length should pass"

    def test_empty_source_rejected(self, validator):
        """Relationships with empty source should be rejected."""
        rel = Relationship(source="", target="PostgreSQL", relation_type="uses")
        assert not validator.is_valid(rel), "Empty source should be rejected"

    def test_empty_target_rejected(self, validator):
        """Relationships with empty target should be rejected."""
        rel = Relationship(source="auth-service", target="", relation_type="uses")
        assert not validator.is_valid(rel), "Empty target should be rejected"


class TestHybridEntityExtractorWithValidation:
    """Integration tests for entity extraction with validation."""

    @pytest.fixture
    def extractor(self):
        """Create a HybridEntityExtractor (without spaCy to avoid dependency issues)."""
        return HybridEntityExtractor(use_spacy=False)

    def test_extract_valid_entities_only(self, extractor):
        """Extractor should filter out invalid entities."""
        text = "The auth-service uses PostgreSQL and THE AND connects to 12345."
        
        entities = extractor.extract(text)
        
        # Should extract valid entities
        names = {e.name for e in entities}
        assert "auth-service" in names, "Valid service should be extracted"
        assert "PostgreSQL" in names, "Valid technology should be extracted"
        
        # Should reject garbage
        assert "THE" not in names, "All-caps stopword should be filtered"
        assert "AND" not in names, "All-caps stopword should be filtered"
        assert "12345" not in names, "Numeric-only should be filtered"

    def test_extract_with_relationships_filters_invalid(self, extractor):
        """Extractor should filter invalid entities and relationships."""
        text = "The auth-service uses PostgreSQL. THE connects to AND."
        
        entities, relationships = extractor.extract_with_relationships(text)
        
        # Valid relationship should pass
        valid_rels = [r for r in relationships if r.source == "auth-service" and r.target == "PostgreSQL"]
        assert len(valid_rels) > 0, "Valid relationship should be extracted"
        
        # Invalid relationship should be filtered
        invalid_rels = [r for r in relationships if r.source == "THE" or r.target == "AND"]
        assert len(invalid_rels) == 0, "Invalid relationships should be filtered"

    def test_extract_filters_self_relationships(self, extractor):
        """Extractor should filter self-relationships."""
        text = "The service connects to service."
        
        entities, relationships = extractor.extract_with_relationships(text)
        
        # Should not have self-relationship
        self_rels = [r for r in relationships if r.source.lower() == r.target.lower()]
        assert len(self_rels) == 0, "Self-relationships should be filtered"

    def test_garbage_text_yields_no_entities(self, extractor):
        """Feeding garbage text should yield no entities."""
        garbage = "THE AND FOR BUT OR 12345 --- ... !!! @#$%"
        
        entities = extractor.extract(garbage)
        
        assert len(entities) == 0, "Garbage text should yield no valid entities"

    def test_mixed_valid_and_garbage_filters_correctly(self, extractor):
        """Mixed valid and garbage text should filter correctly."""
        text = "PostgreSQL and Redis are databases. THE AND 12345 --- are noise."
        
        entities = extractor.extract(text)
        
        names = {e.name for e in entities}
        assert "PostgreSQL" in names, "Valid entity should pass"
        assert "Redis" in names, "Valid entity should pass"
        assert "THE" not in names, "Garbage should be filtered"
        assert "12345" not in names, "Numeric garbage should be filtered"
        assert "---" not in names, "Symbol garbage should be filtered"

    def test_entity_count_matches_valid_entities(self, extractor):
        """Entity count should reflect only valid entities."""
        text = "auth-service uses PostgreSQL and Redis. THE AND 12345."
        
        entities = extractor.extract(text)
        
        # Should have 3 valid entities: auth-service, PostgreSQL, Redis
        assert len(entities) == 3, f"Expected 3 valid entities, got {len(entities)}: {[e.name for e in entities]}"

    def test_empty_text_yields_no_entities(self, extractor):
        """Empty text should yield no entities."""
        assert extractor.extract("") == [], "Empty text should yield no entities"
        assert extractor.extract("   ") == [], "Whitespace text should yield no entities"

    def test_valid_entities_preserved_exactly(self, extractor):
        """Valid entities should be preserved with correct names and types."""
        text = "The auth-service uses PostgreSQL."
        
        entities = extractor.extract(text)
        
        # Check auth-service
        auth_service = next((e for e in entities if "auth-service" in e.name), None)
        assert auth_service is not None, "auth-service should be extracted"
        assert auth_service.entity_type == "service", "auth-service should be typed as service"
        
        # Check PostgreSQL
        postgres = next((e for e in entities if "PostgreSQL" in e.name), None)
        assert postgres is not None, "PostgreSQL should be extracted"
        assert postgres.entity_type == "technology", "PostgreSQL should be typed as technology"
