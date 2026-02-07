"""Tests for regex-based entity extraction cleanup (Issue #130).

This test suite verifies that garbage relationship patterns are removed
and remaining patterns are tightened to require known entities on both sides.
"""

import pytest
from tribalmemory.services.graph_store import (
    EntityExtractor,
    HybridEntityExtractor,
    Entity,
    Relationship,
)


class TestServesPatternRemoval:
    """Test that the overly-broad 'serves' pattern is removed."""
    
    def test_serves_pattern_not_in_relationship_patterns(self):
        """The 'serves' pattern should be completely removed from RELATIONSHIP_PATTERNS."""
        extractor = EntityExtractor()
        
        # Check that no pattern has 'serves' as the relation type
        for pattern, rel_type in extractor.RELATIONSHIP_PATTERNS:
            assert rel_type != 'serves', (
                "The 'serves' relationship pattern should be removed "
                "because it's too broad and produces garbage relationships"
            )
    
    def test_for_pattern_does_not_extract_relationships(self):
        """Text with 'X for Y' should not create relationships."""
        extractor = EntityExtractor()
        
        # Examples that previously triggered 'serves' relationships
        test_cases = [
            "waiting for the bus",
            "looking for my keys",
            "recipe for dinner",
            "time for bed",
        ]
        
        for text in test_cases:
            _, relationships = extractor.extract_with_relationships(text)
            
            # Should produce zero relationships (no known entities match pattern)
            assert len(relationships) == 0, (
                f"Text '{text}' should not produce relationships, "
                f"but got: {relationships}"
            )


class TestKnownEntityValidation:
    """Test that relationship patterns require both sides to be known entities."""
    
    def test_is_known_entity_method_exists(self):
        """EntityExtractor should have _is_known_entity() method."""
        extractor = EntityExtractor()
        assert hasattr(extractor, '_is_known_entity'), (
            "EntityExtractor should have _is_known_entity() method"
        )
        assert callable(extractor._is_known_entity), (
            "_is_known_entity should be callable"
        )
    
    def test_is_known_entity_recognizes_technologies(self):
        """_is_known_entity() should recognize TECHNOLOGIES set members."""
        extractor = EntityExtractor()
        
        # Known technologies
        assert extractor._is_known_entity('postgresql'), (
            "postgresql is in TECHNOLOGIES and should be recognized"
        )
        assert extractor._is_known_entity('PostgreSQL'), (
            "PostgreSQL (mixed case) should be recognized"
        )
        assert extractor._is_known_entity('redis'), (
            "redis is in TECHNOLOGIES and should be recognized"
        )
        
        # Unknown entities
        assert not extractor._is_known_entity('waiting'), (
            "waiting is not a known entity"
        )
        assert not extractor._is_known_entity('bus'), (
            "bus is not a known entity"
        )
    
    def test_is_known_entity_recognizes_service_pattern(self):
        """_is_known_entity() should recognize SERVICE_PATTERN matches."""
        extractor = EntityExtractor()
        
        # Valid service names
        assert extractor._is_known_entity('auth-service'), (
            "auth-service matches SERVICE_PATTERN"
        )
        assert extractor._is_known_entity('payment-gateway'), (
            "payment-gateway matches SERVICE_PATTERN"
        )
        assert extractor._is_known_entity('user-auth-api'), (
            "user-auth-api matches SERVICE_PATTERN"
        )
        
        # Invalid service names (too short, no hyphen, etc.)
        assert not extractor._is_known_entity('user'), (
            "user doesn't match SERVICE_PATTERN (no hyphen)"
        )
        assert not extractor._is_known_entity('my-x'), (
            "my-x is too short (second segment < 4 chars, no known suffix)"
        )
    
    def test_uses_requires_both_sides_known(self):
        """'uses' pattern should only fire when both entities are known."""
        extractor = EntityExtractor()
        
        # Valid: both sides are known
        text = "auth-service uses postgresql"
        _, relationships = extractor.extract_with_relationships(text)
        assert len(relationships) == 1, (
            "Should extract 1 relationship when both sides are known entities"
        )
        assert relationships[0].relation_type == 'uses'
        assert relationships[0].source == 'auth-service'
        assert relationships[0].target == 'postgresql'
        
        # Invalid: only one side is known
        text_invalid = "waiting uses the bus"
        _, relationships_invalid = extractor.extract_with_relationships(text_invalid)
        assert len(relationships_invalid) == 0, (
            "Should not extract relationship when both sides aren't known entities"
        )
    
    def test_connects_to_requires_both_sides_known(self):
        """'connects_to' pattern should only fire when both entities are known."""
        extractor = EntityExtractor()
        
        # Valid: both sides are known
        text = "api-gateway connects to auth-service"
        _, relationships = extractor.extract_with_relationships(text)
        assert len(relationships) == 1
        assert relationships[0].relation_type == 'connects_to'
        
        # Invalid: neither side is known
        text_invalid = "alice connects to bob"
        _, relationships_invalid = extractor.extract_with_relationships(text_invalid)
        assert len(relationships_invalid) == 0, (
            "Should not extract relationship when entities aren't known"
        )


class TestPersonalConversationText:
    """Test that personal conversation text produces zero garbage relationships."""
    
    @pytest.mark.parametrize("text", [
        "I'm waiting for the bus to arrive.",
        "Looking for my keys in the kitchen.",
        "Need to pick up groceries for dinner tonight.",
        "Recipe for chocolate cake.",
        "Appointment for next Tuesday at 3pm.",
        "This is for my grandmother's birthday.",
        "Running late for the meeting.",
        "Thanks for the help!",
    ])
    def test_personal_text_no_garbage_relationships(self, text):
        """Personal conversation text should produce zero garbage relationships."""
        extractor = EntityExtractor()
        _, relationships = extractor.extract_with_relationships(text)
        
        assert len(relationships) == 0, (
            f"Personal text '{text}' should produce zero relationships, "
            f"but got: {relationships}"
        )


class TestSoftwareArchitectureText:
    """Test that software architecture text still produces valid relationships."""
    
    def test_software_entities_and_relationships_still_work(self):
        """Software architecture text should still extract valid entities and relationships."""
        extractor = EntityExtractor()
        
        text = (
            "auth-service uses postgresql for storage. "
            "api-gateway connects to auth-service. "
            "payment-worker stores data in redis cache."
        )
        
        entities, relationships = extractor.extract_with_relationships(text)
        
        # Should extract known entities
        entity_names = {e.name.lower() for e in entities}
        assert 'auth-service' in entity_names
        assert 'postgresql' in entity_names or 'postgres' in entity_names
        assert 'api-gateway' in entity_names
        assert 'payment-worker' in entity_names
        assert 'redis' in entity_names
        
        # Should extract valid relationships
        assert len(relationships) > 0, (
            "Software architecture text should still produce relationships"
        )
        
        # Check specific relationships
        rel_types = {r.relation_type for r in relationships}
        assert 'uses' in rel_types or 'stores_in' in rel_types, (
            "Should extract known relationship types"
        )


class TestExtractionContext:
    """Test extraction_context parameter on HybridEntityExtractor."""
    
    def test_hybrid_extractor_has_context_parameter(self):
        """HybridEntityExtractor.__init__() should accept extraction_context parameter."""
        # Should not raise
        extractor_personal = HybridEntityExtractor(extraction_context="personal")
        extractor_software = HybridEntityExtractor(extraction_context="software")
        
        # Check that context is stored
        assert hasattr(extractor_personal, '_extraction_context'), (
            "HybridEntityExtractor should store extraction_context"
        )
        assert extractor_personal._extraction_context == "personal"
        assert extractor_software._extraction_context == "software"
    
    def test_context_personal_disables_regex_relationships(self):
        """When context='personal', regex relationship extraction should be disabled."""
        extractor = HybridEntityExtractor(
            use_spacy=False,  # Disable spaCy for isolated test
            extraction_context="personal"
        )
        
        # Software-like text that would normally produce relationships
        text = "auth-service uses postgresql"
        entities, relationships = extractor.extract_with_relationships(text)
        
        # Should still extract entities
        assert len(entities) > 0, (
            "Should still extract entities in personal context"
        )
        
        # Should NOT extract relationships
        assert len(relationships) == 0, (
            "Personal context should disable regex relationship extraction entirely"
        )
    
    def test_context_software_enables_regex_relationships(self):
        """When context='software', regex relationship extraction should work."""
        extractor = HybridEntityExtractor(
            use_spacy=False,  # Disable spaCy for isolated test
            extraction_context="software"
        )
        
        text = "auth-service uses postgresql"
        entities, relationships = extractor.extract_with_relationships(text)
        
        # Should extract both entities and relationships
        assert len(entities) > 0
        assert len(relationships) > 0, (
            "Software context should enable regex relationship extraction"
        )
    
    def test_default_context_is_personal(self):
        """Default extraction_context should be 'personal' for safety."""
        extractor = HybridEntityExtractor(use_spacy=False)
        
        # Should default to personal context
        assert hasattr(extractor, '_extraction_context')
        assert extractor._extraction_context == "personal", (
            "Default context should be 'personal' to avoid garbage relationships"
        )
        
        # Verify it behaves like personal context (no relationships on software text)
        text = "auth-service uses postgresql"
        _, relationships = extractor.extract_with_relationships(text)
        assert len(relationships) == 0, (
            "Default personal context should not extract relationships"
        )


class TestBackwardCompatibility:
    """Test that changes don't break existing valid use cases."""
    
    def test_software_entities_still_extracted(self):
        """Known software entities should still be extracted correctly."""
        extractor = EntityExtractor()
        
        text = "auth-service connects to postgresql via pgbouncer"
        entities = extractor.extract(text)
        
        entity_names = {e.name.lower() for e in entities}
        assert 'auth-service' in entity_names
        assert 'postgresql' in entity_names or 'postgres' in entity_names
        assert 'pgbouncer' in entity_names
    
    def test_entity_types_still_inferred(self):
        """Entity types should still be inferred correctly."""
        extractor = EntityExtractor()
        
        entities = extractor.extract("payment-api uses user-database")
        
        # Find the entities
        payment_api = next((e for e in entities if 'payment' in e.name.lower()), None)
        user_db = next((e for e in entities if 'database' in e.name.lower()), None)
        
        assert payment_api is not None
        assert payment_api.entity_type in ('service', 'api')
        
        if user_db:  # Might be extracted
            assert user_db.entity_type in ('database', 'service')
    
    def test_hybrid_extractor_still_combines_extractors(self):
        """HybridEntityExtractor should still combine regex and spaCy (when available)."""
        extractor = HybridEntityExtractor(extraction_context="software")
        
        # Text with both software and personal entities
        text = "Dr. Smith deployed auth-service to AWS"
        entities = extractor.extract(text)
        
        entity_names = {e.name.lower() for e in entities}
        
        # Should extract software entities via regex
        assert 'auth-service' in entity_names
        assert 'aws' in entity_names
        
        # Should extract person via spaCy (if available)
        if extractor.has_spacy:
            # spaCy should extract "Smith" (title stripped)
            assert any('smith' in name for name in entity_names), (
                "spaCy should extract person name (title-stripped)"
            )
