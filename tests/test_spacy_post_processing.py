"""Tests for spaCy entity post-processing (Issue #131).

Tests the SpacyPostProcessor class which fixes misclassified entities
after spaCy NER extraction. Focuses on:
- Rejecting product names misclassified as people
- Rejecting food names misclassified as people
- Reclassifying product-like entities from person to product
- Preserving real person names, places, dates, orgs
"""
import pytest
from tribalmemory.services.graph_store import (
    Entity,
    SpacyPostProcessor,
    SPACY_AVAILABLE,
)


pytestmark = pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")


class TestSpacyPostProcessor:
    """Tests for SpacyPostProcessor entity cleaning."""

    # =========================================================================
    # Product keyword detection tests
    # =========================================================================

    def test_rejects_razer_kraken_x_as_person(self):
        """Should reject 'Razer Kraken X' (product) misclassified as person."""
        processor = SpacyPostProcessor()
        entity = Entity(name="Razer Kraken X", entity_type="person")
        result = processor.process(entity)
        assert result is None, "Product 'Razer Kraken X' should be rejected"

    def test_rejects_kindle_as_person(self):
        """Should reject 'Kindle' (product) misclassified as person."""
        processor = SpacyPostProcessor()
        entity = Entity(name="Kindle", entity_type="person")
        result = processor.process(entity)
        assert result is None, "'Kindle' should be rejected as person"

    def test_rejects_iphone_as_person(self):
        """Should reject 'iPhone' (product) misclassified as person."""
        processor = SpacyPostProcessor()
        entity = Entity(name="iPhone", entity_type="person")
        result = processor.process(entity)
        assert result is None, "'iPhone' should be rejected as person"

    def test_rejects_airpods_pro_max_as_person(self):
        """Should reject 'AirPods Pro Max' (product) misclassified as person."""
        processor = SpacyPostProcessor()
        entity = Entity(name="AirPods Pro Max", entity_type="person")
        result = processor.process(entity)
        assert result is None, "'AirPods Pro Max' should be rejected"

    def test_rejects_product_with_model_number(self):
        """Should reject entities with model numbers like X100, Pro Max."""
        processor = SpacyPostProcessor()
        # Model number pattern: letter + digits
        entity1 = Entity(name="Camera X100", entity_type="person")
        entity2 = Entity(name="Speaker Z300", entity_type="person")
        
        assert processor.process(entity1) is None, (
            "'Camera X100' should be rejected (model number)"
        )
        assert processor.process(entity2) is None, (
            "'Speaker Z300' should be rejected (model number)"
        )

    # =========================================================================
    # Food name detection tests
    # =========================================================================

    def test_rejects_sarson_ka_saag_as_person(self):
        """Should reject 'Sarson Ka Saag' (food) misclassified as person."""
        processor = SpacyPostProcessor()
        entity = Entity(name="Sarson Ka Saag", entity_type="person")
        result = processor.process(entity)
        assert result is None, "'Sarson Ka Saag' should be rejected"

    def test_rejects_biryani_as_person(self):
        """Should reject 'Biryani' (food) misclassified as person."""
        processor = SpacyPostProcessor()
        entity = Entity(name="Biryani", entity_type="person")
        result = processor.process(entity)
        assert result is None, "'Biryani' should be rejected"

    def test_rejects_pad_thai_as_person(self):
        """Should reject 'Pad Thai' (food) misclassified as person."""
        processor = SpacyPostProcessor()
        entity = Entity(name="Pad Thai", entity_type="person")
        result = processor.process(entity)
        assert result is None, "'Pad Thai' should be rejected"

    def test_rejects_common_food_names(self):
        """Should reject a variety of commonly misclassified food items."""
        processor = SpacyPostProcessor()
        food_names = [
            "Butter Chicken",
            "Tikka Masala",
            "Palak Paneer",
            "Kung Pao",
            "Tom Yum",
        ]
        
        for food_name in food_names:
            entity = Entity(name=food_name, entity_type="person")
            result = processor.process(entity)
            assert result is None, f"'{food_name}' should be rejected as person"

    # =========================================================================
    # Reclassification tests (person → product)
    # =========================================================================

    def test_reclassifies_galaxy_to_product(self):
        """Should reclassify 'Galaxy' from person to product."""
        processor = SpacyPostProcessor()
        entity = Entity(name="Galaxy", entity_type="person")
        result = processor.process(entity)
        
        assert result is not None, "'Galaxy' should be reclassified, not rejected"
        assert result.entity_type == "product", (
            f"'Galaxy' should be reclassified to product, got: {result.entity_type}"
        )
        assert result.name == "Galaxy", "Name should be preserved"

    def test_reclassifies_kraken_to_product(self):
        """Should reclassify 'Kraken' from person to product."""
        processor = SpacyPostProcessor()
        entity = Entity(name="Kraken", entity_type="person")
        result = processor.process(entity)
        
        assert result is not None
        assert result.entity_type == "product"

    def test_reclassifies_airpods_to_product(self):
        """Should reclassify 'AirPods' from person to product."""
        processor = SpacyPostProcessor()
        entity = Entity(name="AirPods", entity_type="person")
        result = processor.process(entity)
        
        assert result is not None
        assert result.entity_type == "product"

    # =========================================================================
    # Real person name preservation tests
    # =========================================================================

    def test_preserves_real_person_name_sarah_thompson(self):
        """Should preserve real person name 'Sarah Thompson'."""
        processor = SpacyPostProcessor()
        entity = Entity(name="Sarah Thompson", entity_type="person")
        result = processor.process(entity)
        
        assert result is not None, "'Sarah Thompson' should be preserved"
        assert result.entity_type == "person", (
            "Real person name should remain as 'person' type"
        )
        assert result.name == "Sarah Thompson", "Name should be unchanged"

    def test_preserves_dr_smith_after_normalization(self):
        """Should preserve 'Smith' (after title normalization)."""
        processor = SpacyPostProcessor()
        # Title normalization happens before post-processing
        entity = Entity(name="Smith", entity_type="person")
        result = processor.process(entity)
        
        assert result is not None, "'Smith' should be preserved"
        assert result.entity_type == "person"

    def test_preserves_common_person_names(self):
        """Should preserve a variety of real person names."""
        processor = SpacyPostProcessor()
        person_names = [
            "John Doe",
            "Jane Smith",
            "Robert Johnson",
            "Emily Davis",
            "Michael Chen",
        ]
        
        for name in person_names:
            entity = Entity(name=name, entity_type="person")
            result = processor.process(entity)
            assert result is not None, f"'{name}' should be preserved"
            assert result.entity_type == "person", (
                f"'{name}' should remain as person"
            )

    # =========================================================================
    # No-op for non-person entity types
    # =========================================================================

    def test_does_not_modify_place_entities(self):
        """Should not modify place entities (pass through unchanged)."""
        processor = SpacyPostProcessor()
        entity = Entity(name="New York", entity_type="place")
        result = processor.process(entity)
        
        assert result is not None, "Place entities should pass through"
        assert result.entity_type == "place", "Type should remain 'place'"
        assert result.name == "New York", "Name should be unchanged"

    def test_does_not_modify_date_entities(self):
        """Should not modify date entities (pass through unchanged)."""
        processor = SpacyPostProcessor()
        entity = Entity(name="March 15th", entity_type="date")
        result = processor.process(entity)
        
        assert result is not None
        assert result.entity_type == "date"

    def test_does_not_modify_organization_entities(self):
        """Should not modify organization entities (pass through unchanged)."""
        processor = SpacyPostProcessor()
        entity = Entity(name="Google", entity_type="organization")
        result = processor.process(entity)
        
        assert result is not None
        assert result.entity_type == "organization"

    def test_does_not_modify_event_entities(self):
        """Should not modify event entities (pass through unchanged)."""
        processor = SpacyPostProcessor()
        entity = Entity(name="World Cup", entity_type="event")
        result = processor.process(entity)
        
        assert result is not None
        assert result.entity_type == "event"

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_case_insensitive_product_keyword_matching(self):
        """Should match product keywords case-insensitively."""
        processor = SpacyPostProcessor()
        # Lowercase variant
        entity1 = Entity(name="iphone 15", entity_type="person")
        # Mixed case variant
        entity2 = Entity(name="KINDLE FIRE", entity_type="person")
        
        assert processor.process(entity1) is None, "Lowercase 'iphone' should match"
        assert processor.process(entity2) is None, "Uppercase 'KINDLE' should match"

    def test_handles_entity_with_metadata(self):
        """Should preserve metadata when processing entities."""
        processor = SpacyPostProcessor()
        entity = Entity(
            name="Sarah",
            entity_type="person",
            metadata={"spacy_label": "PERSON", "confidence": 0.95}
        )
        result = processor.process(entity)
        
        assert result is not None
        assert result.metadata == entity.metadata, (
            "Metadata should be preserved"
        )

    def test_handles_empty_entity_name(self):
        """Should handle empty entity name gracefully."""
        processor = SpacyPostProcessor()
        entity = Entity(name="", entity_type="person")
        result = processor.process(entity)
        # Should either reject or pass through (implementation choice)
        assert result is None or result.name == ""

    def test_handles_whitespace_only_name(self):
        """Should handle whitespace-only entity name."""
        processor = SpacyPostProcessor()
        entity = Entity(name="   ", entity_type="person")
        result = processor.process(entity)
        # Should either reject or pass through
        assert result is None or result.name.strip() == ""

    # =========================================================================
    # Mixed case + numbers detection
    # =========================================================================

    def test_rejects_mixed_case_numbers_product_pattern(self):
        """Should detect product-like patterns with mixed case and numbers."""
        processor = SpacyPostProcessor()
        # Patterns like "iPhone 15", "Galaxy S23", "Model X100"
        entity1 = Entity(name="iPhone 15", entity_type="person")
        entity2 = Entity(name="Galaxy S23", entity_type="person")
        
        assert processor.process(entity1) is None
        assert processor.process(entity2) is None

    def test_preserves_person_name_with_numbers_edge_case(self):
        """Should preserve person names that happen to have numbers.
        
        Note: This is an edge case. Names like 'Agent 47' or 'R2D2'
        are rare for real people. The implementation should prioritize
        catching products over this edge case.
        """
        processor = SpacyPostProcessor()
        # Uncommon but valid person name with number
        entity = Entity(name="John 3rd", entity_type="person")
        result = processor.process(entity)
        # Implementation choice: may reject or preserve
        # This test documents the behavior
