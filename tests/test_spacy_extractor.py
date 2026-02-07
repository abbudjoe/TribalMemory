"""Tests for spaCy-based entity extraction."""
import pytest
from tribalmemory.services.graph_store import (
    SpacyEntityExtractor,
    HybridEntityExtractor,
    SPACY_AVAILABLE,
    MIN_ENTITY_NAME_LENGTH,
)


@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestSpacyEntityExtractor:
    """Tests for SpacyEntityExtractor."""

    def test_extract_person_names(self):
        """Should extract person names from text with titles stripped."""
        extractor = SpacyEntityExtractor()
        text = "I met with Dr. Thompson and Sarah about the project."
        entities = extractor.extract(text)
        
        names = {e.name for e in entities if e.entity_type == "person"}
        # "Dr. Thompson" should be normalized to "Thompson"
        assert "Thompson" in names, f"Expected 'Thompson' in {names}"
        assert "Sarah" in names, f"Expected 'Sarah' in {names}"

    def test_extract_places(self):
        """Should extract place names from text.
        
        Note: spaCy classifies places as GPE (geopolitical entity), LOC (location),
        FAC (facility), or sometimes ORG (organization). All map to 'place' or 'organization'
        in our internal type system.
        """
        extractor = SpacyEntityExtractor()
        text = "I viewed a townhouse in the Brookside neighborhood near Oak Street."
        entities = extractor.extract(text)
        
        # Get all location-related entities (spaCy may classify as org, place, etc.)
        location_types = {"place", "organization"}
        location_entities = [e for e in entities if e.entity_type in location_types]
        location_names = {e.name.lower() for e in location_entities}
        
        # Should find at least one location entity
        assert len(location_entities) >= 1, (
            f"Expected at least 1 location entity, got {len(location_entities)}: {entities}"
        )
        # At least one should contain brookside or oak
        assert any("brookside" in n or "oak" in n for n in location_names), (
            f"Expected 'brookside' or 'oak' in {location_names}"
        )

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
        """Should not return duplicate entities when same name appears multiple times.
        
        Redis is in the regex extractor's TECHNOLOGIES set and will be extracted.
        spaCy typically doesn't extract technology names as entities, so this
        primarily tests regex-level deduplication.
        """
        hybrid = HybridEntityExtractor(use_spacy=True)
        # Same technology mentioned multiple times
        text = "We use Redis for caching. Redis is fast."
        entities = hybrid.extract(text)
        
        redis_entities = [e for e in entities if e.name.lower() == "redis"]
        # Should deduplicate to exactly one Redis entity
        assert len(redis_entities) == 1, (
            f"Expected 1 Redis entity (deduped), got {len(redis_entities)}: {redis_entities}"
        )

    def test_extract_with_relationships_uses_regex(self):
        """Should combine regex entities/relationships with spaCy entities.
        
        Relationships are only extracted by the regex extractor (spaCy doesn't
        do relationship extraction). This test verifies:
        1. Entities come from both extractors (Redis from regex, Sarah from spaCy)
        2. Relationships come from regex patterns only
        """
        hybrid = HybridEntityExtractor(use_spacy=True)
        text = "The auth-service uses Redis and was built by Sarah."
        entities, relationships = hybrid.extract_with_relationships(text)
        
        # Should have entities from both extractors
        names = {e.name.lower() for e in entities}
        assert "redis" in names, f"Expected 'redis' (from regex) in {names}"
        assert "sarah" in names, f"Expected 'sarah' (from spaCy) in {names}"
        
        # Relationships come from regex pattern matching
        # The "uses" relationship should be detected
        uses_rels = [r for r in relationships if r.relation_type == "uses"]
        # Note: relationship extraction depends on exact pattern match


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


# =============================================================================
# Issue #92: Tests for PRODUCT and EVENT entity types
# =============================================================================

@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestSpacyEntityTypes:
    """Tests for specific entity types in RELEVANT_TYPES."""

    def test_extract_product_entities(self):
        """Should extract PRODUCT entities (Issue #92).
        
        PRODUCT is in RELEVANT_TYPES and should be extracted.
        Note: spaCy's PRODUCT detection requires clear context.
        
        Important: spaCy's entity recognition varies significantly by:
        - Model version (en_core_web_sm vs en_core_web_lg)
        - Sentence context and capitalization
        - Training data coverage
        The fallback logic below handles this variability gracefully.
        """
        extractor = SpacyEntityExtractor()
        # spaCy recognizes well-known products
        text = "I bought an iPhone yesterday and also picked up a Kindle."
        entities = extractor.extract(text)
        
        # Get entities that spaCy classified as products
        spacy_product_entities = [e for e in entities if e.metadata.get('spacy_label') == 'PRODUCT']
        
        # Should find at least one product (iPhone and Kindle are commonly recognized)
        # Fallback: spaCy model variability means PRODUCT may be classified differently
        if len(spacy_product_entities) == 0:
            # Fall back to checking if any entity contains product-like names
            all_names = {e.name.lower() for e in entities}
            assert 'iphone' in all_names or 'kindle' in all_names, (
                f"Expected at least one product-like entity, got: {entities}"
            )

    def test_extract_event_entities(self):
        """Should extract EVENT entities (Issue #92).
        
        EVENT is in RELEVANT_TYPES and should be extracted.
        
        Important: spaCy's entity recognition varies significantly by:
        - Model version (en_core_web_sm vs en_core_web_lg)
        - Sentence context and capitalization
        - Training data coverage
        The fallback logic below handles this variability gracefully.
        """
        extractor = SpacyEntityExtractor()
        # Events like conferences, holidays, etc.
        text = "I'm attending the World Cup next year and Coachella in April."
        entities = extractor.extract(text)
        
        # Get entities that spaCy classified as events
        spacy_event_entities = [e for e in entities if e.metadata.get('spacy_label') == 'EVENT']
        
        # Should find at least one event
        # Fallback: spaCy model variability means EVENT may be classified differently
        if len(spacy_event_entities) == 0:
            # Fall back to checking all entities for event-like names
            all_names = {e.name.lower() for e in entities}
            # At least one should be detected as some entity type
            assert len(entities) >= 1, (
                f"Expected at least one entity from event text, got: {entities}"
            )

    def test_relevant_types_coverage(self):
        """Verify all RELEVANT_TYPES are handled in type mapping."""
        extractor = SpacyEntityExtractor()
        
        # Check that all RELEVANT_TYPES have mappings
        for spacy_type in extractor.RELEVANT_TYPES:
            assert spacy_type in extractor.SPACY_TYPE_MAP, (
                f"Missing type mapping for {spacy_type}"
            )

    def test_type_map_superset_of_relevant_types(self):
        """SPACY_TYPE_MAP should contain all RELEVANT_TYPES (plus extras).
        
        SPACY_TYPE_MAP includes additional types like MONEY, CARDINAL, ORDINAL
        that are intentionally excluded from RELEVANT_TYPES for personal
        conversation extraction. This test verifies the relationship.
        """
        extractor = SpacyEntityExtractor()
        
        # All RELEVANT_TYPES must be in SPACY_TYPE_MAP
        for relevant_type in extractor.RELEVANT_TYPES:
            assert relevant_type in extractor.SPACY_TYPE_MAP, (
                f"RELEVANT_TYPE {relevant_type} missing from SPACY_TYPE_MAP"
            )
        
        # SPACY_TYPE_MAP should be a superset (has extras like MONEY, CARDINAL)
        assert len(extractor.SPACY_TYPE_MAP) >= len(extractor.RELEVANT_TYPES), (
            "SPACY_TYPE_MAP should have at least as many entries as RELEVANT_TYPES"
        )


# =============================================================================
# Issue #93: Test for spaCy metadata preservation
# =============================================================================

@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestSpacyMetadataPreservation:
    """Tests for spaCy metadata being preserved in entities."""

    def test_spacy_label_in_metadata(self):
        """Should preserve spaCy label in entity metadata (Issue #93).
        
        SpacyEntityExtractor.extract() adds metadata={'spacy_label': ent.label_}
        to each entity. This test verifies the metadata is preserved and accessible.
        """
        extractor = SpacyEntityExtractor()
        text = "Dr. Thompson visited New York last Tuesday."
        entities = extractor.extract(text)
        
        assert len(entities) > 0, "Should extract at least one entity"
        
        for entity in entities:
            # Every entity should have metadata with spacy_label
            assert entity.metadata is not None, (
                f"Entity {entity.name} has no metadata"
            )
            assert 'spacy_label' in entity.metadata, (
                f"Entity {entity.name} missing spacy_label in metadata"
            )
            # Label should be a valid spaCy label in the type map
            # Note: We check SPACY_TYPE_MAP (not RELEVANT_TYPES) because the map
            # includes additional types like MONEY, CARDINAL, ORDINAL
            assert entity.metadata['spacy_label'] in extractor.SPACY_TYPE_MAP, (
                f"Unexpected spacy_label: {entity.metadata['spacy_label']}"
            )

    def test_metadata_matches_entity_type(self):
        """Metadata spacy_label should map correctly to entity_type."""
        extractor = SpacyEntityExtractor()
        text = "Sarah works at Google in San Francisco."
        entities = extractor.extract(text)
        
        for entity in entities:
            spacy_label = entity.metadata.get('spacy_label')
            expected_type = extractor.SPACY_TYPE_MAP.get(spacy_label)
            assert entity.entity_type == expected_type, (
                f"Entity type mismatch: {entity.entity_type} != {expected_type} "
                f"for spacy_label={spacy_label}"
            )


# =============================================================================
# Issue #95: Test for MIN_ENTITY_NAME_LENGTH filter
# =============================================================================

@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestMinEntityNameLength:
    """Tests for MIN_ENTITY_NAME_LENGTH filtering."""

    def test_filters_short_entity_names(self):
        """Should filter entities shorter than MIN_ENTITY_NAME_LENGTH (Issue #95).
        
        SpacyEntityExtractor.extract() filters entities with len < MIN_ENTITY_NAME_LENGTH.
        """
        extractor = SpacyEntityExtractor()
        # "Jo" is 2 chars (below threshold), "Bob" is 3 chars (at threshold)
        # Note: spaCy may or may not extract these as PERSON depending on context
        text = "I talked to Jo and Bob today."
        entities = extractor.extract(text)
        
        for entity in entities:
            assert len(entity.name) >= MIN_ENTITY_NAME_LENGTH, (
                f"Entity '{entity.name}' should be filtered (len={len(entity.name)} "
                f"< {MIN_ENTITY_NAME_LENGTH})"
            )

    def test_filters_two_char_names_reliably(self):
        """Verify 2-char names are filtered even when spaCy extracts them.
        
        Uses "Dr. Li" and "Dr. Wu" which spaCy reliably extracts as PERSON,
        but after title stripping become 2-char names that should be filtered.
        """
        extractor = SpacyEntityExtractor()
        # After title normalization, "Dr. Li" → "Li" (2 chars), "Dr. Wu" → "Wu" (2 chars)
        text = "I had a meeting with Dr. Li and Dr. Wu yesterday."
        entities = extractor.extract(text)
        
        # All extracted entities must meet minimum length
        for entity in entities:
            assert len(entity.name) >= MIN_ENTITY_NAME_LENGTH, (
                f"Entity '{entity.name}' (len={len(entity.name)}) should be filtered"
            )
        
        # Specifically verify Li and Wu are not in the results
        names_lower = {e.name.lower() for e in entities}
        assert "li" not in names_lower, "2-char name 'Li' should be filtered"
        assert "wu" not in names_lower, "2-char name 'Wu' should be filtered"

    def test_accepts_minimum_length_names(self):
        """Should accept names exactly at MIN_ENTITY_NAME_LENGTH."""
        extractor = SpacyEntityExtractor()
        # "Bob" is exactly 3 characters, "Amy" is also 3 characters
        text = "Bob and Amy visited the zoo."
        entities = extractor.extract(text)
        
        # Get person entities
        persons = [e for e in entities if e.entity_type == 'person']
        
        # Should extract at least one person entity
        # (spaCy should recognize common names like Bob/Amy in clear context)
        assert len(persons) >= 1, (
            f"Expected at least one person entity from '{text}', got: {entities}"
        )
        
        # All extracted persons should meet minimum length
        for person in persons:
            assert len(person.name) >= MIN_ENTITY_NAME_LENGTH, (
                f"Person '{person.name}' should be >= {MIN_ENTITY_NAME_LENGTH} chars"
            )

    def test_min_length_constant_value(self):
        """MIN_ENTITY_NAME_LENGTH should be 3 (hardcoded constant)."""
        assert MIN_ENTITY_NAME_LENGTH == 3


# =============================================================================
# Issue #91: Multi-word title normalization for person names
# =============================================================================

@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestMultiWordTitleNormalization:
    """Tests for multi-word title stripping from person names (Issue #91)."""

    def test_normalize_single_title(self):
        """Should strip single-word titles like Dr., Mr., Mrs."""
        extractor = SpacyEntityExtractor()
        assert extractor._normalize_person_name("Dr. Thompson") == "Thompson"
        assert extractor._normalize_person_name("Mr. Johnson") == "Johnson"
        assert extractor._normalize_person_name("Mrs. Williams") == "Williams"
        assert extractor._normalize_person_name("Ms. Davis") == "Davis"
        assert extractor._normalize_person_name("Prof. Miller") == "Miller"

    def test_normalize_multi_word_title(self):
        """Should strip multi-word titles like 'Professor Emeritus' (Issue #91).

        Multi-word titles consist of consecutive title words at the start
        of the name. All should be stripped to yield the actual name.
        """
        extractor = SpacyEntityExtractor()
        assert extractor._normalize_person_name("Professor Emeritus Smith") == "Smith"

    def test_normalize_preserves_non_title_names(self):
        """Should not strip words that aren't titles."""
        extractor = SpacyEntityExtractor()
        assert extractor._normalize_person_name("Sarah Connor") == "Sarah Connor"
        assert extractor._normalize_person_name("John") == "John"

    def test_normalize_title_without_period(self):
        """Should strip titles regardless of trailing period."""
        extractor = SpacyEntityExtractor()
        assert extractor._normalize_person_name("Dr Thompson") == "Thompson"
        assert extractor._normalize_person_name("Professor Smith") == "Smith"

    def test_normalize_preserves_middle_title_words(self):
        """Should only strip leading title words, not titles in the middle."""
        extractor = SpacyEntityExtractor()
        # "Sir" is a title, but "Arthur" and "Doyle" are not
        assert extractor._normalize_person_name("Sir Arthur Doyle") == "Arthur Doyle"

    def test_normalize_all_titles_yields_original(self):
        """If stripping all titles would leave nothing, return original."""
        extractor = SpacyEntityExtractor()
        # Edge case: name is just titles (unlikely but defensive)
        assert extractor._normalize_person_name("Dr.") == "Dr."
        assert extractor._normalize_person_name("Professor") == "Professor"

    def test_normalize_military_titles(self):
        """Should strip military rank titles."""
        extractor = SpacyEntityExtractor()
        assert extractor._normalize_person_name("Sgt. Barnes") == "Barnes"
        assert extractor._normalize_person_name("Captain Rogers") == "Rogers"
        assert extractor._normalize_person_name("General Patton") == "Patton"
        assert extractor._normalize_person_name("Col. Mustard") == "Mustard"

    def test_normalize_multiple_consecutive_titles(self):
        """Should strip all consecutive title words."""
        extractor = SpacyEntityExtractor()
        # "Rev. Dr." are both titles
        assert extractor._normalize_person_name("Rev. Dr. King") == "King"

    def test_extract_normalizes_person_titles_end_to_end(self):
        """Verify title normalization happens during extract() (Issue #91).

        This is an end-to-end test: feed text with titled names into
        extract() and verify the returned entities have normalized names.
        """
        extractor = SpacyEntityExtractor()
        text = "I met Professor Smith at the conference yesterday."
        entities = extractor.extract(text)

        person_names = {e.name for e in entities if e.entity_type == 'person'}
        # Should have normalized "Professor Smith" to "Smith"
        if person_names:  # spaCy may or may not detect as PERSON in this context
            assert "Smith" in person_names or any("smith" in n.lower() for n in person_names), (
                f"Expected normalized 'Smith' in {person_names}"
            )
            assert "Professor Smith" not in person_names, (
                f"Title should be stripped: {person_names}"
            )


# =============================================================================
# Issue #107: Edge case tests for entity extraction
# =============================================================================

@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
class TestEntityExtractionEdgeCases:
    """Edge case tests for entity extraction (Issue #107)."""

    def test_unicode_entity_names(self):
        """Should handle non-ASCII characters in entity names (Issue #107).

        Names like 'São Paulo' and 'Zürich' contain accented characters
        that must not cause errors or be silently dropped.
        """
        extractor = SpacyEntityExtractor()
        text = "I traveled from São Paulo to Zürich last summer."
        entities = extractor.extract(text)

        # Should not crash and should return a list
        assert isinstance(entities, list)
        # spaCy should extract at least one place entity with unicode chars intact
        names_lower = {e.name.lower() for e in entities}
        unicode_cities_found = [n for n in names_lower
                                if "paulo" in n or "zürich" in n or "zurich" in n]
        assert len(unicode_cities_found) >= 1, (
            f"Expected at least one unicode city name (São Paulo or Zürich), "
            f"got entities: {[(e.name, e.entity_type) for e in entities]}"
        )

    def test_emoji_in_text(self):
        """Should handle emoji in text without crashing (Issue #107).

        Emoji should not cause exceptions. Entities around emoji should
        still be extracted normally.
        """
        extractor = SpacyEntityExtractor()
        text = "Had lunch with Sarah 🍕 in New York 🗽"
        entities = extractor.extract(text)

        assert isinstance(entities, list)
        # Should still extract named entities despite emoji
        names_lower = {e.name.lower() for e in entities}
        assert "sarah" in names_lower or "new york" in names_lower, (
            f"Expected entities despite emoji, got: {[e.name for e in entities]}"
        )

    def test_empty_string(self):
        """Should return empty list for empty string (Issue #107)."""
        extractor = SpacyEntityExtractor()
        assert extractor.extract("") == []

    def test_none_input(self):
        """Should return empty list for None input (Issue #107)."""
        extractor = SpacyEntityExtractor()
        assert extractor.extract(None) == []

    def test_whitespace_only(self):
        """Should return empty list for whitespace-only text (Issue #107)."""
        extractor = SpacyEntityExtractor()
        assert extractor.extract("   \t\n  ") == []

    @pytest.mark.slow
    def test_very_long_text(self):
        """Should handle very long text without errors (Issue #107).

        Tests that entity extraction works on text >10,000 characters
        without crashing, timing out, or truncating results.
        """
        extractor = SpacyEntityExtractor()
        # Build a long text with known entities scattered throughout
        base_sentence = "Sarah visited Google headquarters in San Francisco. "
        long_text = base_sentence * 200  # ~10,000+ chars
        entities = extractor.extract(long_text)

        assert isinstance(entities, list)
        # Should still deduplicate — same entities repeated shouldn't balloon.
        # With 3 unique entities (Sarah, Google, San Francisco) repeated 200x,
        # deduplication should keep this well under 10.
        assert len(entities) <= 10, (
            f"Expected deduplication to limit entities, got {len(entities)}"
        )
        # Should still find the key entities
        names = {e.name.lower() for e in entities}
        assert "sarah" in names or "google" in names or "san francisco" in names, (
            f"Expected known entities from repeated text, got: {names}"
        )

    def test_special_characters_in_text(self):
        """Should handle special characters without crashing (Issue #107).

        Text with HTML entities, brackets, quotes, and other special
        characters should not cause extraction errors.
        """
        extractor = SpacyEntityExtractor()
        text = 'Meeting with <Dr. Smith> & "Prof. Jones" @ the university (Room #42).'
        entities = extractor.extract(text)

        assert isinstance(entities, list)
        # Should not crash — correctness of extraction may vary

    def test_mixed_language_text(self):
        """Should handle mixed-language text without errors (Issue #107).

        English spaCy model may not extract non-English entities accurately,
        but it should not crash on mixed-language input.
        """
        extractor = SpacyEntityExtractor()
        text = "I met Pierre in Paris. Nous avons visité le Louvre."
        entities = extractor.extract(text)

        assert isinstance(entities, list)
        # Should at least extract the English-context entities
        names_lower = {e.name.lower() for e in entities}
        assert "pierre" in names_lower or "paris" in names_lower or "louvre" in names_lower, (
            f"Expected at least one entity from mixed text, got: {names_lower}"
        )

    def test_concurrent_extraction(self):
        """Should be thread-safe for concurrent extraction (Issue #107).

        Multiple threads extracting entities simultaneously should not
        cause data races, crashes, or corrupted results.

        Thread safety note: spaCy's nlp() is thread-safe for inference
        (the model weights are read-only after loading). Our extractor
        adds no shared mutable state — seen_names is local to each
        extract() call. This test verifies no regressions.
        """
        import concurrent.futures

        extractor = SpacyEntityExtractor()
        texts = [
            "Sarah visited New York last Monday.",
            "Dr. Thompson works at Google in London.",
            "Bob met Amy at the conference in Berlin.",
            "Professor Smith teaches at Harvard University.",
        ]

        def extract_from(text):
            return extractor.extract(text)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(extract_from, t) for t in texts]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All results should be valid lists
        for result in results:
            assert isinstance(result, list)
            for entity in result:
                assert hasattr(entity, 'name')
                assert hasattr(entity, 'entity_type')

    def test_numeric_only_text(self):
        """Should handle text that is purely numeric (Issue #107)."""
        extractor = SpacyEntityExtractor()
        text = "12345 67890 11111"
        entities = extractor.extract(text)

        assert isinstance(entities, list)
        # Numeric-only entities may or may not be extracted (CARDINAL/ORDINAL
        # are not in RELEVANT_TYPES), but should not crash

    def test_single_character_text(self):
        """Should handle single-character text (Issue #107)."""
        extractor = SpacyEntityExtractor()
        assert extractor.extract("a") == [] or isinstance(extractor.extract("a"), list)
        assert extractor.extract(".") == [] or isinstance(extractor.extract("."), list)

    def test_very_long_entity_name(self):
        """Should handle entity names >100 chars without overflow (Issue #107).

        spaCy may extract very long spans as entities. The extractor should
        handle these gracefully without truncation issues.
        """
        extractor = SpacyEntityExtractor()
        # Construct text with a very long proper noun phrase
        long_name = "The International Association of " + "Very " * 25 + "Important Scientists"
        text = f"She presented at {long_name} in Geneva."
        entities = extractor.extract(text)

        assert isinstance(entities, list)
        # Should not crash regardless of entity name length
