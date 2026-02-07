# Entity Extraction v2 — Spec & Implementation Plan

## Problem Statement

Current entity extraction produces garbage entities and relationships that poison the graph store, hurting multi-session recall accuracy (20% on LongMemEval).

### Root Causes (diagnosed 2026-02-06)

1. **Regex relationship patterns match noise** — `(\S+)\s+for\s+(?:the\s+)?(\S+)` produces garbage like "Useful --[serves]--> longer-range", "Check --[serves]--> Energy"
2. **spaCy NER misclassifies entities** — Products/food classified as PERSON (e.g., "Razer Kraken X", "Sarson Ka Saag" → person)
3. **No entity quality filtering** — Short, numeric, and garbage entities pass through
4. **Regex extractor designed for software architecture** — SERVICE_PATTERN, TECHNOLOGIES set, RELATIONSHIP_PATTERNS all assume code/infra context, fail on personal conversations
5. **17,488 entities but only 238 relationships** — And all relationships are garbage

### Current Architecture

```
HybridEntityExtractor
├── EntityExtractor (regex)     — services, technologies, relationships
└── SpacyEntityExtractor        — NER for people, places, dates, orgs
```

## Solution: 4-Phase Approach

### Phase 1: Entity Quality Filtering (Issue #128)
**Goal:** Stop garbage from entering the graph. Quick win.
**Scope:** Add validation layer, no extraction logic changes.

Changes:
- Add `EntityValidator` class with configurable rules:
  - Min/max name length (already have MIN=3, add MAX=100)
  - Reject names that are all-caps stopwords (THE, AND, FOR, etc.)
  - Reject names that are common English words below a frequency threshold
  - Reject numeric-only entities
  - Reject entities with no alphabetic characters
- Add `RelationshipValidator`:
  - Both source and target must pass entity validation
  - Reject self-relationships (source == target)
  - Reject relationships where source or target is a stopword
- Wire validators into `HybridEntityExtractor.extract()` and `extract_with_relationships()`
- Tests: 15+ test cases for each validator

**Estimated time:** 2-3 hours

### Phase 2: Remove Garbage Regex Patterns (Issue #129)
**Goal:** Stop the regex extractor from generating bad relationships.
**Scope:** Prune/rewrite RELATIONSHIP_PATTERNS.

Changes:
- Remove the `serves` pattern entirely (`X for Y` is too broad)
- Tighten `uses`, `connects_to`, `stores_in` to require both sides to be known entities (from TECHNOLOGIES or SERVICE_PATTERN match)
- Add `_is_known_entity()` check before relationship creation
- Consider making regex extractor opt-in (disabled by default for personal conversations)
- Add `extraction_context` parameter: `"software"` vs `"personal"` to control which patterns activate
- Tests: Verify removed patterns no longer produce false positives

**Estimated time:** 2 hours

### Phase 3: spaCy Entity Post-Processing (Issue #130)
**Goal:** Fix misclassified entities from spaCy NER.
**Scope:** Add post-processing pipeline after spaCy extraction.

Changes:
- Add `SpacyPostProcessor` class:
  - **Person name validation:** Reject PERSON entities that contain product keywords (brand names, model numbers)
  - **Organization validation:** Reject ORG entities that are clearly food/product names
  - **Confidence filtering:** Use spaCy's entity scores where available
  - **Context-aware filtering:** If surrounding text contains "bought", "ordered", "ate" → entity is likely product/food, not person
- Add commonly misclassified entity blocklist (expandable)
- Tests: Feed known misclassified examples, verify correction

**Estimated time:** 3-4 hours

### Phase 4: Dependency-Parsed Relationships (Issue #131)
**Goal:** Replace regex relationship extraction with spaCy dependency parsing.
**Scope:** New relationship extractor using syntactic structure.

Changes:
- Add `DependencyRelationshipExtractor` class:
  - Parse sentence with spaCy's dependency parser
  - Find subject-verb-object triples
  - Map verbs to relationship types ("uses" → uses, "lives in" → located_in, "works at" → works_at)
  - Only create relationships between entities that were already extracted by NER
- Verb-to-relationship mapping (configurable):
  ```python
  VERB_RELATIONS = {
      'use': 'uses', 'uses': 'uses',
      'live': 'located_in', 'lives': 'located_in',
      'work': 'works_at', 'works': 'works_at',
      'visit': 'visited', 'visited': 'visited',
      'meet': 'met', 'met': 'met',
      'like': 'prefers', 'likes': 'prefers', 'love': 'prefers',
      'prefer': 'prefers', 'prefers': 'prefers',
      'buy': 'purchased', 'bought': 'purchased',
  }
  ```
- Replace regex RELATIONSHIP_PATTERNS with dependency-parsed extraction when spaCy is available
- Fall back to (cleaned-up) regex patterns when spaCy is not available
- Tests: 20+ tests with real conversation snippets

**Estimated time:** 4-6 hours

## Success Criteria

- LongMemEval multi-session accuracy: >40% (currently 20%)
- LongMemEval overall accuracy: >50% (currently 40%)
- Entity count reduction: >50% fewer garbage entities
- Relationship quality: >80% of extracted relationships are semantically valid
- Zero regression on single-session accuracy

## Dependencies

- PR #124 (entity extraction improvements) should merge first — it adds the foundation (RELEVANT_TYPES, title normalization, edge case tests)
- spaCy `en_core_web_sm` model (already installed)

## Order of Operations

Phase 1 → Phase 2 → benchmark → Phase 3 → Phase 4 → benchmark → release
