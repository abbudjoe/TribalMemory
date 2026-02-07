# Entity Extraction v2 — Spec

## Problem Statement

Current entity extraction produces garbage:
- "Glen Canyon Dam" → person
- "Razer Kraken X" → person  
- "Tips --[serves]--> Your" relationships

**Root causes:**
1. No entity quality filtering (length, confidence, stopwords)
2. Regex relationship patterns match random text
3. spaCy NER without validation produces noisy output
4. No co-reference resolution ("he" → actual person)

## Goals

1. **Precision over recall** — Better to miss entities than extract garbage
2. **Meaningful relationships** — Only extract real semantic connections
3. **Multi-session linking** — Enable graph to connect conversations about same entities
4. **Minimal latency impact** — Keep query time under 500ms

## Phase 1: Entity Quality Filtering (Quick Fix)

### Changes to `SpacyEntityExtractor.extract()`

```python
# New constants
MIN_ENTITY_LENGTH = 3          # Skip "I", "We", "Dr"
MAX_ENTITY_LENGTH = 50         # Skip long garbage strings
ENTITY_STOPWORDS = {
    'i', 'we', 'you', 'he', 'she', 'they', 'it', 'this', 'that',
    'today', 'tomorrow', 'yesterday', 'now', 'then', 'here', 'there',
    'good', 'great', 'nice', 'bad', 'new', 'old', 'first', 'last',
}

# New validation in extract()
def _is_valid_entity(self, text: str, label: str) -> bool:
    """Filter out garbage entities."""
    # Length bounds
    if len(text) < MIN_ENTITY_LENGTH or len(text) > MAX_ENTITY_LENGTH:
        return False
    
    # Stopwords
    if text.lower() in ENTITY_STOPWORDS:
        return False
    
    # Must contain at least one letter
    if not any(c.isalpha() for c in text):
        return False
    
    # PERSON: must look like a name (capitalized, no special chars)
    if label == 'PERSON':
        if not text[0].isupper():
            return False
        if any(c in text for c in '*[](){}'):
            return False
        # Skip product-looking names
        if any(word in text.lower() for word in ['pro', 'plus', 'max', 'ultra', 'edition']):
            return False
    
    # ORG: skip obvious non-orgs
    if label == 'ORG':
        if len(text.split()) == 1 and text.isupper() and len(text) <= 4:
            pass  # Acronyms OK (FBI, NASA)
        elif text.lower() in {'the', 'a', 'an'}:
            return False
    
    return True
```

### Changes to `EntityExtractor` (regex)

```python
# Disable garbage relationship patterns
RELATIONSHIP_PATTERNS = [
    # Keep only high-precision patterns
    (re.compile(r'(\b[a-z]+-service)\s+uses\s+(\b[a-z]+-\w+)', re.I), 'uses'),
    (re.compile(r'(\b[a-z]+-service)\s+connects?\s+to\s+(\b[a-z]+-\w+)', re.I), 'connects_to'),
    # REMOVE: the "for" pattern that produces garbage
    # (re.compile(r'(\S+)\s+for\s+(?:the\s+)?(\S+)', re.IGNORECASE), 'serves'),
]
```

### Success Metrics (Phase 1)

| Metric | Before | Target |
|--------|--------|--------|
| Person entities that are actual people | ~50% | >90% |
| Relationships that make sense | ~5% | >80% |
| Entities per 1000 chars | ~15 | ~5 |

## Phase 2: Relationship Extraction Rewrite

### Approach: Dependency Parsing

Use spaCy's dependency parser to find real subject-verb-object relationships:

```python
def extract_relationships(self, doc) -> list[Relationship]:
    """Extract relationships using dependency parsing."""
    relationships = []
    
    for token in doc:
        # Find verbs with subject and object
        if token.pos_ == 'VERB':
            subject = None
            obj = None
            
            for child in token.children:
                if child.dep_ in ('nsubj', 'nsubjpass'):
                    # Find the entity this subject belongs to
                    subject = self._find_entity_for_token(child, doc)
                elif child.dep_ in ('dobj', 'pobj', 'attr'):
                    obj = self._find_entity_for_token(child, doc)
            
            if subject and obj:
                relationships.append(Relationship(
                    source=subject,
                    target=obj,
                    relation_type=token.lemma_,  # e.g., "use", "connect", "store"
                ))
    
    return relationships

def _find_entity_for_token(self, token, doc) -> Optional[Entity]:
    """Find the named entity that contains this token."""
    for ent in doc.ents:
        if ent.start <= token.i < ent.end:
            return ent
    return None
```

### Co-reference Resolution (Optional)

```python
# Using spacy-experimental or neuralcoref
def resolve_coreferences(self, doc) -> dict[str, str]:
    """Map pronouns to their referents."""
    # "Sarah went to the store. She bought milk."
    # Returns: {"She": "Sarah"}
    pass
```

### Success Metrics (Phase 2)

| Metric | Target |
|--------|--------|
| multi-session accuracy | >50% (currently 20%) |
| Relationships with valid entity pairs | >95% |
| Query latency (p95) | <1s |

## Implementation Plan

### Phase 1 (PR #117) — 2-3 hours
1. Add `_is_valid_entity()` to SpacyEntityExtractor
2. Remove garbage regex patterns from EntityExtractor
3. Add unit tests for entity filtering
4. Run benchmark with 20-sample

### Phase 2 (PR #118) — 4-6 hours
1. Add `extract_relationships()` with dep parsing
2. Update HybridEntityExtractor to use new method
3. Add integration tests
4. Full benchmark run

## Testing Strategy

```python
def test_entity_filtering():
    extractor = SpacyEntityExtractor()
    
    # Should extract
    assert extract_names("Sarah went to the store") == ["Sarah"]
    assert extract_names("Dr. Thompson called") == ["Thompson"]
    
    # Should NOT extract
    assert extract_names("Tips for Your Garden") == []  # No garbage
    assert extract_names("The Razer Kraken X Pro") == []  # Product, not person
    assert extract_names("I went to the store") == []  # Pronoun

def test_relationship_quality():
    # Should extract
    text = "auth-service uses PostgreSQL for the user database"
    rels = extract_relationships(text)
    assert ("auth-service", "uses", "PostgreSQL") in rels
    
    # Should NOT extract
    text = "Tips for Minimizing Cancer Risk"
    rels = extract_relationships(text)
    assert len(rels) == 0  # No garbage "serves" relationship
```

## Rollback Plan

If accuracy drops:
1. Feature flag: `graph_expansion: false` in config
2. Falls back to pure vector + BM25 search
3. No entity extraction overhead

---

*Author: Clawdio*
*Date: 2026-02-07*
