"""Graph-enriched memory storage for entity and relationship tracking.

Provides a lightweight graph layer alongside vector search to enable:
- Entity-centric queries ("tell me everything about auth-service")
- Relationship traversal ("what does auth-service connect to?")
- Multi-hop reasoning ("what framework does the service that handles auth use?")

Uses SQLite for local-first, zero-cloud constraint.
"""

import re
import sqlite3
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    import spacy.tokens

# Constants
MIN_ENTITY_NAME_LENGTH = 3
MAX_HOP_ITERATIONS = 100  # Safety limit for graph traversal

# Temporal relationship types
TEMPORAL_OCCURRED_ON = "occurred_on"
TEMPORAL_MENTIONED_DATE = "mentioned_date"


@dataclass
class TemporalFact:
    """A temporal fact linking an entity/event to a resolved date.

    Attributes:
        subject: What happened (event or entity name).
        relation: occurred_on or mentioned_date.
        resolved_date: ISO date string (YYYY-MM-DD, YYYY-MM, YYYY).
        original_expression: Raw text ("yesterday").
        precision: day, week, month, or year.
        confidence: Score in [0.0, 1.0].
    """

    subject: str
    relation: str
    resolved_date: str
    original_expression: str
    precision: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate confidence is in [0.0, 1.0]."""
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class Entity:
    """An extracted entity from memory text."""
    
    name: str
    entity_type: str  # service, technology, data, concept, person, etc.
    metadata: dict = field(default_factory=dict)


@dataclass
class Relationship:
    """A relationship between two entities."""
    
    source: str  # Entity name
    target: str  # Entity name
    relation_type: str  # uses, stores, connects_to, depends_on, etc.
    metadata: dict = field(default_factory=dict)


class EntityExtractor:
    """Extract entities and relationships from text.
    
    Uses pattern-based extraction for common software architecture terms.
    Applies EntityValidator and RelationshipValidator to filter garbage.
    Can be upgraded to spaCy NER or LLM extraction later.
    
    Applies EntityValidator and RelationshipValidator to filter garbage entities
    and relationships before returning results.
    
    Attributes:
        SERVICE_PATTERN: Regex for service-like names (kebab-case with suffix or 8+ chars).
        TECHNOLOGIES: Set of known technology names for exact matching.
        RELATIONSHIP_PATTERNS: List of (pattern, relation_type) tuples for extraction.
    """
    
    # Patterns for common entity types
    # Matches: kebab-case identifiers that look like service/component names
    # - Must have at least one hyphen (kebab-case)
    # - Either ends with known suffix OR has 3+ segments OR is 8+ chars
    # - Excludes common false positives via MIN_ENTITY_NAME_LENGTH
    SERVICE_PATTERN = re.compile(
        r'\b('
        r'[a-z][a-z0-9]*-(?:[a-z0-9]+-)*(?:service|api|worker|db|cache|server|client|gateway|proxy|database)'  # Known suffix
        r'|'
        r'[a-z][a-z0-9]*(?:-[a-z0-9]+){2,}'  # 3+ segments
        r'|'
        r'[a-z][a-z0-9]*-[a-z0-9]{4,}'  # 2 segments, second is 4+ chars
        r')\b',
        re.IGNORECASE
    )
    
    # Known technology names (case-insensitive matching)
    TECHNOLOGIES = {
        'postgresql', 'postgres', 'mysql', 'mongodb', 'redis', 'memcached',
        'elasticsearch', 'kafka', 'rabbitmq', 'nginx', 'docker', 'kubernetes',
        'aws', 'gcp', 'azure', 'terraform', 'ansible', 'jenkins', 'github',
        'python', 'javascript', 'typescript', 'rust', 'go', 'java', 'node',
        'react', 'vue', 'angular', 'django', 'flask', 'fastapi', 'express',
        'graphql', 'rest', 'grpc', 'websocket', 'http', 'https',
        'sqlite', 'lancedb', 'chromadb', 'pinecone', 'weaviate',
        'openai', 'anthropic', 'ollama', 'huggingface',
        'pgbouncer', 'haproxy', 'traefik', 'envoy',
    }
    
    # Relationship patterns: (pattern, relation_type)
    # NOTE: Patterns are tightened to require both sides to be known entities
    # (in TECHNOLOGIES or matching SERVICE_PATTERN). This prevents garbage
    # relationships from personal conversation text.
    RELATIONSHIP_PATTERNS = [
        (re.compile(r'(\S+)\s+uses\s+(\S+)', re.IGNORECASE), 'uses'),
        (re.compile(r'(\S+)\s+connects?\s+to\s+(\S+)', re.IGNORECASE), 'connects_to'),
        (re.compile(r'(\S+)\s+stores?\s+(?:data\s+)?in\s+(\S+)', re.IGNORECASE), 'stores_in'),
        (re.compile(r'(\S+)\s+depends?\s+on\s+(\S+)', re.IGNORECASE), 'depends_on'),
        (re.compile(r'(\S+)\s+talks?\s+to\s+(\S+)', re.IGNORECASE), 'connects_to'),
        (re.compile(r'(\S+)\s+calls?\s+(\S+)', re.IGNORECASE), 'calls'),
        (re.compile(r'(\S+)\s+handles?\s+(\S+)', re.IGNORECASE), 'handles'),
        # NOTE: 'serves' pattern removed — too broad, produced garbage relationships
        # Old pattern: (re.compile(r'(\S+)\s+for\s+(?:the\s+)?(\S+)', re.IGNORECASE), 'serves')
    ]
    
    def extract(self, text: str) -> list[Entity]:
        """Extract entities from text.
        
        Args:
            text: Input text to extract entities from.
            
        Returns:
            List of extracted Entity objects.
        """
        if not text or not text.strip():
            return []
        
        entities = []
        seen_names: set[str] = set()
        
        # Extract service-like names (kebab-case identifiers)
        for match in self.SERVICE_PATTERN.finditer(text):
            name = match.group(1)
            if name and name.lower() not in seen_names and len(name) >= MIN_ENTITY_NAME_LENGTH:
                seen_names.add(name.lower())
                entities.append(Entity(
                    name=name,
                    entity_type=self._infer_service_type(name)
                ))
        
        # Extract known technology names
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            word_lower = word.lower()
            if word_lower in self.TECHNOLOGIES and word_lower not in seen_names:
                seen_names.add(word_lower)
                entities.append(Entity(
                    name=word,  # Preserve original case
                    entity_type='technology'
                ))
        
        return entities
    
    def extract_with_relationships(
        self, text: str
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract both entities and relationships from text.
        
        Relationships are only extracted when BOTH sides are known entities
        (in TECHNOLOGIES set or matching SERVICE_PATTERN). This prevents
        garbage relationships from personal conversation text.
        
        Args:
            text: Input text to extract from.
            
        Returns:
            Tuple of (entities, relationships).
        """
        entities = self.extract(text)
        entity_names = {e.name.lower() for e in entities}
        relationships = []
        
        for pattern, rel_type in self.RELATIONSHIP_PATTERNS:
            for match in pattern.finditer(text):
                source = match.group(1).strip('.,;:')
                target = match.group(2).strip('.,;:')
                
                # NOTE: Only create relationship if BOTH sides are known entities
                # This prevents garbage like "waiting for the bus" → (waiting, bus, serves)
                if not self._is_known_entity(source):
                    continue
                if not self._is_known_entity(target):
                    continue
                
                relationships.append(Relationship(
                    source=source,
                    target=target,
                    relation_type=rel_type
                ))
                
                # Add entities if not already present
                if source.lower() not in entity_names:
                    entity_names.add(source.lower())
                    entities.append(Entity(
                        name=source,
                        entity_type=self._infer_type(source)
                    ))
                if target.lower() not in entity_names:
                    entity_names.add(target.lower())
                    entities.append(Entity(
                        name=target,
                        entity_type=self._infer_type(target)
                    ))
        
        # Filter through validators to remove garbage
        return entities, relationships
    
    def _infer_service_type(self, name: str) -> str:
        """Infer entity type from service-like name.
        
        Args:
            name: Service name to analyze.
            
        Returns:
            Entity type string (e.g., 'service', 'database', 'worker').
        """
        name_lower = name.lower()
        if '-db' in name_lower or '-database' in name_lower:
            return 'database'
        if '-api' in name_lower or '-service' in name_lower:
            return 'service'
        if '-worker' in name_lower or '-job' in name_lower:
            return 'worker'
        if '-cache' in name_lower:
            return 'cache'
        if '-gateway' in name_lower or '-proxy' in name_lower:
            return 'gateway'
        if '-server' in name_lower:
            return 'server'
        if '-client' in name_lower:
            return 'client'
        return 'service'
    
    def _infer_type(self, name: str) -> str:
        """Infer entity type from name.
        
        Args:
            name: Entity name to analyze.
            
        Returns:
            Entity type string.
        """
        if name.lower() in self.TECHNOLOGIES:
            return 'technology'
        if self.SERVICE_PATTERN.match(name):
            return self._infer_service_type(name)
        return 'concept'
    
    def _looks_like_entity(self, name: str) -> bool:
        """Check if a string looks like a valid entity name.
        
        Args:
            name: String to check.
            
        Returns:
            True if the string looks like an entity name.
        """
        if not name or len(name) < MIN_ENTITY_NAME_LENGTH:
            return False
        if name.lower() in self.TECHNOLOGIES:
            return True
        if self.SERVICE_PATTERN.match(name):
            return True
        # Capitalized words (proper nouns)
        if name[0].isupper() and name.isalnum():
            return True
        return False
    
    def _is_known_entity(self, name: str) -> bool:
        """Check if a name is a known entity (technology or service).
        
        Used to validate both sides of relationships before extraction.
        This prevents garbage relationships from personal conversation text
        like "waiting for the bus" where neither side is a real entity.
        
        Args:
            name: Entity name to check.
            
        Returns:
            True if the name is a known technology or matches SERVICE_PATTERN.
        """
        if not name or len(name) < MIN_ENTITY_NAME_LENGTH:
            return False
        
        # Check if it's a known technology (case-insensitive)
        if name.lower() in self.TECHNOLOGIES:
            return True
        
        # Check if it matches SERVICE_PATTERN (kebab-case service names)
        if self.SERVICE_PATTERN.match(name):
            return True
        
        return False


# Check if spaCy is available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Common person titles to strip for better entity matching.
# Includes both abbreviated and full forms. The normalization logic
# strips ALL consecutive leading title words, so multi-word titles like
# "Professor Emeritus" are handled by matching each word individually.
PERSON_TITLES = {
    'dr', 'dr.', 'mr', 'mr.', 'mrs', 'mrs.', 'ms', 'ms.', 'miss',
    'prof', 'prof.', 'professor', 'sir', 'madam', 'rev', 'rev.',
    'reverend', 'hon', 'hon.', 'honorable',
    'emeritus', 'emerita',
    'sgt', 'sgt.', 'sergeant', 'cpl', 'cpl.', 'corporal',
    'capt', 'capt.', 'captain', 'lt', 'lt.', 'lieutenant',
    'cmdr', 'cmdr.', 'commander', 'col', 'col.', 'colonel',
    'gen', 'gen.', 'general', 'maj', 'maj.', 'major',
    'pvt', 'pvt.', 'private', 'admiral',
}


class SpacyPostProcessor:
    """Post-processes spaCy entities to fix common misclassifications.
    
    Addresses Issue #131: spaCy often misclassifies products and food items
    as PERSON entities. This processor applies heuristics to:
    - Reject product names (brands, model numbers) misclassified as people
    - Reject food names commonly misclassified as people
    - Reclassify certain ambiguous product-like entities from person to product
    - Preserve real person names and non-person entities
    
    Processing rules (applied only to PERSON entities):
        1. Reclassify if EXACT match of reclassify keywords (Galaxy, Kraken)
        2. Reject if in food blocklist
        3. Reject if contains product brand keywords
        4. Reject if matches model number pattern
        5. Pass through all other entities unchanged
    
    Examples:
        processor = SpacyPostProcessor()
        
        # Rejects misclassified products
        entity = Entity(name="iPhone 15", entity_type="person")
        assert processor.process(entity) is None
        
        # Reclassifies ambiguous product terms
        entity = Entity(name="Galaxy", entity_type="person")
        result = processor.process(entity)
        assert result.entity_type == "product"
        
        # Preserves real person names
        entity = Entity(name="Sarah Thompson", entity_type="person")
        result = processor.process(entity)
        assert result.entity_type == "person"
    """
    
    # Well-known product brand keywords (case-insensitive matching)
    # If a PERSON entity contains any of these, reject it entirely
    PRODUCT_BRANDS = {
        'razer', 'kindle', 'iphone', 'ipad', 'macbook',
        'samsung', 'pixel', 'oneplus', 'xiaomi',
        'playstation', 'xbox', 'nintendo',
        'gopro', 'canon', 'nikon', 'sony',
    }
    
    # Keywords that trigger reclassification from person → product
    # These are ambiguous terms that could be products OR other things
    # ONLY reclassify if it's an EXACT match (standalone word)
    # Example: "Galaxy" alone → reclassify, "Galaxy S23" → reject via brands
    RECLASSIFY_TO_PRODUCT = {
        'galaxy', 'kraken', 'airpods',
    }
    
    # Model number pattern: letter + digits (e.g., X100, S23, Pro Max)
    # Intentionally broad — false positives (e.g., "X23" in military
    # context) are rare and acceptable. Products misclassified as
    # people are far more common in personal conversations.
    MODEL_NUMBER_PATTERN = re.compile(
        r'\b[A-Z]\d{2,}\b|'  # Letter followed by 2+ digits: X100, S23
        r'\bPro\s+Max\b',    # Common product suffix: Pro Max
        re.IGNORECASE
    )
    
    # Food names commonly misclassified as PERSON by spaCy.
    # Curated from LongMemEval benchmark failures where spaCy's
    # en_core_web_sm model classified dish names as PERSON entities.
    # To extend: add lowercase dish names that spaCy misclassifies.
    FOOD_BLOCKLIST = {
        'sarson ka saag', 'biryani', 'pad thai', 'tikka masala',
        'butter chicken', 'palak paneer', 'kung pao', 'tom yum',
        'pad see ew', 'chow mein', 'lo mein', 'pho',
    }
    
    def process(self, entity: Entity) -> Optional[Entity]:
        """Process an entity through post-processing rules.
        
        Args:
            entity: Entity to process.
            
        Returns:
            - None if entity should be rejected
            - Modified Entity if reclassification occurred
            - Original Entity if no changes needed
        """
        # Only apply rules to PERSON entities
        if entity.entity_type != 'person':
            return entity
        
        # Empty or whitespace-only names should be rejected
        if not entity.name or not entity.name.strip():
            return None
        
        name_lower = entity.name.lower().strip()
        
        # Rule 1: Reclassify to product if EXACT match of reclassify keywords
        # (Only for standalone ambiguous words like "Galaxy", "Kraken")
        if name_lower in self.RECLASSIFY_TO_PRODUCT:
            # Create new entity with type changed to 'product'
            return Entity(
                name=entity.name,
                entity_type='product',
                metadata=entity.metadata
            )
        
        # Rule 2: Reject if in food blocklist (exact match, case-insensitive)
        if name_lower in self.FOOD_BLOCKLIST:
            return None
        
        # Rule 3: Reject if contains product brand keywords (substring match)
        for brand in self.PRODUCT_BRANDS:
            if brand in name_lower:
                return None
        
        # Rule 4: Reject if matches model number pattern
        if self.MODEL_NUMBER_PATTERN.search(entity.name):
            return None
        
        # Pass through unchanged (valid person name)
        return entity


class SpacyEntityExtractor:
    """Entity extractor using spaCy NER for personal conversations.
    
    Extracts named entities like people, places, organizations, and dates
    from natural language text. Designed for personal assistant use cases
    where users discuss daily activities, appointments, purchases, etc.
    
    Requires: pip install tribalmemory[spacy] && python -m spacy download en_core_web_sm
    
    Entity types extracted:
        - PERSON: People's names (Dr. Smith, Sarah)
        - GPE: Geopolitical entities / places (Brookside, New York)
        - ORG: Organizations (Google, City Hospital)
        - DATE: Dates and times (March, last Tuesday, 2024)
        - EVENT: Events (the meeting, my appointment)
        - FAC: Facilities (the townhouse, Oak Street)
        - PRODUCT: Products (iPhone, Toyota Camry)
        - MONEY: Monetary values ($500,000)
    """
    
    # Map spaCy entity types to our internal types
    SPACY_TYPE_MAP = {
        'PERSON': 'person',
        'GPE': 'place',
        'LOC': 'place',
        'FAC': 'place',
        'ORG': 'organization',
        'DATE': 'date',
        'TIME': 'date',
        'EVENT': 'event',
        'PRODUCT': 'product',
        'MONEY': 'money',
        'CARDINAL': 'number',
        'ORDINAL': 'number',
    }
    
    # Entity types we care about for personal conversations.
    #
    # NOTE: RELEVANT_TYPES is intentionally a SUBSET of SPACY_TYPE_MAP.
    # SPACY_TYPE_MAP defines how to translate *all* spaCy labels we might
    # encounter into our internal type system (including MONEY, CARDINAL,
    # ORDINAL, TIME). RELEVANT_TYPES controls which of those labels we
    # actually *extract* during personal conversation processing.
    #
    # Types like MONEY, CARDINAL, and ORDINAL are mapped but excluded from
    # extraction because they generate too much noise in personal conversation
    # contexts (e.g., every number or dollar amount becoming an entity).
    # TIME is mapped to 'date' but excluded because standalone time
    # expressions ("3pm") are rarely useful as entities without a date.
    #
    # To add a new entity type:
    #   1. Add the spaCy label -> internal type mapping to SPACY_TYPE_MAP
    #   2. Add the spaCy label to RELEVANT_TYPES if it should be extracted
    RELEVANT_TYPES = {'PERSON', 'GPE', 'LOC', 'FAC', 'ORG', 'DATE', 'EVENT', 'PRODUCT'}
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize spaCy entity extractor.
        
        Args:
            model_name: Name of spaCy model to load. Default is the small
                English model which balances speed and accuracy.
        """
        if not SPACY_AVAILABLE:
            raise ImportError(
                "spaCy is not installed. Install with: pip install tribalmemory[spacy]"
            )
        
        try:
            self._nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"spaCy model '{model_name}' not found. "
                f"Download with: python -m spacy download {model_name}"
            )
        
        # Initialize post-processor for entity cleaning (Issue #131)
        self._post_processor = SpacyPostProcessor()
    
    def _normalize_person_name(self, name: str) -> str:
        """Strip consecutive leading title words from person names.

        Handles both single-word titles ("Dr. Smith" -> "Smith") and
        multi-word titles ("Professor Emeritus Smith" -> "Smith").
        Each leading word is checked against PERSON_TITLES (with and
        without trailing period). Stripping stops at the first non-title
        word. If all words are titles, the original name is returned
        unchanged to avoid producing an empty string.

        Args:
            name: Raw person name (e.g., "Professor Emeritus Smith").

        Returns:
            Normalized name with leading titles removed (e.g., "Smith").
        """
        parts = name.split()
        if len(parts) <= 1:
            return name

        # Strip all consecutive leading title words (not just the first)
        # This handles multi-word titles like "Professor Emeritus"
        first_non_title = 0
        for i, part in enumerate(parts):
            if part.lower().rstrip('.') in PERSON_TITLES:
                first_non_title = i + 1
            else:
                break

        # Don't strip if it would remove all words
        if first_non_title >= len(parts):
            return name

        if first_non_title > 0:
            return ' '.join(parts[first_non_title:])
        return name

    def extract(self, text: Optional[str]) -> list[Entity]:
        """Extract named entities from text using spaCy NER.
        
        Applies post-processing to fix common misclassifications (Issue #131).
        
        Args:
            text: Input text to extract entities from (can be None).
            
        Returns:
            List of extracted and post-processed Entity objects.
        """
        if not text or not text.strip():
            return []
        
        doc = self._nlp(text)
        entities = []
        seen_names: set[str] = set()
        
        for ent in doc.ents:
            if ent.label_ not in self.RELEVANT_TYPES:
                continue
            
            # Normalize entity text
            name = ent.text.strip()
            
            # Strip titles from person names for better entity matching
            if ent.label_ == 'PERSON':
                name = self._normalize_person_name(name)
            
            if len(name) < MIN_ENTITY_NAME_LENGTH:
                continue
            
            # Deduplicate by lowercase name
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)
            
            entity_type = self.SPACY_TYPE_MAP.get(ent.label_, 'concept')
            entity = Entity(
                name=name,
                entity_type=entity_type,
                metadata={'spacy_label': ent.label_}
            )
            
            # Apply post-processing to fix misclassifications (Issue #131)
            processed_entity = self._post_processor.process(entity)
            if processed_entity is not None:
                entities.append(processed_entity)
        
        return entities
    
    def extract_with_relationships(
        self, text: str
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities from text. Relationships not implemented for spaCy.
        
        spaCy doesn't extract relationships directly. For relationship extraction,
        use the regex-based EntityExtractor patterns or combine both extractors.
        
        Args:
            text: Input text to process.
            
        Returns:
            Tuple of (entities, empty relationships list).
        """
        return self.extract(text), []


class EntityValidator:
    """Validates entities before they enter the graph store (Issue #129, #135).
    
    Prevents garbage entities from polluting the knowledge graph by enforcing:
    - Max name length (100 chars)
    - Reject all-caps stopwords (THE, AND, FOR, etc.)
    - Reject numeric-only entities ("12345")
    - Reject entities with no alphabetic characters ("---", "...")
    - Reject single-word common English words when entity_type is 'concept'
    
    Performance optimizations (Issue #135):
    - Batch validation support via validate_batch()
    - LRU cache for common word lookups
    """
    
    # Maximum entity name length (prevents extremely long extractions)
    MAX_ENTITY_NAME_LENGTH = 100
    
    # All-caps stopwords to reject (these are almost never useful entities)
    # These are structural words that appear frequently in text but rarely
    # represent meaningful entities when extracted in all-caps form.
    ALL_CAPS_STOPWORDS = {
        'THE', 'AND', 'FOR', 'BUT', 'OR', 'NOT', 'WITH', 'THIS', 'THAT',
        'FROM', 'THEY', 'HAVE', 'BEEN', 'WILL', 'WOULD', 'COULD', 'SHOULD',
        'THEIR', 'THERE', 'ABOUT', 'WHICH', 'WHEN', 'WHAT', 'WHERE',
        'SOME', 'MUCH', 'MANY', 'ALSO', 'INTO', 'JUST', 'VERY', 'THAN',
        'THEN', 'ONLY', 'EACH', 'BOTH', 'SUCH', 'LIKE', 'MORE', 'MOST',
        'OVER', 'AFTER', 'BEFORE'
    }
    
    # Single-word common English words to reject when entity_type is 'concept'
    # Conservative list - only obvious generic words that are rarely useful as entities.
    # Be careful not to reject domain-specific terms (e.g., "Docker" is valid).
    # Common English words that should NOT be stored as 'concept' entities.
    # Conservative list: only truly meaningless words. Does NOT include
    # subjective quality words (good, bad, useful, check, nice, great)
    # which could be legitimate technical terms (e.g., "health check").
    COMMON_CONCEPT_WORDS = {
        # Articles, conjunctions, prepositions
        'the', 'and', 'a', 'an', 'of', 'to', 'in', 'for', 'is', 'on', 'at',
        'by', 'with', 'from', 'as', 'be', 'it', 'this', 'that', 'which',
        # Verbs (be/have/do/modal)
        'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'can', 'shall', 'need', 'dare', 'ought',
        # Basic affirmatives/negatives
        'yes', 'no', 'ok', 'okay',
        # Common intensifiers/qualifiers
        'very', 'too', 'so', 'just', 'really', 'quite', 'rather',
        'pretty', 'enough', 'also', 'only', 'even', 'well',
    }
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def _is_common_word(name_lower: str) -> bool:
        """Cached check if a lowercase name is a common concept word.
        
        Note: This cache is intentionally global (shared across all EntityValidator
        instances) since the results are deterministic and immutable. This provides
        marginal benefits for workloads that validate the same entity names repeatedly.
        
        Args:
            name_lower: Lowercase entity name.
            
        Returns:
            True if the name is in COMMON_CONCEPT_WORDS.
        """
        return name_lower in EntityValidator.COMMON_CONCEPT_WORDS
    
    @staticmethod
    @lru_cache(maxsize=512)
    def _has_alpha(name: str) -> bool:
        """Cached check if a string contains alphabetic characters.
        
        Note: This cache is intentionally global (shared across all EntityValidator
        instances) since the results are deterministic. The main benefit is avoiding
        repeated iterations for long entity names that are validated multiple times.
        
        Args:
            name: String to check.
            
        Returns:
            True if the string contains at least one alphabetic character.
        """
        return any(c.isalpha() for c in name)
    
    def is_valid(self, entity: Entity) -> bool:
        """Check if an entity is valid for storage.
        
        Validation rules:
            - Name length must be >= MIN_ENTITY_NAME_LENGTH (3) and <= 100 chars
            - Name must contain at least one alphabetic character
            - Rejects all-caps stopwords (THE, AND, FOR, etc.)
            - Rejects numeric-only names ("12345")
            - Rejects single-word common English words when entity_type is 'concept'
        
        Args:
            entity: The entity to validate.
            
        Returns:
            True if the entity passes all validation rules, False otherwise.
        """
        name = entity.name.strip() if entity.name else ""
        
        # Reject empty or whitespace-only names
        if not name:
            return False
        
        # Reject names below minimum length
        if len(name) < MIN_ENTITY_NAME_LENGTH:
            return False
        
        # Reject names over maximum length
        if len(name) > self.MAX_ENTITY_NAME_LENGTH:
            return False
        
        # Reject all-caps stopwords
        if name in self.ALL_CAPS_STOPWORDS:
            return False
        
        # Reject numeric-only entities
        if name.isdigit():
            return False
        
        # Reject entities with no alphabetic characters (cached)
        if not self._has_alpha(name):
            return False
        
        # Reject single-word common English words when entity_type is 'concept'
        # Only apply this rule for 'concept' type - other types may have legitimate
        # single-word names (e.g., organization named "Check")
        if entity.entity_type == 'concept':
            # Check if it's a single word (no spaces)
            if ' ' not in name and self._is_common_word(name.lower()):
                return False
        
        return True
    
    def validate_batch(self, entities: List[Entity]) -> List[bool]:
        """Validate a batch of entities efficiently.
        
        Performance optimization for large-scale validation (Issue #135).
        Reduces Python function call overhead by inlining validation logic
        instead of repeatedly calling is_valid().
        
        Args:
            entities: List of entities to validate.
            
        Returns:
            List of booleans indicating validity for each entity (same order).
        """
        results = []
        all_caps_stopwords = self.ALL_CAPS_STOPWORDS
        max_length = self.MAX_ENTITY_NAME_LENGTH
        min_length = MIN_ENTITY_NAME_LENGTH
        
        for entity in entities:
            name = entity.name.strip() if entity.name else ""
            
            # Fast path: check simple conditions first
            if (not name or 
                len(name) < min_length or 
                len(name) > max_length or
                name in all_caps_stopwords or
                name.isdigit()):
                results.append(False)
                continue
            
            # Check for alphabetic characters (cached)
            if not self._has_alpha(name):
                results.append(False)
                continue
            
            # Check concept-specific filtering
            if entity.entity_type == 'concept' and ' ' not in name and self._is_common_word(name.lower()):
                results.append(False)
                continue
            
            results.append(True)
        
        return results


class RelationshipValidator:
    """Validates relationships before they enter the graph store (Issue #129, #135).
    
    Prevents garbage relationships from polluting the knowledge graph by enforcing:
    - Both source and target must be valid entities
    - No self-relationships (source == target, case-insensitive)
    - Source and target must meet minimum length requirements
    
    Performance optimizations (Issue #135):
    - Batch validation support via validate_batch()
    """
    
    def __init__(self) -> None:
        """Initialize with an entity validator."""
        self._entity_validator = EntityValidator()
    
    def is_valid(self, relationship: Relationship) -> bool:
        """Check if a relationship is valid for storage.
        
        Validation rules:
            - Source and target must not be empty
            - No self-relationships (source == target, case-insensitive)
            - Both source and target must pass basic entity validation
              (length, alphabetic chars) but NOT concept-specific word filtering
        
        Args:
            relationship: The relationship to validate.
            
        Returns:
            True if the relationship passes all validation rules, False otherwise.
        """
        source = relationship.source.strip() if relationship.source else ""
        target = relationship.target.strip() if relationship.target else ""
        
        # Reject if source or target is empty
        if not source or not target:
            return False
        
        # Reject self-relationships (case-insensitive)
        if source.lower() == target.lower():
            return False
        
        # Validate source and target as entities.
        # Use entity_type='unknown' (not 'concept') to avoid triggering
        # concept-specific common word filtering — relationship endpoints
        # may be organization names like "Check" or "Stripe" that happen
        # to be common English words.
        source_entity = Entity(name=source, entity_type='unknown')
        if not self._entity_validator.is_valid(source_entity):
            return False
        
        target_entity = Entity(name=target, entity_type='unknown')
        if not self._entity_validator.is_valid(target_entity):
            return False
        
        return True
    
    def validate_batch(self, relationships: List[Relationship]) -> List[bool]:
        """Validate a batch of relationships efficiently.
        
        Performance optimization for large-scale validation (Issue #135).
        Reduces Python function call overhead by inlining validation logic
        instead of repeatedly calling is_valid().
        
        Args:
            relationships: List of relationships to validate.
            
        Returns:
            List of booleans indicating validity for each relationship (same order).
        """
        results = []
        
        for relationship in relationships:
            source = relationship.source.strip() if relationship.source else ""
            target = relationship.target.strip() if relationship.target else ""
            
            # Fast path: check simple conditions
            if not source or not target or source.lower() == target.lower():
                results.append(False)
                continue
            
            # Validate source and target as entities
            source_entity = Entity(name=source, entity_type='unknown')
            target_entity = Entity(name=target, entity_type='unknown')
            
            if not (self._entity_validator.is_valid(source_entity) and 
                    self._entity_validator.is_valid(target_entity)):
                results.append(False)
                continue
            
            results.append(True)
        
        return results


# Pronoun set for filtering relationship endpoints
PRONOUNS = {
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    'this', 'that', 'these', 'those',
}


class DependencyRelationshipExtractor:
    """Extract relationships using spaCy dependency parsing (Issue #132).
    
    Finds subject-verb-object triples from dependency parse trees and maps
    verbs to relationship types. Only creates relationships between entities
    that were already extracted by NER.
    
    Key features:
        - Walks spaCy dependency tree to find SVO triples
        - Handles prepositional phrases ("lives in Denver")
        - Supports passive voice ("was bought by John")
        - Handles compound subjects/objects ("John and Mary")
        - Maps verbs to semantic relationship types
        - Filters out pronouns as relationship endpoints
        - Matches multi-word entities correctly
    
    Usage:
        extractor = DependencyRelationshipExtractor()
        doc = nlp("John uses Python")
        entities = [Entity("John", "person"), Entity("Python", "technology")]
        relationships = extractor.extract(doc, entities)
        # Returns: [Relationship("John", "Python", "uses")]
    """
    
    # Verb-to-relationship mapping.
    # Maps verb lemmas to semantic relationship types.
    # Covers common verbs in personal conversations and their inflected forms.
    VERB_RELATIONS: dict[str, str] = {
        # Technology/tool usage
        'use': 'uses', 'uses': 'uses', 'used': 'uses', 'using': 'uses',
        
        # Location
        'live': 'located_in', 'lives': 'located_in', 'lived': 'located_in',
        'move': 'moved_to', 'moved': 'moved_to',
        
        # Employment/education
        'work': 'works_at', 'works': 'works_at', 'worked': 'works_at',
        'study': 'studied_at', 'studied': 'studied_at', 'studies': 'studied_at',
        
        # Travel/events
        'visit': 'visited', 'visited': 'visited', 'visiting': 'visited',
        'attend': 'attended', 'attended': 'attended',
        
        # Social
        'meet': 'met', 'met': 'met', 'meeting': 'met',
        'know': 'knows', 'knows': 'knows', 'knew': 'knows',
        
        # Preferences
        'like': 'prefers', 'likes': 'prefers', 'liked': 'prefers',
        'love': 'prefers', 'loves': 'prefers', 'loved': 'prefers',
        'prefer': 'prefers', 'prefers': 'prefers', 'preferred': 'prefers',
        
        # Transactions
        'buy': 'purchased', 'bought': 'purchased', 'buying': 'purchased',
        'own': 'owns', 'owns': 'owns', 'owned': 'owns',
        
        # Information flow
        'recommend': 'recommends', 'recommends': 'recommends',
        
        # Activities
        'play': 'plays', 'plays': 'plays', 'played': 'plays',
        'drive': 'drives', 'drives': 'drives', 'drove': 'drives',
        'eat': 'eats', 'eats': 'eats', 'ate': 'eats',
    }
    
    def extract(self, doc: "spacy.tokens.Doc", entities: list[Entity]) -> list[Relationship]:
        """Extract relationships from a spaCy Doc using dependency parsing.
        
        Examples:
            >>> import spacy
            >>> nlp = spacy.load("en_core_web_sm")
            >>> extractor = DependencyRelationshipExtractor()
            >>> 
            >>> # Basic SVO
            >>> doc = nlp("John uses Python")
            >>> entities = [Entity("John", "person"), Entity("Python", "technology")]
            >>> extractor.extract(doc, entities)
            [Relationship("John", "Python", "uses")]
            >>> 
            >>> # Prepositional relationship
            >>> doc = nlp("Sarah lives in Denver")
            >>> entities = [Entity("Sarah", "person"), Entity("Denver", "place")]
            >>> extractor.extract(doc, entities)
            [Relationship("Sarah", "Denver", "located_in")]
        
        Args:
            doc: spaCy Doc object (already parsed).
            entities: List of entities extracted from the same text.
                      Only relationships between these entities will be created.
        
        Returns:
            List of Relationship objects extracted from dependency parse.
        """
        if not doc or not entities:
            return []
        
        # Build entity name index for fast lookup (case-insensitive)
        # Maps lowercase name -> original case name for case preservation
        entity_map = {e.name.lower(): e.name for e in entities}
        
        relationships = []
        
        # Find all verbs and their subject-object dependencies
        for token in doc:
            if token.pos_ != "VERB":
                continue
            
            # Get verb lemma (lowercase) for mapping
            verb_lemma = token.lemma_.lower()
            
            # Skip unmapped verbs
            if verb_lemma not in self.VERB_RELATIONS:
                continue
            
            relation_type = self.VERB_RELATIONS[verb_lemma]
            
            # Extract relationships from this verb
            verb_relationships = self._extract_from_verb(
                token, entity_map, relation_type
            )
            relationships.extend(verb_relationships)
            
            # Handle coordinated verbs (conjunctions that share the same subject)
            # Example: "Mike works at Google and lives in Seattle"
            # "lives" is a conj of "works", shares subject "Mike"
            for child in token.children:
                if child.dep_ == "conj" and child.pos_ == "VERB":
                    conj_verb_lemma = child.lemma_.lower()
                    if conj_verb_lemma in self.VERB_RELATIONS:
                        conj_relation_type = self.VERB_RELATIONS[conj_verb_lemma]
                        # Extract from coordinated verb, passing parent's subjects
                        conj_relationships = self._extract_from_verb(
                            child, entity_map, conj_relation_type,
                            inherited_subjects=self._find_subjects(token)
                        )
                        relationships.extend(conj_relationships)
        
        return relationships
    
    def _extract_from_verb(
        self,
        verb_token: "spacy.tokens.Token",
        entity_map: dict[str, str],
        relation_type: str,
        inherited_subjects: Optional[list[str]] = None
    ) -> list[Relationship]:
        """Extract all relationships from a verb token.
        
        Handles:
            - Active voice: subject + verb + direct object
            - Passive voice: subject + verb + agent (by-phrase)
            - Prepositional objects: verb + prep + object
            - Coordinated verbs (inherited subjects from parent verb)
        
        Args:
            verb_token: spaCy Token for the verb.
            entity_map: Dict mapping lowercase entity names to original case names.
            relation_type: Relationship type to create.
            inherited_subjects: Subjects inherited from a parent verb (for coordinated verbs).
        
        Returns:
            List of extracted relationships.
        """
        relationships = []
        
        # Find subjects (use inherited if provided and no explicit subjects found)
        subjects = self._find_subjects(verb_token)
        if not subjects and inherited_subjects:
            subjects = inherited_subjects
        
        # Find objects (direct objects, prepositional objects, attributes)
        objects = self._find_objects(verb_token)
        
        # Create relationships between all subject-object pairs
        for subj_text in subjects:
            for obj_text in objects:
                # Skip if either is a pronoun
                if self._is_pronoun(subj_text) or self._is_pronoun(obj_text):
                    continue
                
                # Match to entity names (fuzzy match for multi-word entities)
                subj_entity = self._match_entity(subj_text, entity_map)
                obj_entity = self._match_entity(obj_text, entity_map)
                
                # Only create relationship if both match known entities
                if subj_entity and obj_entity:
                    relationships.append(Relationship(
                        source=subj_entity,
                        target=obj_entity,
                        relation_type=relation_type
                    ))
        
        # Handle passive voice: check for agent phrases (by X)
        passive_rels = self._extract_passive_relationships(
            verb_token, entity_map, relation_type
        )
        relationships.extend(passive_rels)
        
        return relationships
    
    def _find_subjects(self, verb_token: "spacy.tokens.Token") -> list[str]:
        """Find subject spans for a verb token.
        
        Handles:
            - nsubj (nominal subject)
            - nsubjpass (passive nominal subject)
            - Compound subjects (John and Mary)
        
        Args:
            verb_token: spaCy Token for the verb.
        
        Returns:
            List of subject text spans.
        """
        subjects = []
        
        for child in verb_token.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                # Get full noun phrase including compounds
                subj_text = self._get_full_noun_phrase(child)
                subjects.append(subj_text)
                
                # Check for compound subjects (and, or)
                for conj_child in child.children:
                    if conj_child.dep_ == "conj":
                        conj_text = self._get_full_noun_phrase(conj_child)
                        subjects.append(conj_text)
        
        return subjects
    
    def _find_objects(self, verb_token: "spacy.tokens.Token") -> list[str]:
        """Find object spans for a verb token.
        
        Handles:
            - dobj (direct object)
            - pobj (prepositional object via prep children)
            - attr (attribute)
            - oprd (object predicate)
        
        Args:
            verb_token: spaCy Token for the verb.
        
        Returns:
            List of object text spans.
        """
        objects = []
        
        for child in verb_token.children:
            # Direct objects
            if child.dep_ in {"dobj", "attr", "oprd"}:
                obj_text = self._get_full_noun_phrase(child)
                objects.append(obj_text)
                
                # Check for compound objects
                for conj_child in child.children:
                    if conj_child.dep_ == "conj":
                        conj_text = self._get_full_noun_phrase(conj_child)
                        objects.append(conj_text)
            
            # Prepositional objects ("lives in Denver")
            elif child.dep_ == "prep":
                for prep_child in child.children:
                    if prep_child.dep_ == "pobj":
                        pobj_text = self._get_full_noun_phrase(prep_child)
                        objects.append(pobj_text)
                        
                        # Compound prepositional objects
                        for conj_child in prep_child.children:
                            if conj_child.dep_ == "conj":
                                conj_text = self._get_full_noun_phrase(conj_child)
                                objects.append(conj_text)
        
        return objects
    
    def _extract_passive_relationships(
        self,
        verb_token: "spacy.tokens.Token",
        entity_map: dict[str, str],
        relation_type: str
    ) -> list[Relationship]:
        """Extract relationships from passive voice constructions.
        
        Example: "Python was recommended by Alice"
            → Alice recommends Python (reversed subject/object)
        
        Args:
            verb_token: spaCy Token for the verb.
            entity_map: Dict mapping lowercase entity names to original case names.
            relation_type: Relationship type to create.
        
        Returns:
            List of relationships with reversed direction.
        """
        relationships = []
        
        # Find passive subject (the thing being acted upon)
        passive_subjects = []
        for child in verb_token.children:
            if child.dep_ == "nsubjpass":
                subj_text = self._get_full_noun_phrase(child)
                passive_subjects.append(subj_text)
        
        # Find agent ("by X")
        agents = []
        for child in verb_token.children:
            if child.dep_ == "agent":  # "by" phrase
                for agent_child in child.children:
                    if agent_child.dep_ == "pobj":
                        agent_text = self._get_full_noun_phrase(agent_child)
                        agents.append(agent_text)
        
        # Create relationships: agent -> passive_subject
        # (reversed from normal subject -> object)
        for agent_text in agents:
            for psubj_text in passive_subjects:
                if self._is_pronoun(agent_text) or self._is_pronoun(psubj_text):
                    continue
                
                agent_entity = self._match_entity(agent_text, entity_map)
                psubj_entity = self._match_entity(psubj_text, entity_map)
                
                if agent_entity and psubj_entity:
                    # Agent is the source in passive voice
                    relationships.append(Relationship(
                        source=agent_entity,
                        target=psubj_entity,
                        relation_type=relation_type
                    ))
        
        return relationships
    
    def _get_full_noun_phrase(self, token: "spacy.tokens.Token") -> str:
        """Get the full noun phrase including compounds and key modifiers.
        
        Captures:
            - Compounds (New York)
            - Adjectives (Big Apple)
            - Possessives (Joe's Restaurant)
            - Numerals (5 stars)
            - Recursive compounds (San Francisco Bay Area)
        
        Args:
            token: Head token of the noun phrase.
        
        Returns:
            Full noun phrase as a string.
        """
        # Collect all tokens in the noun phrase
        tokens = [token]
        
        # Add relevant modifiers
        for child in token.children:
            if child.dep_ in {"compound", "amod", "poss", "nummod"}:
                tokens.append(child)
                # Recursively get compounds of the modifier
                # (e.g., "San Francisco Bay Area" where "San" is compound of "Francisco")
                for grandchild in child.children:
                    if grandchild.dep_ in {"compound", "amod"}:
                        tokens.append(grandchild)
        
        # Sort by position in sentence
        tokens.sort(key=lambda t: t.i)
        
        # Join into text
        return ' '.join(t.text for t in tokens)
    
    def _is_pronoun(self, text: str) -> bool:
        """Check if text is a pronoun that should be filtered.
        
        Args:
            text: Text to check.
        
        Returns:
            True if text is a pronoun.
        """
        return text.lower().strip() in PRONOUNS
    
    def _match_entity(self, text: str, entity_map: dict[str, str]) -> Optional[str]:
        """Match text span to a known entity name.
        
        Uses case-insensitive matching. Prefers exact matches, then checks
        if text appears as a complete word within a multi-word entity name
        using word boundary matching to prevent false positives (e.g.,
        "can" should NOT match "Glen Canyon Dam").
        
        Args:
            text: Text span from dependency parse.
            entity_map: Dict mapping lowercase entity names to original case names.
        
        Returns:
            Matched entity name (original case), or None if no match.
        """
        text_lower = text.lower().strip()
        
        # Exact match (most common case)
        if text_lower in entity_map:
            return entity_map[text_lower]
        
        # Check if text appears as a complete word within a multi-word entity.
        # Uses word boundary regex to prevent false positives like
        # "can" matching "Glen Canyon Dam" or "go" matching "Chicago".
        # Only match if text is shorter (avoid matching "New York City" to "New York")
        pattern = re.compile(r'\b' + re.escape(text_lower) + r'\b')
        candidates = [
            (entity_lower, entity_original)
            for entity_lower, entity_original in entity_map.items()
            if len(text_lower) < len(entity_lower) and pattern.search(entity_lower)
        ]
        
        if len(candidates) == 1:
            return candidates[0][1]  # Return original case
        elif len(candidates) > 1:
            # Multiple matches - return shortest to prefer more specific match
            # e.g., if text="York" matches both "New York" and "New York City",
            # prefer "New York"
            shortest = min(candidates, key=lambda x: len(x[0]))
            return shortest[1]
        
        return None


class HybridEntityExtractor:
    """Combines regex-based and spaCy-based entity extraction.
    
    Uses regex patterns for software/technical entities and spaCy NER for
    named entities (people, places, dates). This provides broad coverage
    for both code-related and personal conversation use cases.
    
    Falls back to regex-only if spaCy is not available.
    
    Extraction context controls relationship extraction behavior:
        - "personal" (default): Disables regex relationship extraction to avoid
          garbage relationships from casual conversation.
        - "software": Enables regex relationship extraction for technical text.
    """
    
    def __init__(
        self,
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
        extraction_context: str = "personal"
    ) -> None:
        """Initialize hybrid extractor.
        
        Args:
            use_spacy: Whether to use spaCy NER. Set False to use regex only.
            spacy_model: spaCy model name to load.
            extraction_context: Context for extraction ("personal" or "software").
                Defaults to "personal" which disables regex relationship extraction
                to prevent garbage relationships from casual conversation.
        
        Raises:
            ValueError: If extraction_context is not "personal" or "software".
        """
        if extraction_context not in ("personal", "software"):
            raise ValueError(
                f"extraction_context must be 'personal' or 'software', "
                f"got: {extraction_context!r}"
            )
        self._regex_extractor = EntityExtractor()
        self._spacy_extractor: Optional[SpacyEntityExtractor] = None
        self._extraction_context = extraction_context
        
        # Validators for quality filtering (Issue #129)
        self._entity_validator = EntityValidator()
        self._relationship_validator = RelationshipValidator()
        
        if use_spacy and SPACY_AVAILABLE:
            try:
                self._spacy_extractor = SpacyEntityExtractor(spacy_model)
            except (ImportError, OSError) as e:
                # Fall back to regex-only with warning
                logger.warning(
                    "spaCy extraction unavailable (%s), falling back to regex-only: %s",
                    type(e).__name__, e
                )
    
    @property
    def has_spacy(self) -> bool:
        """Whether spaCy extraction is available."""
        return self._spacy_extractor is not None
    
    def extract(self, text: Optional[str]) -> list[Entity]:
        """Extract entities using both regex and spaCy.
        
        Args:
            text: Input text to extract entities from (can be None).
            
        Returns:
            Combined, deduplicated list of valid entities from both extractors.
        """
        if not text or not text.strip():
            return []
        
        # Get regex entities first
        entities = self._regex_extractor.extract(text)
        seen_names = {e.name.lower() for e in entities}
        
        # Add spaCy entities if available
        if self._spacy_extractor:
            spacy_entities = self._spacy_extractor.extract(text)
            for ent in spacy_entities:
                if ent.name.lower() not in seen_names:
                    seen_names.add(ent.name.lower())
                    entities.append(ent)
        
        # Filter through validator to remove garbage entities (Issue #129)
        valid_entities = [e for e in entities if self._entity_validator.is_valid(e)]
        
        return valid_entities
    
    def extract_with_relationships(
        self, text: str
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships.
        
        Respects extraction_context:
            - "personal": Extracts entities via regex+spaCy, relationships via
              dependency parsing (Issue #132).
            - "software": Extracts entities + regex relationships.
        
        Args:
            text: Input text to process.
            
        Returns:
            Tuple of (combined valid entities, valid relationships).
        """
        # Context-aware extraction: personal context uses dependency parsing
        if self._extraction_context == "personal":
            # Extract entities only (no relationships from regex patterns)
            regex_entities = self._regex_extractor.extract(text)
            relationships = []
            
            # Use dependency parsing for relationships if spaCy is available
            if self._spacy_extractor:
                # Parse text with spaCy to get Doc (for dependency parsing)
                doc = self._spacy_extractor._nlp(text)
                
                # We'll combine entities after spaCy extraction below
                # For now, just initialize the dependency extractor
                dep_extractor = DependencyRelationshipExtractor()
            else:
                doc = None
                dep_extractor = None
        else:
            # Software context: extract both entities and relationships
            regex_entities, relationships = self._regex_extractor.extract_with_relationships(text)
            doc = None
            dep_extractor = None
        
        seen_names = {e.name.lower() for e in regex_entities}
        
        # Combine with spaCy entities
        entities = list(regex_entities)
        if self._spacy_extractor:
            spacy_entities = self._spacy_extractor.extract(text)
            for ent in spacy_entities:
                if ent.name.lower() not in seen_names:
                    seen_names.add(ent.name.lower())
                    entities.append(ent)
        
        # Filter entities through validator to remove garbage (Issue #129)
        valid_entities = [e for e in entities if self._entity_validator.is_valid(e)]
        
        # Extract dependency-parsed relationships if in personal context
        if self._extraction_context == "personal" and dep_extractor and doc:
            dep_relationships = dep_extractor.extract(doc, valid_entities)
            relationships.extend(dep_relationships)
        
        # Filter relationships through validator to remove garbage (Issue #129)
        valid_relationships = [r for r in relationships if self._relationship_validator.is_valid(r)]
        
        return valid_entities, valid_relationships


class GraphStore:
    """SQLite-backed graph storage for entities and relationships.
    
    Schema:
    - entities: (id, name, entity_type, metadata_json)
    - entity_memories: (entity_id, memory_id) - many-to-many
    - relationships: (id, source_entity_id, target_entity_id, relation_type, metadata_json)
    - relationship_memories: (relationship_id, memory_id) - many-to-many
    
    Connection management:
        Uses a persistent SQLite connection with WAL mode for better concurrency.
        Thread-safe with an RLock protecting database operations.
        
        Lifecycle:
            # Option 1: Context manager (recommended)
            with GraphStore(db_path) as store:
                store.add_entity(...)
            
            # Option 2: Manual cleanup
            store = GraphStore(db_path)
            try:
                store.add_entity(...)
            finally:
                store.close()
            
            # Option 3: Rely on __del__ (automatic cleanup on garbage collection)
            store = GraphStore(db_path)
            store.add_entity(...)
            # Connection closed when store is garbage collected
        
        After calling close(), the GraphStore instance should not be used.
    """
    
    # Known technology names for type inference
    KNOWN_TECHNOLOGIES = EntityExtractor.TECHNOLOGIES
    
    def __init__(self, db_path: str | Path):
        """Initialize graph store with SQLite database.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create persistent connection with thread safety
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False  # Allow usage across threads
        )
        self._conn.row_factory = sqlite3.Row
        
        # Use RLock for better read concurrency
        # RLock allows same thread to acquire multiple times
        # WAL mode handles most read concurrency at SQLite level
        self._lock = threading.RLock()
        
        # Enable WAL mode for better concurrency
        self._conn.execute("PRAGMA journal_mode=WAL")
        
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get the persistent database connection.
        
        Returns:
            SQLite connection with Row factory.
        """
        return self._conn
    
    def __enter__(self) -> 'GraphStore':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensure cleanup."""
        self.close()
        return None
    
    def __del__(self) -> None:
        """Destructor to ensure connection is closed.
        
        Called when the object is garbage collected.
        Ensures resources are released even if close() wasn't called explicitly.
        """
        self.close()
    
    def close(self) -> None:
        """Close the database connection and release resources.
        
        After calling close(), the GraphStore instance should not be used.
        This method is idempotent - calling it multiple times is safe.
        """
        if hasattr(self, '_conn') and self._conn:
            try:
                self._conn.close()
            except sqlite3.ProgrammingError:
                # Connection already closed, this is expected
                pass
            except Exception as e:
                # Log unexpected errors but don't raise
                # (close should be safe to call multiple times)
                import warnings
                warnings.warn(
                    f"Unexpected error closing GraphStore: {e}",
                    RuntimeWarning,
                    stacklevel=2
                )
    
    def _infer_entity_type(self, name: str) -> str:
        """Infer entity type from name when creating from relationships.
        
        Args:
            name: Entity name to analyze.
            
        Returns:
            Inferred entity type string.
        """
        if name.lower() in self.KNOWN_TECHNOLOGIES:
            return 'technology'
        # Check for service-like patterns
        name_lower = name.lower()
        if '-db' in name_lower or '-database' in name_lower:
            return 'database'
        if '-api' in name_lower or '-service' in name_lower:
            return 'service'
        if '-worker' in name_lower or '-job' in name_lower:
            return 'worker'
        if '-cache' in name_lower:
            return 'cache'
        if '-gateway' in name_lower or '-proxy' in name_lower:
            return 'gateway'
        if '-' in name:  # Generic kebab-case, probably a service
            return 'service'
        return 'concept'
    
    def _init_schema(self) -> None:
        """Initialize database schema.
        
        Thread-safe: Uses lock even though typically called during __init__.
        This ensures safety if schema updates are added later.
        """
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    UNIQUE(name)
                );
                
                CREATE TABLE IF NOT EXISTS entity_memories (
                    entity_id INTEGER NOT NULL,
                    memory_id TEXT NOT NULL,
                    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                    UNIQUE(entity_id, memory_id)
                );
                
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_entity_id INTEGER NOT NULL,
                    target_entity_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (source_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                    UNIQUE(source_entity_id, target_entity_id, relation_type)
                );
                
                CREATE TABLE IF NOT EXISTS relationship_memories (
                    relationship_id INTEGER NOT NULL,
                    memory_id TEXT NOT NULL,
                    FOREIGN KEY (relationship_id) REFERENCES relationships(id) ON DELETE CASCADE,
                    UNIQUE(relationship_id, memory_id)
                );
                
                -- Temporal facts table (Issue #57)
                CREATE TABLE IF NOT EXISTS temporal_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    resolved_date TEXT NOT NULL,
                    original_expression TEXT NOT NULL,
                    precision TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    UNIQUE(memory_id, subject, resolved_date)
                );
                
                CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name);
                CREATE INDEX IF NOT EXISTS idx_entity_memories_memory ON entity_memories(memory_id);
                CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id);
                CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id);
                CREATE INDEX IF NOT EXISTS idx_temporal_date ON temporal_facts(resolved_date);
                CREATE INDEX IF NOT EXISTS idx_temporal_memory ON temporal_facts(memory_id);
            """)
    
    def add_entity(self, entity: Entity, memory_id: str) -> int:
        """Add an entity and associate it with a memory.
        
        Returns the entity ID.
        """
        with self._lock:
            # Upsert entity
            cursor = self._conn.execute(
                """
                INSERT INTO entities (name, entity_type, metadata_json)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    entity_type = COALESCE(excluded.entity_type, entities.entity_type)
                RETURNING id
                """,
                (entity.name, entity.entity_type, '{}')
            )
            entity_id = cursor.fetchone()[0]
            
            # Associate with memory
            self._conn.execute(
                """
                INSERT OR IGNORE INTO entity_memories (entity_id, memory_id)
                VALUES (?, ?)
                """,
                (entity_id, memory_id)
            )
            self._conn.commit()
            
            return entity_id
    
    def add_relationship(self, relationship: Relationship, memory_id: str) -> int:
        """Add a relationship and associate it with a memory.
        
        Args:
            relationship: The relationship to store.
            memory_id: ID of the memory this relationship was extracted from.
        
        Returns:
            The relationship ID.
        """
        with self._lock:
            # Get or create source entity (infer type from name)
            source_row = self._conn.execute(
                "SELECT id FROM entities WHERE name = ?",
                (relationship.source,)
            ).fetchone()
            if not source_row:
                source_type = self._infer_entity_type(relationship.source)
                cursor = self._conn.execute(
                    "INSERT INTO entities (name, entity_type) VALUES (?, ?) RETURNING id",
                    (relationship.source, source_type)
                )
                source_id = cursor.fetchone()[0]
            else:
                source_id = source_row[0]
            
            # Get or create target entity (infer type from name)
            target_row = self._conn.execute(
                "SELECT id FROM entities WHERE name = ?",
                (relationship.target,)
            ).fetchone()
            if not target_row:
                target_type = self._infer_entity_type(relationship.target)
                cursor = self._conn.execute(
                    "INSERT INTO entities (name, entity_type) VALUES (?, ?) RETURNING id",
                    (relationship.target, target_type)
                )
                target_id = cursor.fetchone()[0]
            else:
                target_id = target_row[0]
            
            # Upsert relationship
            cursor = self._conn.execute(
                """
                INSERT INTO relationships (source_entity_id, target_entity_id, relation_type)
                VALUES (?, ?, ?)
                ON CONFLICT(source_entity_id, target_entity_id, relation_type) DO NOTHING
                RETURNING id
                """,
                (source_id, target_id, relationship.relation_type)
            )
            row = cursor.fetchone()
            if row:
                rel_id = row[0]
            else:
                # Relationship already exists, get its ID
                rel_id = self._conn.execute(
                    """
                    SELECT id FROM relationships 
                    WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = ?
                    """,
                    (source_id, target_id, relationship.relation_type)
                ).fetchone()[0]
            
            # Associate with memory
            self._conn.execute(
                """
                INSERT OR IGNORE INTO relationship_memories (relationship_id, memory_id)
                VALUES (?, ?)
                """,
                (rel_id, memory_id)
            )
            self._conn.commit()
            
            return rel_id
    
    def get_entities_for_memory(self, memory_id: str) -> list[Entity]:
        """Get all entities associated with a memory."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT e.name, e.entity_type, e.metadata_json
                FROM entities e
                JOIN entity_memories em ON e.id = em.entity_id
                WHERE em.memory_id = ?
                """,
                (memory_id,)
            ).fetchall()
            
            return [
                Entity(name=row['name'], entity_type=row['entity_type'])
                for row in rows
            ]
    
    def get_relationships_for_entity(self, entity_name: str) -> list[Relationship]:
        """Get all relationships where entity is the source."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT e_source.name as source, e_target.name as target, r.relation_type
                FROM relationships r
                JOIN entities e_source ON r.source_entity_id = e_source.id
                JOIN entities e_target ON r.target_entity_id = e_target.id
                WHERE e_source.name = ?
                """,
                (entity_name,)
            ).fetchall()
            
            return [
                Relationship(
                    source=row['source'],
                    target=row['target'],
                    relation_type=row['relation_type']
                )
                for row in rows
            ]
    
    def get_memories_for_entity(self, entity_name: str) -> list[str]:
        """Get all memory IDs associated with an entity."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT em.memory_id
                FROM entity_memories em
                JOIN entities e ON em.entity_id = e.id
                WHERE e.name = ?
                """,
                (entity_name,)
            ).fetchall()
            
            return [row['memory_id'] for row in rows]
    
    def find_connected(
        self, 
        entity_name: str, 
        hops: int = 1,
        include_source: bool = False
    ) -> list[Entity]:
        """Find entities connected to the given entity within N hops.
        
        Args:
            entity_name: Starting entity name.
            hops: Maximum number of relationship hops (1 = direct connections).
                  Capped at MAX_HOP_ITERATIONS for safety.
            include_source: Whether to include the source entity in results.
        
        Returns:
            List of connected entities.
        """
        # Safety: cap hops to prevent runaway traversal
        safe_hops = min(hops, MAX_HOP_ITERATIONS)
        
        with self._lock:
            # Start with source entity
            source = self._conn.execute(
                "SELECT id, name, entity_type FROM entities WHERE name = ?",
                (entity_name,)
            ).fetchone()
            
            if not source:
                return []
            
            visited: set[int] = {source['id']}
            current_frontier: set[int] = {source['id']}
            result_ids: set[int] = set()
            
            for _ in range(safe_hops):
                if not current_frontier:
                    break
                
                # Find all entities connected to current frontier
                # SECURITY NOTE: placeholders is safe because it's computed from
                # len(current_frontier) (an integer), not user input. The actual
                # values are passed as parameters, not interpolated.
                placeholders = ','.join('?' * len(current_frontier))
                rows = self._conn.execute(
                    f"""
                    SELECT DISTINCT e.id, e.name, e.entity_type
                    FROM entities e
                    JOIN relationships r ON (
                        (r.source_entity_id IN ({placeholders}) AND r.target_entity_id = e.id)
                        OR
                        (r.target_entity_id IN ({placeholders}) AND r.source_entity_id = e.id)
                    )
                    """,
                    list(current_frontier) + list(current_frontier)
                ).fetchall()
                
                next_frontier: set[int] = set()
                for row in rows:
                    if row['id'] not in visited:
                        visited.add(row['id'])
                        next_frontier.add(row['id'])
                        result_ids.add(row['id'])
                
                current_frontier = next_frontier
            
            # Fetch full entity info for results
            if not result_ids:
                return []
            
            placeholders = ','.join('?' * len(result_ids))
            rows = self._conn.execute(
                f"SELECT name, entity_type FROM entities WHERE id IN ({placeholders})",
                list(result_ids)
            ).fetchall()
            
            result = [
                Entity(name=row['name'], entity_type=row['entity_type'])
                for row in rows
            ]
            
            if include_source:
                result.insert(0, Entity(
                    name=source['name'], 
                    entity_type=source['entity_type']
                ))
            
            return result
    
    def delete_memory(self, memory_id: str) -> None:
        """Delete all entity and relationship associations for a memory.
        
        Note: Entities themselves are preserved (they may be referenced by other memories).
        Only the associations are removed.
        """
        with self._lock:
            # Delete relationship associations
            self._conn.execute(
                "DELETE FROM relationship_memories WHERE memory_id = ?",
                (memory_id,)
            )
            
            # Delete entity associations
            self._conn.execute(
                "DELETE FROM entity_memories WHERE memory_id = ?",
                (memory_id,)
            )
            
            # Delete temporal facts
            self._conn.execute(
                "DELETE FROM temporal_facts WHERE memory_id = ?",
                (memory_id,)
            )
            
            # Clean up orphaned relationships (no memory references)
            self._conn.execute("""
                DELETE FROM relationships 
                WHERE id NOT IN (SELECT relationship_id FROM relationship_memories)
            """)
            
            # Optionally clean up orphaned entities (no memory or relationship references)
            self._conn.execute("""
                DELETE FROM entities 
                WHERE id NOT IN (SELECT entity_id FROM entity_memories)
                AND id NOT IN (SELECT source_entity_id FROM relationships)
                AND id NOT IN (SELECT target_entity_id FROM relationships)
            """)
            self._conn.commit()
    
    # =========================================================================
    # Temporal Methods (Issue #57)
    # =========================================================================
    
    def add_temporal_fact(self, fact: TemporalFact, memory_id: str) -> int:
        """Store a temporal fact for a memory.
        
        Args:
            fact: The temporal fact to store.
            memory_id: ID of the memory this fact was extracted from.
        
        Returns:
            The temporal fact ID.
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO temporal_facts 
                    (memory_id, subject, relation, resolved_date, 
                     original_expression, precision, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, subject, resolved_date) DO UPDATE SET
                    confidence = MAX(excluded.confidence, temporal_facts.confidence)
                RETURNING id
                """,
                (memory_id, fact.subject, fact.relation, fact.resolved_date,
                 fact.original_expression, fact.precision, fact.confidence)
            )
            fact_id = cursor.fetchone()[0]
            self._conn.commit()
            return fact_id
    
    def get_temporal_facts_for_memory(self, memory_id: str) -> list[TemporalFact]:
        """Get all temporal facts for a memory."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT subject, relation, resolved_date, original_expression, 
                       precision, confidence
                FROM temporal_facts
                WHERE memory_id = ?
                """,
                (memory_id,)
            ).fetchall()
            
            return [
                TemporalFact(
                    subject=row['subject'],
                    relation=row['relation'],
                    resolved_date=row['resolved_date'],
                    original_expression=row['original_expression'],
                    precision=row['precision'],
                    confidence=row['confidence'],
                )
                for row in rows
            ]
    
    def get_memories_in_date_range(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[str]:
        """Get memory IDs with temporal facts in a date range.

        Args:
            start_date: ISO date string, range start (inclusive).
            end_date: ISO date string, range end (inclusive).

        Returns:
            List of memory IDs.
        """
        with self._lock:
            # Use explicit query variants to avoid f-string SQL
            if start_date and end_date:
                rows = self._conn.execute(
                    """SELECT DISTINCT memory_id
                    FROM temporal_facts
                    WHERE resolved_date >= ?
                      AND resolved_date <= ?
                    ORDER BY resolved_date""",
                    (start_date, end_date),
                ).fetchall()
            elif start_date:
                rows = self._conn.execute(
                    """SELECT DISTINCT memory_id
                    FROM temporal_facts
                    WHERE resolved_date >= ?
                    ORDER BY resolved_date""",
                    (start_date,),
                ).fetchall()
            elif end_date:
                rows = self._conn.execute(
                    """SELECT DISTINCT memory_id
                    FROM temporal_facts
                    WHERE resolved_date <= ?
                    ORDER BY resolved_date""",
                    (end_date,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT DISTINCT memory_id
                    FROM temporal_facts
                    ORDER BY resolved_date"""
                ).fetchall()

            return [row['memory_id'] for row in rows]
    
    def get_memories_for_date(self, date: str) -> list[str]:
        """Get memory IDs with events on a specific date.
        
        Args:
            date: ISO date string (YYYY-MM-DD, YYYY-MM, or YYYY).
        
        Returns:
            List of memory IDs.
        """
        with self._lock:
            # Use LIKE to match partial dates (year, year-month, or full date)
            rows = self._conn.execute(
                """
                SELECT DISTINCT memory_id
                FROM temporal_facts
                WHERE resolved_date LIKE ? || '%'
                """,
                (date,)
            ).fetchall()
            
            return [row['memory_id'] for row in rows]

    def get_memory_counts_batch(
        self, entity_names: list[str],
    ) -> dict[str, int]:
        """Get memory counts for multiple entities in one query.

        Args:
            entity_names: Entity names to look up.

        Returns:
            Dict mapping entity name → memory count.
        """
        if not entity_names:
            return {}
        with self._lock:
            placeholders = ",".join("?" * len(entity_names))
            rows = self._conn.execute(
                f"""
                SELECT e.name, COUNT(em.memory_id) as cnt
                FROM entities e
                LEFT JOIN entity_memories em
                    ON e.id = em.entity_id
                WHERE e.name IN ({placeholders})
                GROUP BY e.name
                """,
                entity_names,
            ).fetchall()
        return {r["name"]: r["cnt"] for r in rows}

    # =====================================================================
    # Graph visualization helpers (Issue #165)
    # =====================================================================

    def get_graph_stats(self) -> dict:
        """Get high-level graph statistics.

        Returns:
            Dict with entity_count, relationship_count,
            entity_types, relationship_types.
        """
        with self._lock:
            entity_count = self._conn.execute(
                "SELECT COUNT(*) FROM entities"
            ).fetchone()[0]

            rel_count = self._conn.execute(
                "SELECT COUNT(*) FROM relationships"
            ).fetchone()[0]

            type_rows = self._conn.execute(
                "SELECT entity_type, COUNT(*) as cnt "
                "FROM entities GROUP BY entity_type"
            ).fetchall()
            entity_types = {
                r["entity_type"]: r["cnt"]
                for r in type_rows
            }

            rel_type_rows = self._conn.execute(
                "SELECT relation_type, COUNT(*) as cnt "
                "FROM relationships "
                "GROUP BY relation_type"
            ).fetchall()
            rel_types = {
                r["relation_type"]: r["cnt"]
                for r in rel_type_rows
            }

        return {
            "entity_count": entity_count,
            "relationship_count": rel_count,
            "entity_types": entity_types,
            "relationship_types": rel_types,
        }

    def list_entities(
        self,
        offset: int = 0,
        limit: int = 50,
        entity_type: Optional[str] = None,
        search: Optional[str] = None,
    ) -> tuple[list[dict], int]:
        """List entities with pagination and filters.

        Returns:
            (entities, total_count) where each entity is a dict
            with name, entity_type, memory_count.
        """
        with self._lock:
            where_clauses: list[str] = []
            params: list = []

            if entity_type:
                where_clauses.append(
                    "e.entity_type = ?"
                )
                params.append(entity_type)

            if search:
                where_clauses.append("e.name LIKE ?")
                params.append(f"%{search}%")

            where_sql = ""
            if where_clauses:
                where_sql = (
                    "WHERE "
                    + " AND ".join(where_clauses)
                )

            total = self._conn.execute(
                "SELECT COUNT(*) "
                f"FROM entities e {where_sql}",
                params,
            ).fetchone()[0]

            rows = self._conn.execute(
                f"""
                SELECT e.name, e.entity_type,
                    COUNT(em.memory_id) as mem_count
                FROM entities e
                LEFT JOIN entity_memories em
                    ON e.id = em.entity_id
                {where_sql}
                GROUP BY e.id, e.name, e.entity_type
                ORDER BY mem_count DESC, e.name
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            ).fetchall()

        entities = [
            {
                "name": r["name"],
                "entity_type": r["entity_type"],
                "memory_count": r["mem_count"],
            }
            for r in rows
        ]
        return entities, total
