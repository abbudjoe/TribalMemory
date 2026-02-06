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
from pathlib import Path
from typing import Optional
import logging

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
    Can be upgraded to spaCy NER or LLM extraction later.
    
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
    RELATIONSHIP_PATTERNS = [
        (re.compile(r'(\S+)\s+uses\s+(\S+)', re.IGNORECASE), 'uses'),
        (re.compile(r'(\S+)\s+connects?\s+to\s+(\S+)', re.IGNORECASE), 'connects_to'),
        (re.compile(r'(\S+)\s+stores?\s+(?:data\s+)?in\s+(\S+)', re.IGNORECASE), 'stores_in'),
        (re.compile(r'(\S+)\s+depends?\s+on\s+(\S+)', re.IGNORECASE), 'depends_on'),
        (re.compile(r'(\S+)\s+talks?\s+to\s+(\S+)', re.IGNORECASE), 'connects_to'),
        (re.compile(r'(\S+)\s+calls?\s+(\S+)', re.IGNORECASE), 'calls'),
        (re.compile(r'(\S+)\s+handles?\s+(\S+)', re.IGNORECASE), 'handles'),
        (re.compile(r'(\S+)\s+for\s+(?:the\s+)?(\S+)', re.IGNORECASE), 'serves'),
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
        """Extract both entities and relationships from text."""
        entities = self.extract(text)
        entity_names = {e.name.lower() for e in entities}
        relationships = []
        
        for pattern, rel_type in self.RELATIONSHIP_PATTERNS:
            for match in pattern.finditer(text):
                source = match.group(1).strip('.,;:')
                target = match.group(2).strip('.,;:')
                
                # Only create relationship if both entities were extracted
                # or if they look like valid entity names
                source_valid = (
                    source.lower() in entity_names or 
                    self._looks_like_entity(source)
                )
                target_valid = (
                    target.lower() in entity_names or 
                    self._looks_like_entity(target)
                )
                
                if source_valid and target_valid:
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


# Check if spaCy is available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Common person titles to strip for better entity matching
PERSON_TITLES = {
    'dr', 'dr.', 'mr', 'mr.', 'mrs', 'mrs.', 'ms', 'ms.', 'miss',
    'prof', 'prof.', 'professor', 'sir', 'madam', 'rev', 'rev.',
}


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
    
    # Entity types we care about for personal conversations
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
    
    def _normalize_person_name(self, name: str) -> str:
        """Strip common titles from person names for better matching.
        
        Args:
            name: Raw person name (e.g., "Dr. Thompson").
            
        Returns:
            Normalized name (e.g., "Thompson").
        """
        parts = name.split()
        if len(parts) > 1 and parts[0].lower().rstrip('.') in PERSON_TITLES:
            return ' '.join(parts[1:])
        return name

    def extract(self, text: Optional[str]) -> list[Entity]:
        """Extract named entities from text using spaCy NER.
        
        Args:
            text: Input text to extract entities from (can be None).
            
        Returns:
            List of extracted Entity objects.
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
            entities.append(Entity(
                name=name,
                entity_type=entity_type,
                metadata={'spacy_label': ent.label_}
            ))
        
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


class HybridEntityExtractor:
    """Combines regex-based and spaCy-based entity extraction.
    
    Uses regex patterns for software/technical entities and spaCy NER for
    named entities (people, places, dates). This provides broad coverage
    for both code-related and personal conversation use cases.
    
    Falls back to regex-only if spaCy is not available.
    """
    
    def __init__(self, use_spacy: bool = True, spacy_model: str = "en_core_web_sm"):
        """Initialize hybrid extractor.
        
        Args:
            use_spacy: Whether to use spaCy NER. Set False to use regex only.
            spacy_model: spaCy model name to load.
        """
        self._regex_extractor = EntityExtractor()
        self._spacy_extractor: Optional[SpacyEntityExtractor] = None
        
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
            Combined, deduplicated list of entities from both extractors.
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
        
        return entities
    
    def extract_with_relationships(
        self, text: str
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships.
        
        Uses regex extractor for relationships (spaCy doesn't extract these).
        Combines entities from both extractors.
        
        Args:
            text: Input text to process.
            
        Returns:
            Tuple of (combined entities, regex-based relationships).
        """
        # Get regex entities + relationships
        regex_entities, relationships = self._regex_extractor.extract_with_relationships(text)
        seen_names = {e.name.lower() for e in regex_entities}
        
        # Combine with spaCy entities
        entities = list(regex_entities)
        if self._spacy_extractor:
            spacy_entities = self._spacy_extractor.extract(text)
            for ent in spacy_entities:
                if ent.name.lower() not in seen_names:
                    seen_names.add(ent.name.lower())
                    entities.append(ent)
        
        return entities, relationships


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
