"""Graph-enriched memory storage for entity and relationship tracking.

Provides a lightweight graph layer alongside vector search to enable:
- Entity-centric queries ("tell me everything about auth-service")
- Relationship traversal ("what does auth-service connect to?")
- Multi-hop reasoning ("what framework does the service that handles auth use?")

Uses SQLite for local-first, zero-cloud constraint.
"""

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    """
    
    # Patterns for common entity types
    SERVICE_PATTERN = re.compile(
        r'\b([a-z][a-z0-9]*(?:-[a-z0-9]+)+(?:-service|-api|-worker|-db|-cache)?)\b',
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
        """Extract entities from text."""
        if not text or not text.strip():
            return []
        
        entities = []
        seen_names = set()
        
        # Extract service-like names (kebab-case identifiers)
        for match in self.SERVICE_PATTERN.finditer(text):
            name = match.group(1)
            if name.lower() not in seen_names and len(name) > 2:
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
        """Infer entity type from service-like name."""
        name_lower = name.lower()
        if '-db' in name_lower or '-database' in name_lower:
            return 'database'
        if '-api' in name_lower or '-service' in name_lower:
            return 'service'
        if '-worker' in name_lower or '-job' in name_lower:
            return 'worker'
        if '-cache' in name_lower:
            return 'cache'
        return 'service'
    
    def _infer_type(self, name: str) -> str:
        """Infer entity type from name."""
        if name.lower() in self.TECHNOLOGIES:
            return 'technology'
        if self.SERVICE_PATTERN.match(name):
            return self._infer_service_type(name)
        return 'concept'
    
    def _looks_like_entity(self, name: str) -> bool:
        """Check if a string looks like a valid entity name."""
        if not name or len(name) < 2:
            return False
        if name.lower() in self.TECHNOLOGIES:
            return True
        if self.SERVICE_PATTERN.match(name):
            return True
        # Capitalized words (proper nouns)
        if name[0].isupper() and name.isalnum():
            return True
        return False


class GraphStore:
    """SQLite-backed graph storage for entities and relationships.
    
    Schema:
    - entities: (id, name, entity_type, metadata_json)
    - entity_memories: (entity_id, memory_id) - many-to-many
    - relationships: (id, source_entity_id, target_entity_id, relation_type, metadata_json)
    - relationship_memories: (relationship_id, memory_id) - many-to-many
    """
    
    def __init__(self, db_path: str | Path):
        """Initialize graph store with SQLite database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
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
                
                CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name);
                CREATE INDEX IF NOT EXISTS idx_entity_memories_memory ON entity_memories(memory_id);
                CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id);
                CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id);
            """)
    
    def add_entity(self, entity: Entity, memory_id: str) -> int:
        """Add an entity and associate it with a memory.
        
        Returns the entity ID.
        """
        with self._get_connection() as conn:
            # Upsert entity
            cursor = conn.execute(
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
            conn.execute(
                """
                INSERT OR IGNORE INTO entity_memories (entity_id, memory_id)
                VALUES (?, ?)
                """,
                (entity_id, memory_id)
            )
            
            return entity_id
    
    def add_relationship(self, relationship: Relationship, memory_id: str) -> int:
        """Add a relationship and associate it with a memory.
        
        Returns the relationship ID.
        """
        with self._get_connection() as conn:
            # Get or create source entity
            source_row = conn.execute(
                "SELECT id FROM entities WHERE name = ?",
                (relationship.source,)
            ).fetchone()
            if not source_row:
                cursor = conn.execute(
                    "INSERT INTO entities (name, entity_type) VALUES (?, 'unknown') RETURNING id",
                    (relationship.source,)
                )
                source_id = cursor.fetchone()[0]
            else:
                source_id = source_row[0]
            
            # Get or create target entity
            target_row = conn.execute(
                "SELECT id FROM entities WHERE name = ?",
                (relationship.target,)
            ).fetchone()
            if not target_row:
                cursor = conn.execute(
                    "INSERT INTO entities (name, entity_type) VALUES (?, 'unknown') RETURNING id",
                    (relationship.target,)
                )
                target_id = cursor.fetchone()[0]
            else:
                target_id = target_row[0]
            
            # Upsert relationship
            cursor = conn.execute(
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
                rel_id = conn.execute(
                    """
                    SELECT id FROM relationships 
                    WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = ?
                    """,
                    (source_id, target_id, relationship.relation_type)
                ).fetchone()[0]
            
            # Associate with memory
            conn.execute(
                """
                INSERT OR IGNORE INTO relationship_memories (relationship_id, memory_id)
                VALUES (?, ?)
                """,
                (rel_id, memory_id)
            )
            
            return rel_id
    
    def get_entities_for_memory(self, memory_id: str) -> list[Entity]:
        """Get all entities associated with a memory."""
        with self._get_connection() as conn:
            rows = conn.execute(
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
        with self._get_connection() as conn:
            rows = conn.execute(
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
        with self._get_connection() as conn:
            rows = conn.execute(
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
            entity_name: Starting entity name
            hops: Maximum number of relationship hops (1 = direct connections)
            include_source: Whether to include the source entity in results
        
        Returns:
            List of connected entities
        """
        with self._get_connection() as conn:
            # Start with source entity
            source = conn.execute(
                "SELECT id, name, entity_type FROM entities WHERE name = ?",
                (entity_name,)
            ).fetchone()
            
            if not source:
                return []
            
            visited = {source['id']}
            current_frontier = {source['id']}
            result_ids = set()
            
            for _ in range(hops):
                if not current_frontier:
                    break
                
                # Find all entities connected to current frontier
                placeholders = ','.join('?' * len(current_frontier))
                rows = conn.execute(
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
                
                next_frontier = set()
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
            rows = conn.execute(
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
        with self._get_connection() as conn:
            # Delete relationship associations
            conn.execute(
                "DELETE FROM relationship_memories WHERE memory_id = ?",
                (memory_id,)
            )
            
            # Delete entity associations
            conn.execute(
                "DELETE FROM entity_memories WHERE memory_id = ?",
                (memory_id,)
            )
            
            # Clean up orphaned relationships (no memory references)
            conn.execute("""
                DELETE FROM relationships 
                WHERE id NOT IN (SELECT relationship_id FROM relationship_memories)
            """)
            
            # Optionally clean up orphaned entities (no memory or relationship references)
            conn.execute("""
                DELETE FROM entities 
                WHERE id NOT IN (SELECT entity_id FROM entity_memories)
                AND id NOT IN (SELECT source_entity_id FROM relationships)
                AND id NOT IN (SELECT target_entity_id FROM relationships)
            """)
