"""Episode storage for Tribal Memory.

Provides SQLite-backed storage for episodic memory narratives.
Episodes group related memories into cohesive stories with auto-summarization.

Thread-safe with persistent connection and WAL mode.
"""

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


@dataclass
class Episode:
    """An episode grouping related memories.
    
    Attributes:
        id: Unique identifier (UUID).
        title: Episode title.
        summary: Generated narrative summary.
        summary_memory_id: FK to vector store memory containing summary.
        status: Episode status (active, closed, archived).
        created_at: When episode was created.
        updated_at: When last memory was added.
        closed_at: When episode was closed (None if active).
        memory_count: Number of memories in this episode.
        metadata: Additional metadata dictionary.
    """
    id: str
    title: str
    summary: str = ""
    summary_memory_id: Optional[str] = None
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    memory_count: int = 0
    metadata: dict = field(default_factory=dict)


class EpisodeStore:
    """SQLite-backed storage for episodes.
    
    Schema:
        episodes: (id, title, summary, summary_memory_id, status, created_at,
                  updated_at, closed_at, memory_count, metadata_json)
        episode_memories: (episode_id, memory_id, added_at, summarized)
    
    Thread-safe with RLock. Uses WAL mode for better concurrency.
    
    Usage:
        # Context manager (recommended)
        with EpisodeStore(db_path) as store:
            episode = store.create_episode("Title")
        
        # Manual cleanup
        store = EpisodeStore(db_path)
        try:
            episode = store.create_episode("Title")
        finally:
            store.close()
    """
    
    def __init__(self, db_path: str | Path):
        """Initialize episode store with SQLite database.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create persistent connection with thread safety
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False  # Allow usage across threads
        )
        self._conn.row_factory = sqlite3.Row
        
        # RLock for thread safety
        self._lock = threading.RLock()
        
        # Enable WAL mode for better concurrency
        self._conn.execute("PRAGMA journal_mode=WAL")
        
        # Enable foreign key constraints
        self._conn.execute("PRAGMA foreign_keys=ON")
        
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get the persistent database connection.
        
        Returns:
            SQLite connection with Row factory.
        """
        return self._conn
    
    def __enter__(self) -> 'EpisodeStore':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensure cleanup."""
        self.close()
        return None
    
    def __del__(self) -> None:
        """Destructor to ensure connection is closed."""
        self.close()
    
    def close(self) -> None:
        """Close database connection and release resources.
        
        Idempotent - safe to call multiple times.
        """
        if hasattr(self, '_conn') and self._conn:
            try:
                # Checkpoint WAL before close (best effort - may fail if DB is locked)
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.OperationalError:
                    # Database locked, skip checkpoint (SQLite will handle on next access)
                    pass
                self._conn.close()
            except sqlite3.ProgrammingError:
                # Connection already closed
                pass
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Unexpected error closing EpisodeStore: {e}",
                    RuntimeWarning,
                    stacklevel=2
                )
    
    def _init_schema(self) -> None:
        """Initialize database schema.
        
        Idempotent - safe to call multiple times.
        """
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_memory_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    closed_at TEXT,
                    memory_count INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}'
                );
                
                CREATE TABLE IF NOT EXISTS episode_memories (
                    episode_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    summarized BOOLEAN NOT NULL DEFAULT 0,
                    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE,
                    PRIMARY KEY (episode_id, memory_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_episode_status ON episodes(status);
                CREATE INDEX IF NOT EXISTS idx_episode_updated ON episodes(updated_at);
                CREATE INDEX IF NOT EXISTS idx_episode_status_updated 
                    ON episodes(status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_episode_memories_memory 
                    ON episode_memories(memory_id);
                CREATE INDEX IF NOT EXISTS idx_episode_summary_memory
                    ON episodes(summary_memory_id);
            """)
            self._conn.commit()
    
    MAX_TITLE_LENGTH = 500

    def create_episode(
        self,
        title: str,
        metadata: Optional[dict] = None
    ) -> Episode:
        """Create a new episode.
        
        Args:
            title: Episode title (max 500 chars).
            metadata: Optional metadata dictionary.
        
        Returns:
            Created Episode object.
        
        Raises:
            ValueError: If title is empty or exceeds MAX_TITLE_LENGTH.
        """
        if not title or not title.strip():
            raise ValueError("Episode title cannot be empty")
        if len(title) > self.MAX_TITLE_LENGTH:
            raise ValueError(
                f"Episode title too long ({len(title)} chars, max "
                f"{self.MAX_TITLE_LENGTH})"
            )
        episode = Episode(
            id=str(uuid.uuid4()),
            title=title,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO episodes 
                (id, title, summary, summary_memory_id, status, created_at, 
                 updated_at, closed_at, memory_count, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.id,
                    episode.title,
                    episode.summary,
                    episode.summary_memory_id,
                    episode.status,
                    episode.created_at.isoformat(),
                    episode.updated_at.isoformat(),
                    None,
                    episode.memory_count,
                    json.dumps(episode.metadata)
                )
            )
            self._conn.commit()
        
        return episode
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get episode by ID.
        
        Args:
            episode_id: Episode UUID.
        
        Returns:
            Episode object or None if not found.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM episodes WHERE id = ?",
                (episode_id,)
            )
            row = cursor.fetchone()
        
        if not row:
            return None
        
        return self._row_to_episode(row)
    
    def list_episodes(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Episode]:
        """List episodes with optional filtering.
        
        Args:
            status: Filter by status (active, closed, archived).
            limit: Maximum number of results.
        
        Returns:
            List of Episode objects.
        """
        with self._lock:
            if status:
                cursor = self._conn.execute(
                    """
                    SELECT * FROM episodes 
                    WHERE status = ? 
                    ORDER BY updated_at DESC 
                    LIMIT ?
                    """,
                    (status, limit)
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT * FROM episodes 
                    ORDER BY updated_at DESC 
                    LIMIT ?
                    """,
                    (limit,)
                )
            rows = cursor.fetchall()
        
        return [self._row_to_episode(row) for row in rows]
    
    def update_episode(self, episode_id: str, **kwargs) -> Episode:
        """Update episode fields.
        
        Args:
            episode_id: Episode UUID.
            **kwargs: Fields to update (title, summary, summary_memory_id, 
                     status, metadata).
        
        Returns:
            Updated Episode object.
        
        Raises:
            ValueError: If episode not found.
        """
        # Build update query
        update_fields = []
        values = []
        
        if 'title' in kwargs:
            update_fields.append("title = ?")
            values.append(kwargs['title'])
        
        if 'summary' in kwargs:
            update_fields.append("summary = ?")
            values.append(kwargs['summary'])
        
        if 'summary_memory_id' in kwargs:
            update_fields.append("summary_memory_id = ?")
            values.append(kwargs['summary_memory_id'])
        
        if 'status' in kwargs:
            update_fields.append("status = ?")
            values.append(kwargs['status'])
        
        if 'metadata' in kwargs:
            update_fields.append("metadata_json = ?")
            values.append(json.dumps(kwargs['metadata']))
        
        if not update_fields:
            # No fields to update, just verify existence
            episode = self.get_episode(episode_id)
            if not episode:
                raise ValueError(f"Episode {episode_id} not found")
            return episode
        
        # Always update updated_at
        update_fields.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        
        values.append(episode_id)
        
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE episodes SET {', '.join(update_fields)} WHERE id = ?",
                values
            )
            self._conn.commit()
            
            if cursor.rowcount == 0:
                raise ValueError(f"Episode {episode_id} not found")
        
        return self.get_episode(episode_id)
    
    def close_episode(self, episode_id: str) -> Episode:
        """Close an episode.
        
        Args:
            episode_id: Episode UUID.
        
        Returns:
            Closed Episode object.
        
        Raises:
            ValueError: If episode not found.
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE episodes 
                SET status = ?, closed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    "closed",
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                    episode_id
                )
            )
            self._conn.commit()
            
            if cursor.rowcount == 0:
                raise ValueError(f"Episode {episode_id} not found")
        
        return self.get_episode(episode_id)
    
    def set_updated_at(self, episode_id: str, updated_at: str) -> None:
        """Set the updated_at timestamp for an episode.
        
        Test helper for simulating stale episodes without breaking
        encapsulation by accessing private SQLite connections.
        
        Args:
            episode_id: Episode UUID.
            updated_at: ISO-8601 timestamp string.
            
        Raises:
            ValueError: If episode not found.
        """
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE episodes SET updated_at = ? WHERE id = ?",
                (updated_at, episode_id)
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                raise ValueError(f"Episode {episode_id} not found")
    
    def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode.
        
        Args:
            episode_id: Episode UUID.
        
        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM episodes WHERE id = ?",
                (episode_id,)
            )
            self._conn.commit()
            return cursor.rowcount > 0
    
    def add_memory(self, episode_id: str, memory_id: str) -> bool:
        """Add a memory to an episode.
        
        Idempotent - adding same memory twice has no effect.
        Updates episode.updated_at and increments memory_count.
        
        Args:
            episode_id: Episode UUID.
            memory_id: Memory UUID.
        
        Returns:
            True if newly added, False if already exists.
        """
        with self._lock:
            # Check if already exists
            cursor = self._conn.execute(
                """
                SELECT 1 FROM episode_memories 
                WHERE episode_id = ? AND memory_id = ?
                """,
                (episode_id, memory_id)
            )
            if cursor.fetchone():
                return False  # Already exists, no action taken
            
            # Add memory
            self._conn.execute(
                """
                INSERT INTO episode_memories (episode_id, memory_id, added_at, summarized)
                VALUES (?, ?, ?, 0)
                """,
                (episode_id, memory_id, datetime.utcnow().isoformat())
            )
            
            # Update episode memory_count and updated_at
            self._conn.execute(
                """
                UPDATE episodes 
                SET memory_count = memory_count + 1, updated_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), episode_id)
            )
            
            self._conn.commit()
        
        return True
    
    def remove_memory(self, episode_id: str, memory_id: str) -> bool:
        """Remove a memory from an episode.
        
        Args:
            episode_id: Episode UUID.
            memory_id: Memory UUID.
        
        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                DELETE FROM episode_memories 
                WHERE episode_id = ? AND memory_id = ?
                """,
                (episode_id, memory_id)
            )
            
            if cursor.rowcount > 0:
                # Update episode memory_count
                self._conn.execute(
                    """
                    UPDATE episodes 
                    SET memory_count = memory_count - 1
                    WHERE id = ?
                    """,
                    (episode_id,)
                )
                self._conn.commit()
                return True
            else:
                return False
    
    def get_episode_memories(self, episode_id: str) -> List[str]:
        """Get all memory IDs for an episode.
        
        Args:
            episode_id: Episode UUID.
        
        Returns:
            List of memory IDs.
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT memory_id FROM episode_memories 
                WHERE episode_id = ?
                ORDER BY added_at
                """,
                (episode_id,)
            )
            rows = cursor.fetchall()
        
        return [row['memory_id'] for row in rows]
    
    def get_unsummarized_memories(self, episode_id: str) -> List[str]:
        """Get memory IDs that haven't been summarized.
        
        Args:
            episode_id: Episode UUID.
        
        Returns:
            List of unsummarized memory IDs.
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT memory_id FROM episode_memories 
                WHERE episode_id = ? AND summarized = 0
                ORDER BY added_at
                """,
                (episode_id,)
            )
            rows = cursor.fetchall()
        
        return [row['memory_id'] for row in rows]
    
    def mark_memories_summarized(
        self,
        episode_id: str,
        memory_ids: List[str]
    ) -> int:
        """Mark memories as summarized.
        
        Args:
            episode_id: Episode UUID.
            memory_ids: List of memory IDs to mark.
        
        Returns:
            Number of memories actually marked.
        """
        if not memory_ids:
            return 0
        
        with self._lock:
            # Safe: placeholders count is from len(), not user input
            placeholders = ','.join('?' * len(memory_ids))
            cursor = self._conn.execute(
                f"""
                UPDATE episode_memories 
                SET summarized = 1
                WHERE episode_id = ? AND memory_id IN ({placeholders})
                """,
                [episode_id] + memory_ids
            )
            self._conn.commit()
            return cursor.rowcount
    
    def get_episode_for_memory(self, memory_id: str) -> Optional[str]:
        """Find which episode owns a memory.
        
        Args:
            memory_id: Memory UUID.
        
        Returns:
            Episode ID or None if not found.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT episode_id FROM episode_memories WHERE memory_id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()
        
        return row['episode_id'] if row else None
    
    def get_active_episodes(self, window_days: int = 14) -> List[Episode]:
        """Get episodes updated within window.
        
        Args:
            window_days: Number of days to look back.
        
        Returns:
            List of Episode objects updated within window.
        """
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT * FROM episodes 
                WHERE updated_at >= ? AND status = 'active'
                ORDER BY updated_at DESC
                """,
                (cutoff.isoformat(),)
            )
            rows = cursor.fetchall()
        
        return [self._row_to_episode(row) for row in rows]
    
    def close_stale_episodes(self, window_days: int = 14) -> List[str]:
        """Auto-close episodes that haven't been updated within window.
        
        Args:
            window_days: Number of days to consider stale.
        
        Returns:
            List of closed episode IDs.
        """
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        
        with self._lock:
            # Find stale active episodes
            cursor = self._conn.execute(
                """
                SELECT id FROM episodes 
                WHERE updated_at < ? AND status = 'active'
                """,
                (cutoff.isoformat(),)
            )
            stale_ids = [row['id'] for row in cursor.fetchall()]
            
            if stale_ids:
                # Close them
                # Safe: placeholders count is from len(), not user input
                placeholders = ','.join('?' * len(stale_ids))
                self._conn.execute(
                    f"""
                    UPDATE episodes 
                    SET status = 'closed', 
                        closed_at = ?,
                        updated_at = ?
                    WHERE id IN ({placeholders})
                    """,
                    [
                        datetime.utcnow().isoformat(),
                        datetime.utcnow().isoformat()
                    ] + stale_ids
                )
                self._conn.commit()
        
        return stale_ids
    
    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        """Convert database row to Episode object.
        
        Args:
            row: SQLite Row object.
        
        Returns:
            Episode object.
        """
        return Episode(
            id=row['id'],
            title=row['title'],
            summary=row['summary'],
            summary_memory_id=row['summary_memory_id'],
            status=row['status'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            closed_at=(
                datetime.fromisoformat(row['closed_at']) 
                if row['closed_at'] else None
            ),
            memory_count=row['memory_count'],
            metadata=json.loads(row['metadata_json'])
        )
