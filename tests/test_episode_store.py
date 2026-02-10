"""Tests for EpisodeStore - Episode memory storage layer.

Comprehensive test coverage for:
- CRUD operations (create, get, list, update, delete)
- Episode-memory associations (add, remove, get, unsummarized tracking)
- Status transitions (active → closed, stale detection)
- Active window filtering
- Edge cases (duplicate adds, remove from wrong episode, get non-existent)
- Thread safety (concurrent operations)
- Idempotent schema creation
- Foreign key constraint enforcement
"""

import pytest
import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from tribalmemory.services.episode_store import Episode, EpisodeStore


class TestEpisodeCRUD:
    """Test basic CRUD operations."""

    def test_create_episode(self, tmp_path: Path):
        """Test creating a new episode."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(
            title="Project Setup",
            metadata={"project": "tribalmemory"}
        )
        
        assert episode.id is not None
        assert episode.title == "Project Setup"
        assert episode.status == "active"
        assert episode.summary == ""
        assert episode.summary_memory_id is None
        assert episode.memory_count == 0
        assert episode.metadata == {"project": "tribalmemory"}
        assert episode.created_at is not None
        assert episode.updated_at is not None
        assert episode.closed_at is None
        
        store.close()

    def test_create_episode_minimal(self, tmp_path: Path):
        """Test creating episode with minimal parameters."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Simple Episode")
        
        assert episode.title == "Simple Episode"
        assert episode.metadata == {}
        assert episode.status == "active"
        
        store.close()

    def test_get_episode(self, tmp_path: Path):
        """Test retrieving an episode by ID."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        created = store.create_episode(title="Test Episode")
        retrieved = store.get_episode(created.id)
        
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.title == created.title
        assert retrieved.status == created.status
        
        store.close()

    def test_get_nonexistent_episode(self, tmp_path: Path):
        """Test getting episode that doesn't exist returns None."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        result = store.get_episode("nonexistent-uuid")
        
        assert result is None
        
        store.close()

    def test_list_episodes(self, tmp_path: Path):
        """Test listing all episodes."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        ep1 = store.create_episode(title="Episode 1")
        ep2 = store.create_episode(title="Episode 2")
        ep3 = store.create_episode(title="Episode 3")
        
        episodes = store.list_episodes()
        
        assert len(episodes) == 3
        episode_ids = {e.id for e in episodes}
        assert ep1.id in episode_ids
        assert ep2.id in episode_ids
        assert ep3.id in episode_ids
        
        store.close()

    def test_list_episodes_by_status(self, tmp_path: Path):
        """Test listing episodes filtered by status."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        ep1 = store.create_episode(title="Active 1")
        ep2 = store.create_episode(title="Active 2")
        ep3 = store.create_episode(title="To Close")
        store.close_episode(ep3.id)
        
        active = store.list_episodes(status="active")
        closed = store.list_episodes(status="closed")
        
        assert len(active) == 2
        assert len(closed) == 1
        assert all(e.status == "active" for e in active)
        assert all(e.status == "closed" for e in closed)
        
        store.close()

    def test_list_episodes_with_limit(self, tmp_path: Path):
        """Test listing episodes with limit."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        for i in range(10):
            store.create_episode(title=f"Episode {i}")
        
        episodes = store.list_episodes(limit=5)
        
        assert len(episodes) == 5
        
        store.close()

    def test_update_episode(self, tmp_path: Path):
        """Test updating episode fields."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Original Title")
        
        updated = store.update_episode(
            episode.id,
            title="Updated Title",
            summary="New summary",
            metadata={"key": "value"}
        )
        
        assert updated.title == "Updated Title"
        assert updated.summary == "New summary"
        assert updated.metadata == {"key": "value"}
        
        # Verify persistence
        retrieved = store.get_episode(episode.id)
        assert retrieved.title == "Updated Title"
        assert retrieved.summary == "New summary"
        
        store.close()

    def test_update_episode_partial(self, tmp_path: Path):
        """Test updating only some fields."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(
            title="Original",
            metadata={"keep": "this"}
        )
        
        updated = store.update_episode(episode.id, title="New Title")
        
        assert updated.title == "New Title"
        assert updated.metadata == {"keep": "this"}  # Unchanged
        
        store.close()

    def test_close_episode(self, tmp_path: Path):
        """Test closing an episode."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="To Close")
        
        closed = store.close_episode(episode.id)
        
        assert closed.status == "closed"
        assert closed.closed_at is not None
        
        # Verify persistence
        retrieved = store.get_episode(episode.id)
        assert retrieved.status == "closed"
        assert retrieved.closed_at is not None
        
        store.close()

    def test_delete_episode(self, tmp_path: Path):
        """Test deleting an episode."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="To Delete")
        
        result = store.delete_episode(episode.id)
        
        assert result is True
        assert store.get_episode(episode.id) is None
        
        store.close()

    def test_delete_nonexistent_episode(self, tmp_path: Path):
        """Test deleting nonexistent episode returns False."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        result = store.delete_episode("nonexistent-uuid")
        
        assert result is False
        
        store.close()


class TestEpisodeMemoryAssociations:
    """Test episode-memory associations."""

    def test_add_memory(self, tmp_path: Path):
        """Test adding a memory to an episode."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Test Episode")
        
        result = store.add_memory(episode.id, "memory-123")
        
        assert result is True
        
        # Verify memory count updated
        retrieved = store.get_episode(episode.id)
        assert retrieved.memory_count == 1
        
        store.close()

    def test_add_multiple_memories(self, tmp_path: Path):
        """Test adding multiple memories to an episode."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Test Episode")
        
        store.add_memory(episode.id, "memory-1")
        store.add_memory(episode.id, "memory-2")
        store.add_memory(episode.id, "memory-3")
        
        retrieved = store.get_episode(episode.id)
        assert retrieved.memory_count == 3
        
        memories = store.get_episode_memories(episode.id)
        assert len(memories) == 3
        assert "memory-1" in memories
        assert "memory-2" in memories
        assert "memory-3" in memories
        
        store.close()

    def test_add_duplicate_memory(self, tmp_path: Path):
        """Test adding same memory twice (should be idempotent)."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Test Episode")
        
        result1 = store.add_memory(episode.id, "memory-123")
        result2 = store.add_memory(episode.id, "memory-123")  # Duplicate
        
        assert result1 is True  # First add succeeds
        assert result2 is False  # Second add is no-op
        
        retrieved = store.get_episode(episode.id)
        assert retrieved.memory_count == 1  # No duplicates
        
        memories = store.get_episode_memories(episode.id)
        assert len(memories) == 1
        
        store.close()

    def test_remove_memory(self, tmp_path: Path):
        """Test removing a memory from an episode."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Test Episode")
        store.add_memory(episode.id, "memory-123")
        
        result = store.remove_memory(episode.id, "memory-123")
        
        assert result is True
        
        retrieved = store.get_episode(episode.id)
        assert retrieved.memory_count == 0
        
        store.close()

    def test_remove_nonexistent_memory(self, tmp_path: Path):
        """Test removing memory that doesn't exist returns False."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Test Episode")
        
        result = store.remove_memory(episode.id, "nonexistent-memory")
        
        assert result is False
        
        store.close()

    def test_remove_memory_from_wrong_episode(self, tmp_path: Path):
        """Test removing memory from wrong episode returns False."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        ep1 = store.create_episode(title="Episode 1")
        ep2 = store.create_episode(title="Episode 2")
        
        store.add_memory(ep1.id, "memory-123")
        
        result = store.remove_memory(ep2.id, "memory-123")
        
        assert result is False
        assert store.get_episode(ep1.id).memory_count == 1
        
        store.close()

    def test_get_episode_memories(self, tmp_path: Path):
        """Test getting all memories for an episode."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Test Episode")
        store.add_memory(episode.id, "memory-1")
        store.add_memory(episode.id, "memory-2")
        
        memories = store.get_episode_memories(episode.id)
        
        assert len(memories) == 2
        assert "memory-1" in memories
        assert "memory-2" in memories
        
        store.close()

    def test_get_episode_memories_empty(self, tmp_path: Path):
        """Test getting memories for episode with no memories."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Empty Episode")
        
        memories = store.get_episode_memories(episode.id)
        
        assert memories == []
        
        store.close()

    def test_get_unsummarized_memories(self, tmp_path: Path):
        """Test getting memories that haven't been summarized."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Test Episode")
        store.add_memory(episode.id, "memory-1")
        store.add_memory(episode.id, "memory-2")
        store.add_memory(episode.id, "memory-3")
        
        # Mark some as summarized
        store.mark_memories_summarized(episode.id, ["memory-1", "memory-2"])
        
        unsummarized = store.get_unsummarized_memories(episode.id)
        
        assert len(unsummarized) == 1
        assert "memory-3" in unsummarized
        
        store.close()

    def test_mark_memories_summarized(self, tmp_path: Path):
        """Test marking memories as summarized."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Test Episode")
        store.add_memory(episode.id, "memory-1")
        store.add_memory(episode.id, "memory-2")
        
        store.mark_memories_summarized(episode.id, ["memory-1"])
        
        unsummarized = store.get_unsummarized_memories(episode.id)
        
        assert len(unsummarized) == 1
        assert "memory-2" in unsummarized
        
        store.close()

    def test_get_episode_for_memory(self, tmp_path: Path):
        """Test finding which episode owns a memory."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        ep1 = store.create_episode(title="Episode 1")
        ep2 = store.create_episode(title="Episode 2")
        
        store.add_memory(ep1.id, "memory-1")
        store.add_memory(ep2.id, "memory-2")
        
        assert store.get_episode_for_memory("memory-1") == ep1.id
        assert store.get_episode_for_memory("memory-2") == ep2.id
        assert store.get_episode_for_memory("nonexistent") is None
        
        store.close()

    def test_add_memory_to_nonexistent_episode_raises(self, tmp_path: Path):
        """Test that foreign key constraints are enforced."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        with pytest.raises(sqlite3.IntegrityError):
            store.add_memory("nonexistent-episode-id", "memory-123")
        
        store.close()


class TestEpisodeStatusTransitions:
    """Test episode status transitions and lifecycle."""

    def test_close_stale_episodes(self, tmp_path: Path):
        """Test auto-closing stale episodes."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        # Create old episode
        old_ep = store.create_episode(title="Old Episode")
        
        # Manually set updated_at to 20 days ago
        conn = store._get_connection()
        old_date = (datetime.utcnow() - timedelta(days=20)).isoformat()
        with store._lock:
            conn.execute(
                "UPDATE episodes SET updated_at = ? WHERE id = ?",
                (old_date, old_ep.id)
            )
            conn.commit()
        
        # Create recent episode
        recent_ep = store.create_episode(title="Recent Episode")
        
        # Close stale episodes (default 14 day window)
        closed_ids = store.close_stale_episodes()
        
        assert old_ep.id in closed_ids
        assert recent_ep.id not in closed_ids
        
        # Verify old episode is closed
        old_retrieved = store.get_episode(old_ep.id)
        assert old_retrieved.status == "closed"
        
        # Verify recent episode is still active
        recent_retrieved = store.get_episode(recent_ep.id)
        assert recent_retrieved.status == "active"
        
        store.close()

    def test_close_stale_episodes_custom_window(self, tmp_path: Path):
        """Test auto-closing with custom window."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        ep = store.create_episode(title="Episode")
        
        # Set to 10 days old
        conn = store._get_connection()
        old_date = (datetime.utcnow() - timedelta(days=10)).isoformat()
        with store._lock:
            conn.execute(
                "UPDATE episodes SET updated_at = ? WHERE id = ?",
                (old_date, ep.id)
            )
            conn.commit()
        
        # Close with 7-day window (should close)
        closed_ids = store.close_stale_episodes(window_days=7)
        assert ep.id in closed_ids
        
        store.close()

    def test_get_active_episodes(self, tmp_path: Path):
        """Test getting episodes updated within window."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        # Create recent episode
        recent = store.create_episode(title="Recent")
        
        # Create old episode
        old = store.create_episode(title="Old")
        conn = store._get_connection()
        old_date = (datetime.utcnow() - timedelta(days=20)).isoformat()
        with store._lock:
            conn.execute(
                "UPDATE episodes SET updated_at = ? WHERE id = ?",
                (old_date, old.id)
            )
            conn.commit()
        
        active = store.get_active_episodes(window_days=14)
        
        assert len(active) == 1
        assert active[0].id == recent.id
        
        store.close()


class TestEpisodeEdgeCases:
    """Test edge cases and error handling."""

    def test_delete_episode_cascades_memories(self, tmp_path: Path):
        """Test that deleting episode removes associated memories."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Test Episode")
        store.add_memory(episode.id, "memory-1")
        store.add_memory(episode.id, "memory-2")
        
        store.delete_episode(episode.id)
        
        # Should not be able to find episode for memories
        assert store.get_episode_for_memory("memory-1") is None
        assert store.get_episode_for_memory("memory-2") is None
        
        store.close()

    def test_update_nonexistent_episode_raises(self, tmp_path: Path):
        """Test updating nonexistent episode raises exception."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        with pytest.raises(ValueError, match="Episode .* not found"):
            store.update_episode("nonexistent-id", title="New Title")
        
        store.close()

    def test_close_nonexistent_episode_raises(self, tmp_path: Path):
        """Test closing nonexistent episode raises exception."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        with pytest.raises(ValueError, match="Episode .* not found"):
            store.close_episode("nonexistent-id")
        
        store.close()

    def test_updated_at_changes_on_memory_add(self, tmp_path: Path):
        """Test that adding memory updates episode.updated_at."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Test")
        original_updated = episode.updated_at
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        store.add_memory(episode.id, "memory-1")
        
        retrieved = store.get_episode(episode.id)
        assert retrieved.updated_at > original_updated
        
        store.close()


class TestThreadSafety:
    """Test thread safety of EpisodeStore."""

    def test_concurrent_episode_creation(self, tmp_path: Path):
        """Test creating episodes concurrently."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        results: List[Episode] = []
        errors: List[Exception] = []
        
        def create_episode(index: int):
            try:
                ep = store.create_episode(title=f"Episode {index}")
                results.append(ep)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=create_episode, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 10
        assert len(set(e.id for e in results)) == 10  # All unique IDs
        
        store.close()

    def test_concurrent_memory_additions(self, tmp_path: Path):
        """Test adding memories concurrently to same episode."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        episode = store.create_episode(title="Concurrent Test")
        errors: List[Exception] = []
        
        def add_memory(index: int):
            try:
                store.add_memory(episode.id, f"memory-{index}")
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=add_memory, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        retrieved = store.get_episode(episode.id)
        assert retrieved.memory_count == 20
        
        store.close()


class TestSchemaInitialization:
    """Test schema initialization and idempotency."""

    def test_schema_idempotent(self, tmp_path: Path):
        """Test that initializing schema multiple times is safe."""
        db_path = tmp_path / "test.db"
        
        # Create first store
        store1 = EpisodeStore(str(db_path))
        ep = store1.create_episode(title="Test")
        store1.close()
        
        # Create second store (should reuse schema)
        store2 = EpisodeStore(str(db_path))
        retrieved = store2.get_episode(ep.id)
        
        assert retrieved is not None
        assert retrieved.title == "Test"
        
        store2.close()

    def test_context_manager(self, tmp_path: Path):
        """Test using EpisodeStore as context manager."""
        db_path = tmp_path / "test.db"
        
        with EpisodeStore(str(db_path)) as store:
            episode = store.create_episode(title="Context Test")
            assert episode.id is not None
        
        # Should be closed after context
        # Reopen to verify data persisted
        with EpisodeStore(str(db_path)) as store:
            retrieved = store.get_episode(episode.id)
            assert retrieved.title == "Context Test"

    def test_close_idempotent(self, tmp_path: Path):
        """Test that calling close() multiple times is safe."""
        db_path = tmp_path / "test.db"
        store = EpisodeStore(str(db_path))
        
        store.close()
        store.close()  # Should not raise
        store.close()  # Should not raise
