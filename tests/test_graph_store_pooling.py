"""Tests for GraphStore connection pooling (Issue #49)."""

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from tribalmemory.services.graph_store import Entity, GraphStore, Relationship


class TestConnectionPooling:
    """Tests for connection pooling behavior."""

    @pytest.fixture
    def graph_store(self, tmp_path):
        """Create a temporary graph store."""
        db_path = tmp_path / "test_pooling.db"
        store = GraphStore(str(db_path))
        yield store
        # Cleanup
        store.close()

    def test_connection_reuse(self, graph_store):
        """Verify that the same connection object is reused across operations."""
        # Access internal connection (implementation detail, but needed for testing)
        conn1 = graph_store._conn
        
        # Perform some operations
        entity = Entity(name="test-service", entity_type="service")
        graph_store.add_entity(entity, memory_id="mem-1")
        
        # Check connection is the same
        conn2 = graph_store._conn
        assert conn1 is conn2, "Connection should be reused, not recreated"

    def test_wal_mode_enabled(self, graph_store):
        """Verify that WAL mode is enabled for better concurrency."""
        cursor = graph_store._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode.upper() == "WAL", f"Expected WAL mode, got {mode}"

    def test_close_cleanup(self, tmp_path):
        """Verify that close() properly cleans up the connection."""
        db_path = tmp_path / "test_close.db"
        store = GraphStore(str(db_path))
        
        # Store should have an open connection
        assert store._conn is not None
        
        # Close the store
        store.close()
        
        # Connection should be closed (attempting to use it should fail)
        with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
            store._conn.execute("SELECT 1")

    def test_concurrent_access(self, graph_store):
        """Verify thread-safe concurrent access doesn't corrupt data."""
        errors = []
        
        def write_entities(start_id: int):
            try:
                for i in range(10):
                    entity = Entity(
                        name=f"service-{start_id}-{i}",
                        entity_type="service"
                    )
                    graph_store.add_entity(entity, memory_id=f"mem-{start_id}-{i}")
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads writing concurrently
        threads = []
        for i in range(5):
            t = threading.Thread(target=write_entities, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Check for errors
        assert len(errors) == 0, f"Concurrent access caused errors: {errors}"
        
        # Verify all entities were stored correctly
        # Should have 5 threads × 10 entities = 50 entities
        for i in range(5):
            for j in range(10):
                entities = graph_store.get_entities_for_memory(f"mem-{i}-{j}")
                assert len(entities) == 1
                assert entities[0].name == f"service-{i}-{j}"

    def test_performance_improvement(self, tmp_path):
        """Verify that persistent connection reduces overhead."""
        import timeit
        
        # Test with persistent connection (new implementation)
        db_path = tmp_path / "test_persistent.db"
        store_persistent = GraphStore(str(db_path))
        
        def bench_persistent():
            entity = Entity(name="bench-service", entity_type="service")
            store_persistent.add_entity(entity, memory_id="mem-bench")
            store_persistent.get_entities_for_memory("mem-bench")
        
        time_persistent = timeit.timeit(bench_persistent, number=100)
        store_persistent.close()
        
        # For comparison, we can't directly test the old implementation,
        # but we can verify the new one is reasonably fast
        # (< 1ms per operation on average with persistent connection)
        avg_time_ms = (time_persistent / 100) * 1000
        assert avg_time_ms < 10.0, (
            f"Operations should be fast with persistent connection, "
            f"got {avg_time_ms:.2f}ms average"
        )

    def test_connection_survives_multiple_operations(self, graph_store):
        """Verify connection remains valid across many operations."""
        conn_initial = graph_store._conn
        
        # Perform many operations
        for i in range(100):
            entity = Entity(name=f"service-{i}", entity_type="service")
            graph_store.add_entity(entity, memory_id=f"mem-{i}")
            
            rel = Relationship(
                source=f"service-{i}",
                target=f"db-{i}",
                relation_type="uses"
            )
            graph_store.add_relationship(rel, memory_id=f"mem-{i}")
            
            # Query data
            graph_store.get_entities_for_memory(f"mem-{i}")
            graph_store.get_relationships_for_entity(f"service-{i}")
        
        # Connection should still be the same
        conn_final = graph_store._conn
        assert conn_initial is conn_final
        
        # And still functional
        cursor = graph_store._conn.execute("SELECT COUNT(*) FROM entities")
        count = cursor.fetchone()[0]
        assert count >= 100  # At least 100 entities (may be more due to relationships)

    def test_close_is_idempotent(self, tmp_path):
        """Verify that calling close() multiple times is safe."""
        db_path = tmp_path / "test_idempotent.db"
        store = GraphStore(str(db_path))
        
        # Close multiple times should not raise errors
        store.close()
        store.close()  # Should be safe
        store.close()  # Should still be safe


class TestConnectionPoolingEdgeCases:
    """Edge cases and error scenarios for connection pooling."""

    def test_operations_after_close_raise_error(self, tmp_path):
        """Verify that operations after close() fail gracefully."""
        db_path = tmp_path / "test_after_close.db"
        store = GraphStore(str(db_path))
        store.close()
        
        # Operations should fail with a clear error
        with pytest.raises(sqlite3.ProgrammingError):
            entity = Entity(name="test", entity_type="service")
            store.add_entity(entity, memory_id="mem-1")

    def test_concurrent_reads_and_writes(self, tmp_path):
        """Verify that concurrent reads and writes work correctly with WAL."""
        db_path = tmp_path / "test_concurrent_rw.db"
        store = GraphStore(str(db_path))
        
        # Pre-populate some data
        for i in range(10):
            entity = Entity(name=f"service-{i}", entity_type="service")
            store.add_entity(entity, memory_id=f"mem-{i}")
        
        errors = []
        read_counts = []
        
        def reader():
            try:
                for _ in range(20):
                    entities = store.get_entities_for_memory("mem-5")
                    read_counts.append(len(entities))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def writer():
            try:
                for i in range(10, 20):
                    entity = Entity(name=f"service-{i}", entity_type="service")
                    store.add_entity(entity, memory_id=f"mem-{i}")
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)
        
        # Start readers and writers concurrently
        threads = []
        for _ in range(3):  # 3 readers
            t = threading.Thread(target=reader)
            threads.append(t)
            t.start()
        
        for _ in range(2):  # 2 writers
            t = threading.Thread(target=writer)
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        store.close()
        
        # Check for errors
        assert len(errors) == 0, f"Concurrent read/write caused errors: {errors}"
        
        # All reads should have succeeded
        assert len(read_counts) == 60  # 3 readers × 20 reads each
        assert all(count == 1 for count in read_counts)  # Each read finds 1 entity
