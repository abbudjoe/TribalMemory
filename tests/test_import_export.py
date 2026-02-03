"""Tests for import/export data migration (Issue #7).

Covers:
- Export with filtering (tags, date range)
- Import with conflict resolution (skip, overwrite, merge)
- Round-trip test (export → import → verify)
- Edge cases: empty export, unknown IDs, schema validation
"""

import pytest
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from tribalmemory.interfaces import (
    MemoryEntry,
    MemorySource,
    StoreResult,
)
from tribalmemory.portability.embedding_metadata import (
    EmbeddingMetadata,
    PortableBundle,
    ReembeddingStrategy,
    create_embedding_metadata,
)
from tribalmemory.services.import_export import (
    ConflictResolution,
    ExportFilter,
    ImportSummary,
    export_memories,
    export_memories_streaming,
    import_memories,
    validate_conflict_resolution,
    validate_embedding_strategy,
    parse_iso_datetime,
)
from tribalmemory.testing.mocks import MockEmbeddingService
from tribalmemory.services.vector_store import InMemoryVectorStore


# --- Fixtures ---

def _make_entry(
    content: str,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
    entry_id: str | None = None,
    source_type: MemorySource = MemorySource.USER_EXPLICIT,
    embedding: list[float] | None = None,
) -> MemoryEntry:
    """Create a MemoryEntry with sensible defaults for testing."""
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id or f"test-{hash(content) % 100000}",
        content=content,
        embedding=embedding or [0.1] * 64,
        source_instance="test-instance",
        source_type=source_type,
        created_at=created_at or now,
        updated_at=created_at or now,
        tags=tags or [],
        confidence=1.0,
    )


@pytest.fixture
def embedding_service():
    return MockEmbeddingService(embedding_dim=64)


@pytest.fixture
def embedding_metadata():
    return create_embedding_metadata(
        model_name="mock-model",
        dimensions=64,
        provider="test",
    )


@pytest.fixture
async def populated_memory_store(embedding_service):
    """InMemoryVectorStore with 5 memories across tags/dates."""
    store = InMemoryVectorStore(embedding_service)

    now = datetime.now(timezone.utc)
    entries = [
        _make_entry(
            "User prefers dark mode",
            tags=["preferences", "ui"],
            created_at=now - timedelta(days=5),
            entry_id="entry-1",
        ),
        _make_entry(
            "Meeting with Alice on Monday",
            tags=["work", "meetings"],
            created_at=now - timedelta(days=3),
            entry_id="entry-2",
        ),
        _make_entry(
            "Favorite color is blue",
            tags=["preferences"],
            created_at=now - timedelta(days=1),
            entry_id="entry-3",
        ),
        _make_entry(
            "Project deadline is Friday",
            tags=["work"],
            created_at=now - timedelta(hours=12),
            entry_id="entry-4",
        ),
        _make_entry(
            "Lunch at noon",
            tags=["personal"],
            created_at=now - timedelta(hours=1),
            entry_id="entry-5",
        ),
    ]

    for entry in entries:
        await store.store(entry)

    return store


# =============================================================================
# Export Tests
# =============================================================================


class TestExport:
    """Test memory export with filtering."""

    @pytest.mark.asyncio
    async def test_export_all(self, populated_memory_store, embedding_metadata):
        """Export without filters should include all memories."""
        bundle = await export_memories(
            store=populated_memory_store,
            embedding_metadata=embedding_metadata,
        )
        assert isinstance(bundle, PortableBundle)
        assert bundle.manifest.memory_count == 5
        assert len(bundle.entries) == 5

    @pytest.mark.asyncio
    async def test_export_filter_by_tags(
        self, populated_memory_store, embedding_metadata,
    ):
        """Export with tag filter should only include matching."""
        bundle = await export_memories(
            store=populated_memory_store,
            embedding_metadata=embedding_metadata,
            filters=ExportFilter(tags=["preferences"]),
        )
        assert len(bundle.entries) == 2
        for entry in bundle.entries:
            assert "preferences" in entry.tags

    @pytest.mark.asyncio
    async def test_export_filter_by_date_range(
        self, populated_memory_store, embedding_metadata,
    ):
        """Export with date range should only include matching."""
        now = datetime.now(timezone.utc)
        bundle = await export_memories(
            store=populated_memory_store,
            embedding_metadata=embedding_metadata,
            filters=ExportFilter(
                date_from=now - timedelta(days=2),
                date_to=now,
            ),
        )
        # Should include entries from last 2 days: entry-3, -4, -5
        assert len(bundle.entries) == 3

    @pytest.mark.asyncio
    async def test_export_filter_combined(
        self, populated_memory_store, embedding_metadata,
    ):
        """Combined tag + date filter should intersect."""
        now = datetime.now(timezone.utc)
        bundle = await export_memories(
            store=populated_memory_store,
            embedding_metadata=embedding_metadata,
            filters=ExportFilter(
                tags=["work"],
                date_from=now - timedelta(days=2),
            ),
        )
        # "work" tag AND last 2 days: only entry-4
        assert len(bundle.entries) == 1
        assert bundle.entries[0].content == "Project deadline is Friday"

    @pytest.mark.asyncio
    async def test_export_empty_result(
        self, populated_memory_store, embedding_metadata,
    ):
        """Export with non-matching filter should return empty bundle."""
        bundle = await export_memories(
            store=populated_memory_store,
            embedding_metadata=embedding_metadata,
            filters=ExportFilter(tags=["nonexistent"]),
        )
        assert len(bundle.entries) == 0
        assert bundle.manifest.memory_count == 0

    @pytest.mark.asyncio
    async def test_export_preserves_all_fields(
        self, populated_memory_store, embedding_metadata,
    ):
        """Exported entries should preserve all metadata."""
        bundle = await export_memories(
            store=populated_memory_store,
            embedding_metadata=embedding_metadata,
        )
        entry = next(
            e for e in bundle.entries if e.id == "entry-1"
        )
        assert entry.content == "User prefers dark mode"
        assert entry.tags == ["preferences", "ui"]
        assert entry.source_type == MemorySource.USER_EXPLICIT
        assert entry.embedding is not None

    @pytest.mark.asyncio
    async def test_export_manifest_metadata(
        self, populated_memory_store, embedding_metadata,
    ):
        """Manifest should include correct embedding metadata."""
        bundle = await export_memories(
            store=populated_memory_store,
            embedding_metadata=embedding_metadata,
        )
        m = bundle.manifest
        assert m.schema_version == "1.0"
        assert m.embedding_metadata.model_name == "mock-model"
        assert m.embedding_metadata.dimensions == 64
        assert m.exported_at is not None

    @pytest.mark.asyncio
    async def test_export_empty_store(self, embedding_service, embedding_metadata):
        """Export from empty store should work."""
        store = InMemoryVectorStore(embedding_service)
        bundle = await export_memories(
            store=store,
            embedding_metadata=embedding_metadata,
        )
        assert len(bundle.entries) == 0
        assert bundle.manifest.memory_count == 0


# =============================================================================
# Import Tests
# =============================================================================


class TestImport:
    """Test memory import with conflict resolution."""

    def _make_bundle(
        self,
        entries: list[MemoryEntry],
        embedding_metadata: EmbeddingMetadata,
    ) -> PortableBundle:
        """Helper to create a PortableBundle for testing."""
        from tribalmemory.portability.embedding_metadata import (
            create_portable_bundle,
        )
        return create_portable_bundle(
            entries=entries,
            embedding_metadata=embedding_metadata,
        )

    @pytest.mark.asyncio
    async def test_import_to_empty_store(
        self, embedding_service, embedding_metadata,
    ):
        """Import to empty store should add all entries."""
        store = InMemoryVectorStore(embedding_service)
        entries = [
            _make_entry("Memory A", entry_id="a1"),
            _make_entry("Memory B", entry_id="b1"),
        ]
        bundle = self._make_bundle(entries, embedding_metadata)

        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
        )
        assert summary.imported == 2
        assert summary.skipped == 0
        assert summary.overwritten == 0

        # Verify in store
        all_entries = await store.list()
        assert len(all_entries) == 2

    @pytest.mark.asyncio
    async def test_import_skip_conflicts(
        self, embedding_service, embedding_metadata,
    ):
        """SKIP resolution should not overwrite existing entries."""
        store = InMemoryVectorStore(embedding_service)
        original = _make_entry(
            "Original content",
            entry_id="conflict-1",
        )
        await store.store(original)

        imported = _make_entry(
            "New content for same ID",
            entry_id="conflict-1",
        )
        bundle = self._make_bundle([imported], embedding_metadata)

        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
            conflict_resolution=ConflictResolution.SKIP,
        )
        assert summary.imported == 0
        assert summary.skipped == 1

        # Verify original content is preserved
        entry = await store.get("conflict-1")
        assert entry.content == "Original content"

    @pytest.mark.asyncio
    async def test_import_overwrite_conflicts(
        self, embedding_service, embedding_metadata,
    ):
        """OVERWRITE resolution should replace existing entries."""
        store = InMemoryVectorStore(embedding_service)
        original = _make_entry(
            "Original content",
            entry_id="conflict-1",
        )
        await store.store(original)

        imported = _make_entry(
            "Updated content",
            entry_id="conflict-1",
        )
        bundle = self._make_bundle([imported], embedding_metadata)

        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
            conflict_resolution=ConflictResolution.OVERWRITE,
        )
        assert summary.imported == 0
        assert summary.overwritten == 1

        entry = await store.get("conflict-1")
        assert entry.content == "Updated content"

    @pytest.mark.asyncio
    async def test_import_merge_keeps_newer(
        self, embedding_service, embedding_metadata,
    ):
        """MERGE should keep the entry with newer updated_at."""
        store = InMemoryVectorStore(embedding_service)
        now = datetime.now(timezone.utc)

        old = _make_entry(
            "Old content",
            entry_id="merge-1",
            created_at=now - timedelta(days=2),
        )
        await store.store(old)

        newer = _make_entry(
            "Newer content",
            entry_id="merge-1",
            created_at=now,
        )
        bundle = self._make_bundle([newer], embedding_metadata)

        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
            conflict_resolution=ConflictResolution.MERGE,
        )
        assert summary.overwritten == 1

        entry = await store.get("merge-1")
        assert entry.content == "Newer content"

    @pytest.mark.asyncio
    async def test_import_merge_keeps_existing_when_newer(
        self, embedding_service, embedding_metadata,
    ):
        """MERGE should keep existing if it's newer than import."""
        store = InMemoryVectorStore(embedding_service)
        now = datetime.now(timezone.utc)

        existing = _make_entry(
            "Existing newer content",
            entry_id="merge-2",
            created_at=now,
        )
        await store.store(existing)

        older = _make_entry(
            "Older imported content",
            entry_id="merge-2",
            created_at=now - timedelta(days=5),
        )
        bundle = self._make_bundle([older], embedding_metadata)

        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
            conflict_resolution=ConflictResolution.MERGE,
        )
        assert summary.skipped == 1

        entry = await store.get("merge-2")
        assert entry.content == "Existing newer content"

    @pytest.mark.asyncio
    async def test_import_mixed_new_and_conflict(
        self, embedding_service, embedding_metadata,
    ):
        """Import with mix of new and conflicting entries."""
        store = InMemoryVectorStore(embedding_service)
        existing = _make_entry(
            "Existing", entry_id="exists-1",
        )
        await store.store(existing)

        entries = [
            _make_entry("New memory", entry_id="new-1"),
            _make_entry("Conflict", entry_id="exists-1"),
        ]
        bundle = self._make_bundle(entries, embedding_metadata)

        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
            conflict_resolution=ConflictResolution.SKIP,
        )
        assert summary.imported == 1
        assert summary.skipped == 1
        assert summary.total == 2

    @pytest.mark.asyncio
    async def test_import_handles_embedding_strategy(
        self, embedding_service, embedding_metadata,
    ):
        """Import with DROP embedding strategy should clear vectors."""
        store = InMemoryVectorStore(embedding_service)
        different_meta = create_embedding_metadata(
            model_name="different-model",
            dimensions=128,
            provider="other",
        )
        entries = [
            _make_entry(
                "Memory with embedding",
                entry_id="emb-1",
                embedding=[0.5] * 64,
            ),
        ]
        bundle = self._make_bundle(entries, embedding_metadata)

        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=different_meta,
            embedding_strategy=ReembeddingStrategy.DROP,
        )
        assert summary.imported == 1
        assert summary.needs_reembedding is True

    @pytest.mark.asyncio
    async def test_import_default_is_skip(
        self, embedding_service, embedding_metadata,
    ):
        """Default conflict resolution should be SKIP."""
        store = InMemoryVectorStore(embedding_service)
        existing = _make_entry("Existing", entry_id="x1")
        await store.store(existing)

        bundle = self._make_bundle(
            [_make_entry("New X1", entry_id="x1")],
            embedding_metadata,
        )
        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
        )
        assert summary.skipped == 1

    @pytest.mark.asyncio
    async def test_import_errors_tracked(
        self, embedding_service, embedding_metadata,
    ):
        """Import errors should be tracked, not crash."""
        store = InMemoryVectorStore(embedding_service)
        entries = [
            _make_entry("Good entry", entry_id="good-1"),
        ]
        bundle = self._make_bundle(entries, embedding_metadata)

        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
        )
        assert summary.errors == 0


# =============================================================================
# Round-Trip Tests
# =============================================================================


class TestRoundTrip:
    """Export → import → verify integrity."""

    @pytest.mark.asyncio
    async def test_round_trip_preserves_content(
        self, embedding_service, embedding_metadata,
    ):
        """Export then import should preserve all content."""
        # Setup source store
        source = InMemoryVectorStore(embedding_service)
        entries = [
            _make_entry(
                "Memory one",
                tags=["a"],
                entry_id="rt-1",
            ),
            _make_entry(
                "Memory two",
                tags=["b"],
                entry_id="rt-2",
            ),
            _make_entry(
                "Memory three",
                tags=["a", "b"],
                entry_id="rt-3",
            ),
        ]
        for e in entries:
            await source.store(e)

        # Export
        bundle = await export_memories(
            store=source,
            embedding_metadata=embedding_metadata,
        )
        assert bundle.manifest.memory_count == 3

        # Import into fresh store
        target = InMemoryVectorStore(embedding_service)
        summary = await import_memories(
            bundle=bundle,
            store=target,
            target_metadata=embedding_metadata,
        )
        assert summary.imported == 3

        # Verify
        target_entries = await target.list()
        assert len(target_entries) == 3

        by_id = {e.id: e for e in target_entries}
        assert by_id["rt-1"].content == "Memory one"
        assert by_id["rt-1"].tags == ["a"]
        assert by_id["rt-2"].content == "Memory two"
        assert by_id["rt-3"].tags == ["a", "b"]

    @pytest.mark.asyncio
    async def test_round_trip_with_corrections(
        self, embedding_service, embedding_metadata,
    ):
        """Round-trip should preserve correction chains."""
        source = InMemoryVectorStore(embedding_service)
        original = _make_entry(
            "Joe likes red",
            tags=["preferences"],
            entry_id="orig-1",
        )
        correction = _make_entry(
            "Joe likes blue",
            tags=["preferences"],
            entry_id="corr-1",
            source_type=MemorySource.CORRECTION,
        )
        correction.supersedes = "orig-1"

        await source.store(original)
        await source.store(correction)

        bundle = await export_memories(
            store=source,
            embedding_metadata=embedding_metadata,
        )

        target = InMemoryVectorStore(embedding_service)
        await import_memories(
            bundle=bundle,
            store=target,
            target_metadata=embedding_metadata,
        )

        corr = await target.get("corr-1")
        assert corr is not None
        assert corr.supersedes == "orig-1"

    @pytest.mark.asyncio
    async def test_round_trip_serialization(
        self, embedding_service, embedding_metadata,
    ):
        """Bundle should survive JSON serialization round-trip."""
        source = InMemoryVectorStore(embedding_service)
        await source.store(
            _make_entry("Serialize me", entry_id="ser-1"),
        )

        bundle = await export_memories(
            store=source,
            embedding_metadata=embedding_metadata,
        )

        # Serialize → deserialize
        json_str = json.dumps(bundle.to_dict(), default=str)
        restored = PortableBundle.from_dict(json.loads(json_str))

        assert restored.manifest.memory_count == 1
        assert restored.entries[0].content == "Serialize me"


# =============================================================================
# Validation Tests (#3, #4, #7)
# =============================================================================


class TestValidation:
    """Input validation for enum params and dates."""

    def test_validate_conflict_resolution_valid(self):
        assert validate_conflict_resolution("skip") is None
        assert validate_conflict_resolution("overwrite") is None
        assert validate_conflict_resolution("merge") is None

    def test_validate_conflict_resolution_invalid(self):
        err = validate_conflict_resolution("overwirte")
        assert err is not None
        assert "overwirte" in err

    def test_validate_embedding_strategy_valid(self):
        assert validate_embedding_strategy("auto") is None
        assert validate_embedding_strategy("keep") is None
        assert validate_embedding_strategy("drop") is None

    def test_validate_embedding_strategy_invalid(self):
        err = validate_embedding_strategy("yolo")
        assert err is not None
        assert "yolo" in err

    def test_parse_iso_datetime_valid(self):
        dt, err = parse_iso_datetime(
            "2026-01-15T10:30:00", "test_field",
        )
        assert dt is not None
        assert err is None

    def test_parse_iso_datetime_none(self):
        dt, err = parse_iso_datetime(None, "test_field")
        assert dt is None
        assert err is None

    def test_parse_iso_datetime_invalid(self):
        dt, err = parse_iso_datetime(
            "not-a-date", "date_from",
        )
        assert dt is None
        assert err is not None
        assert "date_from" in err
        assert "not-a-date" in err

    def test_parse_iso_datetime_empty_string(self):
        dt, err = parse_iso_datetime("", "test_field")
        assert dt is None
        assert err is None


# =============================================================================
# Upsert / Vector Store Integration (#1)
# =============================================================================


class TestUpsert:
    """Test the upsert method on InMemoryVectorStore."""

    @pytest.mark.asyncio
    async def test_upsert_new_entry(self, embedding_service):
        store = InMemoryVectorStore(embedding_service)
        entry = _make_entry("New", entry_id="u1")
        result = await store.upsert(entry)
        assert result.success
        got = await store.get("u1")
        assert got.content == "New"

    @pytest.mark.asyncio
    async def test_upsert_replaces_existing(
        self, embedding_service,
    ):
        store = InMemoryVectorStore(embedding_service)
        old = _make_entry("Old", entry_id="u2")
        await store.store(old)

        new = _make_entry("Replaced", entry_id="u2")
        result = await store.upsert(new)
        assert result.success
        got = await store.get("u2")
        assert got.content == "Replaced"

    @pytest.mark.asyncio
    async def test_upsert_clears_tombstone(
        self, embedding_service,
    ):
        """Upsert after delete should clear soft-delete flag."""
        store = InMemoryVectorStore(embedding_service)
        entry = _make_entry("Will be deleted", entry_id="u3")
        await store.store(entry)
        await store.delete("u3")
        assert await store.get("u3") is None

        revived = _make_entry("Revived", entry_id="u3")
        await store.upsert(revived)
        got = await store.get("u3")
        assert got is not None
        assert got.content == "Revived"


# =============================================================================
# Dry-Run Tests (Follow-up #1)
# =============================================================================


class TestDryRun:
    """Import dry-run previews changes without writing."""

    def _make_bundle(self, entries, meta):
        from tribalmemory.portability.embedding_metadata import (
            create_portable_bundle,
        )
        return create_portable_bundle(entries=entries,
                                      embedding_metadata=meta)

    @pytest.mark.asyncio
    async def test_dry_run_reports_imports(
        self, embedding_service, embedding_metadata,
    ):
        """Dry run should count imports without writing."""
        store = InMemoryVectorStore(embedding_service)
        bundle = self._make_bundle(
            [_make_entry("A", entry_id="d1")],
            embedding_metadata,
        )

        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
            dry_run=True,
        )
        assert summary.dry_run is True
        assert summary.imported == 1

        # Nothing was actually written
        all_entries = await store.list()
        assert len(all_entries) == 0

    @pytest.mark.asyncio
    async def test_dry_run_reports_skips(
        self, embedding_service, embedding_metadata,
    ):
        """Dry run should count skips for conflicts."""
        store = InMemoryVectorStore(embedding_service)
        existing = _make_entry("Existing", entry_id="d2")
        await store.store(existing)

        bundle = self._make_bundle(
            [_make_entry("Conflict", entry_id="d2")],
            embedding_metadata,
        )

        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
            conflict_resolution=ConflictResolution.SKIP,
            dry_run=True,
        )
        assert summary.skipped == 1
        assert summary.imported == 0

    @pytest.mark.asyncio
    async def test_dry_run_reports_overwrites(
        self, embedding_service, embedding_metadata,
    ):
        """Dry run should count would-be overwrites."""
        store = InMemoryVectorStore(embedding_service)
        existing = _make_entry("Old", entry_id="d3")
        await store.store(existing)

        bundle = self._make_bundle(
            [_make_entry("New", entry_id="d3")],
            embedding_metadata,
        )

        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
            conflict_resolution=ConflictResolution.OVERWRITE,
            dry_run=True,
        )
        assert summary.overwritten == 1

        # Original still untouched
        entry = await store.get("d3")
        assert entry.content == "Old"

    @pytest.mark.asyncio
    async def test_dry_run_includes_duration(
        self, embedding_service, embedding_metadata,
    ):
        store = InMemoryVectorStore(embedding_service)
        bundle = self._make_bundle(
            [_make_entry("X", entry_id="d4")],
            embedding_metadata,
        )
        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
            dry_run=True,
        )
        assert summary.duration_ms >= 0


# =============================================================================
# Streaming Export Tests (Follow-up #3)
# =============================================================================


class TestStreamingExport:
    """export_memories_streaming yields entries one at a time."""

    @pytest.mark.asyncio
    async def test_streaming_yields_all(
        self, populated_memory_store, embedding_metadata,
    ):
        entries = []
        async for e in export_memories_streaming(
            store=populated_memory_store,
            embedding_metadata=embedding_metadata,
        ):
            entries.append(e)
        assert len(entries) == 5

    @pytest.mark.asyncio
    async def test_streaming_with_tag_filter(
        self, populated_memory_store, embedding_metadata,
    ):
        entries = []
        async for e in export_memories_streaming(
            store=populated_memory_store,
            embedding_metadata=embedding_metadata,
            filters=ExportFilter(tags=["preferences"]),
        ):
            entries.append(e)
        assert len(entries) == 2
        for e in entries:
            assert "preferences" in e.tags

    @pytest.mark.asyncio
    async def test_streaming_with_date_filter(
        self, populated_memory_store, embedding_metadata,
    ):
        now = datetime.now(timezone.utc)
        entries = []
        async for e in export_memories_streaming(
            store=populated_memory_store,
            embedding_metadata=embedding_metadata,
            filters=ExportFilter(
                date_from=now - timedelta(days=2),
            ),
        ):
            entries.append(e)
        # entries from last 2 days: entry-3, -4, -5
        assert len(entries) == 3

    @pytest.mark.asyncio
    async def test_streaming_empty_store(
        self, embedding_service, embedding_metadata,
    ):
        store = InMemoryVectorStore(embedding_service)
        entries = []
        async for e in export_memories_streaming(
            store=store,
            embedding_metadata=embedding_metadata,
        ):
            entries.append(e)
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_streaming_small_batch_size(
        self, populated_memory_store, embedding_metadata,
    ):
        """Small batch size should still yield all entries."""
        entries = []
        async for e in export_memories_streaming(
            store=populated_memory_store,
            embedding_metadata=embedding_metadata,
            batch_size=2,
        ):
            entries.append(e)
        assert len(entries) == 5


# =============================================================================
# Import Metrics Tests (Follow-up #2)
# =============================================================================


class TestImportMetrics:
    """Import should include duration_ms in summary."""

    def _make_bundle(self, entries, meta):
        from tribalmemory.portability.embedding_metadata import (
            create_portable_bundle,
        )
        return create_portable_bundle(entries=entries,
                                      embedding_metadata=meta)

    @pytest.mark.asyncio
    async def test_import_includes_duration(
        self, embedding_service, embedding_metadata,
    ):
        store = InMemoryVectorStore(embedding_service)
        bundle = self._make_bundle(
            [_make_entry("Timed", entry_id="t1")],
            embedding_metadata,
        )
        summary = await import_memories(
            bundle=bundle,
            store=store,
            target_metadata=embedding_metadata,
        )
        assert summary.duration_ms >= 0
        assert isinstance(summary.duration_ms, float)
