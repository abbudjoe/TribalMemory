"""Import/export service for data migration (Issue #7).

Provides filtered export and conflict-aware import of memory entries
using the portable bundle format defined in
``tribalmemory.portability.embedding_metadata``.

Export supports filtering by:
- Tags (any-match)
- Date range (created_at)

Import supports conflict resolution:
- SKIP (default): ignore entries whose ID already exists
- OVERWRITE: replace existing entries unconditionally
- MERGE: keep whichever entry has the newer ``updated_at``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from ..interfaces import IVectorStore, MemoryEntry
from ..portability.embedding_metadata import (
    EmbeddingMetadata,
    PortableBundle,
    ReembeddingStrategy,
    create_portable_bundle,
    import_bundle as portability_import_bundle,
)


class ConflictResolution(Enum):
    """How to handle ID collisions on import."""
    SKIP = "skip"           # Keep existing, ignore incoming
    OVERWRITE = "overwrite"  # Replace existing with incoming
    MERGE = "merge"          # Keep whichever has newer updated_at


@dataclass
class ExportFilter:
    """Filters for memory export.

    Attributes:
        tags: Include entries matching *any* of these tags.
              ``None`` means no tag filter.
        date_from: Include entries created on or after this time.
        date_to: Include entries created on or before this time.
    """
    tags: Optional[list[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


@dataclass
class ImportSummary:
    """Result summary of an import operation."""
    total: int = 0
    imported: int = 0
    skipped: int = 0
    overwritten: int = 0
    errors: int = 0
    needs_reembedding: bool = False
    error_details: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Export                                                              #
# ------------------------------------------------------------------ #


async def export_memories(
    store: IVectorStore,
    embedding_metadata: EmbeddingMetadata,
    filters: Optional[ExportFilter] = None,
    schema_version: str = "1.0",
) -> PortableBundle:
    """Export memories from a vector store as a portable bundle.

    Args:
        store: Vector store to export from.
        embedding_metadata: Metadata describing the embedding model.
        filters: Optional filters (tags, date range).
        schema_version: Bundle schema version.

    Returns:
        A ``PortableBundle`` ready for serialization.
    """
    # Fetch all non-deleted entries (tag filter handled by store)
    tag_filter = None
    if filters and filters.tags:
        tag_filter = {"tags": filters.tags}

    entries = await store.list(limit=100_000, filters=tag_filter)

    # Apply date range filter in Python (store doesn't support it)
    if filters:
        entries = _apply_date_filter(entries, filters)

    return create_portable_bundle(
        entries=entries,
        embedding_metadata=embedding_metadata,
        schema_version=schema_version,
    )


def _apply_date_filter(
    entries: list[MemoryEntry],
    filters: ExportFilter,
) -> list[MemoryEntry]:
    """Filter entries by date range."""
    result = entries

    if filters.date_from is not None:
        date_from = _ensure_tz_aware(filters.date_from)
        result = [
            e for e in result
            if _ensure_tz_aware(e.created_at) >= date_from
        ]

    if filters.date_to is not None:
        date_to = _ensure_tz_aware(filters.date_to)
        result = [
            e for e in result
            if _ensure_tz_aware(e.created_at) <= date_to
        ]

    return result


# ------------------------------------------------------------------ #
#  Import                                                              #
# ------------------------------------------------------------------ #


async def import_memories(
    bundle: PortableBundle,
    store: IVectorStore,
    target_metadata: EmbeddingMetadata,
    conflict_resolution: ConflictResolution = ConflictResolution.SKIP,
    embedding_strategy: ReembeddingStrategy = ReembeddingStrategy.AUTO,
) -> ImportSummary:
    """Import a portable bundle into a vector store.

    Args:
        bundle: The bundle to import.
        store: Target vector store.
        target_metadata: Embedding metadata of the target system.
        conflict_resolution: How to handle ID collisions.
        embedding_strategy: How to handle embedding model mismatches.

    Returns:
        ``ImportSummary`` with counts and error details.
    """
    summary = ImportSummary(total=len(bundle.entries))

    # Handle embedding compatibility via portability layer
    import_result = portability_import_bundle(
        bundle=bundle,
        target_metadata=target_metadata,
        strategy=embedding_strategy,
    )
    summary.needs_reembedding = import_result.needs_embedding

    for entry in import_result.entries:
        try:
            existing = await store.get(entry.id)

            if existing is None:
                # New entry — store directly
                result = await store.store(entry)
                if result.success:
                    summary.imported += 1
                else:
                    summary.errors += 1
                    summary.error_details.append(
                        f"{entry.id}: {result.error}"
                    )
            else:
                # Conflict — apply resolution strategy
                await _resolve_conflict(
                    entry, existing, store,
                    conflict_resolution, summary,
                )
        except Exception as exc:
            summary.errors += 1
            summary.error_details.append(
                f"{entry.id}: {exc}"
            )

    return summary


async def _resolve_conflict(
    incoming: MemoryEntry,
    existing: MemoryEntry,
    store: IVectorStore,
    resolution: ConflictResolution,
    summary: ImportSummary,
) -> None:
    """Apply conflict resolution for an ID collision."""
    if resolution == ConflictResolution.SKIP:
        summary.skipped += 1
        return

    if resolution == ConflictResolution.OVERWRITE:
        await _overwrite(incoming, store, summary)
        return

    if resolution == ConflictResolution.MERGE:
        incoming_time = _ensure_tz_aware(incoming.updated_at)
        existing_time = _ensure_tz_aware(existing.updated_at)

        if incoming_time > existing_time:
            await _overwrite(incoming, store, summary)
        else:
            summary.skipped += 1
        return


async def _overwrite(
    entry: MemoryEntry,
    store: IVectorStore,
    summary: ImportSummary,
) -> None:
    """Replace existing entry.

    For InMemoryVectorStore, delete() soft-deletes (adds to _deleted set)
    and store() adds a new dict entry. We must clear the soft-delete flag
    so get() can find the replacement. We use a new UUID-suffixed ID for
    the replacement to avoid the tombstone, keeping the original ID in
    the entry's content chain.

    Actually, the simplest approach: delete + store with same ID. The
    store.store() re-adds the key to _store which get() checks before
    _deleted. But InMemoryVectorStore.get() checks _deleted first.

    Workaround: directly manipulate the store if it's InMemoryVectorStore,
    otherwise fall back to delete+store.
    """
    from .vector_store import InMemoryVectorStore

    if isinstance(store, InMemoryVectorStore):
        # Direct overwrite in memory store
        store._store[entry.id] = entry
        store._deleted.discard(entry.id)
        summary.overwritten += 1
    else:
        await store.delete(entry.id)
        result = await store.store(entry)
        if result.success:
            summary.overwritten += 1
        else:
            summary.errors += 1
            summary.error_details.append(
                f"{entry.id}: overwrite failed: {result.error}"
            )


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _ensure_tz_aware(dt: datetime) -> datetime:
    """Make a datetime timezone-aware (UTC) if it's naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
