"""Import/export service for data migration (Issue #7).

Provides filtered export and conflict-aware import of memory
entries using the portable bundle format defined in
``tribalmemory.portability.embedding_metadata``.

Export supports filtering by:
- Tags (any-match)
- Date range (``created_at``)

Import supports conflict resolution:
- SKIP (default): ignore entries whose ID already exists
- OVERWRITE: replace existing entries unconditionally
- MERGE: keep whichever entry has the newer ``updated_at``

Import also supports **dry-run mode**: when ``dry_run=True`` the
import walks every entry and reports what *would* happen without
writing to the store. Useful for previewing changes before commit.

Timezone assumption:
    All naive ``datetime`` objects are treated as UTC. This is
    consistent with ``MemoryEntry.created_at`` and ``updated_at``
    which default to ``datetime.utcnow()`` (naive UTC).

Recommended limits:
    Export loads all matching entries into memory. For stores with
    more than ~50k entries, consider exporting in batches (e.g.
    by date range) to limit peak memory usage. The hard default
    is ``MAX_EXPORT_ENTRIES`` (100 000).

    For very large datasets, use ``export_memories_streaming()``
    which yields one entry at a time via an async generator and
    avoids loading the full result set into memory.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from ..interfaces import IVectorStore, MemoryEntry
from ..portability.embedding_metadata import (
    EmbeddingMetadata,
    EmbeddingManifest,
    PortableBundle,
    ReembeddingStrategy,
    create_portable_bundle,
    import_bundle as portability_import_bundle,
)

logger = logging.getLogger(__name__)

# Hard ceiling on a single export. A warning is logged when the
# result set reaches this limit. For larger stores, export in
# batches (e.g. by date range) or use export_memories_streaming().
MAX_EXPORT_ENTRIES = 100_000

# Valid values for user-facing enum parameters
VALID_CONFLICT_RESOLUTIONS = {"skip", "overwrite", "merge"}
VALID_EMBEDDING_STRATEGIES = {"auto", "keep", "drop"}


class ConflictResolution(Enum):
    """How to handle ID collisions on import."""
    SKIP = "skip"
    OVERWRITE = "overwrite"
    MERGE = "merge"


@dataclass
class ExportFilter:
    """Filters for memory export.

    Attributes:
        tags: Include entries matching *any* of these tags.
              ``None`` means no tag filter.
        date_from: Include entries created on or after this.
        date_to: Include entries created on or before this.
    """
    tags: Optional[list[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


@dataclass
class ImportSummary:
    """Result summary of an import operation.

    Attributes:
        dry_run: True if this was a preview — no writes occurred.
    """
    total: int = 0
    imported: int = 0
    skipped: int = 0
    overwritten: int = 0
    errors: int = 0
    needs_reembedding: bool = False
    dry_run: bool = False
    duration_ms: float = 0.0
    error_details: list[str] = field(default_factory=list)


# -------------------------------------------------------------- #
#  Export                                                          #
# -------------------------------------------------------------- #


async def export_memories(
    store: IVectorStore,
    embedding_metadata: EmbeddingMetadata,
    filters: Optional[ExportFilter] = None,
    schema_version: Literal["1.0"] = "1.0",
    limit: int = MAX_EXPORT_ENTRIES,
) -> PortableBundle:
    """Export memories from a vector store as a portable bundle.

    Args:
        store: Vector store to export from.
        embedding_metadata: Metadata describing the embedding
            model used by this store.
        filters: Optional tag / date filters.
        schema_version: Bundle schema version.
        limit: Max entries to fetch. Defaults to
            ``MAX_EXPORT_ENTRIES``. A warning is logged when
            the result set is truncated.

    Returns:
        A ``PortableBundle`` ready for serialization.

    Note:
        For stores with >50k entries, consider exporting in
        date-range batches or using
        ``export_memories_streaming()`` to limit peak memory.
    """
    t0 = time.monotonic()

    tag_filter = None
    if filters and filters.tags:
        tag_filter = {"tags": filters.tags}

    entries = await store.list(limit=limit, filters=tag_filter)

    if len(entries) >= limit:
        logger.warning(
            "Export hit limit of %d entries — result may be "
            "truncated. Pass a higher limit or export in "
            "batches.",
            limit,
        )

    if filters:
        entries = _apply_date_filter(entries, filters)

    bundle = create_portable_bundle(
        entries=entries,
        embedding_metadata=embedding_metadata,
        schema_version=schema_version,
    )

    elapsed = (time.monotonic() - t0) * 1000
    logger.info(
        "Exported %d memories in %.0fms (filters=%s)",
        len(entries),
        elapsed,
        _describe_filter(filters),
    )

    return bundle


async def export_memories_streaming(
    store: IVectorStore,
    embedding_metadata: EmbeddingMetadata,
    filters: Optional[ExportFilter] = None,
    batch_size: int = 1000,
) -> AsyncIterator[MemoryEntry]:
    """Stream-export memories one at a time.

    Fetches in batches of ``batch_size`` from the store and
    yields individual entries. This avoids loading the entire
    result set into memory, making it suitable for large stores.

    Usage::

        async for entry in export_memories_streaming(store, meta):
            write_jsonl_line(entry)

    Date/tag filtering is applied per-batch.
    """
    offset = 0
    tag_filter = None
    if filters and filters.tags:
        tag_filter = {"tags": filters.tags}

    total_yielded = 0
    while True:
        batch = await store.list(
            limit=batch_size,
            offset=offset,
            filters=tag_filter,
        )
        if not batch:
            break

        if filters:
            batch = _apply_date_filter(batch, filters)

        for entry in batch:
            yield entry
            total_yielded += 1

        offset += batch_size

    logger.info(
        "Streaming export yielded %d entries", total_yielded,
    )


# -------------------------------------------------------------- #
#  Import                                                          #
# -------------------------------------------------------------- #


async def import_memories(
    bundle: PortableBundle,
    store: IVectorStore,
    target_metadata: EmbeddingMetadata,
    conflict_resolution: ConflictResolution = (
        ConflictResolution.SKIP
    ),
    embedding_strategy: ReembeddingStrategy = (
        ReembeddingStrategy.AUTO
    ),
    dry_run: bool = False,
) -> ImportSummary:
    """Import a portable bundle into a vector store.

    Args:
        bundle: The bundle to import.
        store: Target vector store.
        target_metadata: Embedding metadata of the target system.
        conflict_resolution: How to handle ID collisions.
        embedding_strategy: How to handle embedding mismatches.
        dry_run: If True, compute the summary without writing
            anything. Useful for previewing changes.

    Returns:
        ``ImportSummary`` with counts and error details.
    """
    t0 = time.monotonic()
    summary = ImportSummary(
        total=len(bundle.entries),
        dry_run=dry_run,
    )

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
                if dry_run:
                    summary.imported += 1
                else:
                    result = await store.store(entry)
                    if result.success:
                        summary.imported += 1
                    else:
                        summary.errors += 1
                        summary.error_details.append(
                            _safe_error(entry.id, result.error),
                        )
            else:
                if dry_run:
                    _resolve_conflict_dry(
                        entry, existing,
                        conflict_resolution, summary,
                    )
                else:
                    await _resolve_conflict(
                        entry, existing, store,
                        conflict_resolution, summary,
                    )
        except Exception as exc:
            summary.errors += 1
            summary.error_details.append(
                _safe_error(entry.id, str(exc)),
            )

    summary.duration_ms = (time.monotonic() - t0) * 1000

    mode = "dry-run" if dry_run else "live"
    logger.info(
        "Import (%s): %d total, %d imported, %d skipped, "
        "%d overwritten, %d errors in %.0fms",
        mode,
        summary.total,
        summary.imported,
        summary.skipped,
        summary.overwritten,
        summary.errors,
        summary.duration_ms,
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
        await _upsert(incoming, store, summary)
        return

    if resolution == ConflictResolution.MERGE:
        incoming_t = _ensure_tz_aware(incoming.updated_at)
        existing_t = _ensure_tz_aware(existing.updated_at)
        if incoming_t > existing_t:
            await _upsert(incoming, store, summary)
        else:
            summary.skipped += 1


def _resolve_conflict_dry(
    incoming: MemoryEntry,
    existing: MemoryEntry,
    resolution: ConflictResolution,
    summary: ImportSummary,
) -> None:
    """Dry-run conflict resolution (no writes)."""
    if resolution == ConflictResolution.SKIP:
        summary.skipped += 1
        return

    if resolution == ConflictResolution.OVERWRITE:
        summary.overwritten += 1
        return

    if resolution == ConflictResolution.MERGE:
        incoming_t = _ensure_tz_aware(incoming.updated_at)
        existing_t = _ensure_tz_aware(existing.updated_at)
        if incoming_t > existing_t:
            summary.overwritten += 1
        else:
            summary.skipped += 1


async def _upsert(
    entry: MemoryEntry,
    store: IVectorStore,
    summary: ImportSummary,
) -> None:
    """Insert-or-replace via the store's public upsert API."""
    result = await store.upsert(entry)
    if result.success:
        summary.overwritten += 1
    else:
        summary.errors += 1
        summary.error_details.append(
            _safe_error(entry.id, result.error),
        )


# -------------------------------------------------------------- #
#  Validation helpers (for MCP / HTTP layers)                      #
# -------------------------------------------------------------- #


def validate_conflict_resolution(value: str) -> str | None:
    """Return an error message if *value* is not valid."""
    if value not in VALID_CONFLICT_RESOLUTIONS:
        return (
            f"Invalid conflict_resolution '{value}'. "
            f"Must be one of: "
            f"{sorted(VALID_CONFLICT_RESOLUTIONS)}"
        )
    return None


def validate_embedding_strategy(value: str) -> str | None:
    """Return an error message if *value* is not valid."""
    if value not in VALID_EMBEDDING_STRATEGIES:
        return (
            f"Invalid embedding_strategy '{value}'. "
            f"Must be one of: "
            f"{sorted(VALID_EMBEDDING_STRATEGIES)}"
        )
    return None


def parse_iso_datetime(
    value: str | None,
    field_name: str,
) -> tuple[datetime | None, str | None]:
    """Parse an ISO 8601 string.

    Returns:
        ``(datetime, None)`` on success,
        ``(None, error_msg)`` on failure.
    """
    if not value:
        return None, None
    try:
        return datetime.fromisoformat(value), None
    except (ValueError, TypeError) as exc:
        return None, (
            f"Invalid {field_name}: '{value}' "
            f"is not a valid ISO 8601 datetime ({exc})"
        )


# -------------------------------------------------------------- #
#  Internal helpers                                                #
# -------------------------------------------------------------- #


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


def _ensure_tz_aware(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC (project convention)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _safe_error(entry_id: str, detail: str | None) -> str:
    """Sanitize error detail for user-facing output."""
    if not detail:
        return f"{entry_id}: unknown error"
    safe = detail.split("\n")[0][:200]
    return f"{entry_id}: {safe}"


def _describe_filter(
    filters: Optional[ExportFilter],
) -> str:
    """Human-readable description of active filters."""
    if not filters:
        return "none"
    parts = []
    if filters.tags:
        parts.append(f"tags={filters.tags}")
    if filters.date_from:
        parts.append(f"from={filters.date_from.isoformat()}")
    if filters.date_to:
        parts.append(f"to={filters.date_to.isoformat()}")
    return ", ".join(parts) if parts else "none"
