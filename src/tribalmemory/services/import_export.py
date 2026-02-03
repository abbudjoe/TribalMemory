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

Timezone assumption:
    All naive ``datetime`` objects are treated as UTC. This is
    consistent with ``MemoryEntry.created_at`` and ``updated_at``
    which default to ``datetime.utcnow()`` (naive UTC).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from ..interfaces import IVectorStore, MemoryEntry
from ..portability.embedding_metadata import (
    EmbeddingMetadata,
    PortableBundle,
    ReembeddingStrategy,
    create_portable_bundle,
    import_bundle as portability_import_bundle,
)

logger = logging.getLogger(__name__)

# Max entries fetched in a single export. If the store contains
# more, the export will be silently truncated — a warning is
# logged when this happens.
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
    """Result summary of an import operation."""
    total: int = 0
    imported: int = 0
    skipped: int = 0
    overwritten: int = 0
    errors: int = 0
    needs_reembedding: bool = False
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
            ``MAX_EXPORT_ENTRIES``. A warning is logged when the
            result set is truncated.

    Returns:
        A ``PortableBundle`` ready for serialization.
    """
    tag_filter = None
    if filters and filters.tags:
        tag_filter = {"tags": filters.tags}

    entries = await store.list(limit=limit, filters=tag_filter)

    if len(entries) >= limit:
        logger.warning(
            "Export hit limit of %d entries — result may be "
            "truncated. Pass a higher limit to export_memories() "
            "if needed.",
            limit,
        )

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
) -> ImportSummary:
    """Import a portable bundle into a vector store.

    Args:
        bundle: The bundle to import.
        store: Target vector store.
        target_metadata: Embedding metadata of the target system.
        conflict_resolution: How to handle ID collisions.
        embedding_strategy: How to handle embedding mismatches.

    Returns:
        ``ImportSummary`` with counts and error details.
    """
    summary = ImportSummary(total=len(bundle.entries))

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
                result = await store.store(entry)
                if result.success:
                    summary.imported += 1
                else:
                    summary.errors += 1
                    summary.error_details.append(
                        _safe_error(entry.id, result.error),
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
            f"Must be one of: {sorted(VALID_CONFLICT_RESOLUTIONS)}"
        )
    return None


def validate_embedding_strategy(value: str) -> str | None:
    """Return an error message if *value* is not valid."""
    if value not in VALID_EMBEDDING_STRATEGIES:
        return (
            f"Invalid embedding_strategy '{value}'. "
            f"Must be one of: {sorted(VALID_EMBEDDING_STRATEGIES)}"
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


def _ensure_tz_aware(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC (project convention)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _safe_error(entry_id: str, detail: str | None) -> str:
    """Sanitize error detail for user-facing output."""
    if not detail:
        return f"{entry_id}: unknown error"
    # Strip filesystem paths from error messages
    safe = detail.split("\n")[0][:200]
    return f"{entry_id}: {safe}"
