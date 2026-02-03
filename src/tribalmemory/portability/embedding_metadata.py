"""Embedding metadata and portability utilities.

Tracks embedding model info in export bundles so that imports can detect
model mismatches and handle re-embedding when needed.

Strategy chosen for Issue #5: **Metadata + optional re-embedding**.
- Every export bundle includes an EmbeddingMetadata block documenting
  which model/dimensions produced the embeddings.
- On import, the system compares source and target metadata.
- Three strategies: KEEP (use as-is), DROP (clear for re-generation),
  AUTO (keep if compatible, drop if not).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from ..interfaces import MemoryEntry


class ReembeddingStrategy(Enum):
    """Strategy for handling embeddings on import."""
    KEEP = "keep"     # Keep original embeddings regardless of model mismatch
    DROP = "drop"     # Drop embeddings (caller must re-embed later)
    AUTO = "auto"     # Keep if compatible, drop if not


@dataclass
class EmbeddingMetadata:
    """Metadata about the embedding model used to generate vectors.

    Attributes:
        model_name: Model identifier (e.g. "text-embedding-3-small").
        dimensions: Number of dimensions in the embedding vector.
        provider: Optional provider name (e.g. "openai", "sentence-transformers").
        created_at: When this metadata was created.
    """
    model_name: str
    dimensions: int
    provider: Optional[str] = None
    created_at: Optional[str] = None

    def is_compatible_with(self, other: EmbeddingMetadata) -> bool:
        """Check if two embedding configurations are compatible.

        Compatible means same model name AND same dimensions,
        so vectors can be compared directly without re-embedding.
        """
        return (
            self.model_name == other.model_name
            and self.dimensions == other.dimensions
        )

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        d: dict = {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
        }
        if self.provider is not None:
            d["provider"] = self.provider
        if self.created_at is not None:
            d["created_at"] = self.created_at
        return d

    @classmethod
    def from_dict(cls, d: dict) -> EmbeddingMetadata:
        """Deserialize from dict."""
        return cls(
            model_name=d["model_name"],
            dimensions=d["dimensions"],
            provider=d.get("provider"),
            created_at=d.get("created_at"),
        )


@dataclass
class EmbeddingManifest:
    """Manifest included in portable bundles.

    Extends the basic manifest with embedding metadata so importers
    can determine compatibility.
    """
    schema_version: str
    embedding_metadata: EmbeddingMetadata
    memory_count: int
    exported_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        d: dict = {
            "schema_version": self.schema_version,
            "embedding": self.embedding_metadata.to_dict(),
            "memory_count": self.memory_count,
        }
        if self.exported_at:
            d["exported_at"] = self.exported_at
        return d

    @classmethod
    def from_dict(cls, d: dict) -> EmbeddingManifest:
        """Deserialize from dict."""
        return cls(
            schema_version=d["schema_version"],
            embedding_metadata=EmbeddingMetadata.from_dict(d["embedding"]),
            memory_count=d["memory_count"],
            exported_at=d.get("exported_at"),
        )


@dataclass
class PortableBundle:
    """A portable memory bundle with embedding metadata.

    Contains the manifest (with embedding info) and the memory entries.
    Designed for JSON serialization to enable cross-system portability.
    """
    manifest: EmbeddingManifest
    entries: list[MemoryEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize the entire bundle to a dict."""
        return {
            "manifest": self.manifest.to_dict(),
            "entries": [_entry_to_dict(e) for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: dict) -> PortableBundle:
        """Deserialize from dict."""
        manifest = EmbeddingManifest.from_dict(d["manifest"])
        entries = [_entry_from_dict(e) for e in d.get("entries", [])]
        return cls(manifest=manifest, entries=entries)


@dataclass
class ImportResult:
    """Result of importing a portable bundle."""
    entries: list[MemoryEntry]
    reembedded: bool
    source_metadata: EmbeddingMetadata
    target_metadata: EmbeddingMetadata
    strategy_used: ReembeddingStrategy


def create_embedding_metadata(
    model_name: str,
    dimensions: int,
    provider: Optional[str] = None,
) -> EmbeddingMetadata:
    """Create embedding metadata with current timestamp."""
    return EmbeddingMetadata(
        model_name=model_name,
        dimensions=dimensions,
        provider=provider,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def needs_reembedding(
    source: EmbeddingMetadata,
    target: EmbeddingMetadata,
) -> bool:
    """Check if embeddings need to be regenerated.

    Returns True if the source and target models are incompatible.
    """
    return not source.is_compatible_with(target)


def create_portable_bundle(
    entries: list[MemoryEntry],
    embedding_metadata: EmbeddingMetadata,
    schema_version: str = "1.0",
) -> PortableBundle:
    """Create a portable bundle from memory entries and embedding metadata."""
    manifest = EmbeddingManifest(
        schema_version=schema_version,
        embedding_metadata=embedding_metadata,
        memory_count=len(entries),
        exported_at=datetime.now(timezone.utc).isoformat(),
    )
    return PortableBundle(manifest=manifest, entries=list(entries))


def import_bundle(
    bundle: PortableBundle,
    target_metadata: EmbeddingMetadata,
    strategy: ReembeddingStrategy = ReembeddingStrategy.AUTO,
) -> ImportResult:
    """Import a portable bundle with the given re-embedding strategy.

    Args:
        bundle: The bundle to import.
        target_metadata: Embedding metadata of the target system.
        strategy: How to handle embedding model mismatches.

    Returns:
        ImportResult with entries (possibly with cleared embeddings).
    """
    source_meta = bundle.manifest.embedding_metadata
    compatible = source_meta.is_compatible_with(target_metadata)

    # Deep copy entries to avoid mutating the bundle
    imported_entries = [_copy_entry(e) for e in bundle.entries]

    should_drop = False
    if strategy == ReembeddingStrategy.DROP:
        should_drop = True
    elif strategy == ReembeddingStrategy.AUTO and not compatible:
        should_drop = True
    # KEEP and AUTO-compatible: keep embeddings as-is

    if should_drop:
        for entry in imported_entries:
            entry.embedding = None

    return ImportResult(
        entries=imported_entries,
        reembedded=False,  # We only drop; actual re-embedding is caller's job
        source_metadata=source_meta,
        target_metadata=target_metadata,
        strategy_used=strategy,
    )


# --- Serialization helpers for MemoryEntry ---

def _entry_to_dict(entry: MemoryEntry) -> dict:
    """Serialize a MemoryEntry to a dict."""
    return {
        "id": entry.id,
        "content": entry.content,
        "embedding": entry.embedding,
        "source_instance": entry.source_instance,
        "source_type": entry.source_type.value,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "tags": entry.tags,
        "context": entry.context,
        "confidence": entry.confidence,
        "supersedes": entry.supersedes,
        "related_to": entry.related_to,
    }


def _entry_from_dict(d: dict) -> MemoryEntry:
    """Deserialize a MemoryEntry from a dict."""
    from ..interfaces import MemorySource

    return MemoryEntry(
        id=d["id"],
        content=d["content"],
        embedding=d.get("embedding"),
        source_instance=d.get("source_instance", "unknown"),
        source_type=MemorySource(d.get("source_type", "unknown")),
        created_at=datetime.fromisoformat(d["created_at"]),
        updated_at=datetime.fromisoformat(d["updated_at"]),
        tags=d.get("tags", []),
        context=d.get("context"),
        confidence=d.get("confidence", 1.0),
        supersedes=d.get("supersedes"),
        related_to=d.get("related_to", []),
    )


def _copy_entry(entry: MemoryEntry) -> MemoryEntry:
    """Deep copy a MemoryEntry."""
    return MemoryEntry(
        id=entry.id,
        content=entry.content,
        embedding=list(entry.embedding) if entry.embedding else None,
        source_instance=entry.source_instance,
        source_type=entry.source_type,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        tags=list(entry.tags),
        context=entry.context,
        confidence=entry.confidence,
        supersedes=entry.supersedes,
        related_to=list(entry.related_to),
    )
