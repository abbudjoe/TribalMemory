"""Tests for embedding portability (Issue #5).

Tests embedding metadata tracking, export format with model info,
and re-embedding support on import.
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from tribalmemory.interfaces import MemoryEntry, MemorySource
from tribalmemory.portability.embedding_metadata import (
    EmbeddingMetadata,
    EmbeddingManifest,
    PortableBundle,
    ReembeddingStrategy,
    create_embedding_metadata,
    needs_reembedding,
    create_portable_bundle,
    import_bundle,
)


class TestEmbeddingMetadata:
    """Test embedding metadata creation and validation."""

    def test_create_metadata_with_model_info(self):
        """Metadata should capture model name, version, and dimensions."""
        meta = create_embedding_metadata(
            model_name="BAAI/bge-small-en-v1.5",
            dimensions=384,
            provider="fastembed",
        )
        assert meta.model_name == "BAAI/bge-small-en-v1.5"
        assert meta.dimensions == 384
        assert meta.provider == "fastembed"

    def test_create_metadata_defaults(self):
        """Should have sensible defaults for optional fields."""
        meta = create_embedding_metadata(
            model_name="BAAI/bge-small-en-v1.5",
            dimensions=384,
        )
        assert meta.provider is None
        assert meta.created_at is not None

    def test_metadata_equality(self):
        """Two metadata objects with same model should be compatible."""
        meta1 = create_embedding_metadata("BAAI/bge-small-en-v1.5", 384, "fastembed")
        meta2 = create_embedding_metadata("BAAI/bge-small-en-v1.5", 384, "fastembed")
        assert meta1.is_compatible_with(meta2)

    def test_metadata_incompatible_model(self):
        """Different models should be incompatible."""
        meta1 = create_embedding_metadata("BAAI/bge-small-en-v1.5", 384, "fastembed")
        meta2 = create_embedding_metadata("all-MiniLM-L6-v2", 384, "sentence-transformers")
        assert not meta1.is_compatible_with(meta2)

    def test_metadata_incompatible_dimensions(self):
        """Same model but different dimensions should be incompatible."""
        meta1 = create_embedding_metadata("BAAI/bge-small-en-v1.5", 384, "fastembed")
        meta2 = create_embedding_metadata("BAAI/bge-small-en-v1.5", 512, "fastembed")
        assert not meta1.is_compatible_with(meta2)

    def test_metadata_serialization(self):
        """Metadata should serialize to/from dict for JSON export."""
        meta = create_embedding_metadata("BAAI/bge-small-en-v1.5", 384, "fastembed")
        d = meta.to_dict()
        restored = EmbeddingMetadata.from_dict(d)
        assert restored.model_name == meta.model_name
        assert restored.dimensions == meta.dimensions
        assert restored.provider == meta.provider


class TestNeedsReembedding:
    """Test detection of when re-embedding is required."""

    def test_same_model_no_reembedding(self):
        """Same model and dimensions should not need re-embedding."""
        source = create_embedding_metadata("BAAI/bge-small-en-v1.5", 384, "fastembed")
        target = create_embedding_metadata("BAAI/bge-small-en-v1.5", 384, "fastembed")
        assert not needs_reembedding(source, target)

    def test_different_model_needs_reembedding(self):
        """Different model should need re-embedding."""
        source = create_embedding_metadata("BAAI/bge-small-en-v1.5", 384, "fastembed")
        target = create_embedding_metadata("all-MiniLM-L6-v2", 384, "sentence-transformers")
        assert needs_reembedding(source, target)

    def test_different_dimensions_needs_reembedding(self):
        """Different dimensions should need re-embedding."""
        source = create_embedding_metadata("BAAI/bge-small-en-v1.5", 384)
        target = create_embedding_metadata("BAAI/bge-small-en-v1.5", 512)
        assert needs_reembedding(source, target)


class TestEmbeddingManifest:
    """Test the manifest that goes into exported bundles."""

    def test_manifest_includes_embedding_metadata(self):
        """Manifest should include embedding model info."""
        meta = create_embedding_metadata("BAAI/bge-small-en-v1.5", 384, "fastembed")
        manifest = EmbeddingManifest(
            schema_version="1.0",
            embedding_metadata=meta,
            memory_count=42,
        )
        assert manifest.embedding_metadata.model_name == "BAAI/bge-small-en-v1.5"
        assert manifest.memory_count == 42

    def test_manifest_serialization(self):
        """Manifest should serialize to dict for JSON."""
        meta = create_embedding_metadata("BAAI/bge-small-en-v1.5", 384, "fastembed")
        manifest = EmbeddingManifest(
            schema_version="1.0",
            embedding_metadata=meta,
            memory_count=10,
        )
        d = manifest.to_dict()
        assert d["schema_version"] == "1.0"
        assert d["embedding"]["model_name"] == "BAAI/bge-small-en-v1.5"
        assert d["memory_count"] == 10

    def test_manifest_deserialization(self):
        """Manifest should deserialize from dict."""
        d = {
            "schema_version": "1.0",
            "embedding": {
                "model_name": "BAAI/bge-small-en-v1.5",
                "dimensions": 384,
                "provider": "fastembed",
            },
            "memory_count": 10,
        }
        manifest = EmbeddingManifest.from_dict(d)
        assert manifest.schema_version == "1.0"
        assert manifest.embedding_metadata.model_name == "BAAI/bge-small-en-v1.5"


class TestPortableBundle:
    """Test creating and importing portable bundles with embedding metadata."""

    def test_create_bundle_includes_metadata(self):
        """Bundle should include embedding metadata in manifest."""
        entries = [
            MemoryEntry(
                content="User likes dark mode",
                embedding=[0.1] * 384,
                source_type=MemorySource.USER_EXPLICIT,
            ),
        ]
        meta = create_embedding_metadata("BAAI/bge-small-en-v1.5", 384, "fastembed")
        bundle = create_portable_bundle(entries, meta)

        assert bundle.manifest.embedding_metadata.model_name == "BAAI/bge-small-en-v1.5"
        assert len(bundle.entries) == 1

    def test_bundle_serialization_roundtrip(self):
        """Bundle should survive serialization to dict and back."""
        entries = [
            MemoryEntry(
                content="User prefers quiet",
                embedding=[0.5] * 384,
                source_type=MemorySource.USER_EXPLICIT,
            ),
        ]
        meta = create_embedding_metadata("all-MiniLM-L6-v2", 384, "sentence-transformers")
        bundle = create_portable_bundle(entries, meta)

        d = bundle.to_dict()
        restored = PortableBundle.from_dict(d)

        assert restored.manifest.embedding_metadata.model_name == "all-MiniLM-L6-v2"
        assert len(restored.entries) == 1
        assert restored.entries[0].content == "User prefers quiet"

    def test_bundle_preserves_embeddings(self):
        """Embeddings should be preserved in the bundle."""
        embedding = [0.1, 0.2, 0.3]
        entries = [
            MemoryEntry(content="test", embedding=embedding),
        ]
        meta = create_embedding_metadata("test-model", 3)
        bundle = create_portable_bundle(entries, meta)

        d = bundle.to_dict()
        restored = PortableBundle.from_dict(d)
        assert restored.entries[0].embedding == embedding


class TestReembeddingStrategy:
    """Test re-embedding strategies on import."""

    def test_strategy_keep_uses_existing_embeddings(self):
        """KEEP strategy should preserve original embeddings."""
        entries = [MemoryEntry(content="test", embedding=[1.0, 2.0])]
        source_meta = create_embedding_metadata("old-model", 2)
        target_meta = create_embedding_metadata("new-model", 3)
        bundle = create_portable_bundle(entries, source_meta)

        result = import_bundle(bundle, target_meta, strategy=ReembeddingStrategy.KEEP)
        assert result.entries[0].embedding == [1.0, 2.0]
        assert result.needs_embedding is False

    def test_strategy_drop_clears_embeddings(self):
        """DROP strategy should clear embeddings for later re-generation."""
        entries = [MemoryEntry(content="test", embedding=[1.0, 2.0])]
        source_meta = create_embedding_metadata("old-model", 2)
        target_meta = create_embedding_metadata("new-model", 3)
        bundle = create_portable_bundle(entries, source_meta)

        result = import_bundle(bundle, target_meta, strategy=ReembeddingStrategy.DROP)
        assert result.entries[0].embedding is None
        assert result.needs_embedding is True

    def test_strategy_auto_keeps_compatible(self):
        """AUTO strategy should keep embeddings when models are compatible."""
        entries = [MemoryEntry(content="test", embedding=[1.0, 2.0])]
        meta = create_embedding_metadata("same-model", 2)
        bundle = create_portable_bundle(entries, meta)

        result = import_bundle(bundle, meta, strategy=ReembeddingStrategy.AUTO)
        assert result.entries[0].embedding == [1.0, 2.0]
        assert result.needs_embedding is False

    def test_strategy_auto_drops_incompatible(self):
        """AUTO strategy should drop embeddings when models differ."""
        entries = [MemoryEntry(content="test", embedding=[1.0, 2.0])]
        source_meta = create_embedding_metadata("old-model", 2)
        target_meta = create_embedding_metadata("new-model", 3)
        bundle = create_portable_bundle(entries, source_meta)

        result = import_bundle(bundle, target_meta, strategy=ReembeddingStrategy.AUTO)
        assert result.entries[0].embedding is None
        assert result.needs_embedding is True


class TestValidation:
    """Test input validation."""

    def test_dimension_mismatch_raises(self):
        """Creating a bundle with wrong embedding dimensions should raise."""
        entries = [MemoryEntry(content="test", embedding=[1.0, 2.0, 3.0])]
        meta = create_embedding_metadata("model", 384)  # Expects 384, got 3
        with pytest.raises(ValueError, match="3 dimensions.*expected 384"):
            create_portable_bundle(entries, meta)

    def test_entries_without_embeddings_pass_validation(self):
        """Entries with no embedding should not trigger dimension check."""
        entries = [MemoryEntry(content="test", embedding=None)]
        meta = create_embedding_metadata("model", 384)
        bundle = create_portable_bundle(entries, meta)
        assert len(bundle.entries) == 1

    def test_unsupported_schema_version_raises(self):
        """Importing a bundle with unknown schema version should raise."""
        meta = create_embedding_metadata("model", 2)
        manifest = EmbeddingManifest(
            schema_version="99.0",
            embedding_metadata=meta,
            memory_count=0,
        )
        bundle = PortableBundle(manifest=manifest, entries=[])
        target = create_embedding_metadata("model", 2)
        with pytest.raises(ValueError, match="Unsupported schema version"):
            import_bundle(bundle, target)

    def test_supported_schema_version_passes(self):
        """Schema version 1.0 should import without error."""
        entries = [MemoryEntry(content="test", embedding=[1.0, 2.0])]
        meta = create_embedding_metadata("model", 2)
        bundle = create_portable_bundle(entries, meta)
        result = import_bundle(bundle, meta)
        assert len(result.entries) == 1


class TestPortableMemorySpec:
    """Test that the spec document is properly updated."""

    def test_spec_documents_strategy(self):
        """The portable-memory spec should document the embedding strategy."""
        import os
        spec_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "portable-memory.md"
        )
        assert os.path.exists(spec_path), "portable-memory.md should exist"
        content = open(spec_path).read()
        assert "embedding" in content.lower()
        # Should document the metadata approach
        assert "EmbeddingMetadata" in content or "embedding_metadata" in content
