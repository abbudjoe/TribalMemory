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
            model_name="text-embedding-3-small",
            dimensions=1536,
            provider="openai",
        )
        assert meta.model_name == "text-embedding-3-small"
        assert meta.dimensions == 1536
        assert meta.provider == "openai"

    def test_create_metadata_defaults(self):
        """Should have sensible defaults for optional fields."""
        meta = create_embedding_metadata(
            model_name="text-embedding-3-small",
            dimensions=1536,
        )
        assert meta.provider is None
        assert meta.created_at is not None

    def test_metadata_equality(self):
        """Two metadata objects with same model should be compatible."""
        meta1 = create_embedding_metadata("text-embedding-3-small", 1536, "openai")
        meta2 = create_embedding_metadata("text-embedding-3-small", 1536, "openai")
        assert meta1.is_compatible_with(meta2)

    def test_metadata_incompatible_model(self):
        """Different models should be incompatible."""
        meta1 = create_embedding_metadata("text-embedding-3-small", 1536, "openai")
        meta2 = create_embedding_metadata("all-MiniLM-L6-v2", 384, "sentence-transformers")
        assert not meta1.is_compatible_with(meta2)

    def test_metadata_incompatible_dimensions(self):
        """Same model but different dimensions should be incompatible."""
        meta1 = create_embedding_metadata("text-embedding-3-small", 1536, "openai")
        meta2 = create_embedding_metadata("text-embedding-3-small", 512, "openai")
        assert not meta1.is_compatible_with(meta2)

    def test_metadata_serialization(self):
        """Metadata should serialize to/from dict for JSON export."""
        meta = create_embedding_metadata("text-embedding-3-small", 1536, "openai")
        d = meta.to_dict()
        restored = EmbeddingMetadata.from_dict(d)
        assert restored.model_name == meta.model_name
        assert restored.dimensions == meta.dimensions
        assert restored.provider == meta.provider


class TestNeedsReembedding:
    """Test detection of when re-embedding is required."""

    def test_same_model_no_reembedding(self):
        """Same model and dimensions should not need re-embedding."""
        source = create_embedding_metadata("text-embedding-3-small", 1536, "openai")
        target = create_embedding_metadata("text-embedding-3-small", 1536, "openai")
        assert not needs_reembedding(source, target)

    def test_different_model_needs_reembedding(self):
        """Different model should need re-embedding."""
        source = create_embedding_metadata("text-embedding-3-small", 1536, "openai")
        target = create_embedding_metadata("all-MiniLM-L6-v2", 384, "sentence-transformers")
        assert needs_reembedding(source, target)

    def test_different_dimensions_needs_reembedding(self):
        """Different dimensions should need re-embedding."""
        source = create_embedding_metadata("text-embedding-3-small", 1536)
        target = create_embedding_metadata("text-embedding-3-small", 512)
        assert needs_reembedding(source, target)


class TestEmbeddingManifest:
    """Test the manifest that goes into exported bundles."""

    def test_manifest_includes_embedding_metadata(self):
        """Manifest should include embedding model info."""
        meta = create_embedding_metadata("text-embedding-3-small", 1536, "openai")
        manifest = EmbeddingManifest(
            schema_version="1.0",
            embedding_metadata=meta,
            memory_count=42,
        )
        assert manifest.embedding_metadata.model_name == "text-embedding-3-small"
        assert manifest.memory_count == 42

    def test_manifest_serialization(self):
        """Manifest should serialize to dict for JSON."""
        meta = create_embedding_metadata("text-embedding-3-small", 1536, "openai")
        manifest = EmbeddingManifest(
            schema_version="1.0",
            embedding_metadata=meta,
            memory_count=10,
        )
        d = manifest.to_dict()
        assert d["schema_version"] == "1.0"
        assert d["embedding"]["model_name"] == "text-embedding-3-small"
        assert d["memory_count"] == 10

    def test_manifest_deserialization(self):
        """Manifest should deserialize from dict."""
        d = {
            "schema_version": "1.0",
            "embedding": {
                "model_name": "text-embedding-3-small",
                "dimensions": 1536,
                "provider": "openai",
            },
            "memory_count": 10,
        }
        manifest = EmbeddingManifest.from_dict(d)
        assert manifest.schema_version == "1.0"
        assert manifest.embedding_metadata.model_name == "text-embedding-3-small"


class TestPortableBundle:
    """Test creating and importing portable bundles with embedding metadata."""

    def test_create_bundle_includes_metadata(self):
        """Bundle should include embedding metadata in manifest."""
        entries = [
            MemoryEntry(
                content="User likes dark mode",
                embedding=[0.1] * 1536,
                source_type=MemorySource.USER_EXPLICIT,
            ),
        ]
        meta = create_embedding_metadata("text-embedding-3-small", 1536, "openai")
        bundle = create_portable_bundle(entries, meta)

        assert bundle.manifest.embedding_metadata.model_name == "text-embedding-3-small"
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
        assert result.reembedded is False

    def test_strategy_drop_clears_embeddings(self):
        """DROP strategy should clear embeddings for later re-generation."""
        entries = [MemoryEntry(content="test", embedding=[1.0, 2.0])]
        source_meta = create_embedding_metadata("old-model", 2)
        target_meta = create_embedding_metadata("new-model", 3)
        bundle = create_portable_bundle(entries, source_meta)

        result = import_bundle(bundle, target_meta, strategy=ReembeddingStrategy.DROP)
        assert result.entries[0].embedding is None
        assert result.reembedded is False

    def test_strategy_auto_keeps_compatible(self):
        """AUTO strategy should keep embeddings when models are compatible."""
        entries = [MemoryEntry(content="test", embedding=[1.0, 2.0])]
        meta = create_embedding_metadata("same-model", 2)
        bundle = create_portable_bundle(entries, meta)

        result = import_bundle(bundle, meta, strategy=ReembeddingStrategy.AUTO)
        assert result.entries[0].embedding == [1.0, 2.0]
        assert result.reembedded is False

    def test_strategy_auto_drops_incompatible(self):
        """AUTO strategy should drop embeddings when models differ."""
        entries = [MemoryEntry(content="test", embedding=[1.0, 2.0])]
        source_meta = create_embedding_metadata("old-model", 2)
        target_meta = create_embedding_metadata("new-model", 3)
        bundle = create_portable_bundle(entries, source_meta)

        result = import_bundle(bundle, target_meta, strategy=ReembeddingStrategy.AUTO)
        assert result.entries[0].embedding is None
        assert result.reembedded is False


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
