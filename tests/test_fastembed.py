"""Tests for FastEmbed embedding service."""

import math
import pytest

from tribalmemory.interfaces import IEmbeddingService


# Check if fastembed is available
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not FASTEMBED_AVAILABLE,
    reason="fastembed not installed",
)


class TestFastEmbedServiceInit:
    """Test FastEmbedService initialization."""

    def test_implements_interface(self):
        """FastEmbedService must implement IEmbeddingService."""
        from tribalmemory.services.fastembed_service import (
            FastEmbedService,
        )
        assert issubclass(FastEmbedService, IEmbeddingService)

    def test_default_model(self):
        """Default model should be BAAI/bge-small-en-v1.5."""
        from tribalmemory.services.fastembed_service import (
            FastEmbedService,
        )
        service = FastEmbedService()
        assert service.model_name == "BAAI/bge-small-en-v1.5"
        assert service.dimensions == 384

    def test_custom_model(self):
        """Should accept custom model name."""
        from tribalmemory.services.fastembed_service import (
            FastEmbedService,
        )
        service = FastEmbedService(
            model="BAAI/bge-base-en-v1.5",
            dimensions=768,
        )
        assert service.model_name == "BAAI/bge-base-en-v1.5"
        assert service.dimensions == 768

    def test_invalid_dimensions(self):
        """Should reject invalid dimensions."""
        from tribalmemory.services.fastembed_service import (
            FastEmbedService,
        )
        with pytest.raises(ValueError, match="Dimensions must be"):
            FastEmbedService(dimensions=0)
        with pytest.raises(ValueError, match="Dimensions must be"):
            FastEmbedService(dimensions=-1)


class TestFastEmbedServiceEmbed:
    """Test embedding generation."""

    @pytest.fixture
    def service(self):
        from tribalmemory.services.fastembed_service import (
            FastEmbedService,
        )
        return FastEmbedService()

    @pytest.mark.asyncio
    async def test_embed_returns_list_of_floats(self, service):
        """embed() should return a list of floats."""
        result = await service.embed("hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_dimensions(self, service):
        """Embedding should match configured dimensions."""
        result = await service.embed("hello world")
        assert len(result) == service.dimensions

    @pytest.mark.asyncio
    async def test_embed_normalized(self, service):
        """Embeddings should be L2-normalized."""
        result = await service.embed("hello world")
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_embed_empty_string(self, service):
        """Empty string should still produce an embedding."""
        result = await service.embed("")
        assert len(result) == service.dimensions

    @pytest.mark.asyncio
    async def test_embed_deterministic(self, service):
        """Same input should produce same embedding."""
        a = await service.embed("test sentence")
        b = await service.embed("test sentence")
        assert a == b

    @pytest.mark.asyncio
    async def test_embed_different_texts_differ(self, service):
        """Different texts should produce different embeddings."""
        a = await service.embed("cats are fluffy")
        b = await service.embed("quantum physics is complex")
        assert a != b


class TestFastEmbedServiceBatch:
    """Test batch embedding generation."""

    @pytest.fixture
    def service(self):
        from tribalmemory.services.fastembed_service import (
            FastEmbedService,
        )
        return FastEmbedService()

    @pytest.mark.asyncio
    async def test_embed_batch_returns_correct_count(self, service):
        """Batch embed should return one embedding per input."""
        texts = ["hello", "world", "foo"]
        results = await service.embed_batch(texts)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_embed_batch_dimensions(self, service):
        """Each batch embedding should have correct dimensions."""
        texts = ["hello", "world"]
        results = await service.embed_batch(texts)
        for emb in results:
            assert len(emb) == service.dimensions

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self, service):
        """Empty input list should return empty output."""
        results = await service.embed_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_embed_batch_matches_single(self, service):
        """Batch result should match individual embed() calls."""
        texts = ["hello world", "goodbye world"]
        batch = await service.embed_batch(texts)
        single_0 = await service.embed(texts[0])
        single_1 = await service.embed(texts[1])
        assert batch[0] == single_0
        assert batch[1] == single_1


class TestFastEmbedServiceSimilarity:
    """Test cosine similarity calculation."""

    @pytest.fixture
    def service(self):
        from tribalmemory.services.fastembed_service import (
            FastEmbedService,
        )
        return FastEmbedService()

    def test_identical_vectors(self, service):
        """Identical vectors should have similarity 1.0."""
        v = [0.5, 0.5, 0.5]
        assert abs(service.similarity(v, v) - 1.0) < 0.001

    def test_orthogonal_vectors(self, service):
        """Orthogonal vectors should have similarity 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(service.similarity(a, b)) < 0.001

    def test_opposite_vectors(self, service):
        """Opposite vectors should have similarity -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(service.similarity(a, b) + 1.0) < 0.001

    def test_dimension_mismatch_raises(self, service):
        """Mismatched dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="don't match"):
            service.similarity([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_zero_vectors(self, service):
        """Zero vectors should return 0.0 (not crash)."""
        assert service.similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    @pytest.mark.asyncio
    async def test_semantic_similarity(self, service):
        """Semantically similar texts should have higher similarity."""
        cat = await service.embed("the cat sat on the mat")
        dog = await service.embed("the dog lay on the rug")
        quantum = await service.embed("quantum entanglement theory")

        sim_cat_dog = service.similarity(cat, dog)
        sim_cat_quantum = service.similarity(cat, quantum)

        assert sim_cat_dog > sim_cat_quantum


class TestFastEmbedServiceFactory:
    """Test factory integration."""

    def test_create_memory_service_with_fastembed(self, tmp_path):
        """create_memory_service should accept provider='fastembed'."""
        from tribalmemory.services.memory import (
            create_memory_service,
        )
        service = create_memory_service(
            instance_id="test",
            db_path=str(tmp_path / "db"),
            embedding_provider="fastembed",
        )
        from tribalmemory.services.fastembed_service import (
            FastEmbedService,
        )
        assert isinstance(service.embedding_service, FastEmbedService)

    def test_create_memory_service_fastembed_custom_model(
        self, tmp_path
    ):
        """Factory should pass model/dimensions to FastEmbedService."""
        from tribalmemory.services.memory import (
            create_memory_service,
        )
        service = create_memory_service(
            instance_id="test",
            db_path=str(tmp_path / "db"),
            embedding_provider="fastembed",
            embedding_model="BAAI/bge-base-en-v1.5",
            embedding_dimensions=768,
        )
        assert service.embedding_service.dimensions == 768

    def test_config_provider_fastembed(self):
        """Config should accept provider='fastembed'."""
        from tribalmemory.server.config import EmbeddingConfig
        cfg = EmbeddingConfig(provider="fastembed")
        assert cfg.provider == "fastembed"

    def test_config_validate_fastembed_no_api_key_needed(self):
        """FastEmbed shouldn't require an API key."""
        from tribalmemory.server.config import TribalMemoryConfig
        import os
        # Temporarily clear the env var
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            cfg = TribalMemoryConfig.from_dict({
                "embedding": {"provider": "fastembed"},
            })
            errors = cfg.validate()
            assert not errors, f"Unexpected errors: {errors}"
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old
