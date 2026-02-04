"""Tests for local embedding support (api_base, Ollama compatibility).

TDD: RED → GREEN → REFACTOR
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from tribalmemory.services.embeddings import OpenAIEmbeddingService
from tribalmemory.server.config import TribalMemoryConfig, EmbeddingConfig


class TestApiBaseSupport:
    """OpenAIEmbeddingService should support custom api_base for Ollama/local models."""

    def test_default_api_url_unchanged(self):
        """Default URL should remain OpenAI's endpoint."""
        service = OpenAIEmbeddingService(api_key="sk-test")
        assert service.api_url == "https://api.openai.com/v1/embeddings"

    def test_custom_api_base_sets_url(self):
        """Custom api_base should override the API URL."""
        service = OpenAIEmbeddingService(
            api_key="unused",
            api_base="http://localhost:11434/v1",
        )
        assert service.api_url == "http://localhost:11434/v1/embeddings"

    def test_api_base_trailing_slash_stripped(self):
        """Trailing slashes on api_base should be handled."""
        service = OpenAIEmbeddingService(
            api_key="unused",
            api_base="http://localhost:11434/v1/",
        )
        assert service.api_url == "http://localhost:11434/v1/embeddings"

    def test_api_base_with_embeddings_path_not_doubled(self):
        """If api_base already ends with /embeddings, don't append again."""
        service = OpenAIEmbeddingService(
            api_key="unused",
            api_base="http://localhost:11434/v1/embeddings",
        )
        assert service.api_url == "http://localhost:11434/v1/embeddings"

    def test_api_key_not_required_for_local(self, monkeypatch):
        """When api_base is set (local model), api_key should not be required."""
        # Clear env so we can test the no-key path
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        service = OpenAIEmbeddingService(
            api_base="http://localhost:11434/v1",
        )
        assert service.api_key == OpenAIEmbeddingService.LOCAL_API_KEY_PLACEHOLDER

    def test_custom_dimensions(self):
        """Custom dimensions should be configurable (e.g., 768 for nomic-embed-text)."""
        service = OpenAIEmbeddingService(
            api_key="unused",
            api_base="http://localhost:11434/v1",
            model="nomic-embed-text",
            dimensions=768,
        )
        assert service.dimensions == 768
        assert service.model == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_embed_uses_custom_url(self):
        """embed() should POST to the custom api_base URL."""
        service = OpenAIEmbeddingService(
            api_key="unused",
            api_base="http://localhost:11434/v1",
            dimensions=4,  # Small dimension for mock response
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]}]
        }

        with patch.object(service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await service.embed("test text")

            # Verify it called the custom URL
            mock_client.post.assert_called_once()
            call_url = mock_client.post.call_args[0][0]
            assert "localhost:11434" in call_url


class TestEdgeCases:
    """Edge cases identified in review."""

    def test_empty_api_base_treated_as_none(self):
        """Empty string api_base should behave like None (use OpenAI)."""
        service = OpenAIEmbeddingService(
            api_key="sk-test",
            api_base="",
        )
        assert "api.openai.com" in service.api_url

    def test_invalid_api_base_raises(self):
        """Non-HTTP api_base should raise ValueError."""
        with pytest.raises(ValueError, match="HTTP"):
            OpenAIEmbeddingService(
                api_key="sk-test",
                api_base="ftp://bad-protocol.local",
            )

    def test_invalid_dimensions_raises(self):
        """Dimensions outside valid range should raise."""
        with pytest.raises(ValueError, match="Dimensions"):
            OpenAIEmbeddingService(
                api_key="sk-test",
                dimensions=0,
            )
        with pytest.raises(ValueError, match="Dimensions"):
            OpenAIEmbeddingService(
                api_key="sk-test",
                dimensions=10000,
            )


class TestConfigApiBase:
    """EmbeddingConfig should support api_base field."""

    def test_embedding_config_has_api_base(self):
        """EmbeddingConfig should accept api_base parameter."""
        config = EmbeddingConfig(
            provider="openai",
            model="nomic-embed-text",
            api_key="unused",
            api_base="http://localhost:11434/v1",
            dimensions=768,
        )
        assert config.api_base == "http://localhost:11434/v1"
        assert config.dimensions == 768

    def test_config_from_yaml_with_api_base(self, tmp_path):
        """Config loaded from YAML should include api_base and dimensions."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
instance_id: local-test
embedding:
  provider: openai
  model: nomic-embed-text
  api_base: http://localhost:11434/v1
  api_key: unused
  dimensions: 768
db:
  path: /tmp/test-tribal-memory/lancedb
""")
        config = TribalMemoryConfig.from_file(str(config_file))
        assert config.embedding.api_base == "http://localhost:11434/v1"
        assert config.embedding.dimensions == 768

    def test_config_default_api_base_none(self):
        """Default api_base should be None (uses OpenAI)."""
        config = EmbeddingConfig(api_key="sk-test")
        assert config.api_base is None

    def test_config_default_dimensions(self):
        """Default dimensions should be 1536 (OpenAI text-embedding-3-small)."""
        config = EmbeddingConfig(api_key="sk-test")
        assert config.dimensions == 1536

    def test_validation_skips_api_key_for_local(self):
        """Config validation should not require api_key when api_base is set."""
        config = TribalMemoryConfig(
            embedding=EmbeddingConfig(
                api_base="http://localhost:11434/v1",
                model="nomic-embed-text",
                dimensions=768,
            ),
        )
        errors = config.validate()
        assert not any("api_key" in e for e in errors)


class TestCreateMemoryServiceLocal:
    """create_memory_service should support local embedding config."""

    def test_factory_accepts_api_base(self):
        """create_memory_service should accept api_base parameter."""
        from tribalmemory.services.memory import create_memory_service

        # Should not raise — creates service with local embedding URL
        # We can't actually connect, but it should construct fine
        service = create_memory_service(
            instance_id="local-test",
            db_path=None,  # in-memory
            api_base="http://localhost:11434/v1",
            embedding_model="nomic-embed-text",
            embedding_dimensions=768,
        )
        assert service.embedding_service.api_url == "http://localhost:11434/v1/embeddings"
        assert service.embedding_service.model == "nomic-embed-text"
        assert service.embedding_service.dimensions == 768
