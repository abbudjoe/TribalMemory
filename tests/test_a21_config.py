"""Tests for A2.1 configuration classes."""

import os
import pytest
from unittest.mock import patch

from tribalmemory.a21.config import SystemConfig
from tribalmemory.a21.config.providers import (
    EmbeddingConfig,
    StorageConfig,
    TimestampConfig,
    DeduplicationConfig,
    EmbeddingProviderType,
    StorageProviderType,
    TimestampProviderType,
)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""
    
    def test_defaults(self):
        """Test default values."""
        config = EmbeddingConfig()
        
        assert config.provider == EmbeddingProviderType.OPENAI
        assert config.model == "text-embedding-3-small"
        assert config.dimensions == 1536
        assert config.max_retries == 3
        assert config.timeout_seconds == 30.0
        assert config.batch_size == 100
    
    def test_custom_values(self):
        """Test custom values."""
        config = EmbeddingConfig(
            provider=EmbeddingProviderType.MOCK,
            model="custom-model",
            dimensions=768,
            api_key="sk-test",
        )
        
        assert config.provider == EmbeddingProviderType.MOCK
        assert config.model == "custom-model"
        assert config.dimensions == 768
        assert config.api_key == "sk-test"


class TestStorageConfig:
    """Tests for StorageConfig."""
    
    def test_defaults(self):
        """Test default values."""
        config = StorageConfig()
        
        assert config.provider == StorageProviderType.MEMORY
        assert config.path is None
        assert config.uri is None
        assert config.table_name == "memories"
        assert config.embedding_dimensions == 1536
    
    def test_lancedb_config(self):
        """Test LanceDB configuration."""
        config = StorageConfig(
            provider=StorageProviderType.LANCEDB,
            path="/tmp/lancedb",
            table_name="my_memories",
        )
        
        assert config.provider == StorageProviderType.LANCEDB
        assert config.path == "/tmp/lancedb"


class TestDeduplicationConfig:
    """Tests for DeduplicationConfig."""
    
    def test_defaults(self):
        """Test default values."""
        config = DeduplicationConfig()
        
        assert config.enabled is True
        assert config.exact_threshold == 0.98
        assert config.near_threshold == 0.90
        assert config.strategy == "embedding"
    
    def test_custom_thresholds(self):
        """Test custom thresholds."""
        config = DeduplicationConfig(
            exact_threshold=0.95,
            near_threshold=0.80,
        )
        
        assert config.exact_threshold == 0.95
        assert config.near_threshold == 0.80


class TestSystemConfig:
    """Tests for SystemConfig."""
    
    def test_defaults(self):
        """Test default values."""
        config = SystemConfig()
        
        assert config.instance_id == "default"
        assert config.debug is False
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.deduplication, DeduplicationConfig)
    
    def test_for_testing_factory(self):
        """Test for_testing() factory method."""
        config = SystemConfig.for_testing(instance_id="test-instance")
        
        assert config.instance_id == "test-instance"
        assert config.embedding.provider == EmbeddingProviderType.MOCK
        assert config.storage.provider == StorageProviderType.MEMORY
        assert config.debug is True
    
    def test_for_testing_default_instance_id(self):
        """Test for_testing() default instance ID."""
        config = SystemConfig.for_testing()
        
        assert config.instance_id == "test"


class TestSystemConfigFromEnv:
    """Tests for SystemConfig.from_env()."""
    
    def test_from_env_defaults(self):
        """Test from_env() with no env vars."""
        with patch.dict(os.environ, {}, clear=True):
            config = SystemConfig.from_env()
            
            assert config.instance_id == "default"
            assert config.debug is False
    
    def test_from_env_instance_id(self):
        """Test from_env() reads instance ID."""
        with patch.dict(os.environ, {"TRIBAL_MEMORY_INSTANCE_ID": "my-agent"}):
            config = SystemConfig.from_env()
            
            assert config.instance_id == "my-agent"
    
    def test_from_env_debug(self):
        """Test from_env() reads debug flag."""
        with patch.dict(os.environ, {"TRIBAL_MEMORY_DEBUG": "true"}):
            config = SystemConfig.from_env()
            
            assert config.debug is True
    
    def test_from_env_embedding_provider(self):
        """Test from_env() reads embedding provider."""
        with patch.dict(os.environ, {"TRIBAL_MEMORY_EMBEDDING_PROVIDER": "mock"}):
            config = SystemConfig.from_env()
            
            assert config.embedding.provider == EmbeddingProviderType.MOCK
    
    def test_from_env_storage_provider(self):
        """Test from_env() reads storage provider."""
        with patch.dict(os.environ, {
            "TRIBAL_MEMORY_STORAGE_PROVIDER": "lancedb",
            "TRIBAL_MEMORY_STORAGE_PATH": "/tmp/test",
        }):
            config = SystemConfig.from_env()
            
            assert config.storage.provider == StorageProviderType.LANCEDB
            assert config.storage.path == "/tmp/test"
    
    def test_from_env_deduplication(self):
        """Test from_env() reads deduplication config."""
        with patch.dict(os.environ, {
            "TRIBAL_MEMORY_DEDUP_ENABLED": "false",
            "TRIBAL_MEMORY_DEDUP_EXACT_THRESHOLD": "0.95",
            "TRIBAL_MEMORY_DEDUP_NEAR_THRESHOLD": "0.80",
        }):
            config = SystemConfig.from_env()
            
            assert config.deduplication.enabled is False
            assert config.deduplication.exact_threshold == 0.95
            assert config.deduplication.near_threshold == 0.80
    
    def test_from_env_api_key_fallback(self):
        """Test from_env() falls back to OPENAI_API_KEY."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fallback"}):
            config = SystemConfig.from_env()
            
            assert config.embedding.api_key == "sk-fallback"
    
    def test_from_env_custom_prefix(self):
        """Test from_env() with custom prefix."""
        with patch.dict(os.environ, {"MY_PREFIX_INSTANCE_ID": "custom-agent"}):
            config = SystemConfig.from_env(prefix="MY_PREFIX")
            
            assert config.instance_id == "custom-agent"


class TestSystemConfigValidate:
    """Tests for SystemConfig.validate()."""
    
    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = SystemConfig.for_testing()
        errors = config.validate()
        
        assert errors == []
    
    def test_validate_openai_requires_api_key(self):
        """Test validation requires API key for OpenAI."""
        config = SystemConfig(
            embedding=EmbeddingConfig(
                provider=EmbeddingProviderType.OPENAI,
                api_key=None,
            )
        )
        
        errors = config.validate()
        
        assert any("API key" in e for e in errors)
    
    def test_validate_lancedb_requires_path_or_uri(self):
        """Test validation requires path or URI for LanceDB."""
        config = SystemConfig(
            storage=StorageConfig(
                provider=StorageProviderType.LANCEDB,
                path=None,
                uri=None,
            )
        )
        
        errors = config.validate()
        
        assert any("path or uri" in e for e in errors)
    
    def test_validate_dimension_mismatch(self):
        """Test validation catches dimension mismatch."""
        config = SystemConfig(
            embedding=EmbeddingConfig(dimensions=768),
            storage=StorageConfig(embedding_dimensions=1536),
        )
        
        errors = config.validate()
        
        assert any("mismatch" in e.lower() for e in errors)
    
    def test_validate_empty_instance_id(self):
        """Test validation catches empty instance_id."""
        config = SystemConfig(instance_id="")
        
        errors = config.validate()
        
        assert any("instance_id" in e for e in errors)
    
    def test_validate_negative_timeout(self):
        """Test validation catches negative timeout."""
        config = SystemConfig(
            embedding=EmbeddingConfig(
                provider=EmbeddingProviderType.MOCK,
                timeout_seconds=-1,
            )
        )
        
        errors = config.validate()
        
        assert any("timeout" in e.lower() for e in errors)
    
    def test_validate_zero_batch_size(self):
        """Test validation catches zero batch_size."""
        config = SystemConfig(
            embedding=EmbeddingConfig(
                provider=EmbeddingProviderType.MOCK,
                batch_size=0,
            )
        )
        
        errors = config.validate()
        
        assert any("batch_size" in e for e in errors)
    
    def test_validate_zero_dimensions(self):
        """Test validation catches zero dimensions."""
        config = SystemConfig(
            embedding=EmbeddingConfig(
                provider=EmbeddingProviderType.MOCK,
                dimensions=0,
            ),
            storage=StorageConfig(embedding_dimensions=0),
        )
        
        errors = config.validate()
        
        assert any("dimensions" in e for e in errors)
    
    def test_validate_threshold_out_of_range(self):
        """Test validation catches threshold out of range."""
        config = SystemConfig(
            deduplication=DeduplicationConfig(
                enabled=True,
                exact_threshold=1.5,  # > 1.0
            )
        )
        
        errors = config.validate()
        
        assert any("exact_threshold" in e for e in errors)
    
    def test_validate_near_threshold_above_exact(self):
        """Test validation catches near > exact threshold."""
        config = SystemConfig(
            deduplication=DeduplicationConfig(
                enabled=True,
                exact_threshold=0.80,
                near_threshold=0.90,  # Higher than exact
            )
        )
        
        errors = config.validate()
        
        assert any("near_threshold" in e and "exceed" in e for e in errors)
    
    def test_validate_dedup_disabled_skips_threshold_checks(self):
        """Test validation skips threshold checks when dedup disabled."""
        config = SystemConfig(
            deduplication=DeduplicationConfig(
                enabled=False,
                exact_threshold=1.5,  # Invalid but shouldn't matter
            )
        )
        
        errors = config.validate()
        
        # Should not have threshold errors since dedup is disabled
        assert not any("threshold" in e for e in errors)


class TestConfigEnums:
    """Tests for configuration enums."""
    
    def test_embedding_provider_values(self):
        """Test EmbeddingProviderType values."""
        assert EmbeddingProviderType.OPENAI.value == "openai"
        assert EmbeddingProviderType.LOCAL.value == "local"
        assert EmbeddingProviderType.MOCK.value == "mock"
    
    def test_storage_provider_values(self):
        """Test StorageProviderType values."""
        assert StorageProviderType.LANCEDB.value == "lancedb"
        assert StorageProviderType.MEMORY.value == "memory"
    
    def test_timestamp_provider_values(self):
        """Test TimestampProviderType values."""
        assert TimestampProviderType.RFC3161.value == "rfc3161"
        assert TimestampProviderType.MOCK.value == "mock"
        assert TimestampProviderType.NONE.value == "none"
