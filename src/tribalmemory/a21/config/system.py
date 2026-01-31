"""System-wide configuration."""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .providers import (
    EmbeddingConfig,
    StorageConfig,
    TimestampConfig,
    DeduplicationConfig,
    EmbeddingProviderType,
    StorageProviderType,
)


@dataclass
class SystemConfig:
    """Complete system configuration.
    
    Combines all provider configurations into a single object.
    Can be loaded from environment variables or constructed programmatically.
    
    Attributes:
        instance_id: Unique identifier for this agent instance
        embedding: Embedding provider configuration
        storage: Storage provider configuration
        timestamp: Timestamp provider configuration
        deduplication: Deduplication configuration
        debug: Enable debug logging
    """
    instance_id: str = "default"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    timestamp: TimestampConfig = field(default_factory=TimestampConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    debug: bool = False
    
    @classmethod
    def from_env(cls, prefix: str = "TRIBAL_MEMORY") -> "SystemConfig":
        """Load configuration from environment variables.
        
        Environment variables:
            {prefix}_INSTANCE_ID: Instance identifier
            {prefix}_DEBUG: Enable debug mode
            
            {prefix}_EMBEDDING_PROVIDER: openai|local|mock
            {prefix}_EMBEDDING_MODEL: Model name
            {prefix}_EMBEDDING_API_KEY: API key (or OPENAI_API_KEY)
            
            {prefix}_STORAGE_PROVIDER: lancedb|memory
            {prefix}_STORAGE_PATH: Local database path
            {prefix}_STORAGE_URI: Cloud database URI
            
            {prefix}_DEDUP_ENABLED: true|false
            {prefix}_DEDUP_EXACT_THRESHOLD: Float (0-1)
            {prefix}_DEDUP_NEAR_THRESHOLD: Float (0-1)
        """
        def get(key: str, default: str = None) -> Optional[str]:
            return os.environ.get(f"{prefix}_{key}", default)
        
        def get_bool(key: str, default: bool = False) -> bool:
            val = get(key)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")
        
        def get_float(key: str, default: float) -> float:
            val = get(key)
            return float(val) if val else default
        
        # Embedding config
        embedding = EmbeddingConfig(
            provider=EmbeddingProviderType(get("EMBEDDING_PROVIDER", "openai")),
            model=get("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        )
        
        # Storage config
        storage = StorageConfig(
            provider=StorageProviderType(get("STORAGE_PROVIDER", "memory")),
            path=get("STORAGE_PATH"),
            uri=get("STORAGE_URI"),
            api_key=get("STORAGE_API_KEY") or os.environ.get("LANCEDB_API_KEY"),
        )
        
        # Deduplication config
        deduplication = DeduplicationConfig(
            enabled=get_bool("DEDUP_ENABLED", True),
            exact_threshold=get_float("DEDUP_EXACT_THRESHOLD", 0.98),
            near_threshold=get_float("DEDUP_NEAR_THRESHOLD", 0.90),
        )
        
        return cls(
            instance_id=get("INSTANCE_ID", "default"),
            embedding=embedding,
            storage=storage,
            deduplication=deduplication,
            debug=get_bool("DEBUG", False),
        )
    
    @classmethod
    def for_testing(cls, instance_id: str = "test") -> "SystemConfig":
        """Create a configuration suitable for testing.
        
        Uses mock/in-memory providers to avoid external dependencies.
        """
        return cls(
            instance_id=instance_id,
            embedding=EmbeddingConfig(
                provider=EmbeddingProviderType.MOCK,
            ),
            storage=StorageConfig(
                provider=StorageProviderType.MEMORY,
            ),
            deduplication=DeduplicationConfig(
                enabled=True,
                exact_threshold=0.90,  # Lower for deterministic mock embeddings
            ),
            debug=True,
        )
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.
        
        Checks:
        - Required API keys for providers
        - Storage path/uri requirements
        - Dimension consistency
        - Timeout and threshold bounds
        - Batch size validity
        - Instance ID format
        """
        errors = []
        
        # Instance ID validation
        if not self.instance_id or not self.instance_id.strip():
            errors.append("instance_id cannot be empty")
        
        # Embedding validation
        if self.embedding.provider == EmbeddingProviderType.OPENAI:
            if not self.embedding.api_key:
                errors.append("OpenAI embedding requires API key")
        
        # Embedding config bounds
        if self.embedding.timeout_seconds <= 0:
            errors.append(f"embedding.timeout_seconds must be positive, got {self.embedding.timeout_seconds}")
        if self.embedding.batch_size <= 0:
            errors.append(f"embedding.batch_size must be positive, got {self.embedding.batch_size}")
        if self.embedding.dimensions <= 0:
            errors.append(f"embedding.dimensions must be positive, got {self.embedding.dimensions}")
        
        # Storage validation
        if self.storage.provider == StorageProviderType.LANCEDB:
            if not self.storage.path and not self.storage.uri:
                errors.append("LanceDB storage requires path or uri")
        
        # Dimension consistency
        if self.embedding.dimensions != self.storage.embedding_dimensions:
            errors.append(
                f"Embedding dimensions mismatch: "
                f"embedding={self.embedding.dimensions}, storage={self.storage.embedding_dimensions}"
            )
        
        # Deduplication threshold validation
        if self.deduplication.enabled:
            if not (0.0 <= self.deduplication.exact_threshold <= 1.0):
                errors.append(
                    f"deduplication.exact_threshold must be between 0 and 1, "
                    f"got {self.deduplication.exact_threshold}"
                )
            if not (0.0 <= self.deduplication.near_threshold <= 1.0):
                errors.append(
                    f"deduplication.near_threshold must be between 0 and 1, "
                    f"got {self.deduplication.near_threshold}"
                )
            if self.deduplication.near_threshold > self.deduplication.exact_threshold:
                errors.append(
                    f"deduplication.near_threshold ({self.deduplication.near_threshold}) "
                    f"should not exceed exact_threshold ({self.deduplication.exact_threshold})"
                )
        
        return errors
