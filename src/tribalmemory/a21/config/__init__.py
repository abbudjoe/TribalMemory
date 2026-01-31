"""Configuration system for A2.1.

Provides strongly-typed configuration objects that can be loaded from:
- Environment variables
- YAML/JSON files
- Programmatic construction

All configuration is validated at load time.
"""

from .system import SystemConfig
from .providers import EmbeddingConfig, StorageConfig, TimestampConfig, DeduplicationConfig

__all__ = [
    "SystemConfig",
    "EmbeddingConfig",
    "StorageConfig",
    "TimestampConfig",
    "DeduplicationConfig",
]
