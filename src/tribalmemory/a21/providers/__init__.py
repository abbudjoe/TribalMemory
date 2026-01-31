"""Provider interfaces and implementations.

Providers are swappable backends that implement standard interfaces.
Each provider type has an abstract base and concrete implementations.
"""

from .base import (
    EmbeddingProvider,
    StorageProvider,
    TimestampProvider,
    DeduplicationProvider,
)
from .openai import OpenAIEmbeddingProvider
from .lancedb import LanceDBStorageProvider
from .memory import InMemoryStorageProvider
from .timestamp import RFC3161TimestampProvider, MockTimestampProvider

__all__ = [
    # Base interfaces
    "EmbeddingProvider",
    "StorageProvider",
    "TimestampProvider",
    "DeduplicationProvider",
    # Embedding providers
    "OpenAIEmbeddingProvider",
    # Storage providers
    "LanceDBStorageProvider",
    "InMemoryStorageProvider",
    # Timestamp providers
    "RFC3161TimestampProvider",
    "MockTimestampProvider",
]
