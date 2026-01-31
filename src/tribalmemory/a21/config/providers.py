"""Provider-specific configuration classes."""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum


class EmbeddingProviderType(Enum):
    """Available embedding providers."""
    OPENAI = "openai"
    LOCAL = "local"  # Future: local model support
    MOCK = "mock"


class StorageProviderType(Enum):
    """Available storage providers."""
    LANCEDB = "lancedb"
    MEMORY = "memory"
    # Future: PINECONE = "pinecone", POSTGRES = "postgres"


class TimestampProviderType(Enum):
    """Available timestamp providers."""
    RFC3161 = "rfc3161"
    MOCK = "mock"
    NONE = "none"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider.
    
    Attributes:
        provider: Which embedding provider to use
        model: Model name (e.g., "text-embedding-3-small")
        dimensions: Embedding dimension size
        api_key: API key (for cloud providers)
        api_base: Custom API base URL
        max_retries: Max retry attempts
        timeout_seconds: Request timeout
        backoff_base: Exponential backoff base
        backoff_max: Maximum backoff delay
        batch_size: Max texts per batch request
    """
    provider: EmbeddingProviderType = EmbeddingProviderType.OPENAI
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: float = 30.0
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    batch_size: int = 100


@dataclass
class StorageConfig:
    """Configuration for storage provider.
    
    Attributes:
        provider: Which storage provider to use
        path: Local database path (for LanceDB local)
        uri: Cloud database URI (for LanceDB Cloud)
        api_key: API key (for cloud storage)
        table_name: Name of the memories table
        embedding_dimensions: Expected embedding size (for validation)
    """
    provider: StorageProviderType = StorageProviderType.MEMORY
    path: Optional[str] = None
    uri: Optional[str] = None
    api_key: Optional[str] = None
    table_name: str = "memories"
    embedding_dimensions: int = 1536


@dataclass
class TimestampConfig:
    """Configuration for timestamp provider.
    
    Attributes:
        provider: Which timestamp provider to use
        tsa_url: RFC 3161 Time Stamp Authority URL
        tsa_cert_path: Path to TSA certificate for verification
    """
    provider: TimestampProviderType = TimestampProviderType.NONE
    tsa_url: Optional[str] = None
    tsa_cert_path: Optional[str] = None


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication.
    
    Attributes:
        enabled: Whether to check for duplicates
        exact_threshold: Similarity threshold for exact duplicates (reject)
        near_threshold: Similarity threshold for near-duplicates (warn)
        strategy: Deduplication strategy
    """
    enabled: bool = True
    exact_threshold: float = 0.98
    near_threshold: float = 0.90
    strategy: Literal["embedding", "hash", "hybrid"] = "embedding"
