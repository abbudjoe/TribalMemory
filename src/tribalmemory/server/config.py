"""Server configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EmbeddingConfig:
    """Embedding service configuration.
    
    Default: FastEmbed (local, zero cloud, zero cost).
    
    For OpenAI embeddings:
        provider: openai
        model: text-embedding-3-small
        dimensions: 1536
        api_key: <your-api-key>
    """
    provider: str = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    dimensions: int = 384

    def __post_init__(self):
        # Resolve from environment if not set
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_base is None:
            self.api_base = os.environ.get("TRIBAL_MEMORY_EMBEDDING_API_BASE")


@dataclass
class DatabaseConfig:
    """Database configuration."""
    provider: str = "lancedb"
    path: str = "~/.tribal-memory/lancedb"
    uri: Optional[str] = None  # For cloud

    def __post_init__(self):
        # Expand home directory
        self.path = str(Path(self.path).expanduser())


@dataclass
class ServerConfig:
    """HTTP server configuration."""
    host: str = "127.0.0.1"
    port: int = 18790
    session_retention_days: int = 30  # Days to retain session chunks


@dataclass
class SearchConfig:
    """Search configuration for hybrid BM25 + vector search."""
    hybrid_enabled: bool = True
    vector_weight: float = 0.7
    text_weight: float = 0.3
    candidate_multiplier: int = 4
    # Reranking configuration
    reranking: str = "heuristic"  # "auto" | "cross-encoder" | "heuristic" | "none"
    recency_decay_days: float = 30.0  # Half-life for recency boost
    tag_boost_weight: float = 0.1  # Weight for tag match boost
    rerank_pool_multiplier: int = 2  # How many candidates to give reranker (N * limit)
    # Entity extraction configuration
    lazy_spacy: bool = True  # Use fast regex on ingest, spaCy only on recall queries

    def __post_init__(self):
        if self.vector_weight < 0:
            raise ValueError("vector_weight must be non-negative")
        if self.text_weight < 0:
            raise ValueError("text_weight must be non-negative")
        if self.vector_weight == 0 and self.text_weight == 0:
            raise ValueError(
                "At least one of vector_weight or text_weight must be > 0"
            )
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be >= 1")
        if self.reranking not in ("auto", "cross-encoder", "heuristic", "none"):
            raise ValueError(
                f"Invalid reranking mode: {self.reranking}. "
                f"Valid options: 'auto', 'cross-encoder', 'heuristic', 'none'"
            )
        if self.recency_decay_days <= 0:
            raise ValueError("recency_decay_days must be positive")
        if self.tag_boost_weight < 0:
            raise ValueError("tag_boost_weight must be non-negative")
        if self.rerank_pool_multiplier < 1:
            raise ValueError("rerank_pool_multiplier must be >= 1")


@dataclass
class TribalMemoryConfig:
    """Full service configuration."""
    instance_id: str = "default"
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "TribalMemoryConfig":
        """Load configuration from YAML file."""
        path = Path(path).expanduser()
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "TribalMemoryConfig":
        """Create configuration from dictionary."""
        db_data = data.get("db", {})
        embedding_data = data.get("embedding", {})
        server_data = data.get("server", {})
        search_data = data.get("search", {})

        return cls(
            instance_id=data.get("instance_id", "default"),
            db=DatabaseConfig(**db_data) if db_data else DatabaseConfig(),
            embedding=EmbeddingConfig(**embedding_data) if embedding_data else EmbeddingConfig(),
            server=ServerConfig(**server_data) if server_data else ServerConfig(),
            search=SearchConfig(**search_data) if search_data else SearchConfig(),
        )

    @classmethod
    def from_env(cls) -> "TribalMemoryConfig":
        """Create configuration from environment variables."""
        config_path = os.environ.get(
            "TRIBAL_MEMORY_CONFIG",
            "~/.tribal-memory/config.yaml"
        )
        return cls.from_file(config_path)

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []

        # FastEmbed doesn't need an API key at all
        if self.embedding.provider != "fastembed":
            # api_key is only required for OpenAI (no custom api_base)
            api_base = (self.embedding.api_base or "").strip()
            is_local = (
                api_base != ""
                and "api.openai.com" not in api_base.lower()
            )
            if not self.embedding.api_key and not is_local:
                errors.append(
                    "embedding.api_key is required "
                    "(or set OPENAI_API_KEY)"
                )

        if not self.instance_id:
            errors.append("instance_id is required")

        return errors
