"""Server configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EmbeddingConfig:
    """Embedding service configuration.
    
    Uses FastEmbed for local, zero-cloud embeddings.
    Model: BAAI/bge-small-en-v1.5 (384 dimensions).
    """
    provider: str = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = 384


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
class EpisodeConfig:
    """Episode detection and summarization configuration."""
    enabled: bool = False
    detector_strategy: str = "hybrid"  # "embedding", "llm", "hybrid"
    embedding_similarity_threshold: float = 0.75
    active_window_days: int = 14
    max_active_episodes: int = 20
    summarizer_model: str = "gpt-4o-mini"
    summarizer_provider: str = "openai"  # "openai", "anthropic", "ollama"
    summarizer_temperature: float = 0.3
    full_regen_interval: int = 10
    max_llm_calls_per_memory: int = 2
    monthly_cost_ceiling: float = 5.0

    def __post_init__(self):
        if self.detector_strategy not in ("embedding", "llm", "hybrid"):
            raise ValueError(
                f"Invalid detector_strategy: {self.detector_strategy}. "
                f"Valid: embedding, llm, hybrid"
            )
        if not 0.0 <= self.embedding_similarity_threshold <= 1.0:
            raise ValueError("embedding_similarity_threshold must be 0.0-1.0")
        if self.active_window_days < 1:
            raise ValueError("active_window_days must be >= 1")
        if self.max_active_episodes < 1:
            raise ValueError("max_active_episodes must be >= 1")
        # "mock" is allowed for testing only — not intended for production configs
        if self.summarizer_provider not in ("openai", "anthropic", "ollama", "mock"):
            raise ValueError(
                f"Invalid summarizer_provider: {self.summarizer_provider}. "
                f"Valid: openai, anthropic, ollama, mock (test only)"
            )
        if not 0.0 <= self.summarizer_temperature <= 2.0:
            raise ValueError("summarizer_temperature must be 0.0-2.0")
        if self.full_regen_interval < 1:
            raise ValueError("full_regen_interval must be >= 1")
        if self.max_llm_calls_per_memory < 0:
            raise ValueError("max_llm_calls_per_memory must be >= 0")
        if self.monthly_cost_ceiling < 0:
            raise ValueError("monthly_cost_ceiling must be >= 0")


@dataclass
class TribalMemoryConfig:
    """Full service configuration."""
    instance_id: str = "default"
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    episodes: EpisodeConfig = field(default_factory=EpisodeConfig)

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
        episodes_data = data.get("episodes", {})

        return cls(
            instance_id=data.get("instance_id", "default"),
            db=DatabaseConfig(**db_data) if db_data else DatabaseConfig(),
            embedding=EmbeddingConfig(**embedding_data) if embedding_data else EmbeddingConfig(),
            server=ServerConfig(**server_data) if server_data else ServerConfig(),
            search=SearchConfig(**search_data) if search_data else SearchConfig(),
            episodes=EpisodeConfig(**episodes_data) if episodes_data else EpisodeConfig(),
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

        if not self.instance_id:
            errors.append("instance_id is required")

        # Validate FastEmbed is available
        if self.embedding.provider == "fastembed":
            try:
                import fastembed  # noqa: F401
            except ImportError:
                errors.append(
                    "FastEmbed is required but not installed. "
                    "Install with: pip install tribalmemory[fastembed]"
                )

        # Validate embedding dimensions
        if self.embedding.dimensions <= 0:
            errors.append("embedding.dimensions must be positive")

        return errors
