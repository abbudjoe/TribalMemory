"""Server configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EmbeddingConfig:
    """Embedding service configuration.
    
    Supports OpenAI, Ollama, and any OpenAI-compatible embedding API.
    
    For local Ollama embeddings (zero cloud, zero cost):
        api_base: http://localhost:11434/v1
        model: nomic-embed-text
        dimensions: 768
        api_key: unused
    """
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    dimensions: int = 1536

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


@dataclass
class TribalMemoryConfig:
    """Full service configuration."""
    instance_id: str = "default"
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

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

        return cls(
            instance_id=data.get("instance_id", "default"),
            db=DatabaseConfig(**db_data) if db_data else DatabaseConfig(),
            embedding=EmbeddingConfig(**embedding_data) if embedding_data else EmbeddingConfig(),
            server=ServerConfig(**server_data) if server_data else ServerConfig(),
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

        # api_key is only required when using OpenAI (no custom api_base)
        is_local = self.embedding.api_base is not None and "openai.com" not in (
            self.embedding.api_base or ""
        )
        if not self.embedding.api_key and not is_local:
            errors.append("embedding.api_key is required (or set OPENAI_API_KEY)")

        if not self.instance_id:
            errors.append("instance_id is required")

        return errors
