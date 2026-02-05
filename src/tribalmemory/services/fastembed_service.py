"""FastEmbed embedding service.

Local ONNX-based embedding provider using the fastembed library.
CPU-optimized, no external API calls, no API key needed.

Supported models:
    BAAI/bge-small-en-v1.5   384 dims  ~130MB  Good quality
    BAAI/bge-base-en-v1.5    768 dims  ~440MB  Better quality
    nomic-ai/nomic-embed-text-v1.5  768 dims  ~560MB  Great quality

Usage:
    service = FastEmbedService()  # defaults to bge-small-en-v1.5
    embedding = await service.embed("hello world")

    # Custom model
    service = FastEmbedService(
        model="BAAI/bge-base-en-v1.5",
        dimensions=768,
    )
"""

import math
import logging
from typing import Optional

from ..interfaces import IEmbeddingService
from ..utils import normalize_embedding

logger = logging.getLogger(__name__)

# Model name → default dimensions mapping
_MODEL_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedService(IEmbeddingService):
    """FastEmbed local embedding service.

    Uses ONNX runtime for fast, CPU-optimized embeddings.
    No external API calls, no API key required.

    The underlying ``TextEmbedding`` model is initialized lazily
    on first use to avoid slow imports at construction time.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        dimensions: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize FastEmbed service.

        Args:
            model: FastEmbed model name (e.g. "BAAI/bge-small-en-v1.5").
            dimensions: Output embedding dimensions.  When ``None``,
                inferred from the model name (see ``_MODEL_DIMENSIONS``).
            cache_dir: Directory for downloaded model files.
                Defaults to fastembed's built-in cache.
        """
        if dimensions is not None and dimensions < 1:
            raise ValueError(
                f"Dimensions must be >= 1, got {dimensions}"
            )

        self.model_name = model
        self.dimensions = (
            dimensions
            if dimensions is not None
            else _MODEL_DIMENSIONS.get(model, 384)
        )
        self._cache_dir = cache_dir
        self._model: Optional["TextEmbedding"] = None

    def _get_model(self):
        """Lazy-initialize the TextEmbedding model."""
        if self._model is None:
            from fastembed import TextEmbedding

            kwargs: dict = {"model_name": self.model_name}
            if self._cache_dir is not None:
                kwargs["cache_dir"] = self._cache_dir
            self._model = TextEmbedding(**kwargs)
            logger.info(
                "FastEmbed model loaded: %s (%d dims)",
                self.model_name,
                self.dimensions,
            )
        return self._model

    def __repr__(self) -> str:
        return (
            f"FastEmbedService(model={self.model_name!r}, "
            f"dimensions={self.dimensions})"
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Normalized embedding vector as list of floats.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Uses fastembed's native batch processing for efficiency.

        Args:
            texts: List of input texts.

        Returns:
            List of normalized embedding vectors.
        """
        if not texts:
            return []

        model = self._get_model()
        # fastembed returns a generator of numpy arrays
        raw_embeddings = list(model.embed(texts))

        return [
            normalize_embedding(emb.tolist())
            for emb in raw_embeddings
        ]

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Cosine similarity in [-1.0, 1.0].

        Raises:
            ValueError: If vectors have different dimensions.
        """
        if len(a) != len(b):
            raise ValueError(
                f"Embedding dimensions don't match: "
                f"{len(a)} vs {len(b)}"
            )

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
