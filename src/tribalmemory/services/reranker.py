"""Result reranking for improved retrieval quality.

Provides multiple reranking strategies:
- NoopReranker: Pass-through, no reranking
- HeuristicReranker: Fast heuristic scoring (recency, tags, length)
- CrossEncoderReranker: Model-based reranking (sentence-transformers)

Reranking happens after initial retrieval (vector + BM25) to refine ordering.
"""

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from ..interfaces import RecallResult

if TYPE_CHECKING:
    from ..server.config import SearchConfig

logger = logging.getLogger(__name__)

# Lazy import for optional dependency
CROSS_ENCODER_AVAILABLE = False
CrossEncoder = None

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    CrossEncoder = _CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    pass


class IReranker(Protocol):
    """Interface for result reranking."""

    def rerank(
        self, query: str, candidates: list[RecallResult], top_k: int
    ) -> list[RecallResult]:
        """Rerank candidates and return top_k results.

        Args:
            query: Original search query
            candidates: Initial retrieval results
            top_k: Number of results to return

        Returns:
            Reranked results (up to top_k)
        """
        ...


class NoopReranker:
    """Pass-through reranker (no reranking)."""

    def rerank(
        self, query: str, candidates: list[RecallResult], top_k: int
    ) -> list[RecallResult]:
        """Return top_k candidates unchanged."""
        return candidates[:top_k]


class HeuristicReranker:
    """Heuristic reranking with recency, tag match, and length signals.

    Combines multiple quality signals:
    - Recency: newer memories score higher (exponential decay)
    - Tag match: query terms matching tags boost score
    - Length penalty: very short or very long content penalized

    Final score: original_score * (1 + boost_sum)
    """

    def __init__(
        self,
        recency_decay_days: float = 30.0,
        tag_boost_weight: float = 0.1,
        min_length: int = 10,
        max_length: int = 2000,
    ):
        """Initialize heuristic reranker.

        Args:
            recency_decay_days: Half-life for recency boost (days)
            tag_boost_weight: Weight for tag match boost
            min_length: Content shorter than this gets penalty
            max_length: Content longer than this gets penalty
        """
        self.recency_decay_days = recency_decay_days
        self.tag_boost_weight = tag_boost_weight
        self.min_length = min_length
        self.max_length = max_length

    def rerank(
        self, query: str, candidates: list[RecallResult], top_k: int
    ) -> list[RecallResult]:
        """Rerank using heuristic scoring."""
        if not candidates:
            return []

        # Compute boost for each candidate
        scored = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        now = datetime.utcnow()

        for i, candidate in enumerate(candidates):
            boost = 0.0

            # Recency boost (exponential decay)
            # Brand new memory (age=0) gets boost of 1.0, older memories decay exponentially
            age_days = (now - candidate.memory.created_at).total_seconds() / 86400
            recency_boost = math.exp(-age_days / self.recency_decay_days)
            boost += recency_boost

            # Tag match boost (exact term matching, not substring)
            if candidate.memory.tags:
                tag_lower = set(t.lower() for t in candidate.memory.tags)
                # Count query terms that exactly match tags
                matches = sum(1 for term in query_terms if term in tag_lower)
                if matches > 0:
                    boost += self.tag_boost_weight * matches

            # Length penalty
            content_length = len(candidate.memory.content)
            if content_length < self.min_length:
                boost -= 0.05  # Small penalty for very short
            elif content_length > self.max_length:
                boost -= 0.03  # Small penalty for very long

            # Combine with original score
            final_score = candidate.similarity_score * (1.0 + boost)

            scored.append((final_score, i, candidate))

        # Sort by final score (desc), then original index (preserve order on ties)
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Build reranked results with updated scores
        reranked = []
        for final_score, _, candidate in scored[:top_k]:
            reranked.append(
                RecallResult(
                    memory=candidate.memory,
                    similarity_score=final_score,
                    retrieval_time_ms=candidate.retrieval_time_ms,
                )
            )

        return reranked


class CrossEncoderReranker:
    """Cross-encoder model-based reranking.

    Uses a sentence-transformers cross-encoder to score (query, candidate) pairs.
    Model scores relevance directly, producing better ranking than retrieval alone.

    Requires sentence-transformers package.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name

        Raises:
            ImportError: If sentence-transformers not installed
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[RecallResult], top_k: int
    ) -> list[RecallResult]:
        """Rerank using cross-encoder model."""
        if not candidates:
            logger.debug("No candidates to rerank")
            return []

        # Build (query, content) pairs
        pairs = [(query, candidate.memory.content) for candidate in candidates]

        # Score with model
        scores = self.model.predict(pairs)

        # Sort by score descending
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: -x[0])

        # Build reranked results with updated scores
        reranked = []
        for score, candidate in scored[:top_k]:
            reranked.append(
                RecallResult(
                    memory=candidate.memory,
                    similarity_score=float(score),
                    retrieval_time_ms=candidate.retrieval_time_ms,
                )
            )

        return reranked


def create_reranker(config: "SearchConfig") -> IReranker:
    """Factory function to create reranker from config.

    Args:
        config: SearchConfig with reranking settings

    Returns:
        Configured reranker instance

    Raises:
        ValueError: If reranking mode is invalid
        ImportError: If cross-encoder requested but unavailable
    """
    mode = getattr(config, "reranking", "heuristic")

    if mode == "none":
        return NoopReranker()

    elif mode == "heuristic":
        return HeuristicReranker(
            recency_decay_days=getattr(config, "recency_decay_days", 30.0),
            tag_boost_weight=getattr(config, "tag_boost_weight", 0.1),
        )

    elif mode == "cross-encoder":
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "Cross-encoder reranking requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
        return CrossEncoderReranker()

    elif mode == "auto":
        # Try cross-encoder, fall back to heuristic
        if CROSS_ENCODER_AVAILABLE:
            try:
                return CrossEncoderReranker()
            except Exception as e:
                logger.warning(f"Cross-encoder init failed: {e}. Falling back to heuristic.")
        return HeuristicReranker(
            recency_decay_days=getattr(config, "recency_decay_days", 30.0),
            tag_boost_weight=getattr(config, "tag_boost_weight", 0.1),
        )

    else:
        raise ValueError(
            f"Unknown reranking mode: {mode}. "
            f"Valid options: 'none', 'heuristic', 'cross-encoder', 'auto'"
        )
