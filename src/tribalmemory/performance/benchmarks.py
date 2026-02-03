"""Performance benchmarks for Tribal Memory.

Provides functions to measure retrieval latency, embedding throughput,
and cache effectiveness using mock services for CI-friendly execution.
"""

import random
import statistics
import time
from dataclasses import dataclass
from typing import Optional

from ..interfaces import MemoryEntry
from ..testing.mocks import MockEmbeddingService, MockMemoryService, MockVectorStore
from .corpus_generator import CorpusConfig, generate_corpus


@dataclass
class LatencyStats:
    """Latency percentile statistics in milliseconds."""
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float


@dataclass
class BenchmarkResult:
    """Result of a retrieval latency benchmark."""
    corpus_size: int
    num_queries: int
    stats: LatencyStats


@dataclass
class ThroughputResult:
    """Result of an embedding throughput benchmark."""
    total_embeddings: int
    total_time_ms: float
    embeddings_per_second: float
    batch_size: int


@dataclass
class CacheResult:
    """Result of a cache effectiveness benchmark."""
    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_hit_latency_ms: float
    avg_miss_latency_ms: float


async def benchmark_retrieval_latency(
    corpus_size: int = 1000,
    num_queries: int = 50,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark retrieval latency at a given corpus size.

    Populates a mock store with `corpus_size` entries, then
    measures latency of `num_queries` random recall operations.

    Args:
        corpus_size: Number of memories to populate.
        num_queries: Number of recall queries to measure.
        seed: Random seed for reproducibility.

    Returns:
        BenchmarkResult with p50/p95/p99 latency stats.
    """
    rng = random.Random(seed)

    # Use small embedding dimension for benchmark speed (the dimension
    # doesn't affect retrieval algorithm complexity, just constant factors)
    embedding_dim = 64
    embedding_service = MockEmbeddingService(
        embedding_dim=embedding_dim, skip_latency=True
    )
    vector_store = MockVectorStore(embedding_service)

    # Populate corpus
    corpus = generate_corpus(CorpusConfig(size=corpus_size, seed=seed))
    for entry in corpus:
        entry.embedding = await embedding_service.embed(entry.content)
        await vector_store.store(entry)

    # Generate query texts
    queries = [
        rng.choice(corpus).content[:50]  # Use prefix of random entry
        for _ in range(num_queries)
    ]

    # Pre-compute query embeddings (not part of latency measurement)
    query_embeddings = [
        await embedding_service.embed(q) for q in queries
    ]

    # Measure retrieval latencies (vector store only — the core path)
    latencies: list[float] = []
    for qe in query_embeddings:
        start = time.perf_counter()
        await vector_store.recall(qe, limit=5, min_similarity=0.1)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()
    stats = LatencyStats(
        p50=_percentile(latencies, 50),
        p95=_percentile(latencies, 95),
        p99=_percentile(latencies, 99),
        mean=statistics.mean(latencies),
        min=min(latencies),
        max=max(latencies),
    )

    return BenchmarkResult(
        corpus_size=corpus_size,
        num_queries=num_queries,
        stats=stats,
    )


async def benchmark_batch_embedding_throughput(
    num_texts: int = 500,
    batch_size: int = 50,
    seed: int = 42,
) -> ThroughputResult:
    """Benchmark embedding generation throughput.

    Measures how many embeddings per second the service can produce,
    comparing single vs batch modes.

    Args:
        num_texts: Total number of texts to embed.
        batch_size: Size of each batch (1 for single mode).
        seed: Random seed for reproducibility.

    Returns:
        ThroughputResult with throughput metrics.
    """
    rng = random.Random(seed)
    service = MockEmbeddingService(embedding_dim=64, skip_latency=True)

    # Generate texts
    corpus = generate_corpus(CorpusConfig(size=num_texts, seed=seed))
    texts = [entry.content for entry in corpus]

    # Embed in batches
    start = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        await service.embed_batch(batch)
    total_ms = (time.perf_counter() - start) * 1000

    return ThroughputResult(
        total_embeddings=num_texts,
        total_time_ms=total_ms,
        embeddings_per_second=num_texts / (total_ms / 1000) if total_ms > 0 else 0,
        batch_size=batch_size,
    )


async def benchmark_cache_effectiveness(
    corpus_size: int = 500,
    num_queries: int = 100,
    repeat_ratio: float = 0.5,
    seed: int = 42,
) -> CacheResult:
    """Benchmark query cache effectiveness.

    Simulates a realistic workload with a mix of repeated and unique
    queries, measuring cache hit rates and latency impact.

    Args:
        corpus_size: Number of memories in the store.
        num_queries: Total number of queries to run.
        repeat_ratio: Fraction of queries that are repeats (0.0-1.0).
        seed: Random seed for reproducibility.

    Returns:
        CacheResult with hit rate and latency metrics.
    """
    rng = random.Random(seed)

    # Use small embedding dimension for benchmark speed
    embedding_dim = 64
    embedding_service = MockEmbeddingService(
        embedding_dim=embedding_dim, skip_latency=True
    )
    vector_store = MockVectorStore(embedding_service)

    # Populate corpus
    corpus = generate_corpus(CorpusConfig(size=corpus_size, seed=seed))
    for entry in corpus:
        entry.embedding = await embedding_service.embed(entry.content)
        await vector_store.store(entry)

    # Generate query mix: some unique, some repeated
    unique_queries = [
        rng.choice(corpus).content[:50]
        for _ in range(max(10, int(num_queries * (1 - repeat_ratio))))
    ]
    queries: list[str] = []
    seen_queries: set[str] = set()
    cache_hits = 0
    cache_misses = 0
    hit_latencies: list[float] = []
    miss_latencies: list[float] = []

    # Cache of embeddings to simulate cache behavior
    embedding_cache: dict[str, list[float]] = {}

    for i in range(num_queries):
        if rng.random() < repeat_ratio and seen_queries:
            # Pick a previously seen query
            query = rng.choice(list(seen_queries))
            is_repeat = True
        else:
            # Pick a unique query
            query = rng.choice(unique_queries)
            is_repeat = query in seen_queries

        # Simulate cache: reuse embedding if seen before
        if query in embedding_cache:
            query_embedding = embedding_cache[query]
        else:
            query_embedding = await embedding_service.embed(query)
            embedding_cache[query] = query_embedding

        start = time.perf_counter()
        await vector_store.recall(query_embedding, limit=5, min_similarity=0.3)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if is_repeat:
            cache_hits += 1
            hit_latencies.append(elapsed_ms)
        else:
            cache_misses += 1
            miss_latencies.append(elapsed_ms)

        seen_queries.add(query)

    hit_rate = cache_hits / num_queries if num_queries > 0 else 0.0

    return CacheResult(
        total_queries=num_queries,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        hit_rate=hit_rate,
        avg_hit_latency_ms=(
            statistics.mean(hit_latencies) if hit_latencies else 0.0
        ),
        avg_miss_latency_ms=(
            statistics.mean(miss_latencies) if miss_latencies else 0.0
        ),
    )


def _percentile(sorted_data: list[float], pct: int) -> float:
    """Calculate percentile from sorted data."""
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]
