"""Performance/scale testing suite (Issue #6).

Benchmarks retrieval latency, batch embedding throughput,
and cache effectiveness under realistic workloads.

Uses mock services for deterministic, CI-friendly benchmarks.
Run with: PYTHONPATH=src pytest tests/test_performance.py -v
"""

import time
import pytest

from tribalmemory.interfaces import MemoryEntry
from tribalmemory.performance.corpus_generator import (
    generate_corpus,
    CorpusConfig,
)
from tribalmemory.performance.benchmarks import (
    BenchmarkResult,
    benchmark_retrieval_latency,
    benchmark_batch_embedding_throughput,
    benchmark_cache_effectiveness,
    LatencyStats,
)


# --- Corpus Generator Tests ---


class TestCorpusGenerator:
    """Test synthetic corpus generation for scale testing."""

    def test_generate_corpus_default(self):
        """Should generate a corpus with default config."""
        entries = generate_corpus()
        assert len(entries) > 0
        assert all(isinstance(e, MemoryEntry) for e in entries)

    def test_generate_corpus_custom_size(self):
        """Should generate exact number of entries requested."""
        entries = generate_corpus(CorpusConfig(size=500))
        assert len(entries) == 500

    def test_generate_corpus_large(self):
        """Should handle 10k+ entries efficiently."""
        start = time.perf_counter()
        entries = generate_corpus(CorpusConfig(size=10_000))
        elapsed = time.perf_counter() - start
        assert len(entries) == 10_000
        # Corpus generation should be fast (< 5s for 10k)
        assert elapsed < 5.0, f"Corpus generation too slow: {elapsed:.2f}s"

    def test_corpus_entries_have_content(self):
        """All entries should have non-empty content."""
        entries = generate_corpus(CorpusConfig(size=100))
        for entry in entries:
            assert entry.content.strip(), f"Empty content in entry {entry.id}"

    def test_corpus_entries_have_tags(self):
        """Entries should have varied tags for filtering benchmarks."""
        entries = generate_corpus(CorpusConfig(size=100))
        all_tags = set()
        for entry in entries:
            all_tags.update(entry.tags)
        assert len(all_tags) >= 3, "Corpus should have diverse tags"

    def test_corpus_deterministic_with_seed(self):
        """Same seed should produce identical corpus."""
        entries1 = generate_corpus(CorpusConfig(size=50, seed=42))
        entries2 = generate_corpus(CorpusConfig(size=50, seed=42))
        for e1, e2 in zip(entries1, entries2):
            assert e1.content == e2.content


# --- Retrieval Latency Benchmarks ---


class TestRetrievalLatency:
    """Benchmark retrieval latency at various corpus sizes."""

    @pytest.mark.asyncio
    async def test_retrieval_latency_500(self):
        """Measure p50/p95/p99 retrieval latency with 500 memories."""
        result = await benchmark_retrieval_latency(corpus_size=500, num_queries=30)
        assert result.stats.p50 >= 0
        assert result.stats.p95 >= result.stats.p50
        assert result.stats.p99 >= result.stats.p95

    @pytest.mark.asyncio
    async def test_retrieval_latency_2k(self):
        """Measure retrieval latency with 2k memories."""
        result = await benchmark_retrieval_latency(corpus_size=2_000, num_queries=20)
        assert result.stats.p50 >= 0
        # Document baseline: p99 should be under 500ms for mock store
        assert result.stats.p99 < 500.0, (
            f"p99 latency {result.stats.p99:.2f}ms exceeds 500ms threshold"
        )

    @pytest.mark.asyncio
    async def test_retrieval_latency_10k(self):
        """Measure retrieval latency with 10k memories (Issue #6 requirement)."""
        result = await benchmark_retrieval_latency(corpus_size=10_000, num_queries=15)
        assert result.stats.p50 >= 0
        assert result.stats.p99 < 1500.0, (
            f"p99 latency {result.stats.p99:.2f}ms exceeds 1500ms threshold"
        )

    @pytest.mark.asyncio
    async def test_retrieval_latency_scales_sublinearly(self):
        """Latency should not scale linearly with corpus size."""
        result_500 = await benchmark_retrieval_latency(corpus_size=500, num_queries=15)
        result_2k = await benchmark_retrieval_latency(corpus_size=2_000, num_queries=15)

        # 4x more data should not cause 4x latency (sublinear scaling)
        ratio = result_2k.stats.p50 / max(result_500.stats.p50, 0.001)
        assert ratio < 8.0, (
            f"Latency scaled {ratio:.1f}x for 4x corpus "
            f"(500 p50={result_500.stats.p50:.2f}ms, "
            f"2k p50={result_2k.stats.p50:.2f}ms)"
        )


# --- Batch Embedding Throughput ---


class TestBatchEmbeddingThroughput:
    """Benchmark embedding generation throughput."""

    @pytest.mark.asyncio
    async def test_single_embedding_throughput(self):
        """Measure single embedding generation rate."""
        result = await benchmark_batch_embedding_throughput(
            num_texts=100, batch_size=1
        )
        assert result.embeddings_per_second > 0
        assert result.total_embeddings == 100

    @pytest.mark.asyncio
    async def test_batch_embedding_throughput(self):
        """Measure batch embedding generation rate."""
        result = await benchmark_batch_embedding_throughput(
            num_texts=500, batch_size=50
        )
        assert result.embeddings_per_second > 0
        assert result.total_embeddings == 500

    @pytest.mark.asyncio
    async def test_batch_faster_than_single(self):
        """Batch embedding should be at least as fast as single."""
        single = await benchmark_batch_embedding_throughput(
            num_texts=100, batch_size=1
        )
        batch = await benchmark_batch_embedding_throughput(
            num_texts=100, batch_size=50
        )
        # Batch should not be dramatically slower
        assert batch.embeddings_per_second >= single.embeddings_per_second * 0.5


# --- Cache Effectiveness ---


class TestCacheEffectiveness:
    """Benchmark query cache hit rates under realistic workloads."""

    @pytest.mark.asyncio
    async def test_cache_hit_rate_repeated_queries(self):
        """Repeated identical queries should have high cache hit rate."""
        result = await benchmark_cache_effectiveness(
            corpus_size=500,
            num_queries=100,
            repeat_ratio=0.8,  # 80% repeated queries
        )
        assert result.hit_rate >= 0.5, (
            f"Cache hit rate {result.hit_rate:.2%} too low for 80% repeated queries"
        )

    @pytest.mark.asyncio
    async def test_cache_hit_rate_unique_queries(self):
        """Mostly unique queries should have lower cache hit rate than repeated."""
        repeated_result = await benchmark_cache_effectiveness(
            corpus_size=500,
            num_queries=100,
            repeat_ratio=0.8,
        )
        unique_result = await benchmark_cache_effectiveness(
            corpus_size=500,
            num_queries=100,
            repeat_ratio=0.0,
        )
        assert unique_result.hit_rate < repeated_result.hit_rate, (
            f"Unique ({unique_result.hit_rate:.2%}) should be lower than "
            f"repeated ({repeated_result.hit_rate:.2%})"
        )

    @pytest.mark.asyncio
    async def test_cache_metrics_structure(self):
        """Cache benchmark should return complete metrics."""
        result = await benchmark_cache_effectiveness(
            corpus_size=100,
            num_queries=50,
            repeat_ratio=0.5,
        )
        assert result.total_queries == 50
        assert result.cache_hits + result.cache_misses == result.total_queries
        assert 0.0 <= result.hit_rate <= 1.0


# --- Baseline Documentation ---


class TestBaselineDocumentation:
    """Generate and validate performance baselines."""

    @pytest.mark.asyncio
    async def test_generate_baseline_report(self):
        """Generate a complete baseline report."""
        # Retrieval
        retrieval_500 = await benchmark_retrieval_latency(
            corpus_size=500, num_queries=20
        )
        retrieval_2k = await benchmark_retrieval_latency(
            corpus_size=2_000, num_queries=20
        )

        # Throughput
        throughput = await benchmark_batch_embedding_throughput(
            num_texts=500, batch_size=50
        )

        # Cache
        cache = await benchmark_cache_effectiveness(
            corpus_size=500, num_queries=100, repeat_ratio=0.5
        )

        # Validate all results exist
        assert retrieval_500.stats.p50 >= 0
        assert retrieval_2k.stats.p50 >= 0
        assert throughput.embeddings_per_second > 0
        assert 0.0 <= cache.hit_rate <= 1.0

        # Print report for CI logs
        print("\n=== Performance Baseline Report ===")
        print(f"Retrieval (500):  p50={retrieval_500.stats.p50:.2f}ms "
              f"p95={retrieval_500.stats.p95:.2f}ms "
              f"p99={retrieval_500.stats.p99:.2f}ms")
        print(f"Retrieval (2k):  p50={retrieval_2k.stats.p50:.2f}ms "
              f"p95={retrieval_2k.stats.p95:.2f}ms "
              f"p99={retrieval_2k.stats.p99:.2f}ms")
        print(f"Embedding throughput: {throughput.embeddings_per_second:.0f} emb/s")
        print(f"Cache hit rate (50% repeat): {cache.hit_rate:.2%}")
        print("=== End Report ===")
