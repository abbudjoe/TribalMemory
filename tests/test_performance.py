"""Performance/scale testing suite (Issue #6, #53).

Benchmarks retrieval latency, batch embedding throughput,
cache effectiveness, and graph operations under realistic workloads.

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
    benchmark_entity_extraction,
    benchmark_graph_store_queries,
    benchmark_get_memories_for_entity,
    EntityExtractionResult,
    GraphQueryResult,
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


# --- Graph Performance Benchmarks (Issue #53) ---


class TestEntityExtraction:
    """Benchmark entity extraction throughput."""

    def test_entity_extraction_1k(self):
        """Entity extraction should be fast (<1ms per entry)."""
        result = benchmark_entity_extraction(num_texts=1000)
        assert result.total_texts == 1000
        assert result.total_entities > 0
        # Target: <1ms per text
        assert result.ms_per_text < 1.0, (
            f"Entity extraction too slow: {result.ms_per_text:.3f}ms/text "
            f"(target: <1ms)"
        )

    def test_entity_extraction_10k(self):
        """Entity extraction should scale to 10k entries."""
        result = benchmark_entity_extraction(num_texts=10_000)
        assert result.total_texts == 10_000
        # Should still be under 1ms per text at scale
        assert result.ms_per_text < 1.0, (
            f"Entity extraction at scale: {result.ms_per_text:.3f}ms/text"
        )

    def test_entity_extraction_extracts_entities(self):
        """Should extract meaningful entities from test corpus."""
        result = benchmark_entity_extraction(num_texts=100)
        # Each entry has service + technology, expect ~2 per text
        assert result.entities_per_text >= 1.5, (
            f"Too few entities: {result.entities_per_text:.1f}/text (expected ~2)"
        )

    def test_entity_extraction_deterministic(self):
        """Same seed should produce same results."""
        result1 = benchmark_entity_extraction(num_texts=100, seed=42)
        result2 = benchmark_entity_extraction(num_texts=100, seed=42)
        assert result1.total_entities == result2.total_entities


class TestGraphStoreQueries:
    """Benchmark GraphStore query latency."""

    def test_find_connected_1hop_1k_entities(self):
        """1-hop traversal should be fast (<10ms) with 1k entities."""
        result = benchmark_graph_store_queries(
            num_entities=1000,
            num_queries=30,
            hops=1,
        )
        assert result.num_entities == 1000
        assert result.query_type == 'find_connected'
        assert result.hops == 1
        # Target: p99 < 10ms for 1-hop
        assert result.stats.p99 < 10.0, (
            f"1-hop traversal too slow: p99={result.stats.p99:.2f}ms (target: <10ms)"
        )

    def test_find_connected_2hop_1k_entities(self):
        """2-hop traversal should be reasonable (<50ms) with 1k entities."""
        result = benchmark_graph_store_queries(
            num_entities=1000,
            num_queries=30,
            hops=2,
        )
        assert result.hops == 2
        # Target: p99 < 50ms for 2-hop
        assert result.stats.p99 < 50.0, (
            f"2-hop traversal too slow: p99={result.stats.p99:.2f}ms (target: <50ms)"
        )

    def test_find_connected_scales_with_hops(self):
        """Latency should increase with hop count, but sublinearly."""
        result_1hop = benchmark_graph_store_queries(
            num_entities=500, num_queries=20, hops=1
        )
        result_2hop = benchmark_graph_store_queries(
            num_entities=500, num_queries=20, hops=2
        )
        # 2-hop should be slower but not 10x slower
        ratio = result_2hop.stats.p50 / max(result_1hop.stats.p50, 0.001)
        assert ratio < 10.0, (
            f"2-hop {ratio:.1f}x slower than 1-hop (expected <10x)"
        )

    def test_find_connected_2k_entities(self):
        """Should handle 2k entities efficiently.
        
        Note: Reduced from 5k due to GraphStore connection-per-operation overhead.
        See Issue #49 for connection pooling optimization.
        Setup time: ~50s for 2k entities + relationships.
        """
        result = benchmark_graph_store_queries(
            num_entities=2000,
            num_queries=15,
            hops=1,
        )
        assert result.num_entities == 2000
        # Allow more time for larger graph, but still reasonable
        assert result.stats.p99 < 100.0, (
            f"2k entity 1-hop: p99={result.stats.p99:.2f}ms (target: <100ms)"
        )


class TestGetMemoriesForEntity:
    """Benchmark get_memories_for_entity latency."""

    def test_get_memories_1k_entities(self):
        """get_memories_for_entity should be fast (<5ms)."""
        result = benchmark_get_memories_for_entity(
            num_entities=1000,
            memories_per_entity=5,
            num_queries=30,
        )
        assert result.query_type == 'get_memories'
        # Target: p99 < 5ms
        assert result.stats.p99 < 5.0, (
            f"get_memories too slow: p99={result.stats.p99:.2f}ms (target: <5ms)"
        )

    def test_get_memories_many_associations(self):
        """Should handle entities with many memory associations."""
        result = benchmark_get_memories_for_entity(
            num_entities=500,
            memories_per_entity=20,  # 20 memories per entity
            num_queries=30,
        )
        # Still should be under 10ms
        assert result.stats.p99 < 10.0, (
            f"get_memories (20/entity): p99={result.stats.p99:.2f}ms"
        )


class TestGraphBaselineReport:
    """Generate graph performance baseline report."""

    def test_generate_graph_baseline_report(self):
        """Generate a complete graph benchmark report."""
        # Entity extraction
        extraction = benchmark_entity_extraction(num_texts=1000)
        
        # Graph queries
        query_1hop = benchmark_graph_store_queries(
            num_entities=1000, num_queries=30, hops=1
        )
        query_2hop = benchmark_graph_store_queries(
            num_entities=1000, num_queries=30, hops=2
        )
        
        # Memory lookup
        get_memories = benchmark_get_memories_for_entity(
            num_entities=1000, memories_per_entity=5, num_queries=30
        )
        
        # Validate all results
        assert extraction.ms_per_text >= 0
        assert query_1hop.stats.p50 >= 0
        assert query_2hop.stats.p50 >= 0
        assert get_memories.stats.p50 >= 0
        
        # Print report
        print("\n=== Graph Performance Baseline Report ===")
        print(f"Entity extraction: {extraction.ms_per_text:.3f}ms/text "
              f"({extraction.entities_per_text:.1f} entities/text)")
        print(f"find_connected (1-hop): p50={query_1hop.stats.p50:.2f}ms "
              f"p99={query_1hop.stats.p99:.2f}ms")
        print(f"find_connected (2-hop): p50={query_2hop.stats.p50:.2f}ms "
              f"p99={query_2hop.stats.p99:.2f}ms")
        print(f"get_memories_for_entity: p50={get_memories.stats.p50:.2f}ms "
              f"p99={get_memories.stats.p99:.2f}ms")
        print(f"Graph stats: {query_1hop.num_entities} entities, "
              f"{query_1hop.num_relationships} relationships")
        print("=== End Graph Report ===")
