"""Performance benchmarks for Tribal Memory.

Provides functions to measure retrieval latency, embedding throughput,
cache effectiveness, and graph operations using mock services for
CI-friendly execution.
"""

import random
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ..services.graph_store import EntityExtractor, GraphStore, Entity, Relationship
from ..testing.mocks import MockEmbeddingService, MockVectorStore
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
    """Result of a cache effectiveness benchmark.

    'Cache hits' here means queries that were repeats of previously
    seen queries (simulating a query cache). 'Cache misses' are
    first-time queries that would require full embedding + retrieval.
    """
    total_queries: int
    cache_hits: int     # Repeated queries (would be served from cache)
    cache_misses: int   # First-seen queries (require full retrieval)
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

    # Generate a pool of truly unique queries (sample without replacement)
    pool_size = min(len(corpus), num_queries * 2)
    unique_pool = rng.sample(corpus, k=pool_size)
    unique_queries = list(dict.fromkeys(
        entry.content[:50] for entry in unique_pool
    ))  # Deduplicate while preserving order

    seen_queries: list[str] = []  # Ordered list for repeat selection
    seen_set: set[str] = set()
    cache_hits = 0
    cache_misses = 0
    hit_latencies: list[float] = []
    miss_latencies: list[float] = []
    unique_idx = 0  # Track position in unique pool (no replacement)

    # Cache of embeddings to simulate cache behavior
    embedding_cache: dict[str, list[float]] = {}

    for i in range(num_queries):
        if rng.random() < repeat_ratio and seen_queries:
            # Pick a previously seen query (repeat)
            query = rng.choice(seen_queries)
            is_repeat = True
        else:
            # Pick next unique query (sequential, no replacement)
            if unique_idx < len(unique_queries):
                query = unique_queries[unique_idx]
                unique_idx += 1
            else:
                # Exhausted unique pool, fall back to random
                query = rng.choice(unique_queries)
            is_repeat = query in seen_set

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

        if query not in seen_set:
            seen_queries.append(query)
            seen_set.add(query)

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


def _percentile(data: list[float], pct: int) -> float:
    """Calculate percentile using linear interpolation.

    Uses ``statistics.quantiles`` for accurate interpolation, which
    handles small sample sizes better than nearest-rank.  Data is
    sorted internally so callers don't need to pre-sort.
    """
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    # quantiles(n=100) returns 99 cut points dividing the data into
    # 100 equal-probability intervals.  Index (pct - 1) gives the
    # pct-th percentile.  Clamp for edge cases like p100.
    quantile_points = statistics.quantiles(sorted(data), n=100)
    idx = min(pct - 1, len(quantile_points) - 1)
    return quantile_points[idx]


# --- Graph Benchmark Results ---


@dataclass
class EntityExtractionResult:
    """Result of an entity extraction throughput benchmark."""
    total_texts: int
    total_entities: int
    total_time_ms: float
    ms_per_text: float
    entities_per_text: float


@dataclass
class GraphQueryResult:
    """Result of a graph query latency benchmark."""
    num_entities: int
    num_relationships: int
    query_type: str  # 'get_memories' or 'find_connected'
    hops: int  # Only relevant for find_connected
    stats: LatencyStats


@dataclass
class GraphExpansionOverheadResult:
    """Result of a graph expansion overhead benchmark."""
    corpus_size: int
    num_queries: int
    without_graph_stats: LatencyStats
    with_graph_stats: LatencyStats
    overhead_pct: float  # Percentage increase in p50 latency


# --- Graph Corpus Generator ---


def _generate_graph_corpus(
    num_entries: int,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Generate corpus entries with extractable entities.
    
    Creates synthetic text entries containing service names and technology
    references that EntityExtractor can identify. Templates are designed
    to match EntityExtractor.RELATIONSHIP_PATTERNS for realistic benchmarks.
    
    Args:
        num_entries: Number of corpus entries to generate.
        seed: Random seed for reproducibility.
    
    Returns:
        List of (memory_id, content) tuples suitable for entity extraction.
    """
    rng = random.Random(seed)
    
    # Service names match EntityExtractor.SERVICE_PATTERN (kebab-case)
    services = [
        'auth-service', 'user-api', 'payment-gateway', 'order-worker',
        'notification-service', 'cache-proxy', 'data-pipeline',
        'ml-service', 'search-api', 'analytics-worker',
        'billing-service', 'report-api', 'audit-service',
        'config-server', 'gateway-proxy', 'sync-worker',
    ]
    
    # Technologies match EntityExtractor.TECHNOLOGIES set
    technologies = [
        'PostgreSQL', 'Redis', 'Kafka', 'Docker', 'Kubernetes',
        'Python', 'FastAPI', 'MongoDB', 'Elasticsearch', 'Nginx',
    ]
    
    # Templates match EntityExtractor.RELATIONSHIP_PATTERNS for extraction
    templates = [
        "{service} uses {tech} for data storage",
        "{service} connects to {tech} for caching",
        "Configured {service} to handle requests via {tech}",
        "{service} stores metrics in {tech}",
        "The {service} depends on {tech} for message queue",
        "{tech} is used by {service} for persistence",
        "Deployed {service} with {tech} backend",
        "{service} talks to {tech} for real-time updates",
    ]
    
    entries: list[tuple[str, str]] = []
    for i in range(num_entries):
        service = rng.choice(services)
        tech = rng.choice(technologies)
        template = rng.choice(templates)
        content = template.format(service=service, tech=tech)
        entries.append((f"mem_{i:06d}", content))
    
    return entries


# --- Graph Benchmark Functions ---


@dataclass
class RelationshipExtractionResult:
    """Result of a relationship extraction throughput benchmark."""
    total_texts: int
    total_entities: int
    total_relationships: int
    total_time_ms: float
    ms_per_text: float
    entities_per_text: float
    relationships_per_text: float


async def benchmark_entity_extraction(
    num_texts: int = 1000,
    seed: int = 42,
) -> EntityExtractionResult:
    """Benchmark entity extraction throughput.
    
    Measures how fast EntityExtractor can process text and extract
    entities. Target: <1ms per entry.
    
    Note: This benchmarks extract() only. For production usage which
    includes relationship extraction, see benchmark_relationship_extraction().
    
    Args:
        num_texts: Number of text entries to process.
        seed: Random seed for reproducibility.
    
    Returns:
        EntityExtractionResult with timing and entity count metrics.
    """
    corpus = _generate_graph_corpus(num_texts, seed)
    extractor: EntityExtractor = EntityExtractor()
    
    total_entities = 0
    start = time.perf_counter()
    
    for _, content in corpus:
        entities = extractor.extract(content)
        total_entities += len(entities)
    
    total_ms = (time.perf_counter() - start) * 1000
    
    return EntityExtractionResult(
        total_texts=num_texts,
        total_entities=total_entities,
        total_time_ms=total_ms,
        ms_per_text=total_ms / num_texts if num_texts > 0 else 0,
        entities_per_text=total_entities / num_texts if num_texts > 0 else 0,
    )


async def benchmark_relationship_extraction(
    num_texts: int = 1000,
    seed: int = 42,
) -> RelationshipExtractionResult:
    """Benchmark entity + relationship extraction throughput.
    
    Measures extract_with_relationships() which is the method used in
    production when storing memories. This is more representative of
    actual performance than entity-only extraction.
    
    Args:
        num_texts: Number of text entries to process.
        seed: Random seed for reproducibility.
    
    Returns:
        RelationshipExtractionResult with timing and count metrics.
    """
    corpus = _generate_graph_corpus(num_texts, seed)
    extractor: EntityExtractor = EntityExtractor()
    
    total_entities = 0
    total_relationships = 0
    start = time.perf_counter()
    
    for _, content in corpus:
        entities, relationships = extractor.extract_with_relationships(content)
        total_entities += len(entities)
        total_relationships += len(relationships)
    
    total_ms = (time.perf_counter() - start) * 1000
    
    return RelationshipExtractionResult(
        total_texts=num_texts,
        total_entities=total_entities,
        total_relationships=total_relationships,
        total_time_ms=total_ms,
        ms_per_text=total_ms / num_texts if num_texts > 0 else 0,
        entities_per_text=total_entities / num_texts if num_texts > 0 else 0,
        relationships_per_text=(
            total_relationships / num_texts if num_texts > 0 else 0
        ),
    )


async def benchmark_graph_store_queries(
    num_entities: int = 1000,
    num_queries: int = 50,
    hops: int = 1,
    seed: int = 42,
) -> GraphQueryResult:
    """Benchmark GraphStore query latency.
    
    Populates a graph with entities and relationships, then measures
    query latency for find_connected operations.
    
    Args:
        num_entities: Number of entities to populate.
        num_queries: Number of queries to measure.
        hops: Number of hops for find_connected.
        seed: Random seed for reproducibility.
    
    Returns:
        GraphQueryResult with p50/p95/p99 latency stats.
    """
    rng = random.Random(seed)
    
    # Create temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "benchmark_graph.db"
        graph: GraphStore = GraphStore(db_path)
        
        # Generate entities
        entity_names: list[str] = [
            f"entity-{i:04d}" for i in range(num_entities)
        ]
        
        # Create entities with memory associations
        for i, name in enumerate(entity_names):
            entity = Entity(name=name, entity_type='service')
            graph.add_entity(entity, f"mem_{i:06d}")
        
        # Create relationships (sparse graph: ~3 connections per entity)
        num_relationships = 0
        for idx, name in enumerate(entity_names):
            num_connections = rng.randint(1, 5)
            sample_size = min(num_connections, len(entity_names))
            targets = rng.sample(entity_names, sample_size)
            for target in targets:
                if target != name:
                    rel = Relationship(
                        source=name, target=target, relation_type='uses'
                    )
                    graph.add_relationship(rel, f"mem_{idx:06d}")
                    num_relationships += 1
        
        # Select random entities to query
        query_entities = [rng.choice(entity_names) for _ in range(num_queries)]
        
        # Measure query latencies
        latencies: list[float] = []
        for entity_name in query_entities:
            start = time.perf_counter()
            graph.find_connected(entity_name, hops=hops)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        stats = LatencyStats(
            p50=_percentile(latencies, 50),
            p95=_percentile(latencies, 95),
            p99=_percentile(latencies, 99),
            mean=statistics.mean(latencies),
            min=min(latencies),
            max=max(latencies),
        )
        
        return GraphQueryResult(
            num_entities=num_entities,
            num_relationships=num_relationships,
            query_type='find_connected',
            hops=hops,
            stats=stats,
        )


async def benchmark_get_memories_for_entity(
    num_entities: int = 1000,
    memories_per_entity: int = 5,
    num_queries: int = 50,
    seed: int = 42,
) -> GraphQueryResult:
    """Benchmark get_memories_for_entity latency.
    
    Measures the time to look up memory IDs associated with an entity.
    This is a key operation for entity-centric recall.
    
    Args:
        num_entities: Number of entities to create.
        memories_per_entity: Average memories per entity.
        num_queries: Number of queries to measure.
        seed: Random seed for reproducibility.
    
    Returns:
        GraphQueryResult with latency stats.
    """
    rng = random.Random(seed)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "benchmark_graph.db"
        graph: GraphStore = GraphStore(db_path)
        
        entity_names: list[str] = [
            f"entity-{i:04d}" for i in range(num_entities)
        ]
        
        # Create entities with multiple memory associations
        memory_idx = 0
        for name in entity_names:
            entity = Entity(name=name, entity_type='service')
            for _ in range(memories_per_entity):
                graph.add_entity(entity, f"mem_{memory_idx:06d}")
                memory_idx += 1
        
        # Select random entities to query
        query_entities = [rng.choice(entity_names) for _ in range(num_queries)]
        
        # Measure query latencies
        latencies: list[float] = []
        for entity_name in query_entities:
            start = time.perf_counter()
            graph.get_memories_for_entity(entity_name)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        stats = LatencyStats(
            p50=_percentile(latencies, 50),
            p95=_percentile(latencies, 95),
            p99=_percentile(latencies, 99),
            mean=statistics.mean(latencies),
            min=min(latencies),
            max=max(latencies),
        )
        
        return GraphQueryResult(
            num_entities=num_entities,
            num_relationships=0,
            query_type='get_memories',
            hops=0,
            stats=stats,
        )
