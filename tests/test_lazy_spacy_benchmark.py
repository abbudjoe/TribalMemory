"""Benchmark tests for lazy spaCy performance improvement (Issue #97).

Measures the performance difference between:
- lazy_spacy=True: regex extraction on ingest, spaCy NER on recall
- lazy_spacy=False: spaCy NER on both ingest and recall

Expected improvement: ~70x faster ingest in real-world usage (LongMemEval).
With mock services, improvement is smaller (~1.5-2.5x) due to fast mock embeddings.
"""

import time
import random
import pytest

from tribalmemory.services.graph_store import GraphStore, SPACY_AVAILABLE
from tribalmemory.services.memory import TribalMemoryService
from tribalmemory.testing.mocks import MockEmbeddingService, MockVectorStore


# --- Test Data Generation ---


def generate_test_memories(count: int, seed: int = 42) -> list[str]:
    """Generate test memories with entity-rich content."""
    templates = [
        "{person1} discussed {topic} with {person2} at {place}",
        "The {service} uses {technology} for {function}",
        "{person1} deployed {service} to {environment} using {tool}",
        "Integration between {service1} and {service2} via {protocol}",
        "{person1} fixed {bug_type} bug in {service} {component}",
        "{person1} reviewed code for {service} written in {language}",
        "The {service} connects to {technology} on port {port}",
    ]
    
    names = ["John", "Sarah", "Mary", "Tom", "Alice", "Bob", "Jane", "Mike",
            "Emma", "David", "Lisa", "Chris", "Anna", "Paul", "Rachel", "Steve"]
    places = ["Starbucks", "office", "cafe", "meeting room", "park", "library"]
    topics = ["API design", "database schema", "security", "performance",
              "architecture", "deployment", "monitoring", "testing"]
    services = ["auth-service", "api-gateway", "user-service", "payment-gateway",
                "order-service", "notification-worker", "web-service", "cache-layer",
                "search-service", "analytics-service", "logging-service", "admin-portal"]
    technologies = ["PostgreSQL", "Redis", "MongoDB", "Docker", "Kubernetes",
                    "nginx", "RabbitMQ", "Elasticsearch", "Kafka", "Cassandra"]
    functions = ["authentication", "caching", "logging", "monitoring", "analytics",
                 "search", "storage", "messaging", "routing", "processing"]
    environments = ["production", "staging", "development", "testing"]
    tools = ["Docker", "Kubernetes", "Ansible", "Terraform", "Jenkins"]
    protocols = ["REST API", "gRPC", "WebSocket", "message queue", "HTTP"]
    bug_types = ["memory leak", "race condition", "null pointer", "timeout"]
    components = ["handler", "middleware", "controller", "service layer", "repository"]
    languages = ["Python", "TypeScript", "Go", "Java", "Rust"]
    
    random.seed(seed)
    
    memories = []
    for i in range(count):
        template = random.choice(templates)
        memory = template.format(
            person1=random.choice(names),
            person2=random.choice(names),
            place=random.choice(places),
            topic=random.choice(topics),
            service=random.choice(services),
            service1=random.choice(services),
            service2=random.choice(services),
            technology=random.choice(technologies),
            function=random.choice(functions),
            environment=random.choice(environments),
            tool=random.choice(tools),
            protocol=random.choice(protocols),
            bug_type=random.choice(bug_types),
            component=random.choice(components),
            language=random.choice(languages),
            port=random.randint(3000, 9000),
        )
        memories.append(memory)
    
    return memories


# --- Fixtures ---


@pytest.fixture
def mock_embedding_service():
    return MockEmbeddingService(embedding_dim=64)


@pytest.fixture
def mock_vector_store(mock_embedding_service):
    return MockVectorStore(mock_embedding_service)


@pytest.fixture
def mock_graph_store(tmp_path):
    return GraphStore(str(tmp_path / "graph.db"))


# --- Benchmark Tests ---


class TestLazySpacyBenchmark:
    """Benchmark lazy vs eager spaCy modes (Issue #97)."""

    @pytest.mark.asyncio
    async def test_ingest_benchmark_small(
        self, tmp_path, mock_embedding_service, mock_vector_store
    ):
        """Benchmark ingest with 20 memories (smoke test)."""
        memories = generate_test_memories(20, seed=97)
        
        # Lazy mode
        lazy_graph_store = GraphStore(str(tmp_path / "lazy_graph.db"))
        lazy_service = TribalMemoryService(
            instance_id="lazy",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=lazy_graph_store,
            graph_enabled=True,
            lazy_spacy=True,
        )
        
        start = time.perf_counter()
        for content in memories:
            await lazy_service.remember(content)
        lazy_time = time.perf_counter() - start
        
        # Clear and create new stores for eager mode
        mock_vector_store._store.clear()
        mock_vector_store._deleted.clear()
        
        # Eager mode
        eager_graph_store = GraphStore(str(tmp_path / "eager_graph.db"))
        eager_service = TribalMemoryService(
            instance_id="eager",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=eager_graph_store,
            graph_enabled=True,
            lazy_spacy=False,
        )
        
        start = time.perf_counter()
        for content in memories:
            await eager_service.remember(content)
        eager_time = time.perf_counter() - start
        
        speedup = eager_time / max(lazy_time, 0.0001)
        
        print(f"\n=== Small Benchmark (20 memories) ===")
        print(f"Lazy:  {lazy_time:.3f}s ({lazy_time*50:.1f}ms/memory)")
        print(f"Eager: {eager_time:.3f}s ({eager_time*50:.1f}ms/memory)")
        print(f"Speedup: {speedup:.1f}x")
        
        # Lazy should not be slower
        assert lazy_time <= eager_time * 1.2, f"Lazy unexpectedly slower: {speedup:.1f}x"

    @pytest.mark.asyncio
    async def test_ingest_benchmark_100(
        self, tmp_path, mock_embedding_service, mock_vector_store
    ):
        """Benchmark ingest with 100 memories.
        
        Expected: 1.5-2.5x speedup with mocks (70x with real embeddings).
        """
        memories = generate_test_memories(100, seed=97)
        
        # Lazy mode
        lazy_graph_store = GraphStore(str(tmp_path / "lazy_graph.db"))
        lazy_service = TribalMemoryService(
            instance_id="lazy",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=lazy_graph_store,
            graph_enabled=True,
            lazy_spacy=True,
        )
        
        start = time.perf_counter()
        for content in memories:
            await lazy_service.remember(content)
        lazy_time = time.perf_counter() - start
        
        # Clear and create new stores
        mock_vector_store._store.clear()
        mock_vector_store._deleted.clear()
        
        # Eager mode
        eager_graph_store = GraphStore(str(tmp_path / "eager_graph.db"))
        eager_service = TribalMemoryService(
            instance_id="eager",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=eager_graph_store,
            graph_enabled=True,
            lazy_spacy=False,
        )
        
        start = time.perf_counter()
        for content in memories:
            await eager_service.remember(content)
        eager_time = time.perf_counter() - start
        
        speedup = eager_time / max(lazy_time, 0.0001)
        lazy_ms = (lazy_time * 1000) / len(memories)
        eager_ms = (eager_time * 1000) / len(memories)
        
        print(f"\n=== Benchmark (100 memories) ===")
        print(f"Lazy:  {lazy_time:.3f}s ({lazy_ms:.2f}ms/memory)")
        print(f"Eager: {eager_time:.3f}s ({eager_ms:.2f}ms/memory)")
        print(f"Speedup: {speedup:.1f}x")
        
        # With mock embeddings, expect at least 1.3x improvement
        # (Real-world with actual embeddings shows ~70x)
        if SPACY_AVAILABLE:
            assert speedup >= 1.3, (
                f"Expected ≥1.3x speedup, got {speedup:.1f}x. "
                f"Note: Real-world (with real embeddings) shows ~70x."
            )

    @pytest.mark.asyncio
    async def test_comprehensive_benchmark(
        self, tmp_path, mock_embedding_service, mock_vector_store
    ):
        """Comprehensive benchmark documenting lazy spaCy performance (Issue #97).
        
        This is the canonical performance documentation for the issue.
        Measures:
        - Ingest latency (total and per-memory)
        - Vector-only recall performance
        - Documents expected vs actual speedup
        """
        memories = generate_test_memories(200, seed=97)
        
        print(f"\n{'='*70}")
        print(f"Lazy spaCy Performance Benchmark (Issue #97)")
        print(f"{'='*70}")
        print(f"Corpus: {len(memories)} memories with entity-rich content")
        print()
        
        # === LAZY MODE ===
        lazy_graph_store = GraphStore(str(tmp_path / "lazy_graph.db"))
        lazy_service = TribalMemoryService(
            instance_id="lazy",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=lazy_graph_store,
            graph_enabled=True,
            lazy_spacy=True,
        )
        
        start = time.perf_counter()
        for content in memories:
            await lazy_service.remember(content)
        lazy_ingest_time = time.perf_counter() - start
        
        # Test vector-only recall (no graph expansion)
        lazy_recall_start = time.perf_counter()
        lazy_results = await lazy_service.recall("api-gateway", limit=5, graph_expansion=False)
        lazy_recall_time = time.perf_counter() - lazy_recall_start
        
        # Clear and create new stores
        mock_vector_store._store.clear()
        mock_vector_store._deleted.clear()
        
        # === EAGER MODE ===
        eager_graph_store = GraphStore(str(tmp_path / "eager_graph.db"))
        eager_service = TribalMemoryService(
            instance_id="eager",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=eager_graph_store,
            graph_enabled=True,
            lazy_spacy=False,
        )
        
        start = time.perf_counter()
        for content in memories:
            await eager_service.remember(content)
        eager_ingest_time = time.perf_counter() - start
        
        # Test vector-only recall
        eager_recall_start = time.perf_counter()
        eager_results = await eager_service.recall("api-gateway", limit=5, graph_expansion=False)
        eager_recall_time = time.perf_counter() - eager_recall_start
        
        # === METRICS ===
        speedup = eager_ingest_time / max(lazy_ingest_time, 0.0001)
        lazy_ms_per = (lazy_ingest_time * 1000) / len(memories)
        eager_ms_per = (eager_ingest_time * 1000) / len(memories)
        
        # === REPORT ===
        print(f"Ingest Performance:")
        print(f"  Lazy mode:  {lazy_ingest_time:.3f}s ({lazy_ms_per:.2f}ms/memory)")
        print(f"  Eager mode: {eager_ingest_time:.3f}s ({eager_ms_per:.2f}ms/memory)")
        print(f"  Speedup:    {speedup:.1f}x")
        print()
        print(f"Vector-Only Recall Performance:")
        print(f"  Lazy:  {lazy_recall_time*1000:.2f}ms ({len(lazy_results)} results)")
        print(f"  Eager: {eager_recall_time*1000:.2f}ms ({len(eager_results)} results)")
        print()
        print(f"Expected vs Actual:")
        print(f"  Target (real-world): ~70x faster ingest with lazy spaCy")
        print(f"  Actual (mock services): {speedup:.1f}x")
        print()
        if SPACY_AVAILABLE:
            print(f"  ✓ spaCy available: Ingest speedup from skipping NER")
            print(f"  Note: Mock embeddings reduce the gap. Real embeddings (FastEmbed/")
            print(f"        OpenAI) dominate ingest cost, amplifying the lazy spaCy benefit")
        else:
            print(f"  ⚠ spaCy not installed: Both modes use regex only")
        print()
        print(f"  Vector recall performance is identical (both use embeddings).")
        print(f"  Graph expansion may differ (lazy: regex entities, eager: spaCy entities)")
        print(f"{'='*70}\n")
        
        # === ASSERTIONS ===
        # Lazy should be at least as fast (or within 20% for variance)
        assert lazy_ingest_time <= eager_ingest_time * 1.2, (
            f"Lazy mode slower than expected: {speedup:.1f}x"
        )
        
        # Vector-only recall should return similar results
        # (may differ by 1-2 due to tie-breaking in mock similarity scores)
        assert abs(len(lazy_results) - len(eager_results)) <= 2, (
            f"Vector recall differs significantly: lazy={len(lazy_results)}, "
            f"eager={len(eager_results)}"
        )
        
        # If spaCy available, expect measurable improvement (at least 1.2x)
        if SPACY_AVAILABLE:
            assert speedup >= 1.2, (
                f"Expected ≥1.2x speedup with spaCy, got {speedup:.1f}x"
            )
        
        print(f"✅ Benchmark complete: {speedup:.1f}x faster ingest with lazy spaCy")

    @pytest.mark.asyncio
    async def test_recall_vector_only_identical(
        self, tmp_path, mock_embedding_service, mock_vector_store
    ):
        """Verify vector-only recall is identical between modes.
        
        Vector similarity search should return the same results regardless
        of lazy vs eager mode (both use the same embeddings).
        
        Graph expansion results will differ (by design) because lazy uses
        regex entities on ingest while eager uses spaCy entities.
        """
        memories = [
            "The auth-service uses PostgreSQL for user storage",
            "John deployed the api-gateway to production",
            "The payment-service connects to Stripe API",
        ]
        
        # Lazy mode
        lazy_graph_store = GraphStore(str(tmp_path / "lazy_graph.db"))
        lazy_service = TribalMemoryService(
            instance_id="lazy",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=lazy_graph_store,
            graph_enabled=True,
            lazy_spacy=True,
        )
        
        for content in memories:
            await lazy_service.remember(content)
        
        lazy_results = await lazy_service.recall(
            "auth-service", limit=10, graph_expansion=False
        )
        
        # Clear and switch to eager
        mock_vector_store._store.clear()
        mock_vector_store._deleted.clear()
        
        eager_graph_store = GraphStore(str(tmp_path / "eager_graph.db"))
        eager_service = TribalMemoryService(
            instance_id="eager",
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            graph_store=eager_graph_store,
            graph_enabled=True,
            lazy_spacy=False,
        )
        
        for content in memories:
            await eager_service.remember(content)
        
        eager_results = await eager_service.recall(
            "auth-service", limit=10, graph_expansion=False
        )
        
        # Vector-only results should be identical
        assert len(lazy_results) == len(eager_results), (
            f"Vector recall differs: lazy={len(lazy_results)}, eager={len(eager_results)}"
        )
        
        lazy_contents = {r.memory.content for r in lazy_results}
        eager_contents = {r.memory.content for r in eager_results}
        assert lazy_contents == eager_contents, "Vector recall content differs"
