"""Benchmark tests for lazy spaCy performance improvement.

Measures the performance difference between:
- lazy_spacy=True: regex extraction on ingest, spaCy NER on recall
- lazy_spacy=False: spaCy NER on both ingest and recall

Issue #97: https://github.com/abbudjoe/TribalMemory/issues/97
"""

import asyncio
import time
from dataclasses import dataclass

import pytest

from tribalmemory.services.graph_store import (
    EntityExtractor,
    GraphStore,
    HybridEntityExtractor,
    SPACY_AVAILABLE,
)
from tribalmemory.services.memory import TribalMemoryService
from tribalmemory.testing.mocks import MockEmbeddingService, MockVectorStore


# --- Sample data ---

SAMPLE_MEMORIES = [
    "Sarah deployed the user-service to production on Monday",
    "John reviewed the auth-api pull request and approved it",
    "The payment-gateway connects to Stripe via the api-proxy",
    "Dr. Smith discussed the migration plan with the DevOps team",
    "Alice configured Redis caching for the session-manager service",
    "Bob fixed a critical bug in the notification-worker",
    "The data-pipeline uses Kafka for event streaming",
    "Carol optimized PostgreSQL queries in the analytics-service",
    "David set up monitoring with Grafana for the order-service",
    "Eve integrated OpenAI embeddings into the search-service",
    "Frank deployed Kubernetes clusters on AWS for staging",
    "Grace wrote unit tests for the inventory-api endpoints",
    "Hank migrated the legacy-backend from Python 2 to Python 3",
    "Iris configured Nginx reverse proxy for the frontend-gateway",
    "Jake implemented WebSocket support in the chat-service",
    "Karen reviewed the security audit findings for the auth-gateway",
    "Leo added rate limiting to the public-api using Redis",
    "Mona designed the database schema for the billing-service",
    "Nick troubleshot the connection pool issues in pgbouncer",
    "Olivia set up CI/CD pipelines with GitHub Actions for the core-api",
]

SAMPLE_QUERIES = [
    "What did Sarah deploy?",
    "Who reviewed the auth-api?",
    "How does the payment system work?",
    "Tell me about the data pipeline",
    "What services use Redis?",
]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    label: str
    lazy_spacy: bool
    ingest_total_ms: float
    ingest_per_item_ms: float
    recall_total_ms: float
    recall_per_query_ms: float
    num_ingested: int
    num_queries: int


@pytest.fixture
def mock_embedding_service():
    return MockEmbeddingService(embedding_dim=64)


@pytest.fixture
def mock_vector_store(mock_embedding_service):
    return MockVectorStore(mock_embedding_service)


def _create_graph_store(tmp_path, suffix: str) -> GraphStore:
    """Create a fresh GraphStore in a unique path."""
    return GraphStore(str(tmp_path / f"graph_{suffix}.db"))


async def _run_benchmark(
    tmp_path,
    mock_embedding_service: MockEmbeddingService,
    lazy_spacy: bool,
    memories: list[str],
    queries: list[str],
) -> BenchmarkResult:
    """Run a single benchmark configuration.

    Creates a fresh service + stores for each run to avoid cross-contamination.
    """
    label = "lazy" if lazy_spacy else "eager"
    suffix = f"{label}_{id(memories)}"
    vector_store = MockVectorStore(mock_embedding_service)
    graph_store = _create_graph_store(tmp_path, suffix)

    service = TribalMemoryService(
        instance_id=f"bench-{label}",
        embedding_service=mock_embedding_service,
        vector_store=vector_store,
        graph_store=graph_store,
        graph_enabled=True,
        lazy_spacy=lazy_spacy,
    )

    # --- Ingest benchmark ---
    ingest_start = time.perf_counter()
    for mem in memories:
        await service.remember(mem)
    ingest_elapsed = time.perf_counter() - ingest_start
    ingest_ms = ingest_elapsed * 1000

    # --- Recall benchmark ---
    recall_start = time.perf_counter()
    for query in queries:
        await service.recall(query, limit=5, graph_expansion=True)
    recall_elapsed = time.perf_counter() - recall_start
    recall_ms = recall_elapsed * 1000

    return BenchmarkResult(
        label=label,
        lazy_spacy=lazy_spacy,
        ingest_total_ms=ingest_ms,
        ingest_per_item_ms=ingest_ms / len(memories),
        recall_total_ms=recall_ms,
        recall_per_query_ms=recall_ms / len(queries),
        num_ingested=len(memories),
        num_queries=len(queries),
    )


def _print_summary(lazy_result: BenchmarkResult, eager_result: BenchmarkResult) -> str:
    """Build and return a formatted summary table."""

    def speedup(eager_val: float, lazy_val: float) -> str:
        if lazy_val == 0:
            return "N/A"
        ratio = eager_val / lazy_val
        return f"{ratio:.2f}x"

    lines = [
        "",
        "=" * 72,
        "  Lazy spaCy Benchmark Results",
        "=" * 72,
        f"  Memories ingested : {lazy_result.num_ingested}",
        f"  Queries executed  : {lazy_result.num_queries}",
        f"  spaCy available   : {SPACY_AVAILABLE}",
        "-" * 72,
        f"  {'Metric':<30} {'Lazy (ms)':>12} {'Eager (ms)':>12} {'Speedup':>10}",
        "-" * 72,
        f"  {'Ingest total':<30} {lazy_result.ingest_total_ms:>12.2f}"
        f" {eager_result.ingest_total_ms:>12.2f}"
        f" {speedup(eager_result.ingest_total_ms, lazy_result.ingest_total_ms):>10}",
        f"  {'Ingest per item':<30} {lazy_result.ingest_per_item_ms:>12.2f}"
        f" {eager_result.ingest_per_item_ms:>12.2f}"
        f" {speedup(eager_result.ingest_per_item_ms, lazy_result.ingest_per_item_ms):>10}",
        f"  {'Recall total':<30} {lazy_result.recall_total_ms:>12.2f}"
        f" {eager_result.recall_total_ms:>12.2f}"
        f" {speedup(eager_result.recall_total_ms, lazy_result.recall_total_ms):>10}",
        f"  {'Recall per query':<30} {lazy_result.recall_per_query_ms:>12.2f}"
        f" {eager_result.recall_per_query_ms:>12.2f}"
        f" {speedup(eager_result.recall_per_query_ms, lazy_result.recall_per_query_ms):>10}",
        "-" * 72,
    ]

    if SPACY_AVAILABLE:
        lines.append(
            "  Note: spaCy is installed — lazy mode defers NER cost to recall."
        )
    else:
        lines.append(
            "  Note: spaCy not installed — both modes fall back to regex."
        )
        lines.append(
            "  Install spaCy for meaningful lazy vs eager comparison:"
        )
        lines.append(
            "    pip install tribalmemory[spacy] && python -m spacy download en_core_web_sm"
        )

    lines.append("=" * 72)
    return "\n".join(lines)


class TestSpacyBenchmark:
    """Benchmark comparing lazy_spacy=True vs lazy_spacy=False."""

    @pytest.mark.asyncio
    async def test_ingest_lazy_not_slower_than_eager(
        self, tmp_path, mock_embedding_service
    ):
        """Lazy spaCy ingest should be at least as fast as eager spaCy ingest.

        With lazy mode, ingest uses only regex (no spaCy NER), so it should
        never be slower than eager mode which runs spaCy on every memory.
        """
        lazy = await _run_benchmark(
            tmp_path, mock_embedding_service,
            lazy_spacy=True, memories=SAMPLE_MEMORIES, queries=SAMPLE_QUERIES,
        )
        eager = await _run_benchmark(
            tmp_path, mock_embedding_service,
            lazy_spacy=False, memories=SAMPLE_MEMORIES, queries=SAMPLE_QUERIES,
        )

        # Lazy ingest should not be significantly slower than eager.
        # 20% tolerance accounts for:
        # - Timer granularity on fast operations (sub-ms noise)
        # - GC pauses and OS scheduling jitter
        # - pytest fixture overhead variance between runs
        assert lazy.ingest_total_ms <= eager.ingest_total_ms * 1.20, (
            f"Lazy ingest ({lazy.ingest_total_ms:.2f}ms) was more than 20% "
            f"slower than eager ({eager.ingest_total_ms:.2f}ms)"
        )

    @pytest.mark.asyncio
    async def test_benchmark_produces_valid_results(
        self, tmp_path, mock_embedding_service
    ):
        """Both configurations should complete successfully with positive timings."""
        lazy = await _run_benchmark(
            tmp_path, mock_embedding_service,
            lazy_spacy=True, memories=SAMPLE_MEMORIES, queries=SAMPLE_QUERIES,
        )
        eager = await _run_benchmark(
            tmp_path, mock_embedding_service,
            lazy_spacy=False, memories=SAMPLE_MEMORIES, queries=SAMPLE_QUERIES,
        )

        # Both should produce positive timings
        for label, result in [("lazy", lazy), ("eager", eager)]:
            assert result.ingest_total_ms > 0, f"{label} ingest_total_ms should be positive"
            assert result.ingest_per_item_ms > 0, f"{label} ingest_per_item_ms should be positive"
            assert result.recall_total_ms > 0, f"{label} recall_total_ms should be positive"
            assert result.recall_per_query_ms > 0, f"{label} recall_per_query_ms should be positive"
            assert result.num_ingested == len(SAMPLE_MEMORIES), (
                f"{label} num_ingested: expected {len(SAMPLE_MEMORIES)}, got {result.num_ingested}"
            )
            assert result.num_queries == len(SAMPLE_QUERIES), (
                f"{label} num_queries: expected {len(SAMPLE_QUERIES)}, got {result.num_queries}"
            )

    @pytest.mark.asyncio
    async def test_benchmark_summary_table(
        self, tmp_path, mock_embedding_service, capsys
    ):
        """Benchmark should print a readable summary table."""
        lazy = await _run_benchmark(
            tmp_path, mock_embedding_service,
            lazy_spacy=True, memories=SAMPLE_MEMORIES, queries=SAMPLE_QUERIES,
        )
        eager = await _run_benchmark(
            tmp_path, mock_embedding_service,
            lazy_spacy=False, memories=SAMPLE_MEMORIES, queries=SAMPLE_QUERIES,
        )

        summary = _print_summary(lazy, eager)
        print(summary)

        captured = capsys.readouterr()
        assert "Lazy spaCy Benchmark Results" in captured.out
        assert "Ingest total" in captured.out
        assert "Recall total" in captured.out
        assert "Speedup" in captured.out

    @pytest.mark.asyncio
    async def test_lazy_uses_regex_extractor_for_ingest(
        self, tmp_path, mock_embedding_service
    ):
        """Verify lazy mode uses EntityExtractor (regex) for ingest path."""
        vector_store = MockVectorStore(mock_embedding_service)
        graph_store = _create_graph_store(tmp_path, "lazy_verify")

        service = TribalMemoryService(
            instance_id="verify-lazy",
            embedding_service=mock_embedding_service,
            vector_store=vector_store,
            graph_store=graph_store,
            graph_enabled=True,
            lazy_spacy=True,
        )

        assert isinstance(service.ingest_entity_extractor, EntityExtractor)
        assert isinstance(service.query_entity_extractor, HybridEntityExtractor)

    @pytest.mark.asyncio
    async def test_eager_uses_hybrid_extractor_for_both(
        self, tmp_path, mock_embedding_service
    ):
        """Verify eager mode uses HybridEntityExtractor for both paths."""
        vector_store = MockVectorStore(mock_embedding_service)
        graph_store = _create_graph_store(tmp_path, "eager_verify")

        service = TribalMemoryService(
            instance_id="verify-eager",
            embedding_service=mock_embedding_service,
            vector_store=vector_store,
            graph_store=graph_store,
            graph_enabled=True,
            lazy_spacy=False,
        )

        assert isinstance(service.ingest_entity_extractor, HybridEntityExtractor)
        assert isinstance(service.query_entity_extractor, HybridEntityExtractor)
        assert service.ingest_entity_extractor is service.query_entity_extractor

    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy not installed")
    @pytest.mark.asyncio
    async def test_spacy_ingest_speedup_with_lazy(
        self, tmp_path, mock_embedding_service
    ):
        """When spaCy is available, lazy ingest should be measurably faster.

        This test only runs when spaCy is installed, where the difference
        between regex-only (lazy ingest) and regex+spaCy (eager ingest)
        should be observable.
        """
        lazy = await _run_benchmark(
            tmp_path, mock_embedding_service,
            lazy_spacy=True, memories=SAMPLE_MEMORIES, queries=SAMPLE_QUERIES,
        )
        eager = await _run_benchmark(
            tmp_path, mock_embedding_service,
            lazy_spacy=False, memories=SAMPLE_MEMORIES, queries=SAMPLE_QUERIES,
        )

        # With spaCy, lazy ingest should be faster (regex-only vs regex+spaCy).
        # We just verify it's not slower — the speedup magnitude depends on
        # the machine and spaCy model loading overhead.
        # 10% tolerance (tighter than mock test) because spaCy NER is the
        # dominant cost, making the signal-to-noise ratio much higher.
        assert lazy.ingest_total_ms <= eager.ingest_total_ms * 1.10, (
            f"With spaCy installed, lazy ingest ({lazy.ingest_total_ms:.2f}ms) "
            f"should not be slower than eager ({eager.ingest_total_ms:.2f}ms)"
        )

    @pytest.mark.asyncio
    async def test_benchmark_with_empty_corpus(
        self, tmp_path, mock_embedding_service
    ):
        """Benchmark handles edge case of minimal input gracefully."""
        single_memory = ["Alice deployed auth-service"]
        single_query = ["What did Alice deploy?"]

        lazy = await _run_benchmark(
            tmp_path, mock_embedding_service,
            lazy_spacy=True, memories=single_memory, queries=single_query,
        )
        eager = await _run_benchmark(
            tmp_path, mock_embedding_service,
            lazy_spacy=False, memories=single_memory, queries=single_query,
        )

        assert lazy.num_ingested == 1
        assert eager.num_ingested == 1
        assert lazy.ingest_total_ms > 0
        assert eager.ingest_total_ms > 0
