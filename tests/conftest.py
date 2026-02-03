"""Pytest fixtures for Tribal Memory tests."""

import pytest
from pathlib import Path

from src.tribal_memory.testing import (
    MockEmbeddingService,
    MockVectorStore,
    MockMemoryService,
    LatencyTracker,
    TestResultLogger,
    load_test_data,
)


@pytest.fixture
def embedding_service():
    """Provide a mock embedding service."""
    return MockEmbeddingService()


@pytest.fixture
def vector_store(embedding_service):
    """Provide a mock vector store."""
    return MockVectorStore(embedding_service)


@pytest.fixture
def memory_service(embedding_service, vector_store):
    """Provide a mock memory service."""
    return MockMemoryService(
        instance_id="test-instance",
        embedding_service=embedding_service,
        vector_store=vector_store
    )


@pytest.fixture
def memory_service_b(embedding_service):
    """Provide a second mock memory service (different instance)."""
    # Shares embedding service but has own vector store for isolation testing
    return MockMemoryService(
        instance_id="test-instance-b",
        embedding_service=embedding_service
    )


@pytest.fixture
def latency_tracker():
    """Provide a latency tracker."""
    tracker = LatencyTracker()
    yield tracker
    tracker.clear()


@pytest.fixture
def result_logger(tmp_path):
    """Provide a test result logger."""
    logger = TestResultLogger(output_dir=tmp_path / "results")
    yield logger
    logger.clear()


@pytest.fixture
def test_data():
    """Load test data sets."""
    data_dir = Path(__file__).parent.parent / "data"
    return load_test_data(data_dir)


@pytest.fixture
def slow_embedding_service():
    """Embedding service with simulated latency."""
    return MockEmbeddingService(latency_ms=100)


@pytest.fixture
def failing_embedding_service():
    """Embedding service that fails sometimes."""
    return MockEmbeddingService(failure_rate=0.5)


@pytest.fixture
def timeout_embedding_service():
    """Embedding service that times out after N calls."""
    return MockEmbeddingService(timeout_after_n=3)


@pytest.fixture
def capacity_limited_store(embedding_service):
    """Vector store with limited capacity."""
    return MockVectorStore(embedding_service, max_capacity=10)
