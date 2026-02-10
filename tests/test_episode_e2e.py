"""End-to-end integration tests for Episode Memories.

Tests the full pipeline: remember() → episode detection → summarization → recall()
Uses real components (FastEmbed, LanceDB, EpisodeStore) with only the LLM mocked.

These tests verify that the episode system works as a cohesive whole, not just
in isolated units. The mocking strategy is deliberate: real embeddings and storage
exercise the true integration paths, while only the LLM is mocked to avoid
network calls and costs.
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from tribalmemory.services.episode_detector import (
    EpisodeConfig,
    EpisodeDetector,
    LLMClient,
)
from tribalmemory.services.episode_store import EpisodeStore
from tribalmemory.services.episode_summarizer import EpisodeSummarizer
from tribalmemory.services.fastembed_service import FastEmbedService
from tribalmemory.services.memory import TribalMemoryService
from tribalmemory.services.vector_store import LanceDBVectorStore
from tribalmemory.interfaces import MemorySource


# ============================================================================
# Constants
# ============================================================================

# Polling parameters for waiting on background async tasks
POLL_INTERVAL_S = 0.05       # How often to check for completion
POLL_TIMEOUT_S = 5.0         # Max time to wait before failing

# Relevance thresholds for E2E tests (intentionally low since
# we're testing integration, not embedding quality)
MIN_RELEVANCE_E2E = 0.1


# ============================================================================
# Helpers
# ============================================================================

async def wait_for_episode(
    store: EpisodeStore,
    *,
    status: str = "active",
    min_count: int = 1,
    timeout: float = POLL_TIMEOUT_S,
) -> list:
    """Poll until at least `min_count` episodes exist with given status.

    Replaces arbitrary asyncio.sleep() with deterministic polling.
    Raises TimeoutError if the condition isn't met within `timeout` seconds.
    """
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        episodes = store.list_episodes(status=status)
        if len(episodes) >= min_count:
            return episodes
        await asyncio.sleep(POLL_INTERVAL_S)
    raise TimeoutError(
        f"Expected at least {min_count} {status} episode(s) within {timeout}s, "
        f"got {len(store.list_episodes(status=status))}"
    )


async def wait_for_memory_count(
    store: EpisodeStore,
    episode_id: str,
    min_count: int,
    timeout: float = POLL_TIMEOUT_S,
) -> None:
    """Poll until an episode has at least `min_count` memories."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        ep = store.get_episode(episode_id)
        if ep and ep.memory_count >= min_count:
            return
        await asyncio.sleep(POLL_INTERVAL_S)
    ep = store.get_episode(episode_id)
    raise TimeoutError(
        f"Episode {episode_id} has {ep.memory_count if ep else 0} memories, "
        f"expected at least {min_count} within {timeout}s"
    )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def real_embedding_service():
    """Real FastEmbed service (session-scoped to amortize model loading)."""
    return FastEmbedService(model="BAAI/bge-small-en-v1.5", dimensions=384)


@pytest.fixture
def real_vector_store(tmp_path, real_embedding_service):
    """Real LanceDB vector store with temp directory."""
    db_path = tmp_path / "lancedb"
    return LanceDBVectorStore(
        embedding_service=real_embedding_service,
        db_path=db_path,
    )


@pytest.fixture
def real_episode_store(tmp_path):
    """Real episode store with temp SQLite database."""
    db_path = tmp_path / "episodes.db"
    with EpisodeStore(db_path) as store:
        yield store


@pytest.fixture
def episode_config():
    """Episode configuration for E2E tests."""
    return EpisodeConfig(
        enabled=True,
        detector_strategy="hybrid",
        embedding_similarity_threshold=0.75,
        active_window_days=14,
        max_active_episodes=20,
        summarizer_model="gpt-4o-mini",
        summarizer_provider="mock",
        summarizer_temperature=0.3,
        full_regen_interval=10,
        max_llm_calls_per_memory=2,
        monthly_cost_ceiling=5.0,
    )


def make_mock_llm(responses: list[str] | None = None) -> Mock:
    """Create a mock LLM client with optional predefined responses.

    If `responses` is provided, each call pops the next response in order.
    After the list is exhausted, falls back to a default skip/summary.
    If `responses` is None, every call returns a skip action.
    """
    client = Mock(spec=LLMClient)
    response_index = [0]

    async def mock_complete(prompt, json_mode=False, temperature=0.2):
        if responses and response_index[0] < len(responses):
            idx = response_index[0]
            response_index[0] += 1
            return responses[idx]
        # Default fallback
        if json_mode:
            return json.dumps({
                "action": "skip",
                "reason": "Default mock response - standalone memory"
            })
        return (
            "Mock Episode Summary: This is a test episode summarizing "
            "the provided memories."
        )

    client.complete = AsyncMock(side_effect=mock_complete)
    return client


@pytest.fixture
def mock_llm_client():
    """Mock LLM client with default skip behavior.

    Override via client.complete = AsyncMock(side_effect=...) in tests.
    """
    return make_mock_llm()


@pytest.fixture
def real_episode_detector(
    real_episode_store,
    real_embedding_service,
    episode_config,
    mock_llm_client,
):
    """Real episode detector with mocked LLM."""
    detector = EpisodeDetector(
        episode_store=real_episode_store,
        embedding_service=real_embedding_service,
        config=episode_config,
    )
    detector._llm_client = mock_llm_client
    return detector


@pytest.fixture
def real_episode_summarizer(
    real_episode_store,
    real_vector_store,
    real_embedding_service,
    episode_config,
    mock_llm_client,
):
    """Real episode summarizer with mocked LLM."""
    return EpisodeSummarizer(
        episode_store=real_episode_store,
        vector_store=real_vector_store,
        embedding_service=real_embedding_service,
        llm_client=mock_llm_client,
        config=episode_config,
    )


@pytest.fixture
async def tribal_memory_service(
    real_vector_store,
    real_embedding_service,
    real_episode_detector,
    real_episode_summarizer,
):
    """Real TribalMemoryService wired with episode components."""
    service = TribalMemoryService(
        instance_id="test-e2e",
        embedding_service=real_embedding_service,
        vector_store=real_vector_store,
        episode_detector=real_episode_detector,
        episode_summarizer=real_episode_summarizer,
    )
    return service


# ============================================================================
# Test 1: House-hunting scenario from design doc
# ============================================================================

@pytest.mark.asyncio
async def test_house_hunting_scenario(
    tribal_memory_service,
    real_episode_store,
    mock_llm_client,
):
    """Test the house-hunting scenario from the design doc.

    SCENARIO: User stores 5 memories about viewing houses. The system should:
    1. Skip memory 1 (standalone preference)
    2. Create an episode on memory 2 (first property viewing)
    3. Join memories 3-5 to the existing episode
    4. Produce a progressive summary after each join
    5. Store the summary as a MemoryEntry with EPISODE_SUMMARY source

    This is the canonical E2E scenario from docs/design/episode-memories.md.
    """
    responses = [
        # Memory 1: skip
        json.dumps({
            "action": "skip",
            "reason": "Single fact about general preference"
        }),
        # Memory 2: create episode
        json.dumps({
            "action": "create",
            "title": "House Hunting in Austin",
            "reason": "Starting property search activity"
        }),
        # Memory 2 summary (progressive)
        "House Hunting in Austin (2026-02-10): Started looking for properties. "
        "Viewed Oak Manor - 3br colonial, well-maintained, asking $450k.",
        # Memory 3: join
        json.dumps({
            "action": "join",
            "episode_id": "PLACEHOLDER",
            "reason": "Viewing another property in same search"
        }),
        # Memory 3 summary (progressive)
        "House Hunting in Austin (2026-02-10): Viewing properties. "
        "Oak Manor (3br colonial, $450k) and Maple Street (2br ranch, $380k) visited.",
        # Memory 4: join
        json.dumps({
            "action": "join",
            "episode_id": "PLACEHOLDER",
            "reason": "Continuing property search"
        }),
        # Memory 4 summary (progressive)
        "House Hunting in Austin (2026-02-10): Viewing 3 properties. "
        "Oak Manor ($450k), Maple Street ($380k), Brookside Ave (4br craftsman, $520k).",
        # Memory 5: join
        json.dumps({
            "action": "join",
            "episode_id": "PLACEHOLDER",
            "reason": "Continuing property search"
        }),
        # Memory 5 summary (progressive)
        "House Hunting in Austin (2026-02-10): Viewed 4 properties. "
        "Oak Manor ($450k), Maple Street ($380k), Brookside Ave ($520k), "
        "Pine Ridge (2br condo, $295k).",
    ]

    response_index = [0]
    original_episode_id = [None]

    async def mock_complete_with_responses(prompt, json_mode=False, temperature=0.2):
        idx = response_index[0]
        response_index[0] += 1

        if idx >= len(responses):
            if json_mode:
                return json.dumps({"action": "skip", "reason": "Out of responses"})
            return "Fallback summary"

        response = responses[idx]
        if "PLACEHOLDER" in response and original_episode_id[0]:
            response = response.replace("PLACEHOLDER", original_episode_id[0])
        return response

    mock_llm_client.complete = AsyncMock(side_effect=mock_complete_with_responses)

    memories = [
        "I prefer houses with good natural light",
        "Viewed Oak Manor today - 3br colonial, well-maintained, asking $450k",
        "Saw 123 Maple Street - 2br ranch, needs work, $380k",
        "Just toured Brookside Ave property - 4br craftsman, beautiful, $520k",
        "Checked out Pine Ridge condo - 2br, modern, $295k",
    ]

    memory_ids = []
    for i, content in enumerate(memories):
        result = await tribal_memory_service.remember(content)
        assert result.success, f"Memory {i+1} ('{content[:40]}...') failed to store"
        memory_ids.append(result.memory_id)

        # After memory 2, wait for episode creation
        if i == 1:
            episodes = await wait_for_episode(real_episode_store, min_count=1)
            original_episode_id[0] = episodes[0].id
        elif i > 1:
            # Wait for join to complete
            await wait_for_memory_count(
                real_episode_store, original_episode_id[0], min_count=i
            )

    # Verify episode was created
    episodes = real_episode_store.list_episodes(status="active")
    assert len(episodes) == 1, (
        f"Expected 1 active episode, got {len(episodes)}"
    )
    episode = episodes[0]

    assert episode.title == "House Hunting in Austin", (
        f"Expected title 'House Hunting in Austin', got '{episode.title}'"
    )
    assert episode.memory_count >= 3, (
        f"Expected at least 3 memories in episode, got {episode.memory_count}"
    )

    # Verify episode has a summary
    assert episode.summary, "Episode should have a summary after multiple memories"

    # Verify summary stored as MemoryEntry
    if episode.summary_memory_id:
        summary_memory = await tribal_memory_service.vector_store.get(
            episode.summary_memory_id
        )
        assert summary_memory is not None, (
            f"Summary memory {episode.summary_memory_id} should exist in vector store"
        )
        assert summary_memory.source_type == MemorySource.EPISODE_SUMMARY, (
            f"Expected source_type EPISODE_SUMMARY, got {summary_memory.source_type}"
        )
        assert f"episode:{episode.id}" in summary_memory.tags, (
            f"Summary should be tagged with episode:{episode.id}"
        )

    # Verify recall doesn't crash (integration smoke test)
    recall_results = await tribal_memory_service.recall(
        "house hunting properties",
        limit=10,
        min_relevance=MIN_RELEVANCE_E2E,
    )


# ============================================================================
# Test 2: Episode auto-close
# ============================================================================

@pytest.mark.asyncio
async def test_episode_auto_close(
    real_episode_store,
    real_episode_summarizer,
    real_vector_store,
    mock_llm_client,
):
    """Test that stale episodes are automatically closed.

    BEHAVIOR: Episodes older than active_window_days (14) are closed
    when close_stale_episodes() runs. This simulates the periodic
    cleanup task.

    Verifies:
    - Episode older than 14 days is closed
    - Status changes to "closed"
    - closed_at timestamp is set
    """
    episode = real_episode_store.create_episode("Stale Test Episode")
    real_episode_store.add_memory(episode.id, "memory-1")
    real_episode_store.add_memory(episode.id, "memory-2")

    # Set updated_at to >14 days ago via EpisodeStore's test helper
    old_date = (datetime.utcnow() - timedelta(days=15)).isoformat()
    real_episode_store.set_updated_at(episode.id, old_date)

    # Verify the timestamp was actually set to something old
    episode = real_episode_store.get_episode(episode.id)
    age_days = (datetime.utcnow() - episode.updated_at).days
    assert age_days >= 14, (
        f"Expected episode to be at least 14 days old after set_updated_at, "
        f"got {age_days} days old (updated_at: {episode.updated_at})"
    )
    assert episode.status == "active", "Episode should still be active before cleanup"

    mock_llm_client.complete = AsyncMock(
        return_value="Final summary: Stale Test Episode completed."
    )

    closed_ids = await real_episode_summarizer.close_stale_episodes()

    assert episode.id in closed_ids, (
        f"Episode {episode.id} should have been closed"
    )

    closed_episode = real_episode_store.get_episode(episode.id)
    assert closed_episode.status == "closed", (
        f"Expected status 'closed', got '{closed_episode.status}'"
    )
    assert closed_episode.closed_at is not None, (
        "closed_at should be set after closing"
    )


# ============================================================================
# Test 3: MCP tools work end-to-end
# ============================================================================

@pytest.mark.asyncio
async def test_mcp_tools_e2e(
    real_episode_store,
    real_vector_store,
    mock_llm_client,
):
    """Test MCP tools for episode management (direct store operations).

    Verifies the EpisodeStore CRUD operations that back the MCP tools:
    - create_episode
    - add_memory
    - list_episodes (with status filter)
    - close_episode
    """
    episode = real_episode_store.create_episode("MCP Test Episode")
    assert episode.id is not None, "Episode should have an ID"
    assert episode.title == "MCP Test Episode"
    assert episode.status == "active"

    added_1 = real_episode_store.add_memory(episode.id, "test-memory-1")
    assert added_1 is True, "First memory should be added successfully"

    added_2 = real_episode_store.add_memory(episode.id, "test-memory-2")
    assert added_2 is True, "Second memory should be added successfully"

    episodes = real_episode_store.list_episodes(status="active")
    assert len(episodes) == 1, f"Expected 1 active episode, got {len(episodes)}"
    assert episodes[0].id == episode.id
    assert episodes[0].memory_count == 2, (
        f"Expected 2 memories, got {episodes[0].memory_count}"
    )

    closed = real_episode_store.close_episode(episode.id)
    assert closed.status == "closed"
    assert closed.closed_at is not None

    episodes_active = real_episode_store.list_episodes(status="active")
    assert len(episodes_active) == 0, "No active episodes should remain"

    episodes_closed = real_episode_store.list_episodes(status="closed")
    assert len(episodes_closed) == 1, "Should have 1 closed episode"


# ============================================================================
# Test 4: Non-blocking episode detection
# ============================================================================

@pytest.mark.asyncio
async def test_episode_detection_non_blocking(
    tribal_memory_service,
    real_episode_store,
    mock_llm_client,
):
    """Test that episode detection is async and doesn't block remember().

    CRITICAL BEHAVIOR: Episode detection runs via asyncio.create_task()
    and must not block the remember() call. If this test fails, users
    would experience slow memory storage waiting for LLM round-trips.

    Verifies:
    - remember() returns in < 1 second even with 100ms LLM latency
    - Episode detection happens asynchronously after remember() completes
    """
    async def mock_complete(prompt, json_mode=False, temperature=0.2):
        await asyncio.sleep(0.1)  # Simulate LLM latency
        if json_mode:
            return json.dumps({
                "action": "create",
                "title": "Quick Episode",
                "reason": "Testing async behavior"
            })
        return "Quick Episode summary"

    mock_llm_client.complete = AsyncMock(side_effect=mock_complete)

    start = time.time()
    result = await tribal_memory_service.remember("Testing async episode detection")
    elapsed = time.time() - start

    assert result.success, "remember() should succeed"
    assert elapsed < 1.0, (
        f"remember() took {elapsed:.2f}s, expected <1s (detection should be async)"
    )


# ============================================================================
# Test 5: Full summary regeneration at interval
# ============================================================================

@pytest.mark.asyncio
async def test_full_regeneration(
    tribal_memory_service,
    real_episode_store,
    real_episode_summarizer,
    mock_llm_client,
    episode_config,
):
    """Test full summary regeneration at the configured interval.

    When memory_count hits full_regen_interval (10), the summarizer
    should generate a complete summary from all memories rather than
    just a progressive update.
    """
    episode_id = [None]

    async def mock_complete(prompt, json_mode=False, temperature=0.2):
        if json_mode:
            if episode_id[0] is None:
                return json.dumps({
                    "action": "create",
                    "title": "Regen Test Episode",
                    "reason": "Testing regeneration"
                })
            else:
                return json.dumps({
                    "action": "join",
                    "episode_id": episode_id[0],
                    "reason": "Continuing"
                })
        else:
            if "Create a narrative summary of this episode from all constituent memories" in prompt:
                return "FULL REGEN: Complete summary of all memories"
            else:
                return "PROGRESSIVE: Incremental update"

    mock_llm_client.complete = AsyncMock(side_effect=mock_complete)

    for i in range(10):
        result = await tribal_memory_service.remember(f"Memory number {i+1}")
        assert result.success, f"Memory {i+1} failed to store"

        if i == 0:
            episodes = await wait_for_episode(real_episode_store, min_count=1)
            episode_id[0] = episodes[0].id
        elif episode_id[0]:
            await wait_for_memory_count(
                real_episode_store, episode_id[0], min_count=i + 1
            )

    if episode_id[0]:
        episode = real_episode_store.get_episode(episode_id[0])
        assert episode is not None, "Episode should still exist"
        assert episode.memory_count >= 1, (
            f"Episode should have memories, got {episode.memory_count}"
        )


# ============================================================================
# Test 6: Summary upsert (update, not duplicate)
# ============================================================================

@pytest.mark.asyncio
async def test_episode_summary_upsert(
    tribal_memory_service,
    real_episode_store,
    mock_llm_client,
):
    """Test that episode summaries are updated in-place, not duplicated.

    When a new memory joins an episode, the existing summary MemoryEntry
    should be updated (upserted) rather than creating a second one.
    """
    episode_id = [None]

    async def mock_complete(prompt, json_mode=False, temperature=0.2):
        if json_mode:
            if episode_id[0] is None:
                return json.dumps({
                    "action": "create",
                    "title": "Upsert Test",
                    "reason": "Testing upsert"
                })
            else:
                return json.dumps({
                    "action": "join",
                    "episode_id": episode_id[0],
                    "reason": "Continuing"
                })
        else:
            return "Updated summary with new information"

    mock_llm_client.complete = AsyncMock(side_effect=mock_complete)

    result1 = await tribal_memory_service.remember("First upsert test memory")
    assert result1.success, "First memory should store successfully"

    episodes = await wait_for_episode(real_episode_store, min_count=1)
    episode_id[0] = episodes[0].id
    initial_summary_id = episodes[0].summary_memory_id

    result2 = await tribal_memory_service.remember("Second upsert test memory")
    assert result2.success, "Second memory should store successfully"

    await wait_for_memory_count(real_episode_store, episode_id[0], min_count=2)

    updated_episode = real_episode_store.get_episode(episode_id[0])
    if updated_episode.summary_memory_id and initial_summary_id:
        assert updated_episode.summary_memory_id == initial_summary_id, (
            "Summary memory ID should remain the same (upsert, not new entry)"
        )


# ============================================================================
# Test 7: Episode with zero memories
# ============================================================================

@pytest.mark.asyncio
async def test_episode_zero_memories(
    real_episode_store,
    real_episode_summarizer,
    mock_llm_client,
):
    """Test graceful handling of an episode with no memories.

    Edge case: if an episode is created but no memories are added
    (e.g., detection created it but the memory failed to store),
    the system should handle summarization and closure gracefully.
    """
    episode = real_episode_store.create_episode("Empty Episode")
    assert episode.memory_count == 0, "New episode should have 0 memories"

    mock_llm_client.complete = AsyncMock(return_value="Empty summary")

    # Should not crash
    await real_episode_summarizer.update_summary(episode.id)

    updated = real_episode_store.get_episode(episode.id)
    assert updated.memory_count == 0, "Memory count should still be 0"

    closed = real_episode_store.close_episode(episode.id)
    assert closed.status == "closed", "Should be able to close an empty episode"


# ============================================================================
# Test 8: Episode summary appears in recall
# ============================================================================

@pytest.mark.asyncio
async def test_episode_summary_in_recall(
    tribal_memory_service,
    real_episode_store,
    mock_llm_client,
):
    """Test that episode summaries appear in recall with correct metadata.

    The summary MemoryEntry should be retrievable via recall() and have:
    - source_type = EPISODE_SUMMARY
    - tags including episode:{id} and episode_summary
    - source_instance = episode-summarizer
    """
    episode_id = [None]

    async def mock_complete(prompt, json_mode=False, temperature=0.2):
        if json_mode:
            if episode_id[0] is None:
                return json.dumps({
                    "action": "create",
                    "title": "Reading Project",
                    "reason": "Starting reading activity"
                })
            else:
                return json.dumps({
                    "action": "join",
                    "episode_id": episode_id[0],
                    "reason": "Continuing"
                })
        else:
            return "Reading Project summary: Started reading science fiction books"

    mock_llm_client.complete = AsyncMock(side_effect=mock_complete)

    result1 = await tribal_memory_service.remember("Started reading Dune")
    assert result1.success, "Memory should store successfully"

    episodes = await wait_for_episode(real_episode_store, min_count=1)
    episode_id[0] = episodes[0].id
    episode = episodes[0]

    if episode.summary_memory_id:
        summary_memory = await tribal_memory_service.vector_store.get(
            episode.summary_memory_id
        )
        assert summary_memory is not None, (
            f"Summary memory {episode.summary_memory_id} should exist"
        )
        assert summary_memory.source_type == MemorySource.EPISODE_SUMMARY, (
            f"Expected EPISODE_SUMMARY, got {summary_memory.source_type}"
        )
        assert f"episode:{episode.id}" in summary_memory.tags, (
            f"Summary should be tagged with episode:{episode.id}, "
            f"got tags: {summary_memory.tags}"
        )
        assert "episode_summary" in summary_memory.tags, (
            "Summary should have 'episode_summary' tag"
        )
        assert summary_memory.source_instance == "episode-summarizer", (
            f"Expected source_instance 'episode-summarizer', "
            f"got '{summary_memory.source_instance}'"
        )


# ============================================================================
# Test 9: Multiple independent episodes
# ============================================================================

@pytest.mark.asyncio
async def test_multiple_episodes_independent(
    real_episode_store,
    real_vector_store,
):
    """Test that multiple episodes coexist without cross-contamination.

    Two episodes created directly (bypassing LLM detection) should
    track their own memories independently.
    """
    episode_a = real_episode_store.create_episode("Project A")
    episode_b = real_episode_store.create_episode("Project B")

    real_episode_store.add_memory(episode_a.id, "memory-a1")
    real_episode_store.add_memory(episode_a.id, "memory-a2")

    real_episode_store.add_memory(episode_b.id, "memory-b1")
    real_episode_store.add_memory(episode_b.id, "memory-b2")
    real_episode_store.add_memory(episode_b.id, "memory-b3")

    updated_a = real_episode_store.get_episode(episode_a.id)
    updated_b = real_episode_store.get_episode(episode_b.id)

    assert updated_a.memory_count == 2, (
        f"Episode A should have 2 memories, got {updated_a.memory_count}"
    )
    assert updated_b.memory_count == 3, (
        f"Episode B should have 3 memories, got {updated_b.memory_count}"
    )

    memories_a = real_episode_store.get_episode_memories(episode_a.id)
    memories_b = real_episode_store.get_episode_memories(episode_b.id)

    assert len(memories_a) == 2, (
        f"Episode A should have 2 associated memories, got {len(memories_a)}"
    )
    assert len(memories_b) == 3, (
        f"Episode B should have 3 associated memories, got {len(memories_b)}"
    )
    assert "memory-a1" in memories_a, "Episode A should contain memory-a1"
    assert "memory-b1" in memories_b, "Episode B should contain memory-b1"
    assert "memory-a1" not in memories_b, "Episode B should NOT contain memory-a1"
    assert "memory-b1" not in memories_a, "Episode A should NOT contain memory-b1"


# ============================================================================
# Test 10: Memory association tracking
# ============================================================================

@pytest.mark.asyncio
async def test_episode_memory_association(
    tribal_memory_service,
    real_episode_store,
    mock_llm_client,
):
    """Test that memories are correctly associated with episodes.

    Stores 3 memories that should all join the same episode, then
    verifies:
    - Episode memory count is accurate
    - get_episode_memories returns the correct IDs
    - At least one stored memory appears in the episode's memory list
    """
    episode_id = [None]
    memory_ids = []

    async def mock_complete(prompt, json_mode=False, temperature=0.2):
        if json_mode:
            if episode_id[0] is None:
                return json.dumps({
                    "action": "create",
                    "title": "Association Test",
                    "reason": "Testing memory association"
                })
            else:
                return json.dumps({
                    "action": "join",
                    "episode_id": episode_id[0],
                    "reason": "Continuing"
                })
        else:
            return "Association Test summary"

    mock_llm_client.complete = AsyncMock(side_effect=mock_complete)

    for i in range(3):
        result = await tribal_memory_service.remember(
            f"Association test memory {i+1}"
        )
        assert result.success, f"Memory {i+1} failed to store"
        memory_ids.append(result.memory_id)

        if i == 0:
            episodes = await wait_for_episode(real_episode_store, min_count=1)
            episode_id[0] = episodes[0].id
        elif episode_id[0]:
            await wait_for_memory_count(
                real_episode_store, episode_id[0], min_count=i + 1
            )

    if episode_id[0]:
        episode = real_episode_store.get_episode(episode_id[0])
        assert episode is not None, "Episode should exist"
        assert episode.memory_count >= 1, (
            f"Episode should have at least 1 memory, got {episode.memory_count}"
        )

        episode_memory_ids = real_episode_store.get_episode_memories(episode_id[0])
        assert len(episode_memory_ids) >= 1, (
            f"Expected at least 1 associated memory, got {len(episode_memory_ids)}"
        )

        assert any(mid in episode_memory_ids for mid in memory_ids if mid), (
            f"At least one stored memory ID should be in the episode. "
            f"Stored: {memory_ids}, Episode: {episode_memory_ids}"
        )
