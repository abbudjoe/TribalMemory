"""Tests for episode summarization.

Tests progressive summarization, full regeneration, and integration with the remember() flow.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from tribalmemory.services.episode_store import Episode, EpisodeStore
from tribalmemory.services.episode_detector import EpisodeConfig, LLMClient
from tribalmemory.services.episode_summarizer import EpisodeSummarizer
from tribalmemory.interfaces import (
    IEmbeddingService,
    IVectorStore,
    MemoryEntry,
    MemorySource,
    StoreResult,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def episode_config():
    """Default episode configuration."""
    return EpisodeConfig(
        enabled=True,
        detector_strategy="hybrid",
        embedding_similarity_threshold=0.75,
        active_window_days=14,
        max_active_episodes=20,
        summarizer_model="gpt-4o-mini",
        summarizer_provider="openai",
        full_regen_interval=10,
        max_llm_calls_per_memory=2,
        monthly_cost_ceiling=5.0,
    )


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = Mock(spec=IEmbeddingService)
    service.embed = AsyncMock(return_value=[0.1] * 384)
    service.similarity = Mock(return_value=0.5)
    return service


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = Mock(spec=IVectorStore)
    store.store = AsyncMock(return_value=StoreResult(success=True, memory_id="summary-id"))
    store.upsert = AsyncMock(return_value=StoreResult(success=True, memory_id="summary-id"))
    store.get = AsyncMock(return_value=None)
    store.recall = AsyncMock(return_value=[])  # Return empty list for dedup checks
    return store


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = Mock(spec=LLMClient)
    client.complete = AsyncMock(return_value="Generated summary")
    return client


@pytest.fixture(scope="function")
def episode_store(tmp_path):
    """Episode store with temporary database."""
    db_path = tmp_path / "test_episodes.db"
    with EpisodeStore(db_path) as store:
        yield store


@pytest.fixture
def summarizer(episode_store, mock_vector_store, mock_embedding_service, mock_llm_client, episode_config):
    """Episode summarizer with mocked dependencies."""
    return EpisodeSummarizer(
        episode_store=episode_store,
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        llm_client=mock_llm_client,
        config=episode_config,
    )


# ============================================================================
# EpisodeSummarizer Basic Tests
# ============================================================================

@pytest.mark.asyncio
async def test_summarizer_initialization(summarizer, episode_store, mock_vector_store, mock_embedding_service, mock_llm_client, episode_config):
    """Test EpisodeSummarizer initialization."""
    assert summarizer.episode_store == episode_store
    assert summarizer.vector_store == mock_vector_store
    assert summarizer.embedding_service == mock_embedding_service
    assert summarizer.llm_client == mock_llm_client
    assert summarizer.config == episode_config


@pytest.mark.asyncio
async def test_update_summary_no_new_memories(summarizer, episode_store):
    """Test update_summary with no new memories - should be no-op."""
    # Create episode
    episode = episode_store.create_episode("Test Episode")
    
    # Call update_summary - no memories added yet
    await summarizer.update_summary(episode.id)
    
    # Should not have called LLM
    summarizer.llm_client.complete.assert_not_called()


@pytest.mark.asyncio
async def test_update_summary_nonexistent_episode(summarizer):
    """Test update_summary with nonexistent episode."""
    # Should handle gracefully (no exception)
    await summarizer.update_summary("nonexistent-id")
    
    # Should not have called LLM
    summarizer.llm_client.complete.assert_not_called()


# ============================================================================
# Progressive Summary Tests
# ============================================================================

@pytest.mark.asyncio
async def test_progressive_summary_first_memory(summarizer, episode_store, mock_vector_store):
    """Test progressive summary generation for first memory."""
    # Create episode and add first memory
    episode = episode_store.create_episode("House Hunting")
    memory_id = "mem-1"
    episode_store.add_memory(episode.id, memory_id)
    
    # Mock vector store to return memory content
    mock_vector_store.get = AsyncMock(return_value=MemoryEntry(
        id=memory_id,
        content="Viewed Elm Street bungalow, 3 bed, needs roof, $385k",
        embedding=[0.1] * 384,
    ))
    
    # Mock LLM to return summary
    summarizer.llm_client.complete = AsyncMock(
        return_value="House Hunting (2026-02-10): Viewed 1 property - Elm Street bungalow (3 bed, needs roof, $385k)."
    )
    
    # Update summary
    await summarizer.update_summary(episode.id)
    
    # Verify LLM was called
    summarizer.llm_client.complete.assert_called_once()
    
    # Verify episode summary was updated
    updated_episode = episode_store.get_episode(episode.id)
    assert updated_episode.summary != ""
    assert "Elm Street" in updated_episode.summary
    assert updated_episode.summary_memory_id is not None
    
    # Verify summary was stored in vector store
    mock_vector_store.store.assert_called_once()
    call_args = mock_vector_store.store.call_args[0][0]
    assert call_args.source_type == MemorySource.EPISODE_SUMMARY
    assert "episode:" + episode.id in call_args.tags
    assert "episode_summary" in call_args.tags


@pytest.mark.asyncio
async def test_progressive_summary_subsequent_memory(summarizer, episode_store, mock_vector_store):
    """Test progressive summary with subsequent memories."""
    # Create episode with existing summary
    episode = episode_store.create_episode("House Hunting")
    episode_store.update_episode(
        episode.id,
        summary="House Hunting (2026-02-10): Viewed 1 property - Elm Street.",
        summary_memory_id="summary-1"
    )
    
    # Add first memory and mark as summarized
    memory_id_1 = "mem-1"
    episode_store.add_memory(episode.id, memory_id_1)
    episode_store.mark_memories_summarized(episode.id, [memory_id_1])
    
    # Add second memory
    memory_id_2 = "mem-2"
    episode_store.add_memory(episode.id, memory_id_2)
    
    # Mock vector store to return new memory
    mock_vector_store.get = AsyncMock(return_value=MemoryEntry(
        id=memory_id_2,
        content="Toured 44 Oak Avenue, gorgeous kitchen, tiny yard, $420k",
        embedding=[0.2] * 384,
    ))
    
    # Mock LLM to return updated summary
    summarizer.llm_client.complete = AsyncMock(
        return_value="House Hunting (2026-02-10 - 2026-02-11): Viewed 2 properties - Elm Street and Oak Avenue."
    )
    
    # Update summary
    await summarizer.update_summary(episode.id)
    
    # Verify LLM was called with progressive update prompt
    call_args = summarizer.llm_client.complete.call_args[0][0]
    assert "Update this episode summary" in call_args
    assert "Current summary:" in call_args
    assert "Elm Street" in call_args
    assert "Oak Avenue" in call_args
    
    # Verify episode was updated
    updated_episode = episode_store.get_episode(episode.id)
    assert "2 properties" in updated_episode.summary or "two properties" in updated_episode.summary.lower()
    
    # Verify memory was marked as summarized
    unsummarized = episode_store.get_unsummarized_memories(episode.id)
    assert memory_id_2 not in unsummarized


# ============================================================================
# Full Regeneration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_regeneration_at_interval(summarizer, episode_store, mock_vector_store):
    """Test full regeneration is triggered at configured interval."""
    # Create episode with 9 memories (next one will be 10th)
    episode = episode_store.create_episode("House Hunting")
    for i in range(9):
        mem_id = f"mem-{i}"
        episode_store.add_memory(episode.id, mem_id)
        episode_store.mark_memories_summarized(episode.id, [mem_id])
    
    # Add 10th memory
    mem_id_10 = "mem-10"
    episode_store.add_memory(episode.id, mem_id_10)
    
    # Mock vector store to return all memories
    async def mock_get(memory_id):
        return MemoryEntry(
            id=memory_id,
            content=f"Memory content {memory_id}",
            embedding=[0.1] * 384,
        )
    mock_vector_store.get = AsyncMock(side_effect=mock_get)
    
    # Mock LLM
    summarizer.llm_client.complete = AsyncMock(
        return_value="Full regenerated summary with all 10 properties."
    )
    
    # Update summary
    await summarizer.update_summary(episode.id)
    
    # Verify LLM was called with full regeneration prompt
    call_args = summarizer.llm_client.complete.call_args[0][0]
    assert "Create a narrative summary" in call_args
    assert "Episode title:" in call_args
    assert "Memories (chronological):" in call_args
    
    # Verify all memories were included (0-8 + mem-10)
    for i in range(9):
        assert f"mem-{i}" in call_args or f"Memory content mem-{i}" in call_args
    assert "mem-10" in call_args or "Memory content mem-10" in call_args


@pytest.mark.asyncio
async def test_full_regeneration_custom_interval(episode_store, mock_vector_store, mock_embedding_service, mock_llm_client):
    """Test full regeneration with custom interval."""
    config = EpisodeConfig(full_regen_interval=5)
    summarizer = EpisodeSummarizer(
        episode_store=episode_store,
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        llm_client=mock_llm_client,
        config=config,
    )
    
    # Create episode with 4 memories
    episode = episode_store.create_episode("Test")
    for i in range(4):
        mem_id = f"mem-{i}"
        episode_store.add_memory(episode.id, mem_id)
        episode_store.mark_memories_summarized(episode.id, [mem_id])
    
    # Add 5th memory (should trigger full regen)
    episode_store.add_memory(episode.id, "mem-5")
    
    # Mock vector store
    async def mock_get(memory_id):
        return MemoryEntry(
            id=memory_id,
            content=f"Content {memory_id}",
            embedding=[0.1] * 384,
        )
    mock_vector_store.get = AsyncMock(side_effect=mock_get)
    
    # Mock LLM
    mock_llm_client.complete = AsyncMock(return_value="Full regen")
    
    # Update summary
    await summarizer.update_summary(episode.id)
    
    # Verify full regeneration was called
    call_args = mock_llm_client.complete.call_args[0][0]
    assert "Create a narrative summary" in call_args


# ============================================================================
# Summary Storage Tests
# ============================================================================

@pytest.mark.asyncio
async def test_summary_stored_with_correct_metadata(summarizer, episode_store, mock_vector_store):
    """Test summary is stored as MemoryEntry with correct metadata."""
    episode = episode_store.create_episode("Test Episode")
    memory_id = "mem-1"
    episode_store.add_memory(episode.id, memory_id)
    
    mock_vector_store.get = AsyncMock(return_value=MemoryEntry(
        id=memory_id,
        content="Test content",
        embedding=[0.1] * 384,
    ))
    
    summarizer.llm_client.complete = AsyncMock(return_value="Generated summary")
    
    await summarizer.update_summary(episode.id)
    
    # Verify store was called
    mock_vector_store.store.assert_called_once()
    stored_entry = mock_vector_store.store.call_args[0][0]
    
    # Verify metadata
    assert stored_entry.source_type == MemorySource.EPISODE_SUMMARY
    assert "episode:" + episode.id in stored_entry.tags
    assert "episode_summary" in stored_entry.tags
    assert stored_entry.content == "Generated summary"
    assert stored_entry.embedding is not None


@pytest.mark.asyncio
async def test_summary_updated_not_duplicated(summarizer, episode_store, mock_vector_store):
    """Test summary is updated (upserted) not duplicated on re-summarization."""
    # Create episode with existing summary
    episode = episode_store.create_episode("Test")
    existing_summary_id = "existing-summary-id"
    episode_store.update_episode(
        episode.id,
        summary="Old summary",
        summary_memory_id=existing_summary_id
    )
    
    # Add new memory
    memory_id = "mem-1"
    episode_store.add_memory(episode.id, memory_id)
    
    mock_vector_store.get = AsyncMock(return_value=MemoryEntry(
        id=memory_id,
        content="New content",
        embedding=[0.1] * 384,
    ))
    
    summarizer.llm_client.complete = AsyncMock(return_value="Updated summary")
    
    await summarizer.update_summary(episode.id)
    
    # Verify upsert was called (not store) since summary already exists
    mock_vector_store.upsert.assert_called_once()
    upserted_entry = mock_vector_store.upsert.call_args[0][0]
    
    # Verify it used the existing summary ID
    assert upserted_entry.id == existing_summary_id
    assert upserted_entry.content == "Updated summary"


@pytest.mark.asyncio
async def test_summary_embedding_generated(summarizer, episode_store, mock_vector_store, mock_embedding_service):
    """Test summary embedding is generated and stored."""
    episode = episode_store.create_episode("Test")
    memory_id = "mem-1"
    episode_store.add_memory(episode.id, memory_id)
    
    mock_vector_store.get = AsyncMock(return_value=MemoryEntry(
        id=memory_id,
        content="Test",
        embedding=[0.1] * 384,
    ))
    
    summarizer.llm_client.complete = AsyncMock(return_value="Summary text")
    
    expected_embedding = [0.5] * 384
    mock_embedding_service.embed = AsyncMock(return_value=expected_embedding)
    
    await summarizer.update_summary(episode.id)
    
    # Verify embedding was generated for summary
    mock_embedding_service.embed.assert_called()
    call_args = [call[0][0] for call in mock_embedding_service.embed.call_args_list]
    assert "Summary text" in call_args
    
    # Verify stored entry has embedding
    stored_entry = mock_vector_store.store.call_args[0][0]
    assert stored_entry.embedding == expected_embedding


# ============================================================================
# Episode Store Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_episode_updated_with_summary_memory_id(summarizer, episode_store, mock_vector_store):
    """Test episode is updated with summary_memory_id."""
    episode = episode_store.create_episode("Test")
    memory_id = "mem-1"
    episode_store.add_memory(episode.id, memory_id)
    
    mock_vector_store.get = AsyncMock(return_value=MemoryEntry(
        id=memory_id,
        content="Test",
        embedding=[0.1] * 384,
    ))
    
    summarizer.llm_client.complete = AsyncMock(return_value="Summary")
    
    summary_memory_id = "summary-mem-id"
    mock_vector_store.store = AsyncMock(
        return_value=StoreResult(success=True, memory_id=summary_memory_id)
    )
    
    await summarizer.update_summary(episode.id)
    
    # Verify episode was updated with summary_memory_id
    updated_episode = episode_store.get_episode(episode.id)
    assert updated_episode.summary_memory_id == summary_memory_id
    assert updated_episode.summary == "Summary"


@pytest.mark.asyncio
async def test_memories_marked_summarized(summarizer, episode_store, mock_vector_store):
    """Test memories are marked as summarized after processing."""
    episode = episode_store.create_episode("Test")
    mem1 = "mem-1"
    mem2 = "mem-2"
    episode_store.add_memory(episode.id, mem1)
    episode_store.add_memory(episode.id, mem2)
    
    async def mock_get(memory_id):
        return MemoryEntry(
            id=memory_id,
            content=f"Content {memory_id}",
            embedding=[0.1] * 384,
        )
    mock_vector_store.get = AsyncMock(side_effect=mock_get)
    
    summarizer.llm_client.complete = AsyncMock(return_value="Summary")
    
    # Verify memories are not summarized initially
    unsummarized = episode_store.get_unsummarized_memories(episode.id)
    assert mem1 in unsummarized
    assert mem2 in unsummarized
    
    await summarizer.update_summary(episode.id)
    
    # Verify memories are now marked as summarized
    unsummarized_after = episode_store.get_unsummarized_memories(episode.id)
    assert mem1 not in unsummarized_after
    assert mem2 not in unsummarized_after


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_llm_failure_handled_gracefully(summarizer, episode_store, mock_vector_store):
    """Test LLM failure doesn't crash - logs error and continues."""
    episode = episode_store.create_episode("Test")
    memory_id = "mem-1"
    episode_store.add_memory(episode.id, memory_id)
    
    mock_vector_store.get = AsyncMock(return_value=MemoryEntry(
        id=memory_id,
        content="Test",
        embedding=[0.1] * 384,
    ))
    
    # Mock LLM to fail
    summarizer.llm_client.complete = AsyncMock(side_effect=Exception("LLM API error"))
    
    # Should not raise exception
    await summarizer.update_summary(episode.id)
    
    # Episode should remain unchanged
    updated_episode = episode_store.get_episode(episode.id)
    assert updated_episode.summary == ""


@pytest.mark.asyncio
async def test_vector_store_failure_handled(summarizer, episode_store, mock_vector_store):
    """Test vector store failure is handled gracefully (logged but episode not updated)."""
    episode = episode_store.create_episode("Test")
    memory_id = "mem-1"
    episode_store.add_memory(episode.id, memory_id)
    
    mock_vector_store.get = AsyncMock(return_value=MemoryEntry(
        id=memory_id,
        content="Test",
        embedding=[0.1] * 384,
    ))
    
    summarizer.llm_client.complete = AsyncMock(return_value="Summary")
    
    # Mock vector store to fail
    mock_vector_store.store = AsyncMock(
        return_value=StoreResult(success=False, error="Storage failed")
    )
    
    # Should not raise exception (caught in outer try/except)
    await summarizer.update_summary(episode.id)
    
    # Episode summary should NOT be updated when storage fails (prevents data loss)
    updated_episode = episode_store.get_episode(episode.id)
    assert updated_episode.summary == ""  # Still empty due to storage failure


@pytest.mark.asyncio
async def test_embedding_failure_handled(summarizer, episode_store, mock_vector_store, mock_embedding_service):
    """Test embedding generation failure is handled."""
    episode = episode_store.create_episode("Test")
    memory_id = "mem-1"
    episode_store.add_memory(episode.id, memory_id)
    
    mock_vector_store.get = AsyncMock(return_value=MemoryEntry(
        id=memory_id,
        content="Test",
        embedding=[0.1] * 384,
    ))
    
    summarizer.llm_client.complete = AsyncMock(return_value="Summary")
    
    # Mock embedding service to fail
    mock_embedding_service.embed = AsyncMock(side_effect=Exception("Embedding failed"))
    
    # Should not raise exception
    await summarizer.update_summary(episode.id)


# ============================================================================
# Auto-close Tests
# ============================================================================

@pytest.mark.asyncio
async def test_auto_close_stale_episodes(summarizer, episode_store, mock_vector_store):
    """Test stale episodes are auto-closed with final summary."""
    # Create episode with old updated_at
    episode = episode_store.create_episode("Old Episode")
    memory_id = "mem-1"
    episode_store.add_memory(episode.id, memory_id)
    episode_store.mark_memories_summarized(episode.id, [memory_id])
    
    # Manually set updated_at to 20 days ago
    import sqlite3
    conn = episode_store._get_connection()
    old_date = (datetime.utcnow() - timedelta(days=20)).isoformat()
    conn.execute(
        "UPDATE episodes SET updated_at = ? WHERE id = ?",
        (old_date, episode.id)
    )
    conn.commit()
    
    # Mock vector store
    mock_vector_store.get = AsyncMock(return_value=MemoryEntry(
        id=memory_id,
        content="Content",
        embedding=[0.1] * 384,
    ))
    
    # Mock LLM
    summarizer.llm_client.complete = AsyncMock(return_value="Final summary")
    
    # Call auto-close
    closed_ids = await summarizer.close_stale_episodes()
    
    # Verify episode was closed
    assert episode.id in closed_ids
    
    updated_episode = episode_store.get_episode(episode.id)
    assert updated_episode.status == "closed"
    assert updated_episode.closed_at is not None
    assert updated_episode.summary == "Final summary"


@pytest.mark.asyncio
async def test_auto_close_respects_window(summarizer, episode_store):
    """Test auto-close only closes episodes outside window."""
    # Create recent episode (10 days ago - within 14 day window)
    recent = episode_store.create_episode("Recent Episode")
    episode_store.add_memory(recent.id, "mem-recent")
    
    import sqlite3
    conn = episode_store._get_connection()
    recent_date = (datetime.utcnow() - timedelta(days=10)).isoformat()
    conn.execute(
        "UPDATE episodes SET updated_at = ? WHERE id = ?",
        (recent_date, recent.id)
    )
    conn.commit()
    
    # Create stale episode (20 days ago - outside window)
    stale = episode_store.create_episode("Stale Episode")
    episode_store.add_memory(stale.id, "mem-stale")
    
    stale_date = (datetime.utcnow() - timedelta(days=20)).isoformat()
    conn.execute(
        "UPDATE episodes SET updated_at = ? WHERE id = ?",
        (stale_date, stale.id)
    )
    conn.commit()
    
    # Call auto-close
    closed_ids = await summarizer.close_stale_episodes()
    
    # Only stale should be closed
    assert stale.id in closed_ids
    assert recent.id not in closed_ids
    
    recent_episode = episode_store.get_episode(recent.id)
    stale_episode = episode_store.get_episode(stale.id)
    
    assert recent_episode.status == "active"
    assert stale_episode.status == "closed"


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_multiple_progressive_updates(summarizer, episode_store, mock_vector_store):
    """Test multiple progressive updates in sequence."""
    episode = episode_store.create_episode("Multi-update Episode")
    
    # Add and summarize 3 memories progressively
    for i in range(3):
        mem_id = f"mem-{i}"
        episode_store.add_memory(episode.id, mem_id)
        
        mock_vector_store.get = AsyncMock(return_value=MemoryEntry(
            id=mem_id,
            content=f"Content {i}",
            embedding=[0.1] * 384,
        ))
        
        summarizer.llm_client.complete = AsyncMock(
            return_value=f"Summary after memory {i}"
        )
        
        await summarizer.update_summary(episode.id)
        
        # Verify summary was updated
        updated = episode_store.get_episode(episode.id)
        assert f"memory {i}" in updated.summary.lower()


@pytest.mark.asyncio
async def test_empty_episode_no_crash(summarizer, episode_store):
    """Test handling of episode with no memories."""
    episode = episode_store.create_episode("Empty Episode")
    
    # Should not crash
    await summarizer.update_summary(episode.id)
    
    # Summary should remain empty
    updated = episode_store.get_episode(episode.id)
    assert updated.summary == ""


@pytest.mark.asyncio
async def test_summary_context_includes_episode_title(summarizer, episode_store, mock_vector_store):
    """Test that stored summary includes episode title in context."""
    episode = episode_store.create_episode("Test Title Here")
    memory_id = "mem-1"
    episode_store.add_memory(episode.id, memory_id)
    
    mock_vector_store.get = AsyncMock(return_value=MemoryEntry(
        id=memory_id,
        content="Test",
        embedding=[0.1] * 384,
    ))
    
    summarizer.llm_client.complete = AsyncMock(return_value="Summary")
    
    await summarizer.update_summary(episode.id)
    
    # Verify context includes episode title
    stored_entry = mock_vector_store.store.call_args[0][0]
    assert "Test Title Here" in stored_entry.context


@pytest.mark.asyncio
async def test_date_range_formatting(summarizer, episode_store):
    """Test _format_date_range helper method."""
    # Same day
    episode = episode_store.create_episode("Same Day")
    date_range = summarizer._format_date_range(episode)
    # Single date should have format YYYY-MM-DD (hyphens in date, but no " - " separator)
    assert " - " not in date_range  # No range separator
    assert date_range.count("-") == 2  # Only date hyphens
    
    # Different days
    import sqlite3
    conn = episode_store._get_connection()
    old_date = (datetime.utcnow() - timedelta(days=5)).isoformat()
    conn.execute(
        "UPDATE episodes SET created_at = ? WHERE id = ?",
        (old_date, episode.id)
    )
    conn.commit()
    
    episode = episode_store.get_episode(episode.id)
    date_range = summarizer._format_date_range(episode)
    assert " - " in date_range  # Date range with separator
    assert date_range.count("-") == 5  # 2 date hyphens + separator + 2 more date hyphens


# ============================================================================
# remember() Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_remember_integration_with_episodes(episode_store, mock_vector_store, mock_embedding_service, episode_config):
    """Test remember() calls episode detector and summarizer."""
    from tribalmemory.services.memory import TribalMemoryService
    from tribalmemory.services.episode_detector import EpisodeDetector, LLMClient
    
    # Create mock LLM client
    mock_llm = Mock(spec=LLMClient)
    mock_llm.complete = AsyncMock(return_value="Episode summary")
    
    # Create detector and summarizer
    detector = EpisodeDetector(
        episode_store=episode_store,
        embedding_service=mock_embedding_service,
        config=episode_config,
    )
    
    summarizer = EpisodeSummarizer(
        episode_store=episode_store,
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        llm_client=mock_llm,
        config=episode_config,
    )
    
    # Patch detector.detect to return an episode ID
    detector.detect = AsyncMock(return_value="test-episode-id")
    
    # Patch summarizer.update_summary to track calls
    summarizer.update_summary = AsyncMock()
    
    # Create memory service with episode support
    service = TribalMemoryService(
        instance_id="test",
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        episode_detector=detector,
        episode_summarizer=summarizer,
    )
    
    # Call remember
    result = await service.remember("Test memory content")
    
    # Verify remember succeeded
    assert result.success
    
    # Verify detector was called
    detector.detect.assert_called_once()
    
    # Verify summarizer was called with the episode ID
    summarizer.update_summary.assert_called_once_with("test-episode-id")


@pytest.mark.asyncio
async def test_remember_integration_no_episode_match(episode_store, mock_vector_store, mock_embedding_service, episode_config):
    """Test remember() when detector returns None (no episode)."""
    from tribalmemory.services.memory import TribalMemoryService
    from tribalmemory.services.episode_detector import EpisodeDetector, LLMClient
    
    mock_llm = Mock(spec=LLMClient)
    
    detector = EpisodeDetector(
        episode_store=episode_store,
        embedding_service=mock_embedding_service,
        config=episode_config,
    )
    
    summarizer = EpisodeSummarizer(
        episode_store=episode_store,
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        llm_client=mock_llm,
        config=episode_config,
    )
    
    # Patch detector to return None (no episode match)
    detector.detect = AsyncMock(return_value=None)
    summarizer.update_summary = AsyncMock()
    
    service = TribalMemoryService(
        instance_id="test",
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        episode_detector=detector,
        episode_summarizer=summarizer,
    )
    
    result = await service.remember("Test content")
    
    # Verify remember succeeded
    assert result.success
    
    # Verify detector was called
    detector.detect.assert_called_once()
    
    # Verify summarizer was NOT called (no episode)
    summarizer.update_summary.assert_not_called()


@pytest.mark.asyncio
async def test_remember_integration_episode_error_doesnt_fail(episode_store, mock_vector_store, mock_embedding_service, episode_config):
    """Test remember() doesn't fail when episode processing errors."""
    from tribalmemory.services.memory import TribalMemoryService
    from tribalmemory.services.episode_detector import EpisodeDetector, LLMClient
    
    mock_llm = Mock(spec=LLMClient)
    
    detector = EpisodeDetector(
        episode_store=episode_store,
        embedding_service=mock_embedding_service,
        config=episode_config,
    )
    
    summarizer = EpisodeSummarizer(
        episode_store=episode_store,
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        llm_client=mock_llm,
        config=episode_config,
    )
    
    # Make detector raise an exception
    detector.detect = AsyncMock(side_effect=Exception("Detector error"))
    
    service = TribalMemoryService(
        instance_id="test",
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        episode_detector=detector,
        episode_summarizer=summarizer,
    )
    
    # Should not raise exception - remember() should succeed
    result = await service.remember("Test content")
    
    # Verify remember still succeeded despite episode error
    assert result.success
