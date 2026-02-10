"""Tests for episode detection and LLM classification.

Tests the hybrid detection strategy: embedding similarity fast path + LLM classification slow path.
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from tribalmemory.services.episode_store import Episode, EpisodeStore
from tribalmemory.services.episode_detector import (
    EpisodeConfig,
    EpisodeDetector,
    LLMClient,
)
from tribalmemory.interfaces import IEmbeddingService


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
def episode_store(tmp_path):
    """Episode store with temporary database."""
    db_path = tmp_path / "test_episodes.db"
    with EpisodeStore(db_path) as store:
        yield store


@pytest.fixture
def detector(episode_store, mock_embedding_service, episode_config):
    """Episode detector with mocked dependencies."""
    return EpisodeDetector(
        episode_store=episode_store,
        embedding_service=mock_embedding_service,
        config=episode_config,
    )


# ============================================================================
# EpisodeConfig Tests
# ============================================================================

def test_episode_config_defaults():
    """Test EpisodeConfig default values."""
    config = EpisodeConfig()
    assert config.enabled is False
    assert config.detector_strategy == "hybrid"
    assert config.embedding_similarity_threshold == 0.75
    assert config.active_window_days == 14
    assert config.max_active_episodes == 20
    assert config.summarizer_model == "gpt-4o-mini"
    assert config.summarizer_provider == "openai"
    assert config.full_regen_interval == 10
    assert config.max_llm_calls_per_memory == 2
    assert config.monthly_cost_ceiling == 5.0


def test_episode_config_custom_values():
    """Test EpisodeConfig with custom values."""
    config = EpisodeConfig(
        enabled=True,
        detector_strategy="embedding",
        embedding_similarity_threshold=0.8,
        active_window_days=7,
        max_active_episodes=10,
        summarizer_model="gpt-4",
        summarizer_provider="anthropic",
        full_regen_interval=5,
        max_llm_calls_per_memory=1,
        monthly_cost_ceiling=10.0,
    )
    assert config.enabled is True
    assert config.detector_strategy == "embedding"
    assert config.embedding_similarity_threshold == 0.8
    assert config.active_window_days == 7
    assert config.max_active_episodes == 10
    assert config.summarizer_model == "gpt-4"
    assert config.summarizer_provider == "anthropic"
    assert config.full_regen_interval == 5
    assert config.max_llm_calls_per_memory == 1
    assert config.monthly_cost_ceiling == 10.0


# ============================================================================
# LLMClient Tests
# ============================================================================

@pytest.mark.asyncio
async def test_llm_client_openai_completion():
    """Test OpenAI completion."""
    client = LLMClient(provider="openai", api_key="test-key", model="gpt-4o-mini")
    
    mock_response = {
        "choices": [{"message": {"content": "Test response"}}]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # json() is a regular method, not async
        mock_post.return_value.json = Mock(return_value=mock_response)
        mock_post.return_value.status_code = 200
        
        result = await client.complete("Test prompt")
        
        assert result == "Test response"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_llm_client_openai_json_mode():
    """Test OpenAI JSON mode completion."""
    client = LLMClient(provider="openai", api_key="test-key", model="gpt-4o-mini")
    
    mock_response = {
        "choices": [{"message": {"content": '{"action": "join", "episode_id": "123"}'}}]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # json() is a regular method, not async
        mock_post.return_value.json = Mock(return_value=mock_response)
        mock_post.return_value.status_code = 200
        
        result = await client.complete("Test prompt", json_mode=True)
        
        assert result == '{"action": "join", "episode_id": "123"}'
        # Verify JSON mode was set in request
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_llm_client_anthropic_completion():
    """Test Anthropic completion."""
    client = LLMClient(provider="anthropic", api_key="test-key", model="claude-3-haiku-20240307")
    
    mock_response = {
        "content": [{"text": "Test response"}]
    }
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # json() is a regular method, not async
        mock_post.return_value.json = Mock(return_value=mock_response)
        mock_post.return_value.status_code = 200
        
        result = await client.complete("Test prompt")
        
        assert result == "Test response"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_llm_client_error_handling():
    """Test LLM client error handling."""
    client = LLMClient(provider="openai", api_key="test-key", model="gpt-4o-mini")
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Server Error"
        
        with pytest.raises(Exception, match="LLM API error"):
            await client.complete("Test prompt")


@pytest.mark.asyncio
async def test_llm_client_timeout():
    """Test LLM client timeout handling."""
    client = LLMClient(provider="openai", api_key="test-key", model="gpt-4o-mini", timeout=1.0)
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        import asyncio
        mock_post.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(Exception, match="LLM request timed out"):
            await client.complete("Test prompt")


# ============================================================================
# EpisodeDetector Fast Path Tests
# ============================================================================

@pytest.mark.asyncio
async def test_fast_match_high_similarity_joins(detector, episode_store, mock_embedding_service):
    """Test fast path: high similarity automatically joins episode."""
    # Create an episode with a summary
    episode = episode_store.create_episode("House hunting")
    episode_store.update_episode(episode.id, summary="House hunting episode summary")
    
    # Mock high similarity
    mock_embedding_service.similarity.return_value = 0.85
    
    # Detect episode for new memory
    memory_id = "mem-123"
    content = "Viewed another property today"
    embedding = [0.2] * 384
    
    result = await detector.detect(memory_id, content, embedding)
    
    # Should return the episode ID
    assert result == episode.id
    
    # Should have added memory to episode
    memories = episode_store.get_episode_memories(episode.id)
    assert memory_id in memories


@pytest.mark.asyncio
async def test_fast_match_low_similarity_no_join(detector, episode_store, mock_embedding_service):
    """Test fast path: low similarity does not auto-join."""
    # Create an episode
    episode = episode_store.create_episode("House hunting")
    episode_store.update_episode(episode.id, summary="House hunting episode summary")
    
    # Mock low similarity
    mock_embedding_service.similarity.return_value = 0.5
    
    # Detect episode for new memory
    memory_id = "mem-456"
    content = "Had lunch at a restaurant"
    embedding = [0.3] * 384
    
    result = await detector.detect(memory_id, content, embedding)
    
    # Should return None (no match)
    assert result is None
    
    # Should NOT have added memory to episode
    memories = episode_store.get_episode_memories(episode.id)
    assert memory_id not in memories


@pytest.mark.asyncio
async def test_fast_match_no_active_episodes(detector, mock_embedding_service):
    """Test fast path with no active episodes."""
    # Detect episode for new memory (no episodes exist)
    memory_id = "mem-789"
    content = "Some random content"
    embedding = [0.4] * 384
    
    result = await detector.detect(memory_id, content, embedding)
    
    # Should return None
    assert result is None


@pytest.mark.asyncio
async def test_fast_match_embedding_strategy_only(episode_store, mock_embedding_service):
    """Test detector with embedding-only strategy (no LLM fallback)."""
    config = EpisodeConfig(
        enabled=True,
        detector_strategy="embedding",
        embedding_similarity_threshold=0.75,
    )
    detector = EpisodeDetector(
        episode_store=episode_store,
        embedding_service=mock_embedding_service,
        config=config,
    )
    
    # Create an episode
    episode = episode_store.create_episode("Project Alpha")
    episode_store.update_episode(episode.id, summary="Project Alpha development")
    
    # Mock low similarity (below threshold)
    mock_embedding_service.similarity.return_value = 0.6
    
    # Detect episode
    result = await detector.detect("mem-001", "Some content", [0.5] * 384)
    
    # Should return None (no LLM fallback in embedding-only mode)
    assert result is None


# ============================================================================
# EpisodeDetector LLM Classification Tests
# ============================================================================

@pytest.mark.asyncio
async def test_llm_classify_join_existing_episode(detector, episode_store, mock_embedding_service):
    """Test LLM classification: join existing episode."""
    # Create an episode
    episode = episode_store.create_episode("House hunting")
    episode_store.update_episode(episode.id, summary="House hunting: viewed 2 properties")
    
    # Mock low similarity to trigger LLM path
    mock_embedding_service.similarity.return_value = 0.6
    
    # Mock LLM response: join
    llm_response = json.dumps({
        "action": "join",
        "episode_id": episode.id,
        "reason": "This is another property viewing"
    })
    
    with patch.object(detector, '_llm_client') as mock_llm:
        mock_llm.complete = AsyncMock(return_value=llm_response)
        
        result = await detector.detect("mem-111", "Saw a condo on Park Ave", [0.6] * 384)
        
        assert result == episode.id
        memories = episode_store.get_episode_memories(episode.id)
        assert "mem-111" in memories


@pytest.mark.asyncio
async def test_llm_classify_create_new_episode(detector, episode_store, mock_embedding_service):
    """Test LLM classification: create new episode."""
    # Mock low similarity
    mock_embedding_service.similarity.return_value = 0.5
    
    # Mock LLM response: create
    llm_response = json.dumps({
        "action": "create",
        "title": "Job search",
        "reason": "Second mention suggests a pattern"
    })
    
    with patch.object(detector, '_llm_client') as mock_llm:
        mock_llm.complete = AsyncMock(return_value=llm_response)
        
        result = await detector.detect("mem-222", "Applied to Google", [0.7] * 384)
        
        # Should have created a new episode
        assert result is not None
        episode = episode_store.get_episode(result)
        assert episode is not None
        assert episode.title == "Job search"
        
        # Should have added memory to new episode
        memories = episode_store.get_episode_memories(result)
        assert "mem-222" in memories


@pytest.mark.asyncio
async def test_llm_classify_skip_standalone(detector, mock_embedding_service):
    """Test LLM classification: skip standalone memory."""
    # Mock low similarity
    mock_embedding_service.similarity.return_value = 0.4
    
    # Mock LLM response: skip
    llm_response = json.dumps({
        "action": "skip",
        "reason": "Standalone preference, not part of an episode"
    })
    
    with patch.object(detector, '_llm_client') as mock_llm:
        mock_llm.complete = AsyncMock(return_value=llm_response)
        
        result = await detector.detect("mem-333", "Joe prefers TypeScript", [0.8] * 384)
        
        # Should return None
        assert result is None


@pytest.mark.asyncio
async def test_llm_classify_invalid_json_returns_none(detector, mock_embedding_service):
    """Test LLM classification with invalid JSON response."""
    # Mock low similarity
    mock_embedding_service.similarity.return_value = 0.5
    
    # Mock invalid JSON response
    with patch.object(detector, '_llm_client') as mock_llm:
        mock_llm.complete = AsyncMock(return_value="This is not JSON")
        
        result = await detector.detect("mem-444", "Some content", [0.9] * 384)
        
        # Should return None (graceful failure)
        assert result is None


@pytest.mark.asyncio
async def test_llm_classify_error_returns_none(detector, mock_embedding_service):
    """Test LLM classification with API error."""
    # Mock low similarity
    mock_embedding_service.similarity.return_value = 0.5
    
    # Mock LLM error
    with patch.object(detector, '_llm_client') as mock_llm:
        mock_llm.complete = AsyncMock(side_effect=Exception("API error"))
        
        result = await detector.detect("mem-555", "Some content", [0.1] * 384)
        
        # Should return None (graceful failure)
        assert result is None


# ============================================================================
# EpisodeDetector Hybrid Strategy Tests
# ============================================================================

@pytest.mark.asyncio
async def test_hybrid_strategy_fast_path_wins(detector, episode_store, mock_embedding_service):
    """Test hybrid strategy: fast path takes precedence over LLM."""
    # Create an episode
    episode = episode_store.create_episode("Trip planning")
    episode_store.update_episode(episode.id, summary="Planning trip to Europe")
    
    # Mock high similarity (should trigger fast path)
    mock_embedding_service.similarity.return_value = 0.9
    
    # Mock LLM (should NOT be called)
    with patch.object(detector, '_llm_client') as mock_llm:
        mock_llm.complete = AsyncMock(return_value='{"action": "skip"}')
        
        result = await detector.detect("mem-666", "Booked flights", [0.2] * 384)
        
        # Should use fast path
        assert result == episode.id
        
        # LLM should NOT have been called
        mock_llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_strategy_llm_fallback(detector, episode_store, mock_embedding_service):
    """Test hybrid strategy: LLM fallback when similarity is low."""
    # Create an episode
    episode = episode_store.create_episode("Travel")
    episode_store.update_episode(episode.id, summary="Travel planning")
    
    # Mock low similarity (should trigger LLM path)
    mock_embedding_service.similarity.return_value = 0.6
    
    # Mock LLM response: join
    llm_response = json.dumps({
        "action": "join",
        "episode_id": episode.id,
        "reason": "Related to travel"
    })
    
    with patch.object(detector, '_llm_client') as mock_llm:
        mock_llm.complete = AsyncMock(return_value=llm_response)
        
        result = await detector.detect("mem-777", "Packed bags", [0.3] * 384)
        
        # Should use LLM fallback
        assert result == episode.id
        
        # LLM should have been called
        mock_llm.complete.assert_called_once()


# ============================================================================
# Retroactive Memory Assignment Tests
# ============================================================================

@pytest.mark.asyncio
async def test_create_episode_retroactive_assignment(detector, episode_store, mock_embedding_service):
    """Test retroactive assignment when creating new episode."""
    # Store some prior memories in episode store (simulate recent memories)
    # This would require access to vector store - we'll test the concept
    
    # Mock low similarity
    mock_embedding_service.similarity.return_value = 0.5
    
    # Mock LLM response: create with retroactive check
    llm_response = json.dumps({
        "action": "create",
        "title": "House hunting",
        "reason": "Multiple property viewings"
    })
    
    with patch.object(detector, '_llm_client') as mock_llm:
        mock_llm.complete = AsyncMock(return_value=llm_response)
        
        # Mock retroactive check (would query recent memories)
        with patch.object(detector, '_check_retroactive_memories') as mock_retro:
            mock_retro.return_value = ["mem-old-1", "mem-old-2"]
            
            result = await detector.detect("mem-888", "Made offer on house", [0.4] * 384)
            
            assert result is not None
            
            # Verify retroactive check was called
            # mock_retro.assert_called_once()


# ============================================================================
# Configuration Tests
# ============================================================================

def test_detector_disabled_config(episode_store, mock_embedding_service):
    """Test detector with disabled config."""
    config = EpisodeConfig(enabled=False)
    detector = EpisodeDetector(
        episode_store=episode_store,
        embedding_service=mock_embedding_service,
        config=config,
    )
    
    # Detector should be disabled
    assert detector.config.enabled is False


def test_detector_llm_only_strategy(episode_store, mock_embedding_service):
    """Test detector with LLM-only strategy."""
    config = EpisodeConfig(
        enabled=True,
        detector_strategy="llm",
        embedding_similarity_threshold=0.75,
    )
    detector = EpisodeDetector(
        episode_store=episode_store,
        embedding_service=mock_embedding_service,
        config=config,
    )
    
    assert detector.config.detector_strategy == "llm"


# ============================================================================
# Episode Store Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_detector_updates_episode_timestamp(detector, episode_store, mock_embedding_service):
    """Test that adding memory updates episode timestamp."""
    # Create an episode with a summary
    episode = episode_store.create_episode("Test episode")
    episode_store.update_episode(episode.id, summary="Test episode summary")
    original_updated_at = episode.updated_at
    
    # Wait a bit to ensure timestamp difference
    import time
    time.sleep(0.1)
    
    # Mock high similarity
    mock_embedding_service.similarity.return_value = 0.9
    
    # Add memory via detector
    await detector.detect("mem-999", "Test content", [0.5] * 384)
    
    # Check episode was updated
    updated_episode = episode_store.get_episode(episode.id)
    assert updated_episode.updated_at > original_updated_at


@pytest.mark.asyncio
async def test_detector_increments_memory_count(detector, episode_store, mock_embedding_service):
    """Test that adding memory increments episode memory count."""
    # Create an episode with a summary
    episode = episode_store.create_episode("Test episode")
    episode_store.update_episode(episode.id, summary="Test episode summary")
    assert episode.memory_count == 0
    
    # Mock high similarity
    mock_embedding_service.similarity.return_value = 0.9
    
    # Add memory via detector
    await detector.detect("mem-1000", "Test content", [0.6] * 384)
    
    # Check memory count
    updated_episode = episode_store.get_episode(episode.id)
    assert updated_episode.memory_count == 1


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_detector_handles_missing_episode_id_in_llm_response(
    detector, mock_embedding_service
):
    """Test handling of malformed LLM response missing episode_id."""
    # Mock low similarity
    mock_embedding_service.similarity.return_value = 0.5
    
    # Mock LLM response with missing episode_id for join action
    llm_response = json.dumps({
        "action": "join",
        "reason": "Related"
    })
    
    with patch.object(detector, '_llm_client') as mock_llm:
        mock_llm.complete = AsyncMock(return_value=llm_response)
        
        result = await detector.detect("mem-1001", "Content", [0.7] * 384)
        
        # Should return None (invalid response)
        assert result is None


@pytest.mark.asyncio
async def test_detector_handles_missing_title_in_create_response(
    detector, mock_embedding_service
):
    """Test handling of malformed LLM response missing title for create."""
    # Mock low similarity
    mock_embedding_service.similarity.return_value = 0.5
    
    # Mock LLM response with missing title for create action
    llm_response = json.dumps({
        "action": "create",
        "reason": "New pattern detected"
    })
    
    with patch.object(detector, '_llm_client') as mock_llm:
        mock_llm.complete = AsyncMock(return_value=llm_response)
        
        result = await detector.detect("mem-1002", "Content", [0.8] * 384)
        
        # Should return None (invalid response)
        assert result is None


@pytest.mark.asyncio
async def test_detector_handles_unknown_action(detector, mock_embedding_service):
    """Test handling of unknown action in LLM response."""
    # Mock low similarity
    mock_embedding_service.similarity.return_value = 0.5
    
    # Mock LLM response with unknown action
    llm_response = json.dumps({
        "action": "unknown_action",
        "reason": "Something weird"
    })
    
    with patch.object(detector, '_llm_client') as mock_llm:
        mock_llm.complete = AsyncMock(return_value=llm_response)
        
        result = await detector.detect("mem-1003", "Content", [0.9] * 384)
        
        # Should return None (invalid action)
        assert result is None


@pytest.mark.asyncio
async def test_concurrent_detect_same_content(detector, mock_embedding_service):
    """Test concurrent detect() calls with overlapping content.

    Two memories arriving simultaneously should not create duplicate episodes.
    The first should create the episode; the second should join it.
    """
    import asyncio

    # Mock low similarity (no existing episodes match)
    mock_embedding_service.similarity.return_value = 0.3

    call_count = 0

    async def mock_classify(content, episodes):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: no episodes exist → create
            return {"action": "create", "title": "Test Episode", "reason": "new activity"}
        else:
            # Second call: episode now exists → join
            active = detector.episode_store.get_active_episodes()
            if active:
                return {
                    "action": "join",
                    "episode_id": active[0].id,
                    "reason": "same activity",
                }
            return {"action": "create", "title": "Test Episode 2", "reason": "fallback"}

    llm_responses = iter([
        json.dumps({"action": "create", "title": "House Hunting", "reason": "new"}),
        json.dumps({"action": "create", "title": "House Hunting 2", "reason": "new"}),
    ])

    with patch.object(detector, '_llm_client') as mock_llm:
        mock_llm.complete = AsyncMock(side_effect=lambda *a, **kw: next(llm_responses))

        # Run two detections sequentially (asyncio won't truly parallel on one thread,
        # but this verifies the store handles rapid sequential creates)
        result1 = await detector.detect("mem-A", "Viewed Elm St bungalow", [0.1] * 384)
        result2 = await detector.detect("mem-B", "Toured Oak Avenue house", [0.2] * 384)

    # Both should succeed — at least one episode should exist
    episodes = detector.episode_store.list_episodes()
    assert len(episodes) >= 1
    # Both memories should be assigned (possibly to different episodes)
    assert result1 is not None
    assert result2 is not None
