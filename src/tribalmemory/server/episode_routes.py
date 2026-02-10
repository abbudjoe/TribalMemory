"""HTTP API routes for episode management."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..services import TribalMemoryService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/episodes", tags=["episodes"])


def get_memory_service() -> TribalMemoryService:
    """Dependency injection for memory service."""
    from .app import _memory_service
    if _memory_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _memory_service


# ============================================================================
# Request/Response Models
# ============================================================================

class EpisodeSummary(BaseModel):
    """Episode summary for list responses."""
    id: str
    title: str
    summary: str = Field(description="Truncated summary (max 200 chars)")
    status: str
    memory_count: int
    created_at: str
    updated_at: str
    closed_at: Optional[str] = None


class EpisodeDetail(BaseModel):
    """Full episode details."""
    id: str
    title: str
    summary: str
    summary_memory_id: Optional[str] = None
    status: str
    memory_count: int
    created_at: str
    updated_at: str
    closed_at: Optional[str] = None
    metadata: dict = {}


class ListEpisodesResponse(BaseModel):
    """Response for GET /episodes."""
    episodes: list[EpisodeSummary]
    count: int
    status: Optional[str] = None


class GetEpisodeResponse(BaseModel):
    """Response for GET /episodes/{id}."""
    episode: EpisodeDetail
    memory_ids: list[str]


class CreateEpisodeRequest(BaseModel):
    """Request for POST /episodes."""
    title: str = Field(min_length=1, description="Episode title")
    memory_ids: list[str] = Field(default_factory=list)


class CreateEpisodeResponse(BaseModel):
    """Response for POST /episodes."""
    success: bool
    episode_id: Optional[str] = None
    memory_count: int = 0
    error: Optional[str] = None


class AddMemoryRequest(BaseModel):
    """Request for POST /episodes/{id}/memories."""
    memory_id: str = Field(min_length=1)


class GenericResponse(BaseModel):
    """Generic success/error response."""
    success: bool
    error: Optional[str] = None


# ============================================================================
# Route Handlers
# ============================================================================

@router.get("", response_model=ListEpisodesResponse)
async def list_episodes(
    status: Optional[str] = None,
    limit: int = 50,
    service: TribalMemoryService = Depends(get_memory_service),
) -> ListEpisodesResponse:
    """List episodes with optional filtering.
    
    Args:
        status: Filter by status (active, closed, archived).
        limit: Maximum number of results (1-100, default 50).
    
    Returns:
        List of episodes with summary information.
    """
    if not service.episode_detector:
        raise HTTPException(status_code=501, detail="Episode feature not enabled")
    
    # Validate status
    if status and status not in ("active", "closed", "archived"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: {status}. Valid: active, closed, archived"
        )
    
    # Clamp limit
    limit = max(1, min(100, limit))
    
    try:
        episode_store = service.episode_detector.episode_store
        episodes = episode_store.list_episodes(status=status, limit=limit)
        
        return ListEpisodesResponse(
            episodes=[
                EpisodeSummary(
                    id=ep.id,
                    title=ep.title,
                    summary=ep.summary[:200] + "..." if len(ep.summary) > 200 else ep.summary,
                    status=ep.status,
                    memory_count=ep.memory_count,
                    created_at=ep.created_at.isoformat(),
                    updated_at=ep.updated_at.isoformat(),
                    closed_at=ep.closed_at.isoformat() if ep.closed_at else None,
                )
                for ep in episodes
            ],
            count=len(episodes),
            status=status,
        )
    
    except Exception as e:
        logger.exception("Failed to list episodes")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{episode_id}", response_model=GetEpisodeResponse)
async def get_episode(
    episode_id: str,
    service: TribalMemoryService = Depends(get_memory_service),
) -> GetEpisodeResponse:
    """Get episode details with full summary and memory IDs.
    
    Args:
        episode_id: Episode UUID.
    
    Returns:
        Full episode details and constituent memory IDs.
    """
    if not service.episode_detector:
        raise HTTPException(status_code=501, detail="Episode feature not enabled")
    
    try:
        episode_store = service.episode_detector.episode_store
        episode = episode_store.get_episode(episode_id)
        
        if not episode:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
        
        memory_ids = episode_store.get_episode_memories(episode_id)
        
        return GetEpisodeResponse(
            episode=EpisodeDetail(
                id=episode.id,
                title=episode.title,
                summary=episode.summary,
                summary_memory_id=episode.summary_memory_id,
                status=episode.status,
                memory_count=episode.memory_count,
                created_at=episode.created_at.isoformat(),
                updated_at=episode.updated_at.isoformat(),
                closed_at=episode.closed_at.isoformat() if episode.closed_at else None,
                metadata=episode.metadata,
            ),
            memory_ids=memory_ids,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get episode")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=CreateEpisodeResponse)
async def create_episode(
    request: CreateEpisodeRequest,
    service: TribalMemoryService = Depends(get_memory_service),
) -> CreateEpisodeResponse:
    """Manually create an episode from existing memories.
    
    Creates a new episode, adds the specified memories, and triggers
    initial summary generation.
    
    Args:
        request: Episode creation request with title and memory IDs.
    
    Returns:
        Created episode ID and memory count.
    """
    if not service.episode_detector:
        raise HTTPException(status_code=501, detail="Episode feature not enabled")
    
    try:
        episode_store = service.episode_detector.episode_store
        
        # Create episode
        episode = episode_store.create_episode(request.title.strip())
        
        # Add memories
        added_count = 0
        for memory_id in request.memory_ids:
            if episode_store.add_memory(episode.id, memory_id):
                added_count += 1
        
        # Trigger initial summary if memories added
        if added_count > 0 and service.episode_summarizer:
            try:
                await service.episode_summarizer.update_summary(episode.id)
            except Exception as e:
                logger.warning(f"Failed to generate initial summary for {episode.id}: {e}")
        
        return CreateEpisodeResponse(
            success=True,
            episode_id=episode.id,
            memory_count=added_count,
        )
    
    except Exception as e:
        logger.exception("Failed to create episode")
        return CreateEpisodeResponse(
            success=False,
            error=str(e),
        )


@router.post("/{episode_id}/memories", response_model=GenericResponse)
async def add_memory_to_episode(
    episode_id: str,
    request: AddMemoryRequest,
    service: TribalMemoryService = Depends(get_memory_service),
) -> GenericResponse:
    """Add a memory to an episode.
    
    Adds the memory and triggers progressive summary update.
    
    Args:
        episode_id: Episode UUID.
        request: Memory ID to add.
    
    Returns:
        Success status.
    """
    if not service.episode_detector:
        raise HTTPException(status_code=501, detail="Episode feature not enabled")
    
    try:
        episode_store = service.episode_detector.episode_store
        
        # Verify episode exists
        episode = episode_store.get_episode(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
        
        # Add memory
        added = episode_store.add_memory(episode_id, request.memory_id)
        
        # Trigger summary update if newly added
        if added and service.episode_summarizer:
            try:
                await service.episode_summarizer.update_summary(episode_id)
            except Exception as e:
                logger.warning(f"Failed to update summary for {episode_id}: {e}")
        
        return GenericResponse(success=True)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to add memory to episode")
        return GenericResponse(success=False, error=str(e))


@router.delete("/{episode_id}/memories/{memory_id}", response_model=GenericResponse)
async def remove_memory_from_episode(
    episode_id: str,
    memory_id: str,
    service: TribalMemoryService = Depends(get_memory_service),
) -> GenericResponse:
    """Remove a memory from an episode.
    
    Args:
        episode_id: Episode UUID.
        memory_id: Memory UUID to remove.
    
    Returns:
        Success status.
    """
    if not service.episode_detector:
        raise HTTPException(status_code=501, detail="Episode feature not enabled")
    
    try:
        episode_store = service.episode_detector.episode_store
        
        # Remove memory
        removed = episode_store.remove_memory(episode_id, memory_id)
        
        if not removed:
            return GenericResponse(
                success=False,
                error=f"Memory {memory_id} not found in episode {episode_id}"
            )
        
        return GenericResponse(success=True)
    
    except Exception as e:
        logger.exception("Failed to remove memory from episode")
        return GenericResponse(success=False, error=str(e))


@router.post("/{episode_id}/close", response_model=GenericResponse)
async def close_episode(
    episode_id: str,
    service: TribalMemoryService = Depends(get_memory_service),
) -> GenericResponse:
    """Close an episode.
    
    Sets status to 'closed', sets closed_at timestamp, and triggers
    final full summary regeneration.
    
    Args:
        episode_id: Episode UUID.
    
    Returns:
        Success status.
    """
    if not service.episode_detector:
        raise HTTPException(status_code=501, detail="Episode feature not enabled")
    
    try:
        episode_store = service.episode_detector.episode_store
        
        # Close episode
        episode = episode_store.close_episode(episode_id)
        
        # Trigger final full regeneration
        if service.episode_summarizer:
            try:
                await service.episode_summarizer.regenerate_summary(episode_id)
            except Exception as e:
                logger.warning(f"Failed to regenerate summary for {episode_id}: {e}")
        
        return GenericResponse(success=True)
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to close episode")
        return GenericResponse(success=False, error=str(e))


@router.post("/{episode_id}/regenerate", response_model=GenericResponse)
async def regenerate_episode_summary(
    episode_id: str,
    service: TribalMemoryService = Depends(get_memory_service),
) -> GenericResponse:
    """Force full summary regeneration for an episode.
    
    Rebuilds the summary from all constituent memories (expensive).
    Use for manual corrections or after bulk memory changes.
    
    Args:
        episode_id: Episode UUID.
    
    Returns:
        Success status.
    """
    if not service.episode_detector:
        raise HTTPException(status_code=501, detail="Episode feature not enabled")
    
    try:
        await service.episode_summarizer.regenerate_summary(episode_id)
        return GenericResponse(success=True)
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to regenerate episode summary")
        return GenericResponse(success=False, error=str(e))
