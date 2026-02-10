"""Episode summarization for Tribal Memory.

Generates progressive and full narrative summaries of episodes.
Summaries are stored as MemoryEntry objects with source_type=EPISODE_SUMMARY.

Design doc: docs/design/episode-memories.md
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from ..interfaces import (
    IEmbeddingService,
    IVectorStore,
    MemoryEntry,
    MemorySource,
)
from .episode_detector import EpisodeConfig, LLMClient
from .episode_store import Episode, EpisodeStore

logger = logging.getLogger(__name__)


def _format_id(id_: Optional[str]) -> str:
    """Format ID for logging (first 8 chars or 'None').
    
    Args:
        id_: ID string to format.
    
    Returns:
        First 8 characters of ID, or 'None' if ID is None.
    """
    return id_[:8] if id_ else "None"


class EpisodeSummarizer:
    """Generate and maintain episode summaries.
    
    Supports progressive summarization (incremental updates) and full
    regeneration (rebuild from all memories).
    
    Summaries are stored as MemoryEntry objects in the vector store with
    source_type=EPISODE_SUMMARY, making them automatically discoverable via
    standard recall() queries.
    
    Usage:
        summarizer = EpisodeSummarizer(
            episode_store=episode_store,
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_client=llm_client,
            config=config,
        )
        await summarizer.update_summary(episode_id)
    """
    
    # Progressive update prompt
    PROGRESSIVE_PROMPT = """Update this episode summary with new information.

Current summary:
{current_summary}

New memories:
{new_memories}

Write an updated narrative summary that:
1. Incorporates the new information naturally
2. Maintains chronological order
3. Preserves all specific details (names, numbers, dates, decisions)
4. Updates counts and aggregates as needed
5. Keeps the same format: title, date range, narrative, key details, status

Updated summary:"""
    
    # Full regeneration prompt
    FULL_REGEN_PROMPT = """Create a narrative summary of this episode from all constituent memories.

Episode title: {title}
Memories (chronological):
{all_memories}

Write a comprehensive narrative summary that:
1. Tells the story chronologically
2. Preserves all specific details (names, numbers, dates, decisions)
3. Includes counts and aggregates where relevant
4. Notes the current status (ongoing, completed, abandoned)

Format:
<title> (<date_range>): <narrative>

Key details: <important facts>

Status: <status>"""
    
    def __init__(
        self,
        episode_store: EpisodeStore,
        vector_store: IVectorStore,
        embedding_service: IEmbeddingService,
        llm_client: LLMClient,
        config: EpisodeConfig,
    ):
        """Initialize episode summarizer.
        
        Args:
            episode_store: Episode storage backend.
            vector_store: Vector store for summary memories.
            embedding_service: Embedding generation service.
            llm_client: LLM client for summary generation.
            config: Episode configuration.
        """
        self.episode_store = episode_store
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_client = llm_client
        self.config = config
    
    async def update_summary(self, episode_id: str) -> None:
        """Update episode summary with new memories.
        
        Determines whether to do progressive update or full regeneration
        based on memory count and config.full_regen_interval.
        
        Args:
            episode_id: Episode UUID.
        """
        # Get episode
        episode = self.episode_store.get_episode(episode_id)
        if not episode:
            logger.warning("Episode %s not found", episode_id)
            return
        
        # Get unsummarized memories
        unsummarized_ids = self.episode_store.get_unsummarized_memories(
            episode_id
        )
        if not unsummarized_ids:
            logger.debug("No unsummarized memories for episode %s", episode_id)
            return
        
        try:
            # Determine whether to do progressive update or full regeneration
            should_full_regen = (
                episode.memory_count % self.config.full_regen_interval == 0
                and episode.memory_count > 0
            )
            
            if should_full_regen:
                # Full regeneration: rebuild from all memories
                summary = await self._full_regeneration(episode)
            else:
                # Progressive update: current summary + new memories
                summary = await self._progressive_update(
                    episode, unsummarized_ids
                )
            
            if not summary:
                logger.warning(
                    "Summary generation failed for episode %s", episode_id
                )
                return
            
            # Store summary as MemoryEntry in vector store
            summary_memory_id = await self._store_summary_memory(
                episode, summary
            )
            
            # Update episode with new summary
            self.episode_store.update_episode(
                episode_id,
                summary=summary,
                summary_memory_id=summary_memory_id,
            )
            
            # Mark memories as summarized
            self.episode_store.mark_memories_summarized(
                episode_id, unsummarized_ids
            )
            
            logger.info(
                "Updated summary for episode %s (%d new memories)",
                _format_id(episode_id),
                len(unsummarized_ids),
            )
        
        except (ValueError, TypeError, AttributeError) as e:
            # Log with traceback for debugging
            logger.error(
                "Failed to update summary for episode %s: %s",
                episode_id[:8] if episode_id else "None",
                e,
                exc_info=True,
            )
        except Exception as e:
            # Catch unexpected errors but log with traceback
            logger.error(
                "Unexpected error updating summary for episode %s: %s",
                episode_id[:8] if episode_id else "None",
                e,
                exc_info=True,
            )
    
    async def regenerate_summary(self, episode_id: str) -> None:
        """Force full regeneration of episode summary.
        
        Rebuilds the summary from all constituent memories, regardless
        of full_regen_interval. Use for manual corrections or after
        bulk memory changes.
        
        Args:
            episode_id: Episode UUID.
        
        Raises:
            ValueError: If episode not found.
        """
        # Get episode
        episode = self.episode_store.get_episode(episode_id)
        if not episode:
            raise ValueError(f"Episode {episode_id} not found")
        
        try:
            # Force full regeneration
            summary = await self._full_regeneration(episode)
            
            if not summary:
                logger.warning(
                    "Summary generation failed for episode %s", episode_id
                )
                return
            
            # Store summary as MemoryEntry in vector store
            summary_memory_id = await self._store_summary_memory(
                episode, summary
            )
            
            # Update episode with new summary
            self.episode_store.update_episode(
                episode_id,
                summary=summary,
                summary_memory_id=summary_memory_id,
            )
            
            # Mark all memories as summarized
            all_memory_ids = self.episode_store.get_episode_memories(episode_id)
            if all_memory_ids:
                self.episode_store.mark_memories_summarized(
                    episode_id, all_memory_ids
                )
            
            logger.info(
                "Regenerated summary for episode %s (%d total memories)",
                _format_id(episode_id),
                episode.memory_count,
            )
        
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to regenerate summary for episode %s: %s",
                episode_id[:8] if episode_id else "None",
                e,
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error regenerating summary for episode %s: %s",
                episode_id[:8] if episode_id else "None",
                e,
                exc_info=True,
            )
            raise
    
    async def _progressive_update(
        self,
        episode: Episode,
        new_memory_ids: list[str],
    ) -> str:
        """Generate progressive summary update.
        
        Args:
            episode: Episode object.
            new_memory_ids: List of new memory IDs to incorporate.
        
        Returns:
            Updated summary text.
        """
        # Fetch new memory contents
        new_memories = []
        for memory_id in new_memory_ids:
            memory = await self.vector_store.get(memory_id)
            if memory:
                new_memories.append(memory.content)
        
        if not new_memories:
            return episode.summary or ""
        
        # Build prompt
        prompt = self.PROGRESSIVE_PROMPT.format(
            current_summary=episode.summary or "(no summary yet)",
            new_memories="\n".join(f"- {m}" for m in new_memories),
        )
        
        # Call LLM with configured temperature
        summary = await self.llm_client.complete(
            prompt, temperature=self.config.summarizer_temperature
        )
        return summary.strip()
    
    async def _full_regeneration(self, episode: Episode) -> str:
        """Generate full summary from all constituent memories.
        
        Args:
            episode: Episode object.
        
        Returns:
            Full regenerated summary text.
        """
        # Get all memory IDs for this episode
        all_memory_ids = self.episode_store.get_episode_memories(episode.id)
        
        if not all_memory_ids:
            return episode.summary
        
        # Fetch all memory contents
        all_memories = []
        for memory_id in all_memory_ids:
            memory = await self.vector_store.get(memory_id)
            if memory:
                timestamp = memory.created_at.strftime("%Y-%m-%d %H:%M")
                all_memories.append(f"[{timestamp}] {memory.content}")
        
        if not all_memories:
            return episode.summary or ""
        
        # Build prompt
        prompt = self.FULL_REGEN_PROMPT.format(
            title=episode.title,
            all_memories="\n".join(all_memories),
        )
        
        # Call LLM with configured temperature
        summary = await self.llm_client.complete(
            prompt, temperature=self.config.summarizer_temperature
        )
        return summary.strip()
    
    async def _store_summary_memory(
        self,
        episode: Episode,
        summary: str,
    ) -> str:
        """Store summary as MemoryEntry in vector store.
        
        Args:
            episode: Episode object.
            summary: Summary text.
        
        Returns:
            Memory ID of stored summary.
        """
        # Generate embedding for summary
        try:
            embedding = await self.embedding_service.embed(summary)
        except Exception as e:
            logger.error("Failed to generate embedding for summary: %s", e)
            embedding = None
        
        # Determine if this is an update or new summary
        if episode.summary_memory_id:
            # Update existing summary memory (upsert)
            summary_memory = MemoryEntry(
                id=episode.summary_memory_id,
                content=summary,
                embedding=embedding,
                source_instance="episode-summarizer",
                source_type=MemorySource.EPISODE_SUMMARY,
                created_at=episode.created_at,
                updated_at=datetime.utcnow(),
                tags=[f"episode:{episode.id}", "episode_summary"],
                context=f"Summary of episode: {episode.title}",
                confidence=1.0,
            )
            
            result = await self.vector_store.upsert(summary_memory)
        else:
            # Create new summary memory
            summary_memory = MemoryEntry(
                id=str(uuid.uuid4()),
                content=summary,
                embedding=embedding,
                source_instance="episode-summarizer",
                source_type=MemorySource.EPISODE_SUMMARY,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                tags=[f"episode:{episode.id}", "episode_summary"],
                context=f"Summary of episode: {episode.title}",
                confidence=1.0,
            )
            
            result = await self.vector_store.store(summary_memory)
        
        if not result.success:
            error_msg = f"Failed to store summary memory: {result.error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return result.memory_id
    
    async def close_stale_episodes(self) -> list[str]:
        """Auto-close stale episodes with final summary.
        
        Episodes are considered stale if they haven't been updated within
        config.active_window_days.
        
        Returns:
            List of closed episode IDs.
        """
        # Find stale episodes
        stale_ids = self.episode_store.close_stale_episodes(
            window_days=self.config.active_window_days
        )
        
        if not stale_ids:
            return []
        
        # Generate final summaries for closed episodes
        for episode_id in stale_ids:
            try:
                episode = self.episode_store.get_episode(episode_id)
                if not episode:
                    continue
                
                # Generate final full regeneration
                summary = await self._full_regeneration(episode)
                
                if summary:
                    # Store summary
                    summary_memory_id = await self._store_summary_memory(
                        episode, summary
                    )
                    
                    # Update episode
                    self.episode_store.update_episode(
                        episode_id,
                        summary=summary,
                        summary_memory_id=summary_memory_id,
                    )
                    
                    logger.info(
                        "Closed stale episode %s with final summary",
                        _format_id(episode_id),
                    )
            
            except Exception as e:
                logger.error(
                    "Failed to generate final summary for episode %s: %s",
                    _format_id(episode_id),
                    e,
                    exc_info=True,
                )
        
        return stale_ids
    
    def _format_date_range(self, episode: Episode) -> str:
        """Format episode date range.
        
        Args:
            episode: Episode object.
        
        Returns:
            Formatted date range string (e.g., "2026-01-15 - 2026-01-28").
        """
        start = episode.created_at.strftime("%Y-%m-%d")
        end = episode.updated_at.strftime("%Y-%m-%d")
        
        if start == end:
            return start
        else:
            return f"{start} - {end}"
