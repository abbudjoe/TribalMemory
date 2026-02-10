"""Episode detection for Tribal Memory.

Hybrid detection strategy:
1. Fast path: Embedding similarity against active episode summaries (>0.75 = auto-join)
2. Slow path: LLM classification for edge cases

Design doc: docs/design/episode-memories.md
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import httpx

from ..interfaces import IEmbeddingService
from .episode_store import Episode, EpisodeStore

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EpisodeConfig:
    """Configuration for episode detection and summarization.
    
    Attributes:
        enabled: Feature flag for episode detection.
        detector_strategy: Detection strategy ("embedding", "llm", "hybrid").
        embedding_similarity_threshold: Threshold for fast-path auto-join (0.0-1.0).
        active_window_days: Days before episode auto-closes due to inactivity.
        max_active_episodes: Maximum concurrent active episodes.
        summarizer_model: LLM model for summary generation.
        summarizer_provider: LLM provider ("openai", "anthropic", "ollama").
        summarizer_temperature: Temperature for summary generation (0.0-2.0).
        full_regen_interval: Full summary regeneration every N memories.
        max_llm_calls_per_memory: Max LLM calls per remember() invocation.
        monthly_cost_ceiling: Pause episode processing if monthly cost exceeded.
    """
    enabled: bool = False
    detector_strategy: str = "hybrid"  # "embedding", "llm", "hybrid"
    embedding_similarity_threshold: float = 0.75
    active_window_days: int = 14
    max_active_episodes: int = 20
    summarizer_model: str = "gpt-4o-mini"
    summarizer_provider: str = "openai"  # "openai", "anthropic", "ollama"
    summarizer_temperature: float = 0.3
    full_regen_interval: int = 10
    max_llm_calls_per_memory: int = 2
    monthly_cost_ceiling: float = 5.0


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """Lightweight LLM client for episode detection and summarization.
    
    Supports:
    - OpenAI-compatible APIs (OpenAI, Ollama, etc.)
    - Anthropic API
    
    Usage:
        client = LLMClient(provider="openai", api_key="...", model="gpt-4o-mini")
        response = await client.complete("What is 2+2?")
    """
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize LLM client.
        
        Args:
            provider: Provider type ("openai", "anthropic", "ollama").
            api_key: API key for authentication. Falls back to env vars.
            model: Model name/ID.
            base_url: Custom base URL (for Ollama, etc.). Falls back to env var.
            timeout: Request timeout in seconds.
        """
        self.provider = provider.lower()
        self.model = model
        self.timeout = timeout
        
        # Resolve API key from args or environment
        if api_key:
            self.api_key = api_key
        elif self.provider == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            self.api_key = None
        
        # Anthropic API version (configurable)
        self.anthropic_version = os.environ.get("ANTHROPIC_API_VERSION", "2024-01-01")
        
        # Resolve base URL
        if base_url:
            self.base_url = base_url
        elif self.provider == "openai":
            self.base_url = os.environ.get(
                "EPISODE_LLM_BASE_URL",
                "https://api.openai.com/v1"
            )
        elif self.provider == "anthropic":
            self.base_url = "https://api.anthropic.com/v1"
        else:
            # Assume Ollama or custom provider
            self.base_url = os.environ.get(
                "EPISODE_LLM_BASE_URL",
                "http://localhost:11434/v1"
            )
    
    async def complete(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        """Generate LLM completion.
        
        Args:
            prompt: Text prompt.
            json_mode: If True, request JSON output format.
            temperature: Sampling temperature (0.0-2.0).
        
        Returns:
            LLM response text.
        
        Raises:
            Exception: On API errors or timeouts.
        """
        if self.provider == "anthropic":
            return await self._complete_anthropic(prompt, json_mode, temperature)
        else:
            return await self._complete_openai(prompt, json_mode, temperature)
    
    async def _complete_openai(
        self,
        prompt: str,
        json_mode: bool,
        temperature: float,
    ) -> str:
        """OpenAI-compatible completion."""
        url = f"{self.base_url}/chat/completions"
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    # Truncate error message to avoid exposing sensitive data
                    error_text = response.text[:200] if response.text else "No error message"
                    raise Exception(
                        f"LLM API error {response.status_code}: {error_text}"
                    )
                
                data = response.json()
                return data["choices"][0]["message"]["content"]
            
            except (httpx.TimeoutException, asyncio.TimeoutError):
                raise Exception("LLM request timed out")
            except Exception as e:
                if "LLM" in str(e):
                    raise
                raise Exception(f"LLM API error: {type(e).__name__}") from e
    
    async def _complete_anthropic(
        self,
        prompt: str,
        json_mode: bool,
        temperature: float,
    ) -> str:
        """Anthropic API completion."""
        url = f"{self.base_url}/messages"
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        
        # For JSON mode, append instruction to prompt
        if json_mode:
            prompt += "\n\nRespond with valid JSON only."
        
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    # Truncate error message to avoid exposing sensitive data
                    error_text = response.text[:200] if response.text else "No error message"
                    raise Exception(
                        f"LLM API error {response.status_code}: {error_text}"
                    )
                
                data = response.json()
                return data["content"][0]["text"]
            
            except (httpx.TimeoutException, asyncio.TimeoutError):
                raise Exception("LLM request timed out")
            except Exception as e:
                if "LLM" in str(e):
                    raise
                raise Exception(f"LLM API error: {type(e).__name__}") from e


# ============================================================================
# Episode Detector
# ============================================================================

class EpisodeDetector:
    """Detects whether memories belong to episodes.
    
    Hybrid detection strategy:
    1. Fast path: Embedding similarity to active episode summaries (>0.75 = auto-join)
    2. Slow path: LLM classification for borderline cases
    
    Usage:
        detector = EpisodeDetector(episode_store, embedding_service, config)
        episode_id = await detector.detect(memory_id, content, embedding)
    """
    
    # Classification prompt template
    CLASSIFY_PROMPT = """You are classifying a memory for episode detection.

Active episodes:
{episodes}

New memory:
"{content}"

Respond with JSON:
- If this memory belongs to an existing episode:
  {{"action": "join", "episode_id": "<id>", "reason": "<brief reason>"}}
- If this memory starts a new goal-directed activity:
  {{"action": "create", "title": "<episode title>", "reason": "<brief reason>"}}
- If this memory is standalone (a fact, preference, or one-off event):
  {{"action": "skip", "reason": "<brief reason>"}}

Rules:
- Episodes are GOAL-DIRECTED activities spanning multiple sessions
- Single facts, preferences, or one-off events are NOT episodes
- When in doubt, skip — we can always retroactively assign later"""
    
    def __init__(
        self,
        episode_store: EpisodeStore,
        embedding_service: IEmbeddingService,
        config: EpisodeConfig,
    ):
        """Initialize episode detector.
        
        Args:
            episode_store: Episode storage backend.
            embedding_service: Embedding generation service.
            config: Episode detection configuration.
        """
        self.episode_store = episode_store
        self.embedding_service = embedding_service
        self.config = config
        
        # Initialize LLM client if needed
        if config.detector_strategy in ("llm", "hybrid"):
            self._llm_client = LLMClient(
                provider=config.summarizer_provider,
                model=config.summarizer_model,
            )
        else:
            self._llm_client = None
    
    async def detect(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
    ) -> Optional[str]:
        """Detect episode for a memory.
        
        Returns episode_id if memory joined/created an episode, None otherwise.
        
        Args:
            memory_id: Memory UUID.
            content: Memory content text.
            embedding: Memory embedding vector.
        
        Returns:
            Episode ID if matched/created, None if standalone or error.
        """
        if not self.config.enabled:
            return None
        
        # Fast path: embedding similarity
        if self.config.detector_strategy in ("embedding", "hybrid"):
            match = await self._fast_match(embedding)
            if match:
                episode_id, similarity = match
                logger.info(
                    "Fast match: memory %s → episode %s (similarity %.3f)",
                    memory_id[:8], episode_id[:8], similarity
                )
                # Add memory to episode
                self.episode_store.add_memory(episode_id, memory_id)
                return episode_id
        
        # Slow path: LLM classification
        if self.config.detector_strategy in ("llm", "hybrid"):
            try:
                active_episodes = self.episode_store.get_active_episodes(
                    window_days=self.config.active_window_days
                )
                
                classification = await self._llm_classify(content, active_episodes)
                
                action = classification.get("action")
                
                if action == "join":
                    episode_id = classification.get("episode_id")
                    if episode_id:
                        logger.info(
                            "LLM join: memory %s → episode %s (%s)",
                            memory_id[:8], episode_id[:8],
                            classification.get("reason", "no reason")
                        )
                        self.episode_store.add_memory(episode_id, memory_id)
                        return episode_id
                
                elif action == "create":
                    title = classification.get("title")
                    if title:
                        logger.info(
                            "LLM create: memory %s → new episode '%s' (%s)",
                            memory_id[:8], title,
                            classification.get("reason", "no reason")
                        )
                        episode = self.episode_store.create_episode(title)
                        self.episode_store.add_memory(episode.id, memory_id)
                        return episode.id
                
                # "skip" or invalid action
                return None
            
            except Exception as e:
                logger.warning(
                    "Episode detection failed for %s: %s",
                    memory_id[:8], e
                )
                return None
        
        # No strategy matched
        return None
    
    async def _fast_match(
        self,
        embedding: list[float],
    ) -> Optional[tuple[str, float]]:
        """Fast embedding similarity match against active episodes.
        
        Args:
            embedding: Memory embedding vector.
        
        Returns:
            Tuple of (episode_id, similarity) if match found, None otherwise.
        """
        active_episodes = self.episode_store.get_active_episodes(
            window_days=self.config.active_window_days
        )
        
        if not active_episodes:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for episode in active_episodes:
            if not episode.summary:
                continue
            
            # Get episode summary embedding
            summary_embedding = await self.embedding_service.embed(episode.summary)
            
            # Calculate similarity
            similarity = self.embedding_service.similarity(embedding, summary_embedding)
            
            # Validate similarity (must be between 0.0 and 1.0)
            if similarity is None or not (0.0 <= similarity <= 1.0):
                logger.warning(
                    "Invalid similarity value %.3f for episode %s, skipping",
                    similarity if similarity is not None else float('nan'),
                    episode.id[:8]
                )
                continue
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = episode.id
        
        if best_similarity >= self.config.embedding_similarity_threshold:
            return (best_match, best_similarity)
        
        return None
    
    async def _llm_classify(
        self,
        content: str,
        active_episodes: list[Episode],
    ) -> dict:
        """LLM-based classification for edge cases.
        
        Args:
            content: Memory content text.
            active_episodes: List of active Episode objects.
        
        Returns:
            Classification dict with keys: action, episode_id/title, reason.
            Returns empty dict on error.
        """
        if not self._llm_client:
            return {}
        
        # Build prompt with episode context
        prompt = await self._build_classification_prompt(content, active_episodes)
        
        try:
            # Call LLM in JSON mode
            response = await self._llm_client.complete(prompt, json_mode=True)
            
            # Parse JSON response
            classification = json.loads(response)
            
            # Validate required fields
            action = classification.get("action")
            if action not in ("join", "create", "skip"):
                logger.warning("Invalid action in LLM response: %s", action)
                return {}
            
            if action == "join" and "episode_id" not in classification:
                logger.warning("Missing episode_id in join response")
                return {}
            
            if action == "create" and "title" not in classification:
                logger.warning("Missing title in create response")
                return {}
            
            return classification
        
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in LLM response: %s", e)
            return {}
        
        except Exception as e:
            logger.warning("LLM classification failed: %s", e)
            return {}
    
    async def _build_classification_prompt(
        self,
        content: str,
        episodes: list[Episode],
    ) -> str:
        """Build LLM classification prompt.
        
        Args:
            content: Memory content text.
            episodes: List of active Episode objects.
        
        Returns:
            Formatted prompt string.
        """
        if not episodes:
            episodes_text = "(No active episodes)"
        else:
            episodes_lines = []
            for ep in episodes:
                summary = ep.summary if ep.summary else "(no summary yet)"
                episodes_lines.append(
                    f"- Episode {ep.id}: {ep.title}\n  Summary: {summary}"
                )
            episodes_text = "\n".join(episodes_lines)
        
        # Sanitize content to mitigate prompt injection
        sanitized = content.replace("{", "{{").replace("}", "}}")
        return self.CLASSIFY_PROMPT.format(
            episodes=episodes_text,
            content=sanitized,
        )
    
    async def _check_retroactive_memories(
        self,
        episode_id: str,
        lookback_hours: int = 24,
    ) -> list[str]:
        """Check recent memories for retroactive assignment.
        
        When a new episode is created, check if recent memories should be
        assigned to it.
        
        TODO: Implement in Phase 3 with vector store integration.
        Requires querying recent memories and classifying them against
        the new episode via LLM.
        
        Args:
            episode_id: Newly created episode ID.
            lookback_hours: Hours to look back for candidates.
        
        Returns:
            List of memory IDs retroactively assigned.
        """
        return []
