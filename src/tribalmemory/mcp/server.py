"""MCP server for Tribal Memory.

Exposes Tribal Memory as MCP tools for Claude Code and other MCP clients.
Uses stdio transport for integration with Claude Code.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from ..interfaces import MemorySource
from ..server.config import TribalMemoryConfig
from ..services import create_memory_service, TribalMemoryService

logger = logging.getLogger(__name__)

# Global service instance (initialized on first use)
_memory_service: Optional[TribalMemoryService] = None
_service_lock = asyncio.Lock()


async def get_memory_service() -> TribalMemoryService:
    """Get or create the memory service singleton (thread-safe)."""
    global _memory_service
    
    # Fast path: already initialized
    if _memory_service is not None:
        return _memory_service
    
    # Slow path: initialize with lock to prevent race conditions
    async with _service_lock:
        # Double-check after acquiring lock
        if _memory_service is not None:
            return _memory_service
            
        config = TribalMemoryConfig.from_env()

        # Override instance_id for MCP context
        instance_id = os.environ.get("TRIBAL_MEMORY_INSTANCE_ID", "mcp-claude-code")

        # Ensure db directory exists
        db_path = Path(config.db.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        _memory_service = create_memory_service(
            instance_id=instance_id,
            db_path=config.db.path,
            openai_api_key=config.embedding.api_key,
        )
        logger.info(f"Memory service initialized (instance: {instance_id}, db: {config.db.path})")

    return _memory_service


def create_server() -> FastMCP:
    """Create and configure the MCP server with all tools."""
    mcp = FastMCP("tribal-memory")

    @mcp.tool()
    async def tribal_remember(
        content: str,
        source_type: str = "auto_capture",
        context: Optional[str] = None,
        tags: Optional[list[str]] = None,
        skip_dedup: bool = False,
    ) -> str:
        """Store a new memory with semantic deduplication.

        Args:
            content: Memory content to store (required)
            source_type: How this memory was captured - one of:
                - "user_explicit": User explicitly asked to remember
                - "auto_capture": Automatically detected important info
                - "cross_instance": From another agent instance
            context: Additional context about when/why this was captured
            tags: Categorization tags for filtering (e.g., ["preferences", "work"])
            skip_dedup: If True, store even if a similar memory exists

        Returns:
            JSON with: success, memory_id, duplicate_of (if rejected), error
        """
        # Input validation
        if not content or not content.strip():
            return json.dumps({
                "success": False,
                "memory_id": None,
                "duplicate_of": None,
                "error": "Content cannot be empty",
            })
        
        service = await get_memory_service()

        # Map string to MemorySource enum
        source_map = {
            "user_explicit": MemorySource.USER_EXPLICIT,
            "auto_capture": MemorySource.AUTO_CAPTURE,
            "cross_instance": MemorySource.CROSS_INSTANCE,
        }
        source = source_map.get(source_type, MemorySource.AUTO_CAPTURE)

        result = await service.remember(
            content=content,
            source_type=source,
            context=context,
            tags=tags,
            skip_dedup=skip_dedup,
        )

        return json.dumps({
            "success": result.success,
            "memory_id": result.memory_id,
            "duplicate_of": result.duplicate_of,
            "error": result.error,
        })

    @mcp.tool()
    async def tribal_recall(
        query: str,
        limit: int = 5,
        min_relevance: float = 0.3,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Search memories by semantic similarity.

        Args:
            query: Natural language search query (required)
            limit: Maximum number of results (1-50, default 5)
            min_relevance: Minimum similarity score (0.0-1.0, default 0.3)
            tags: Filter results to only memories with these tags

        Returns:
            JSON with: results (list of memories with similarity scores), query, count
        """
        # Input validation
        if not query or not query.strip():
            return json.dumps({
                "results": [],
                "query": query,
                "count": 0,
                "error": "Query cannot be empty",
            })
        
        service = await get_memory_service()

        # Clamp limit to valid range
        limit = max(1, min(50, limit))
        min_relevance = max(0.0, min(1.0, min_relevance))

        results = await service.recall(
            query=query,
            limit=limit,
            min_relevance=min_relevance,
            tags=tags,
        )

        return json.dumps({
            "results": [
                {
                    "memory_id": r.memory.id,
                    "content": r.memory.content,
                    "similarity_score": round(r.similarity_score, 4),
                    "source_type": r.memory.source_type.value,
                    "source_instance": r.memory.source_instance,
                    "tags": r.memory.tags,
                    "created_at": r.memory.created_at.isoformat(),
                    "context": r.memory.context,
                }
                for r in results
            ],
            "query": query,
            "count": len(results),
        })

    @mcp.tool()
    async def tribal_correct(
        original_id: str,
        corrected_content: str,
        context: Optional[str] = None,
    ) -> str:
        """Update/correct an existing memory.

        Creates a new memory that supersedes the original. The original is
        preserved for audit trail but the new memory will be preferred in searches.

        Args:
            original_id: ID of the memory to correct (required)
            corrected_content: The corrected information (required)
            context: Why this correction was made

        Returns:
            JSON with: success, memory_id (of new correction), error
        """
        # Input validation
        if not original_id or not original_id.strip():
            return json.dumps({
                "success": False,
                "memory_id": None,
                "error": "Original ID cannot be empty",
            })
        if not corrected_content or not corrected_content.strip():
            return json.dumps({
                "success": False,
                "memory_id": None,
                "error": "Corrected content cannot be empty",
            })
        
        service = await get_memory_service()

        result = await service.correct(
            original_id=original_id,
            corrected_content=corrected_content,
            context=context,
        )

        return json.dumps({
            "success": result.success,
            "memory_id": result.memory_id,
            "error": result.error,
        })

    @mcp.tool()
    async def tribal_forget(memory_id: str) -> str:
        """Delete a memory (GDPR-compliant soft delete).

        Args:
            memory_id: ID of the memory to delete (required)

        Returns:
            JSON with: success, memory_id
        """
        # Input validation
        if not memory_id or not memory_id.strip():
            return json.dumps({
                "success": False,
                "memory_id": memory_id,
                "error": "Memory ID cannot be empty",
            })
        
        service = await get_memory_service()

        success = await service.forget(memory_id)

        return json.dumps({
            "success": success,
            "memory_id": memory_id,
        })

    @mcp.tool()
    async def tribal_stats() -> str:
        """Get memory statistics.

        Returns:
            JSON with: total_memories, by_source_type, by_tag, by_instance, corrections
        """
        service = await get_memory_service()

        stats = await service.get_stats()

        return json.dumps(stats)

    return mcp


def main():
    """Entry point for the MCP server."""
    import sys

    # Configure logging to stderr (stdout is for MCP protocol)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    mcp = create_server()
    mcp.run()


if __name__ == "__main__":
    main()
