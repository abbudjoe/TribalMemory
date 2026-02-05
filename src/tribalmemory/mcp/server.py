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
from ..services.session_store import (
    SessionStore,
    SessionMessage,
    LanceDBSessionStore,
    InMemorySessionStore,
)

logger = logging.getLogger(__name__)

# Global service instance (initialized on first use)
_memory_service: Optional[TribalMemoryService] = None
_session_store: Optional[SessionStore] = None
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
            api_base=config.embedding.api_base,
            embedding_model=config.embedding.model,
            embedding_dimensions=config.embedding.dimensions,
            embedding_provider=config.embedding.provider,
        )
        logger.info(f"Memory service initialized (instance: {instance_id}, db: {config.db.path})")

    return _memory_service


async def get_session_store() -> SessionStore:
    """Get or create the session store singleton (thread-safe)."""
    global _session_store
    
    if _session_store is not None:
        return _session_store
    
    memory_service = await get_memory_service()
    
    async with _service_lock:
        if _session_store is not None:
            return _session_store
        
        config = TribalMemoryConfig.from_env()
        instance_id = os.environ.get("TRIBAL_MEMORY_INSTANCE_ID", "mcp-claude-code")
        
        # Use LanceDB session store when db_path is available
        if config.db.path:
            try:
                session_db_path = Path(config.db.path) / "session_chunks"
                _session_store = LanceDBSessionStore(
                    instance_id=instance_id,
                    embedding_service=memory_service.embedding_service,
                    vector_store=memory_service.vector_store,
                    db_path=session_db_path,
                )
                logger.info("LanceDB session store initialized (db: %s)", session_db_path)
            except ImportError:
                logger.warning(
                    "LanceDB not installed. Falling back to in-memory session storage. "
                    "Session data will NOT persist across restarts. "
                    "Install with: pip install lancedb"
                )
                _session_store = InMemorySessionStore(
                    instance_id=instance_id,
                    embedding_service=memory_service.embedding_service,
                    vector_store=memory_service.vector_store,
                )
            except (OSError, PermissionError, ValueError) as exc:
                logger.warning(
                    "LanceDB session store init failed (%s). "
                    "Falling back to in-memory session storage.",
                    exc,
                )
                _session_store = InMemorySessionStore(
                    instance_id=instance_id,
                    embedding_service=memory_service.embedding_service,
                    vector_store=memory_service.vector_store,
                )
        else:
            _session_store = InMemorySessionStore(
                instance_id=instance_id,
                embedding_service=memory_service.embedding_service,
                vector_store=memory_service.vector_store,
            )
            logger.info("In-memory session store initialized")
    
    return _session_store


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
        sources: str = "memories",
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> str:
        """Search memories and/or session transcripts by semantic similarity.

        Args:
            query: Natural language search query (required)
            limit: Maximum number of results (1-50, default 5)
            min_relevance: Minimum similarity score (0.0-1.0, default 0.3)
            tags: Filter results to only memories with these tags
            sources: What to search - "memories" (default), "sessions", or "all"
            after: Only include memories with events on/after this date (ISO or natural language)
            before: Only include memories with events on/before this date (ISO or natural language)

        Returns:
            JSON with: results (list of memories/chunks with similarity scores), query, count
        """
        # Input validation
        if not query or not query.strip():
            return json.dumps({
                "results": [],
                "query": query,
                "count": 0,
                "error": "Query cannot be empty",
            })
        
        valid_sources = {"memories", "sessions", "all"}
        if sources not in valid_sources:
            return json.dumps({
                "results": [],
                "query": query,
                "count": 0,
                "error": f"Invalid sources: {sources}. Valid options: {', '.join(sorted(valid_sources))}",
            })

        # Clamp limit to valid range
        limit = max(1, min(50, limit))
        min_relevance = max(0.0, min(1.0, min_relevance))

        all_results = []

        # Search memories
        if sources in ("memories", "all"):
            service = await get_memory_service()
            memory_results = await service.recall(
                query=query,
                limit=limit,
                min_relevance=min_relevance,
                tags=tags,
                after=after,
                before=before,
            )
            all_results.extend([
                {
                    "type": "memory",
                    "memory_id": r.memory.id,
                    "content": r.memory.content,
                    "similarity_score": round(r.similarity_score, 4),
                    "source_type": r.memory.source_type.value,
                    "source_instance": r.memory.source_instance,
                    "tags": r.memory.tags,
                    "created_at": r.memory.created_at.isoformat(),
                    "context": r.memory.context,
                }
                for r in memory_results
            ])

        # Search sessions
        if sources in ("sessions", "all"):
            session_store = await get_session_store()
            session_results = await session_store.search(
                query=query,
                limit=limit,
                min_relevance=min_relevance,
            )
            all_results.extend([
                {
                    "type": "session",
                    "chunk_id": r["chunk_id"],
                    "session_id": r["session_id"],
                    "instance_id": r["instance_id"],
                    "content": r["content"],
                    "similarity_score": round(r["similarity_score"], 4),
                    "start_time": r["start_time"].isoformat() if hasattr(r["start_time"], "isoformat") else str(r["start_time"]),
                    "end_time": r["end_time"].isoformat() if hasattr(r["end_time"], "isoformat") else str(r["end_time"]),
                    "chunk_index": r["chunk_index"],
                }
                for r in session_results
            ])

        # Sort combined results by score, take top limit
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        all_results = all_results[:limit]

        return json.dumps({
            "results": all_results,
            "query": query,
            "count": len(all_results),
            "sources": sources,
        })

    @mcp.tool()
    async def tribal_sessions_ingest(
        session_id: str,
        messages: str,
        instance_id: Optional[str] = None,
    ) -> str:
        """Ingest a session transcript for indexing.

        Chunks conversation messages into ~400 token windows and indexes them
        for semantic search. Supports delta ingestion — only new messages
        since last ingest are processed.

        Args:
            session_id: Unique identifier for the session (required)
            messages: JSON array of messages, each with "role", "content",
                and optional "timestamp" (ISO 8601). Example:
                [{"role": "user", "content": "What is Docker?"},
                 {"role": "assistant", "content": "Docker is a container platform"}]
            instance_id: Override the agent instance ID (optional)

        Returns:
            JSON with: success, chunks_created, messages_processed
        """
        if not session_id or not session_id.strip():
            return json.dumps({
                "success": False,
                "error": "session_id cannot be empty",
            })

        try:
            raw_messages = json.loads(messages)
        except (json.JSONDecodeError, TypeError) as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid messages JSON: {e}",
            })

        if not isinstance(raw_messages, list):
            return json.dumps({
                "success": False,
                "error": "messages must be a JSON array",
            })

        from datetime import datetime, timezone
        parsed_messages = []
        for i, msg in enumerate(raw_messages):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return json.dumps({
                    "success": False,
                    "error": f"Message {i} must have 'role' and 'content' fields",
                })
            
            ts = datetime.now(timezone.utc)
            if "timestamp" in msg:
                try:
                    ts = datetime.fromisoformat(msg["timestamp"])
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Invalid timestamp '%s' in message %d, using current time: %s",
                        msg.get("timestamp"), i, e,
                    )
            
            parsed_messages.append(SessionMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=ts,
            ))

        session_store = await get_session_store()
        result = await session_store.ingest(
            session_id=session_id,
            messages=parsed_messages,
            instance_id=instance_id,
        )

        return json.dumps(result)

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

    @mcp.tool()
    async def tribal_recall_entity(
        entity_name: str,
        hops: int = 1,
        limit: int = 10,
    ) -> str:
        """Recall memories associated with an entity and its connections.

        Enables entity-centric queries like:
        - "Tell me everything about auth-service"
        - "What do we know about PostgreSQL?"
        - "What services connect to the user database?"

        Args:
            entity_name: Name of the entity to query (required).
                Examples: "auth-service", "PostgreSQL", "user-db"
            hops: Number of relationship hops to traverse (default 1).
                1 = direct connections only
                2 = connections of connections
            limit: Maximum number of results (1-50, default 10)

        Returns:
            JSON with: results (list of memories), entity, hops, count
        """
        if not entity_name or not entity_name.strip():
            return json.dumps({
                "results": [],
                "entity": entity_name,
                "hops": hops,
                "count": 0,
                "error": "Entity name cannot be empty",
            })

        hops = max(1, min(10, hops))  # Clamp to reasonable range
        limit = max(1, min(50, limit))

        service = await get_memory_service()
        
        if not service.graph_enabled:
            return json.dumps({
                "results": [],
                "entity": entity_name,
                "hops": hops,
                "count": 0,
                "error": "Graph search not enabled. Requires db_path for persistent storage.",
            })

        results = await service.recall_entity(
            entity_name=entity_name,
            hops=hops,
            limit=limit,
        )

        return json.dumps({
            "results": [
                {
                    "memory_id": r.memory.id,
                    "content": r.memory.content,
                    "source_type": r.memory.source_type.value,
                    "source_instance": r.memory.source_instance,
                    "tags": r.memory.tags,
                    "created_at": r.memory.created_at.isoformat(),
                }
                for r in results
            ],
            "entity": entity_name,
            "hops": hops,
            "count": len(results),
        })

    @mcp.tool()
    async def tribal_entity_graph(
        entity_name: str,
        hops: int = 2,
    ) -> str:
        """Get the relationship graph around an entity.

        Useful for understanding how concepts/services/technologies
        are connected in your project knowledge base.

        Args:
            entity_name: Name of the entity to explore (required)
            hops: How many relationship hops to include (default 2)

        Returns:
            JSON with: entities (list with name/type), relationships (list with source/target/type)
        """
        if not entity_name or not entity_name.strip():
            return json.dumps({
                "entities": [],
                "relationships": [],
                "error": "Entity name cannot be empty",
            })

        hops = max(1, min(5, hops))  # Clamp to reasonable range

        service = await get_memory_service()
        
        if not service.graph_enabled:
            return json.dumps({
                "entities": [],
                "relationships": [],
                "error": "Graph search not enabled. Requires db_path for persistent storage.",
            })

        graph = service.get_entity_graph(
            entity_name=entity_name,
            hops=hops,
        )

        return json.dumps(graph)

    @mcp.tool()
    async def tribal_export(
        tags: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Export memories to a portable JSON bundle.

        Args:
            tags: Only export memories with any of these tags.
            date_from: ISO 8601 lower bound (created_at).
            date_to: ISO 8601 upper bound (created_at).
            output_path: Write bundle to file (else inline).

        Returns:
            JSON with: success, memory_count, manifest, entries.
        """
        from ..portability.embedding_metadata import (
            create_embedding_metadata,
        )
        from ..services.import_export import (
            ExportFilter,
            export_memories as do_export,
            parse_iso_datetime,
        )

        # Validate dates
        parsed_from, err = parse_iso_datetime(
            date_from, "date_from",
        )
        if err:
            return json.dumps({"success": False, "error": err})
        parsed_to, err = parse_iso_datetime(
            date_to, "date_to",
        )
        if err:
            return json.dumps({"success": False, "error": err})

        service = await get_memory_service()
        emb = service.embedding_service
        meta = create_embedding_metadata(
            model_name=getattr(
                emb, "model",
                getattr(emb, "model_name", "unknown"),
            ),
            dimensions=getattr(emb, "dimensions", 1536),
            provider=getattr(emb, "provider_name", "openai"),
        )

        flt = None
        if tags or parsed_from or parsed_to:
            flt = ExportFilter(
                tags=tags,
                date_from=parsed_from,
                date_to=parsed_to,
            )

        try:
            bundle = await do_export(
                store=service.vector_store,
                embedding_metadata=meta,
                filters=flt,
            )
        except Exception as e:
            return json.dumps({
                "success": False, "error": str(e),
            })

        bundle_dict = bundle.to_dict()

        if output_path:
            try:
                with open(output_path, "w") as f:
                    json.dump(bundle_dict, f, default=str)
                return json.dumps({
                    "success": True,
                    "memory_count": bundle.manifest.memory_count,
                    "output_path": output_path,
                })
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "error": f"Write failed: {e}",
                })

        return json.dumps({
            "success": True,
            "memory_count": bundle.manifest.memory_count,
            "manifest": bundle_dict["manifest"],
            "entries": bundle_dict["entries"],
        }, default=str)

    @mcp.tool()
    async def tribal_import(
        input_path: Optional[str] = None,
        bundle_json: Optional[str] = None,
        conflict_resolution: str = "skip",
        embedding_strategy: str = "auto",
        dry_run: bool = False,
    ) -> str:
        """Import memories from a portable JSON bundle.

        Args:
            input_path: Path to a bundle JSON file.
            bundle_json: Inline JSON string of the bundle.
            conflict_resolution: skip | overwrite | merge.
            embedding_strategy: auto | keep | drop.
            dry_run: Preview what would change without writing.

        Returns:
            JSON with import summary.
        """
        from ..portability.embedding_metadata import (
            PortableBundle,
            ReembeddingStrategy,
            create_embedding_metadata,
        )
        from ..services.import_export import (
            ConflictResolution,
            import_memories as do_import,
            validate_conflict_resolution,
            validate_embedding_strategy,
        )

        if not input_path and not bundle_json:
            return json.dumps({
                "success": False,
                "error": "Provide input_path or bundle_json",
            })

        # Validate enum params
        err = validate_conflict_resolution(conflict_resolution)
        if err:
            return json.dumps({"success": False, "error": err})
        err = validate_embedding_strategy(embedding_strategy)
        if err:
            return json.dumps({"success": False, "error": err})

        # Parse bundle
        try:
            if input_path:
                with open(input_path) as f:
                    raw = json.load(f)
            else:
                raw = json.loads(bundle_json)
            bundle = PortableBundle.from_dict(raw)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to parse bundle: {e}",
            })

        service = await get_memory_service()
        emb = service.embedding_service
        target_meta = create_embedding_metadata(
            model_name=getattr(
                emb, "model",
                getattr(emb, "model_name", "unknown"),
            ),
            dimensions=getattr(emb, "dimensions", 1536),
            provider=getattr(emb, "provider_name", "openai"),
        )

        cr_map = {
            "skip": ConflictResolution.SKIP,
            "overwrite": ConflictResolution.OVERWRITE,
            "merge": ConflictResolution.MERGE,
        }
        es_map = {
            "auto": ReembeddingStrategy.AUTO,
            "keep": ReembeddingStrategy.KEEP,
            "drop": ReembeddingStrategy.DROP,
        }

        try:
            summary = await do_import(
                bundle=bundle,
                store=service.vector_store,
                target_metadata=target_meta,
                conflict_resolution=cr_map[conflict_resolution],
                embedding_strategy=es_map[embedding_strategy],
                dry_run=dry_run,
            )
        except Exception as e:
            return json.dumps({
                "success": False, "error": str(e),
            })

        return json.dumps({
            "success": True,
            "dry_run": summary.dry_run,
            "total": summary.total,
            "imported": summary.imported,
            "skipped": summary.skipped,
            "overwritten": summary.overwritten,
            "errors": summary.errors,
            "needs_reembedding": summary.needs_reembedding,
            "duration_ms": round(summary.duration_ms, 1),
            "error_details": summary.error_details,
        })

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
