"""FastAPI application for tribal-memory service."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..services import create_memory_service, TribalMemoryService
from ..services.session_store import (
    SessionStore,
    LanceDBSessionStore,
    InMemorySessionStore,
)
from .config import TribalMemoryConfig
from .routes import router

# Global service instance (set during lifespan)
_memory_service: Optional[TribalMemoryService] = None
_session_store: Optional[SessionStore] = None
_instance_id: Optional[str] = None

logger = logging.getLogger("tribalmemory.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _memory_service, _session_store, _instance_id

    config: TribalMemoryConfig = app.state.config

    # Validate config
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {errors}")

    logger.info(f"Starting tribal-memory service (instance: {config.instance_id})")

    # Create memory service
    _instance_id = config.instance_id
    _memory_service = create_memory_service(
        instance_id=config.instance_id,
        db_path=config.db.path,
        openai_api_key=config.embedding.api_key,
        api_base=config.embedding.api_base,
        embedding_model=config.embedding.model,
        embedding_dimensions=config.embedding.dimensions,
        embedding_provider=config.embedding.provider,
        hybrid_search=config.search.hybrid_enabled,
        hybrid_vector_weight=config.search.vector_weight,
        hybrid_text_weight=config.search.text_weight,
        hybrid_candidate_multiplier=config.search.candidate_multiplier,
    )

    # Create session store (shares embedding service and vector store)
    # Use LanceDB session store when db_path is available
    if config.db.path:
        try:
            session_db_path = Path(config.db.path) / "session_chunks"
            _session_store = LanceDBSessionStore(
                instance_id=config.instance_id,
                embedding_service=_memory_service.embedding_service,
                vector_store=_memory_service.vector_store,
                db_path=session_db_path,
            )
        except ImportError:
            logger.warning(
                "LanceDB not installed. Falling back to in-memory session storage. "
                "Session data will NOT persist across restarts. "
                "Install with: pip install lancedb"
            )
            _session_store = InMemorySessionStore(
                instance_id=config.instance_id,
                embedding_service=_memory_service.embedding_service,
                vector_store=_memory_service.vector_store,
            )
        except (OSError, PermissionError, ValueError) as exc:
            logger.warning(
                "LanceDB session store init failed (%s). "
                "Falling back to in-memory session storage.",
                exc,
            )
            _session_store = InMemorySessionStore(
                instance_id=config.instance_id,
                embedding_service=_memory_service.embedding_service,
                vector_store=_memory_service.vector_store,
            )
    else:
        _session_store = InMemorySessionStore(
            instance_id=config.instance_id,
            embedding_service=_memory_service.embedding_service,
            vector_store=_memory_service.vector_store,
        )

    search_mode = "hybrid (vector + BM25)" if config.search.hybrid_enabled else "vector-only"
    logger.info(f"Memory service initialized (db: {config.db.path}, search: {search_mode})")
    logger.info(f"Session store initialized (retention: {config.server.session_retention_days} days)")

    # Start background session cleanup task
    cleanup_task = asyncio.create_task(
        _session_cleanup_loop(
            _session_store,
            config.server.session_retention_days,
        )
    )

    yield

    # Cleanup
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down tribal-memory service")
    _memory_service = None
    _session_store = None
    _instance_id = None


async def _session_cleanup_loop(
    session_store: SessionStore,
    retention_days: int,
) -> None:
    """Background task that periodically cleans up expired session chunks.
    
    Runs every 6 hours. Deletes session chunks older than retention_days.
    """
    cleanup_interval = 6 * 60 * 60  # 6 hours in seconds
    while True:
        try:
            await asyncio.sleep(cleanup_interval)
            deleted = await session_store.cleanup(retention_days=retention_days)
            if deleted > 0:
                logger.info(f"Session cleanup: deleted {deleted} expired chunks (retention: {retention_days} days)")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Session cleanup failed")


def create_app(config: Optional[TribalMemoryConfig] = None) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        config: Service configuration. If None, loads from environment.
    
    Returns:
        Configured FastAPI application.
    """
    if config is None:
        config = TribalMemoryConfig.from_env()

    app = FastAPI(
        title="Tribal Memory",
        description="Long-term memory service for AI agents with provenance tracking",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store config for lifespan access
    app.state.config = config

    # CORS middleware (localhost only)
    # Uses regex to match any port on localhost - OpenClaw Gateway runs on
    # user-configurable ports (default 18789). Server is bound to 127.0.0.1
    # so only local processes can reach it regardless of CORS settings.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "tribal-memory",
            "version": "0.1.0",
            "docs": "/docs",
        }

    return app


def run_server(
    config: Optional[TribalMemoryConfig] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    log_level: str = "info",
):
    """Run the HTTP server.
    
    Args:
        config: Service configuration. If None, loads from environment.
        host: Override host from config.
        port: Override port from config.
        log_level: Logging level.
    """
    if config is None:
        config = TribalMemoryConfig.from_env()

    # Ensure db directory exists
    db_path = Path(config.db.path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    app = create_app(config)

    uvicorn.run(
        app,
        host=host or config.server.host,
        port=port or config.server.port,
        log_level=log_level,
    )


# CLI entry point
def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Tribal Memory HTTP Server")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file (default: ~/.tribal-memory/config.yaml)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port to bind to (default: 18790)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        config = TribalMemoryConfig.from_file(args.config)
    else:
        config = TribalMemoryConfig.from_env()

    run_server(
        config=config,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
