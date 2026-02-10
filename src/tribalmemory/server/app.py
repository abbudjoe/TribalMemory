"""FastAPI application for tribal-memory service."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tribalmemory import __version__
from ..services import create_memory_service, TribalMemoryService
from ..services.session_store import (
    SessionStore,
    LanceDBSessionStore,
    InMemorySessionStore,
)
from .auth import TokenAuthMiddleware, load_token
from .config import TribalMemoryConfig
from .episode_routes import router as episode_router
from .graph_routes import router as graph_router
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
        embedding_model=config.embedding.model,
        embedding_dimensions=config.embedding.dimensions,
        hybrid_search=config.search.hybrid_enabled,
        hybrid_vector_weight=config.search.vector_weight,
        hybrid_text_weight=config.search.text_weight,
        hybrid_candidate_multiplier=config.search.candidate_multiplier,
        lazy_spacy=config.search.lazy_spacy,
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

    # Initialize episode components if enabled
    if config.episodes.enabled and config.db.path:
        try:
            from ..services.episode_store import EpisodeStore
            from ..services.episode_detector import EpisodeDetector, EpisodeConfig as EpConfig, LLMClient
            from ..services.episode_summarizer import EpisodeSummarizer
            
            episode_db_path = str(Path(config.db.path) / "episodes.db")
            episode_store = EpisodeStore(episode_db_path)
            
            # Convert server config to episode config
            ep_config = EpConfig(
                enabled=config.episodes.enabled,
                detector_strategy=config.episodes.detector_strategy,
                embedding_similarity_threshold=config.episodes.embedding_similarity_threshold,
                active_window_days=config.episodes.active_window_days,
                max_active_episodes=config.episodes.max_active_episodes,
                summarizer_model=config.episodes.summarizer_model,
                summarizer_provider=config.episodes.summarizer_provider,
                summarizer_temperature=config.episodes.summarizer_temperature,
                full_regen_interval=config.episodes.full_regen_interval,
                max_llm_calls_per_memory=config.episodes.max_llm_calls_per_memory,
                monthly_cost_ceiling=config.episodes.monthly_cost_ceiling,
            )
            
            # Create detector
            episode_detector = EpisodeDetector(
                episode_store=episode_store,
                embedding_service=_memory_service.embedding_service,
                config=ep_config,
            )
            
            # Create LLM client for summarizer
            llm_client = LLMClient(
                provider=config.episodes.summarizer_provider,
                model=config.episodes.summarizer_model,
            )
            
            # Create summarizer
            episode_summarizer = EpisodeSummarizer(
                episode_store=episode_store,
                vector_store=_memory_service.vector_store,
                embedding_service=_memory_service.embedding_service,
                llm_client=llm_client,
                config=ep_config,
            )
            
            # Wire up to service
            _memory_service.episode_detector = episode_detector
            _memory_service.episode_summarizer = episode_summarizer
            
            logger.info(f"Episode memories enabled (strategy: {config.episodes.detector_strategy})")
        
        except Exception as e:
            logger.warning(f"Failed to initialize episode components: {e}")
            logger.warning("Episode feature disabled")
    
    search_mode = "hybrid (vector + BM25)" if config.search.hybrid_enabled else "vector-only"
    logger.info(f"Memory service initialized (db: {config.db.path}, search: {search_mode})")
    retention = config.server.session_retention_days
    logger.info(f"Session store initialized (retention: {retention} days)")

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
                logger.info(
                    "Session cleanup: deleted %d expired chunks "
                    "(retention: %d days)",
                    deleted,
                    retention_days,
                )
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
        version=__version__,
        lifespan=lifespan,
    )

    # Store config for lifespan access
    app.state.config = config

    # Token authentication middleware
    # Precedence: environment variable > .env file > no token (legacy mode)
    api_token = os.environ.get("TRIBAL_MEMORY_API_TOKEN") or load_token()
    if api_token:
        source = (
            "environment variable"
            if os.environ.get("TRIBAL_MEMORY_API_TOKEN")
            else "~/.tribal-memory/.env"
        )
        logger.info("API token loaded from %s", source)
    app.add_middleware(TokenAuthMiddleware, token=api_token)

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
    app.include_router(graph_router)
    app.include_router(episode_router)

    # Serve static files + graph UI
    from fastapi.staticfiles import StaticFiles
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount(
            "/static", StaticFiles(directory=str(static_dir)),
            name="static",
        )

    @app.get("/graph")
    async def graph_ui():
        """Serve the knowledge graph explorer."""
        from fastapi.responses import HTMLResponse
        html_path = static_dir / "graph.html"
        if not html_path.exists():
            from fastapi import HTTPException
            raise HTTPException(
                status_code=404,
                detail="Graph UI not found",
            )
        content = await asyncio.to_thread(
            html_path.read_text,
        )
        return HTMLResponse(content)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "tribal-memory",
            "version": __version__,
            "docs": "/docs",
            "graph": "/graph",
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
