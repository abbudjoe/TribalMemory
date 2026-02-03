"""FastAPI application for tribal-memory service."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..services import create_memory_service, TribalMemoryService
from .config import TribalMemoryConfig
from .routes import router

# Global service instance (set during lifespan)
_memory_service: Optional[TribalMemoryService] = None
_instance_id: Optional[str] = None

logger = logging.getLogger("tribalmemory.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _memory_service, _instance_id

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
    )

    logger.info(f"Memory service initialized (db: {config.db.path})")

    yield

    # Cleanup
    logger.info("Shutting down tribal-memory service")
    _memory_service = None
    _instance_id = None


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
