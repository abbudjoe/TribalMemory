"""Tribal Memory HTTP Server.

FastAPI-based HTTP interface for tribal-memory service.
Designed for integration with OpenClaw's memory-tribal extension.
"""

from .app import create_app, run_server

__all__ = ["create_app", "run_server"]
