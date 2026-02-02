"""MCP server for Tribal Memory.

This module provides an MCP (Model Context Protocol) server that exposes
Tribal Memory as tools for Claude Code and other MCP-compatible clients.
"""

from .server import create_server, main

__all__ = ["create_server", "main"]
