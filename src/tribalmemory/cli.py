"""Tribal Memory CLI — init, serve, and MCP entry points.

Usage:
    tribalmemory init [--local]     # Set up config + MCP integration
    tribalmemory serve              # Start the HTTP server
    tribalmemory mcp                # Start the MCP server (stdio)
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    tomllib = None  # type: ignore

try:
    import tomli_w  # For writing TOML
except ImportError:
    tomli_w = None  # type: ignore

TRIBAL_DIR = Path.home() / ".tribal-memory"
CONFIG_FILE = TRIBAL_DIR / "config.yaml"

# MCP config for Claude Code (claude_desktop_config.json)
CLAUDE_CODE_MCP_CONFIG = {
    "mcpServers": {
        "tribal-memory": {
            "command": "tribalmemory-mcp",
            "env": {}
        }
    }
}

OPENAI_CONFIG_TEMPLATE = """\
# Tribal Memory Configuration
# Docs: https://github.com/abbudjoe/TribalMemory

instance_id: {instance_id}

embedding:
  provider: openai
  model: text-embedding-3-small
  dimensions: 1536
  # api_key: sk-...  # Or set OPENAI_API_KEY env var

db:
  provider: lancedb
  path: {db_path}

server:
  host: 127.0.0.1
  port: 18790
"""

LOCAL_CONFIG_TEMPLATE = """\
# Tribal Memory Configuration — Local Mode (Zero Cloud)
# Uses Ollama for embeddings — no API keys needed!
# Docs: https://github.com/abbudjoe/TribalMemory

instance_id: {instance_id}

embedding:
  provider: openai          # Uses OpenAI-compatible API
  model: nomic-embed-text   # Run: ollama pull nomic-embed-text
  api_base: http://localhost:11434/v1
  dimensions: 768
  api_key: unused           # Ollama doesn't need a key

db:
  provider: lancedb
  path: {db_path}

server:
  host: 127.0.0.1
  port: 18790
"""


def cmd_init(args):
    """Initialize Tribal Memory configuration and MCP integration."""
    instance_id = args.instance_id or "default"
    db_path = str(TRIBAL_DIR / "lancedb")

    # Create config directory
    TRIBAL_DIR.mkdir(parents=True, exist_ok=True)

    # Choose template
    if args.local:
        config_content = LOCAL_CONFIG_TEMPLATE.format(
            instance_id=instance_id,
            db_path=db_path,
        )
    else:
        config_content = OPENAI_CONFIG_TEMPLATE.format(
            instance_id=instance_id,
            db_path=db_path,
        )

    # Write config
    if CONFIG_FILE.exists() and not args.force:
        print(f"⚠️  Config already exists: {CONFIG_FILE}")
        print("   Use --force to overwrite.")
        return 1
    
    CONFIG_FILE.write_text(config_content)
    print(f"✅ Config written: {CONFIG_FILE}")

    if args.local:
        print()
        print("📦 Local mode — make sure Ollama is running:")
        print("   curl -fsSL https://ollama.com/install.sh | sh")
        print("   ollama pull nomic-embed-text")
        print("   ollama serve  # if not already running")

    # Set up MCP integrations
    if args.claude_code:
        _setup_claude_code_mcp(args.local)
    
    if args.codex:
        _setup_codex_mcp(args.local)

    print()
    print("🚀 Start the server:")
    print("   tribalmemory serve")
    print()
    print("🧠 Or use with Claude Code (MCP):")
    print("   tribalmemory-mcp")
    
    return 0


def _setup_claude_code_mcp(is_local: bool):
    """Add Tribal Memory to Claude Code's MCP configuration."""
    # Claude Code stores MCP config in different locations depending on platform
    claude_config_paths = [
        Path.home() / ".claude" / "claude_desktop_config.json",  # macOS/Linux
        Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
    ]

    config_path = None
    for p in claude_config_paths:
        if p.exists():
            config_path = p
            break

    if config_path is None:
        # Create default location
        config_path = claude_config_paths[0]
        config_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing config or start fresh
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}

    # Merge MCP server config
    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    mcp_entry = {
        "command": "tribalmemory-mcp",
        "env": {},
    }
    
    if is_local:
        mcp_entry["env"]["TRIBAL_MEMORY_EMBEDDING_API_BASE"] = "http://localhost:11434/v1"

    existing["mcpServers"]["tribal-memory"] = mcp_entry

    config_path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"✅ Claude Code MCP config updated: {config_path}")


def _setup_codex_mcp(is_local: bool):
    """Add Tribal Memory to Codex CLI's MCP configuration (~/.codex/config.toml)."""
    codex_config_path = Path.home() / ".codex" / "config.toml"
    codex_config_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the TOML section manually (avoid tomli_w dependency)
    # Codex uses [mcp_servers.name] sections in config.toml
    section_marker = "[mcp_servers.tribal-memory]"
    
    mcp_lines = [
        "",
        "# Tribal Memory — shared memory for AI agents",
        section_marker,
        'command = "tribalmemory-mcp"',
    ]
    
    if is_local:
        mcp_lines.append("")
        mcp_lines.append("[mcp_servers.tribal-memory.env]")
        mcp_lines.append('TRIBAL_MEMORY_EMBEDDING_API_BASE = "http://localhost:11434/v1"')
    
    mcp_block = "\n".join(mcp_lines) + "\n"

    if codex_config_path.exists():
        existing = codex_config_path.read_text()
        if section_marker in existing:
            print(f"⚠️  Codex config already has tribal-memory: {codex_config_path}")
            print("   Remove the existing section first, or edit manually.")
            return
        # Append to existing config
        if not existing.endswith("\n"):
            existing += "\n"
        codex_config_path.write_text(existing + mcp_block)
    else:
        codex_config_path.write_text(mcp_block.lstrip("\n"))

    print(f"✅ Codex CLI MCP config updated: {codex_config_path}")


def cmd_serve(args):
    """Start the HTTP server."""
    from .server.app import main as server_main
    # Re-inject args for the server's argparse
    sys.argv = ["tribalmemory"]
    if args.host:
        sys.argv.extend(["--host", args.host])
    if args.port:
        sys.argv.extend(["--port", str(args.port)])
    if args.config:
        sys.argv.extend(["--config", args.config])
    if args.log_level:
        sys.argv.extend(["--log-level", args.log_level])
    server_main()


def cmd_mcp(args):
    """Start the MCP server (stdio)."""
    from .mcp.server import main as mcp_main
    mcp_main()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tribalmemory",
        description="Tribal Memory — Shared memory for AI agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize config and MCP integration")
    init_parser.add_argument("--local", action="store_true",
                             help="Use local Ollama embeddings (no API key needed)")
    init_parser.add_argument("--claude-code", action="store_true",
                             help="Configure Claude Code MCP integration")
    init_parser.add_argument("--codex", action="store_true",
                             help="Configure Codex CLI MCP integration")
    init_parser.add_argument("--instance-id", type=str, default=None,
                             help="Instance identifier (default: 'default')")
    init_parser.add_argument("--force", action="store_true",
                             help="Overwrite existing config")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the HTTP server")
    serve_parser.add_argument("--host", type=str, default=None)
    serve_parser.add_argument("--port", "-p", type=int, default=None)
    serve_parser.add_argument("--config", "-c", type=str, default=None)
    serve_parser.add_argument("--log-level", type=str, default=None,
                              choices=["debug", "info", "warning", "error"])

    # mcp
    subparsers.add_parser("mcp", help="Start the MCP server (stdio transport)")

    args = parser.parse_args()

    if args.command == "init":
        sys.exit(cmd_init(args))
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "mcp":
        cmd_mcp(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
