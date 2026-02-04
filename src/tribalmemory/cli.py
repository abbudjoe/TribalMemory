"""Tribal Memory CLI — init, serve, and MCP entry points.

Usage:
    tribalmemory init [--local]     # Set up config + MCP integration
    tribalmemory serve              # Start the HTTP server
    tribalmemory mcp                # Start the MCP server (stdio)
"""

import argparse
import json
import os
import shutil
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
DEFAULT_INSTANCE_ID = "default"

# Path to the global Claude Code instructions file (relative to home)
CLAUDE_INSTRUCTIONS_FILE = Path(".claude") / "CLAUDE.md"

# Section marker used to detect if auto-capture instructions already exist
_AUTO_CAPTURE_MARKER = "## Tribal Memory — Auto-Capture"

AUTO_CAPTURE_INSTRUCTIONS = f"""\

{_AUTO_CAPTURE_MARKER}

You have access to Tribal Memory — a shared memory system for AI agents.
Use it proactively:

**After completing tasks, making decisions, or learning something:**
- Call `tribal_remember` to store the insight with relevant tags
- Good memories: architectural decisions, bug fixes, "gotchas", user preferences,
  project conventions, API quirks, things that took a while to figure out
- Bad memories: trivial actions ("I ran the tests"), obvious facts, temporary state

**Before answering questions about prior work:**
- Call `tribal_recall` to search for relevant context first
- Search by topic, not exact phrases — it's semantic search

**Tips:**
- Use tags to organize: `["python", "debugging"]`, `["api", "auth"]`
- One clear insight per memory is better than a wall of text
- If you're unsure whether to remember something, remember it — recall is cheap
"""

# MCP config for Claude Code CLI and Claude Desktop
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
{auto_capture_line}"""

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
  # api_key not needed for local Ollama

db:
  provider: lancedb
  path: {db_path}

server:
  host: 127.0.0.1
  port: 18790
{auto_capture_line}"""


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize Tribal Memory config and MCP integration."""
    instance_id = args.instance_id or DEFAULT_INSTANCE_ID
    db_path = str(TRIBAL_DIR / "lancedb")

    # Create config directory
    TRIBAL_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-capture config line (only included when flag is set)
    auto_capture_line = ""
    if args.auto_capture:
        auto_capture_line = "\nauto_capture: true\n"

    # Choose template
    if args.local:
        config_content = LOCAL_CONFIG_TEMPLATE.format(
            instance_id=instance_id,
            db_path=db_path,
            auto_capture_line=auto_capture_line,
        )
    else:
        config_content = OPENAI_CONFIG_TEMPLATE.format(
            instance_id=instance_id,
            db_path=db_path,
            auto_capture_line=auto_capture_line,
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

    # Set up auto-capture instructions
    if args.auto_capture:
        _setup_auto_capture()

    print()
    print("🚀 Start the server:")
    print("   tribalmemory serve")
    print()
    print("🧠 Or use with Claude Code (MCP):")
    print("   tribalmemory-mcp")

    if not args.auto_capture:
        print()
        print("💡 Want Claude to remember things automatically?")
        print("   tribalmemory init --auto-capture --force")
    
    return 0


def _setup_auto_capture() -> None:
    """Write auto-capture instructions to ~/.claude/CLAUDE.md.
    
    Appends memory usage instructions so Claude Code proactively uses
    tribal_remember and tribal_recall without being explicitly asked.
    Skips if instructions are already present (idempotent).
    """
    claude_md = Path.home() / CLAUDE_INSTRUCTIONS_FILE
    claude_md.parent.mkdir(parents=True, exist_ok=True)

    if claude_md.exists():
        existing = claude_md.read_text()
        if _AUTO_CAPTURE_MARKER in existing:
            print(f"✅ Auto-capture instructions already present: {claude_md}")
            return
        # Append to existing file
        if not existing.endswith("\n"):
            existing += "\n"
        claude_md.write_text(existing + AUTO_CAPTURE_INSTRUCTIONS)
    else:
        claude_md.write_text(AUTO_CAPTURE_INSTRUCTIONS.lstrip("\n"))

    print(f"✅ Auto-capture instructions written: {claude_md}")


def _setup_claude_code_mcp(is_local: bool) -> None:
    """Add Tribal Memory to Claude Code's MCP configuration.
    
    Claude Code CLI reads MCP servers from ~/.claude.json (user scope).
    Claude Desktop reads from platform-specific claude_desktop_config.json.
    We update both if they exist, and always ensure ~/.claude.json is set.
    """
    # Claude Code CLI config (primary — this is what `claude` CLI reads)
    claude_cli_config = Path.home() / ".claude.json"
    
    # Claude Desktop config paths (secondary — update if they exist)
    claude_desktop_paths = [
        Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",  # macOS
        Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",  # Windows
        Path.home() / ".claude" / "claude_desktop_config.json",  # Legacy / Linux
    ]

    # Resolve full path to tribalmemory-mcp binary.
    # Claude Desktop doesn't inherit the user's shell PATH (e.g. ~/.local/bin),
    # so we need the absolute path for it to find the command.
    mcp_command = _resolve_mcp_command()

    mcp_entry = {
        "command": mcp_command,
        "env": {},
    }
    
    if is_local:
        mcp_entry["env"]["TRIBAL_MEMORY_EMBEDDING_API_BASE"] = "http://localhost:11434/v1"

    # Always update Claude Code CLI config (~/.claude.json)
    _update_mcp_config(claude_cli_config, mcp_entry, create_if_missing=True)
    print(f"✅ Claude Code CLI config updated: {claude_cli_config}")

    # Also update Claude Desktop config (create platform-appropriate path)
    desktop_path = _get_claude_desktop_config_path()
    _update_mcp_config(desktop_path, mcp_entry, create_if_missing=True)
    print(f"✅ Claude Desktop config updated: {desktop_path}")


def _resolve_mcp_command() -> str:
    """Resolve the full path to the tribalmemory-mcp binary.
    
    Claude Desktop doesn't inherit the user's shell PATH (e.g. ~/.local/bin
    from uv/pipx installs), so bare command names like "tribalmemory-mcp"
    fail with "No such file or directory". We resolve the absolute path at
    init time so the config works regardless of the app's PATH.
    
    Falls back to the bare command name if not found on PATH (e.g. user
    hasn't installed yet and will do so later).
    """
    resolved = shutil.which("tribalmemory-mcp")
    if resolved:
        return resolved
    
    # Check common tool install locations that might not be on PATH
    base_name = "tribalmemory-mcp"
    search_dirs = [
        Path.home() / ".local" / "bin",   # uv/pipx (Linux/macOS)
        Path.home() / ".cargo" / "bin",    # unlikely but possible
    ]
    # On Windows, executables may have .exe/.cmd extensions
    suffixes = [""]
    if sys.platform == "win32":
        suffixes = [".exe", ".cmd", ""]
    
    for search_dir in search_dirs:
        for suffix in suffixes:
            candidate = search_dir / (base_name + suffix)
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)
    
    # Fall back to bare command — will work if PATH is set correctly
    return "tribalmemory-mcp"


def _get_claude_desktop_config_path() -> Path:
    """Get the platform-appropriate Claude Desktop config path."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif sys.platform == "win32":
        return Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
    else:
        return Path.home() / ".claude" / "claude_desktop_config.json"


def _update_mcp_config(
    config_path: Path, mcp_entry: dict, create_if_missing: bool = False
) -> None:
    """Update an MCP config file with the tribal-memory server entry."""
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except json.JSONDecodeError as e:
            backup_path = config_path.with_suffix(".json.bak")
            config_path.rename(backup_path)
            print(f"⚠️  Existing config has invalid JSON: {e}")
            print(f"   Backed up to {backup_path}")
            print(f"   Creating fresh config at {config_path}")
            existing = {}
    elif create_if_missing:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
    else:
        return

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"]["tribal-memory"] = mcp_entry
    config_path.write_text(json.dumps(existing, indent=2) + "\n")


def _setup_codex_mcp(is_local: bool) -> None:
    """Add Tribal Memory to Codex CLI's MCP configuration (~/.codex/config.toml)."""
    codex_config_path = Path.home() / ".codex" / "config.toml"
    codex_config_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve full path (same reason as Claude Desktop — Codex may not
    # inherit the user's full shell PATH)
    mcp_command = _resolve_mcp_command()

    # Build the TOML section manually (avoid tomli_w dependency)
    # Codex uses [mcp_servers.name] sections in config.toml
    section_marker = "[mcp_servers.tribal-memory]"
    
    mcp_lines = [
        "",
        "# Tribal Memory — shared memory for AI agents",
        section_marker,
        f'command = "{mcp_command}"',
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


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the HTTP server."""
    from .server.app import run_server
    from .server.config import TribalMemoryConfig

    config_path = args.config
    if config_path:
        config = TribalMemoryConfig.from_file(config_path)
    else:
        config = TribalMemoryConfig.from_env()

    run_server(
        config=config,
        host=args.host,
        port=args.port,
        log_level=args.log_level or "info",
    )


def cmd_mcp(args: argparse.Namespace) -> None:
    """Start the MCP server (stdio)."""
    from .mcp.server import main as mcp_main
    mcp_main()


def main() -> None:
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
    init_parser.add_argument("--auto-capture", action="store_true",
                             help="Enable auto-capture (writes CLAUDE.md instructions)")
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
