"""Tribal Memory CLI — init, serve, and MCP entry points.

Usage:
    tribalmemory init               # FastEmbed (local, zero cloud)
    tribalmemory serve              # Start the HTTP server
    tribalmemory mcp                # Start the MCP server (stdio)
"""

import argparse
import json
import os
import shutil
import subprocess
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

# Paths to global instructions files (relative to home)
CLAUDE_INSTRUCTIONS_FILE = Path(".claude") / "CLAUDE.md"
CODEX_INSTRUCTIONS_FILE = Path(".codex") / "AGENTS.md"

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

ENV_FILE = TRIBAL_DIR / ".env"

FASTEMBED_CONFIG_TEMPLATE = """\
# Tribal Memory Configuration — FastEmbed (Zero Cloud)
# Local ONNX embeddings — no API keys, no external services.
# Docs: https://github.com/abbudjoe/TribalMemory

instance_id: {instance_id}

embedding:
  provider: fastembed
  model: BAAI/bge-small-en-v1.5
  dimensions: 384

db:
  provider: lancedb
  path: {db_path}

server:
  host: 127.0.0.1
  port: 18790
{auto_capture_line}"""

def _write_env_file(key: str, value: str) -> None:
    """Write or update a key in ~/.tribal-memory/.env.

    The file is created with 600 permissions (owner-only read/write)
    so API keys stay out of config.yaml and are harder to leak.
    """
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    existing[key] = value

    content = "# Tribal Memory secrets — auto-generated, do not commit\n"
    for k, v in existing.items():
        content += f"{k}={v}\n"

    ENV_FILE.write_text(content)
    ENV_FILE.chmod(0o600)


def load_env_file() -> None:
    """Load ~/.tribal-memory/.env into os.environ if it exists.

    Called at server startup so any custom environment variables
    are available without the user manually exporting them.
    """
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            # Don't overwrite explicit env vars
            if k not in os.environ:
                os.environ[k] = v


def _is_uv_environment() -> bool:
    """Detect if we're running inside a uv-managed tool environment."""
    # uv tool environments have uv-specific paths and lack pip.
    # Normalize to forward slashes for cross-platform consistency.
    venv_path = str(Path(sys.executable).resolve()).replace("\\", "/")
    if "/uv/tools/" in venv_path:
        return True
    # Also check: pip module missing + uv available on PATH
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
    )
    if pip_check.returncode != 0:
        uv_check = shutil.which("uv")
        if uv_check:
            return True
    return False


def _install_fastembed(interactive: bool) -> bool:
    """Try to install fastembed using the best available installer.

    Uses ``uv pip install`` in uv-managed environments (which lack pip),
    standard ``pip install`` otherwise.

    Returns True if installation succeeded.
    """
    suppress = not interactive
    out = subprocess.DEVNULL if suppress else None

    # In uv environments, pip doesn't exist — use uv exclusively.
    if _is_uv_environment():
        uv_bin = shutil.which("uv")
        if not uv_bin:
            print("   Warning: uv environment detected but uv not found on PATH.")
            return False
        print("   Installing fastembed via uv (detected uv environment)...")
        cmd = [
            uv_bin, "pip", "install",
            "--python", sys.executable,
            "fastembed",
        ]
        try:
            subprocess.check_call(cmd, stdout=out, stderr=out)
            return True
        except subprocess.CalledProcessError:
            return False

    # Standard pip install (non-uv environments)
    print("   Installing fastembed via pip...")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "fastembed"]
    try:
        subprocess.check_call(cmd, stdout=out, stderr=out)
        return True
    except subprocess.CalledProcessError:
        return False


def _print_manual_install_hint() -> None:
    """Print the correct manual install command for the current environment."""
    if _is_uv_environment():
        uv_bin = shutil.which("uv")
        if uv_bin:
            print(f"   Try: uv pip install --python {sys.executable} fastembed")
        else:
            print("   uv environment detected but uv not found on PATH.")
            print("   Install uv: https://docs.astral.sh/uv/")
    else:
        print(f"   Try: {sys.executable} -m pip install fastembed")


def _auto_install_fastembed() -> bool:
    """Prompt to install fastembed, then install via pip or uv.

    Detects uv tool environments (which lack pip) and uses
    ``uv pip install`` instead. Works in uv tool environments,
    regular venvs, and system Python alike.

    Returns:
        True if fastembed is available after the attempt (import succeeds).
        False if installation was declined, failed, or import still fails.
    """
    print("📦 FastEmbed is not installed (needed for local embeddings).")

    interactive = sys.stdin.isatty()
    if interactive:
        answer = input("   Install it now? [Y/n] ").strip().lower()
        if answer and answer not in ("y", "yes"):
            print()
            print("   To install manually:")
            _print_manual_install_hint()
            print("   FastEmbed is required for embeddings.")
            return False
    else:
        print("   Auto-installing (non-interactive)...")

    if not _install_fastembed(interactive):
        print("❌ Installation failed.")
        _print_manual_install_hint()
        return False

    # Verify in a clean subprocess — the current process may have
    # cached the failed import in sys.modules.
    result = subprocess.run(
        [sys.executable, "-c", "import fastembed"],
        capture_output=True,
    )
    if result.returncode == 0:
        print("✅ FastEmbed installed successfully.")
        return True

    print("❌ Install completed but import still fails.")
    _print_manual_install_hint()
    return False


def _detect_provider(args: argparse.Namespace) -> str:
    """Determine which embedding provider to use.

    FastEmbed is the only supported provider.
    """
    return "fastembed"

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

    # Validate FastEmbed is installed — auto-install if missing
    try:
        import fastembed as _  # noqa: F401
    except ImportError:
        if not _auto_install_fastembed():
            return 1

    # Build config
    config_content = FASTEMBED_CONFIG_TEMPLATE.format(
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

    # Post-install guidance
    print()
    print("📦 FastEmbed — local ONNX embeddings, zero cloud.")
    print("   First run downloads a ~130MB model, then it's instant.")

    # Set up MCP integrations
    if args.claude_code:
        _setup_claude_code_mcp()

    if getattr(args, "claude_desktop", False):
        _setup_claude_desktop_mcp()

    if args.codex:
        _setup_codex_mcp()

    # Set up auto-capture instructions
    if args.auto_capture:
        _setup_auto_capture(
            claude_code=args.claude_code,
            codex=args.codex,
        )

    # Set up background service if requested
    if getattr(args, "service", False):
        from .service import cmd_service
        print()
        result = cmd_service("install")
        if result == 0:
            cmd_service("start")
    else:
        print()
        print("🚀 Start the server:")
        print("   tribalmemory serve")
        print()
        print("💡 Or run as a background service:")
        print("   tribalmemory service install && tribalmemory service start")

    print()
    print("🧠 Or use with Claude Code (MCP):")
    print("   tribalmemory-mcp")

    if not args.auto_capture:
        print()
        print("💡 Want your agents to remember things automatically?")
        print("   tribalmemory init --auto-capture --force")

    return 0


def _setup_auto_capture(claude_code: bool = False, codex: bool = False) -> None:
    """Write auto-capture instructions to agent instruction files.
    
    Appends memory usage instructions so agents proactively use
    tribal_remember and tribal_recall without being explicitly asked.
    
    Writes to:
    - ~/.claude/CLAUDE.md (Claude Code) — when --claude-code is set
    - ~/.codex/AGENTS.md (Codex CLI) — when --codex is set
    - Both files if neither flag is set (covers the common case)
    
    Skips if instructions are already present (idempotent).
    """
    # If no specific flag, write to both (default behavior)
    if not claude_code and not codex:
        claude_code = codex = True

    targets = []
    if claude_code:
        targets.append(("Claude Code", Path.home() / CLAUDE_INSTRUCTIONS_FILE))
    if codex:
        targets.append(("Codex CLI", Path.home() / CODEX_INSTRUCTIONS_FILE))

    for label, instructions_path in targets:
        _write_instructions_file(instructions_path, label)


def _write_instructions_file(instructions_path: Path, label: str) -> None:
    """Write auto-capture instructions to a single instructions file."""
    instructions_path.parent.mkdir(parents=True, exist_ok=True)

    if instructions_path.exists():
        existing = instructions_path.read_text()
        if _AUTO_CAPTURE_MARKER in existing:
            print(f"✅ Auto-capture already present in {label}: {instructions_path}")
            return
        # Append to existing file
        if not existing.endswith("\n"):
            existing += "\n"
        instructions_path.write_text(existing + AUTO_CAPTURE_INSTRUCTIONS)
    else:
        instructions_path.write_text(AUTO_CAPTURE_INSTRUCTIONS.lstrip("\n"))

    print(f"✅ Auto-capture instructions written for {label}: {instructions_path}")


def _build_mcp_entry(mcp_command: str) -> dict:
    """Build MCP config entry."""
    mcp_entry: dict = {"command": mcp_command}
    return mcp_entry


def _setup_claude_code_mcp() -> None:
    """Add Tribal Memory to Claude Code CLI's MCP configuration.

    Claude Code CLI reads MCP servers from ``~/.claude.json`` (user scope).
    The CLI inherits the user's shell PATH, so the bare command name works
    fine — but we still resolve the absolute path for robustness.
    """
    claude_cli_config = Path.home() / ".claude.json"
    mcp_command = _resolve_mcp_command()
    mcp_entry = _build_mcp_entry(mcp_command)

    _update_mcp_config(claude_cli_config, mcp_entry, create_if_missing=True)
    print(f"✅ Claude Code CLI config updated: {claude_cli_config}")


def _setup_claude_desktop_mcp() -> None:
    """Add Tribal Memory to Claude Desktop's MCP configuration.

    Claude Desktop does NOT inherit the user's shell PATH (e.g.
    ``~/.local/bin`` from uv/pipx). The absolute path to the binary
    is resolved at init time and written into the config so the app
    can find it regardless of its limited PATH.

    Config path is platform-specific:
      - macOS: ``~/Library/Application Support/Claude/claude_desktop_config.json``
      - Windows: ``%APPDATA%/Claude/claude_desktop_config.json``
      - Linux: ``~/.claude/claude_desktop_config.json``
    """
    desktop_path = _get_claude_desktop_config_path()
    mcp_command = _resolve_mcp_command()

    if mcp_command == "tribalmemory-mcp":
        # Couldn't resolve — warn the user early
        print("⚠️  Could not find tribalmemory-mcp on PATH or in ~/.local/bin")
        print("   Claude Desktop needs the absolute path to the binary.")
        print("   After installing, run: tribalmemory init --claude-desktop --force")
        print()

    mcp_entry = _build_mcp_entry(mcp_command)

    # Ensure parent directory exists (macOS Application Support/Claude/)
    desktop_path.parent.mkdir(parents=True, exist_ok=True)

    _update_mcp_config(desktop_path, mcp_entry, create_if_missing=True)
    print(f"✅ Claude Desktop config updated: {desktop_path}")
    print(f"   Binary: {mcp_command}")


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


def _setup_codex_mcp() -> None:
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
    # Load .env before config so API keys are in os.environ
    load_env_file()

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
    load_env_file()

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
    init_parser = subparsers.add_parser(
        "init", help="Initialize config and MCP integration"
    )
    init_parser.add_argument("--claude-code", action="store_true",
                             help="Configure Claude Code CLI MCP integration")
    init_parser.add_argument("--claude-desktop", action="store_true",
                             help="Configure Claude Desktop app MCP integration")
    init_parser.add_argument("--codex", action="store_true",
                             help="Configure Codex CLI MCP integration")
    init_parser.add_argument("--auto-capture", action="store_true",
                             help="Enable auto-capture (writes instructions to agent config files)")
    init_parser.add_argument("--service", action="store_true",
                             help="Install and start as a background service (systemd/launchd)")
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

    # service
    service_parser = subparsers.add_parser(
        "service", help="Manage the background server service"
    )
    service_parser.add_argument(
        "action",
        choices=["install", "uninstall", "start", "stop", "status"],
        help="Service action to perform",
    )

    args = parser.parse_args()

    if args.command == "init":
        sys.exit(cmd_init(args))
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "mcp":
        cmd_mcp(args)
    elif args.command == "service":
        from .service import cmd_service
        sys.exit(cmd_service(args.action))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
