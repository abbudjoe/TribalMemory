"""Tests for the tribalmemory CLI (init, serve, mcp commands).

TDD: RED → GREEN → REFACTOR
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from tribalmemory.cli import cmd_init, main, _resolve_mcp_command


class FakeArgs:
    """Fake argparse namespace."""
    def __init__(self, **kwargs):
        self.local = kwargs.get("local", False)
        self.claude_code = kwargs.get("claude_code", False)
        self.codex = kwargs.get("codex", False)
        self.instance_id = kwargs.get("instance_id", None)
        self.force = kwargs.get("force", False)


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Set up isolated CLI environment using tmp_path as home."""
    tribal_dir = tmp_path / ".tribal-memory"
    config_file = tribal_dir / "config.yaml"
    monkeypatch.setattr("tribalmemory.cli.TRIBAL_DIR", tribal_dir)
    monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", config_file)
    # Patch Path.home() so _setup_claude_code_mcp and
    # _setup_codex_mcp write into tmp_path
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    return tmp_path


class TestInitCommand:
    """Tests for `tribalmemory init`."""

    def test_init_creates_config_file(self, cli_env):
        """init should create ~/.tribal-memory/config.yaml."""
        result = cmd_init(FakeArgs())

        assert result == 0
        config = (cli_env / ".tribal-memory" / "config.yaml")
        assert config.exists()
        content = config.read_text()
        assert "instance_id: default" in content
        assert "text-embedding-3-small" in content

    def test_init_local_mode_uses_ollama(self, cli_env):
        """init --local should generate Ollama config."""
        result = cmd_init(FakeArgs(local=True))

        assert result == 0
        content = (cli_env / ".tribal-memory" / "config.yaml").read_text()
        assert "localhost:11434" in content
        assert "nomic-embed-text" in content
        assert "768" in content
        assert "api_key not needed" in content

    def test_init_custom_instance_id(self, cli_env):
        """init --instance-id should set custom ID."""
        result = cmd_init(FakeArgs(instance_id="my-agent"))

        assert result == 0
        content = (cli_env / ".tribal-memory" / "config.yaml").read_text()
        assert "instance_id: my-agent" in content

    def test_init_refuses_overwrite_without_force(self, cli_env, monkeypatch):
        """init should refuse to overwrite existing config."""
        config_dir = cli_env / ".tribal-memory"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text("existing config")
        monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", config_file)

        result = cmd_init(FakeArgs())

        assert result == 1
        assert config_file.read_text() == "existing config"

    def test_init_force_overwrites(self, cli_env, monkeypatch):
        """init --force should overwrite existing config."""
        config_dir = cli_env / ".tribal-memory"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text("old config")
        monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", config_file)

        result = cmd_init(FakeArgs(force=True))

        assert result == 0
        assert "old config" not in config_file.read_text()

    def test_init_claude_code_creates_mcp_config(self, cli_env):
        """init --claude-code should write Claude Code CLI config (~/.claude.json)."""
        result = cmd_init(FakeArgs(claude_code=True))

        assert result == 0
        claude_config = cli_env / ".claude.json"
        assert claude_config.exists()
        mcp = json.loads(claude_config.read_text())
        assert "tribal-memory" in mcp["mcpServers"]
        # Command should be the resolved path (or bare name as fallback)
        cmd = mcp["mcpServers"]["tribal-memory"]["command"]
        assert cmd.endswith("tribalmemory-mcp")

    def test_init_claude_code_local_adds_env(self, cli_env):
        """init --local --claude-code should set api_base env."""
        result = cmd_init(FakeArgs(local=True, claude_code=True))

        assert result == 0
        claude_config = cli_env / ".claude.json"
        mcp = json.loads(claude_config.read_text())
        env = mcp["mcpServers"]["tribal-memory"]["env"]
        assert "TRIBAL_MEMORY_EMBEDDING_API_BASE" in env
        assert "localhost:11434" in env["TRIBAL_MEMORY_EMBEDDING_API_BASE"]

    def test_init_claude_code_creates_desktop_config(self, cli_env):
        """init --claude-code should create both CLI and Desktop configs."""
        result = cmd_init(FakeArgs(claude_code=True))

        assert result == 0
        # CLI config
        cli_config = cli_env / ".claude.json"
        assert cli_config.exists()
        cli_mcp = json.loads(cli_config.read_text())
        assert "tribal-memory" in cli_mcp["mcpServers"]
        # Desktop config (Linux path under fake home)
        desktop_config = cli_env / ".claude" / "claude_desktop_config.json"
        assert desktop_config.exists()
        desktop_mcp = json.loads(desktop_config.read_text())
        assert "tribal-memory" in desktop_mcp["mcpServers"]

    def test_init_claude_code_backs_up_invalid_json(self, cli_env):
        """init --claude-code should backup invalid JSON config before replacing."""
        cli_config = cli_env / ".claude.json"
        cli_config.write_text("not valid json {{{")

        result = cmd_init(FakeArgs(claude_code=True))

        assert result == 0
        # Original should be replaced with valid config
        mcp = json.loads(cli_config.read_text())
        assert "tribal-memory" in mcp["mcpServers"]
        # Backup should exist with the old content
        backup = cli_env / ".claude.json.bak"
        assert backup.exists()
        assert backup.read_text() == "not valid json {{{"

    def test_init_claude_code_preserves_existing_desktop_entries(self, cli_env):
        """init --claude-code should not clobber existing Desktop MCP entries."""
        desktop_dir = cli_env / ".claude"
        desktop_dir.mkdir(parents=True, exist_ok=True)
        desktop_config = desktop_dir / "claude_desktop_config.json"
        desktop_config.write_text(json.dumps({"mcpServers": {"other": {"command": "other-cmd"}}}) + "\n")

        result = cmd_init(FakeArgs(claude_code=True))

        assert result == 0
        desktop_mcp = json.loads(desktop_config.read_text())
        assert "tribal-memory" in desktop_mcp["mcpServers"]
        assert "other" in desktop_mcp["mcpServers"]  # preserved


class TestResolveMcpCommand:
    """Tests for _resolve_mcp_command — full path resolution."""

    def test_resolve_uses_shutil_which(self):
        """Should use shutil.which to find the binary."""
        with patch("tribalmemory.cli.shutil.which", return_value="/usr/local/bin/tribalmemory-mcp"):
            result = _resolve_mcp_command()
        assert result == "/usr/local/bin/tribalmemory-mcp"

    def test_resolve_checks_local_bin_fallback(self, tmp_path):
        """Should check ~/.local/bin when shutil.which fails."""
        local_bin = tmp_path / ".local" / "bin"
        local_bin.mkdir(parents=True)
        mcp_binary = local_bin / "tribalmemory-mcp"
        mcp_binary.touch()
        mcp_binary.chmod(0o755)  # Must be executable

        with patch("tribalmemory.cli.shutil.which", return_value=None), \
             patch.object(Path, "home", staticmethod(lambda: tmp_path)):
            result = _resolve_mcp_command()
        assert result == str(mcp_binary)

    def test_resolve_skips_non_executable_fallback(self, tmp_path):
        """Should skip files in fallback dirs that aren't executable."""
        local_bin = tmp_path / ".local" / "bin"
        local_bin.mkdir(parents=True)
        mcp_binary = local_bin / "tribalmemory-mcp"
        mcp_binary.touch()
        mcp_binary.chmod(0o644)  # NOT executable

        with patch("tribalmemory.cli.shutil.which", return_value=None), \
             patch.object(Path, "home", staticmethod(lambda: tmp_path)):
            result = _resolve_mcp_command()
        assert result == "tribalmemory-mcp"  # Falls back to bare name

    def test_resolve_falls_back_to_bare_name(self, tmp_path):
        """Should fall back to bare command name when not found anywhere."""
        with patch("tribalmemory.cli.shutil.which", return_value=None), \
             patch.object(Path, "home", staticmethod(lambda: tmp_path)):
            result = _resolve_mcp_command()
        assert result == "tribalmemory-mcp"

    def test_init_claude_code_uses_full_path(self, cli_env):
        """init --claude-code should write the resolved full path to configs."""
        fake_path = "/home/test/.local/bin/tribalmemory-mcp"
        with patch("tribalmemory.cli.shutil.which", return_value=fake_path):
            result = cmd_init(FakeArgs(claude_code=True))

        assert result == 0

        # CLI config should have full path
        cli_config = cli_env / ".claude.json"
        mcp = json.loads(cli_config.read_text())
        assert mcp["mcpServers"]["tribal-memory"]["command"] == fake_path

        # Desktop config should also have full path
        desktop_config = cli_env / ".claude" / "claude_desktop_config.json"
        desktop_mcp = json.loads(desktop_config.read_text())
        assert desktop_mcp["mcpServers"]["tribal-memory"]["command"] == fake_path


class TestCodexIntegration:
    """Tests for Codex CLI MCP integration."""

    def test_codex_creates_config_toml(self, cli_env):
        """init --codex should write Codex MCP config."""
        result = cmd_init(FakeArgs(codex=True))

        assert result == 0
        codex_config = cli_env / ".codex" / "config.toml"
        assert codex_config.exists()
        content = codex_config.read_text()
        assert "[mcp_servers.tribal-memory]" in content
        assert "tribalmemory-mcp" in content  # resolved or bare

    def test_codex_local_adds_env(self, cli_env):
        """init --local --codex should add api_base env."""
        result = cmd_init(FakeArgs(local=True, codex=True))

        assert result == 0
        codex_config = cli_env / ".codex" / "config.toml"
        content = codex_config.read_text()
        assert "TRIBAL_MEMORY_EMBEDDING_API_BASE" in content
        assert "localhost:11434" in content

    def test_codex_uses_full_path(self, cli_env):
        """init --codex should write the resolved full path."""
        fake_path = "/home/test/.local/bin/tribalmemory-mcp"
        with patch("tribalmemory.cli.shutil.which", return_value=fake_path):
            result = cmd_init(FakeArgs(codex=True))

        assert result == 0
        codex_config = cli_env / ".codex" / "config.toml"
        content = codex_config.read_text()
        assert fake_path in content


class TestMainEntrypoint:
    """Tests for the main() CLI dispatcher."""

    def test_no_command_shows_help(self):
        """Running with no args should show help and exit."""
        with patch("sys.argv", ["tribalmemory"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_init_dispatches(self, cli_env):
        """main() should dispatch 'init' to cmd_init."""
        with patch("sys.argv", ["tribalmemory", "init"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        assert (cli_env / ".tribal-memory" / "config.yaml").exists()
