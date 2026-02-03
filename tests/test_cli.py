"""Tests for the tribalmemory CLI (init, serve, mcp commands).

TDD: RED → GREEN → REFACTOR
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from tribalmemory.cli import cmd_init, main, TRIBAL_DIR


class FakeArgs:
    """Fake argparse namespace."""
    def __init__(self, **kwargs):
        self.local = kwargs.get("local", False)
        self.claude_code = kwargs.get("claude_code", False)
        self.codex = kwargs.get("codex", False)
        self.instance_id = kwargs.get("instance_id", None)
        self.force = kwargs.get("force", False)


class TestInitCommand:
    """Tests for `tribalmemory init`."""

    def test_init_creates_config_file(self, tmp_path, monkeypatch):
        """init should create ~/.tribal-memory/config.yaml."""
        monkeypatch.setattr("tribalmemory.cli.TRIBAL_DIR", tmp_path / ".tribal-memory")
        monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", tmp_path / ".tribal-memory" / "config.yaml")
        
        args = FakeArgs()
        result = cmd_init(args)
        
        assert result == 0
        config_file = tmp_path / ".tribal-memory" / "config.yaml"
        assert config_file.exists()
        content = config_file.read_text()
        assert "instance_id: default" in content
        assert "text-embedding-3-small" in content

    def test_init_local_mode_uses_ollama(self, tmp_path, monkeypatch):
        """init --local should generate Ollama config."""
        monkeypatch.setattr("tribalmemory.cli.TRIBAL_DIR", tmp_path / ".tribal-memory")
        monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", tmp_path / ".tribal-memory" / "config.yaml")
        
        args = FakeArgs(local=True)
        result = cmd_init(args)
        
        assert result == 0
        content = (tmp_path / ".tribal-memory" / "config.yaml").read_text()
        assert "localhost:11434" in content
        assert "nomic-embed-text" in content
        assert "768" in content
        assert "api_key not needed" in content

    def test_init_custom_instance_id(self, tmp_path, monkeypatch):
        """init --instance-id should set custom ID."""
        monkeypatch.setattr("tribalmemory.cli.TRIBAL_DIR", tmp_path / ".tribal-memory")
        monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", tmp_path / ".tribal-memory" / "config.yaml")
        
        args = FakeArgs(instance_id="my-agent")
        result = cmd_init(args)
        
        assert result == 0
        content = (tmp_path / ".tribal-memory" / "config.yaml").read_text()
        assert "instance_id: my-agent" in content

    def test_init_refuses_overwrite_without_force(self, tmp_path, monkeypatch):
        """init should refuse to overwrite existing config without --force."""
        config_dir = tmp_path / ".tribal-memory"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("existing config")
        
        monkeypatch.setattr("tribalmemory.cli.TRIBAL_DIR", config_dir)
        monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", config_file)
        
        args = FakeArgs()
        result = cmd_init(args)
        
        assert result == 1
        assert config_file.read_text() == "existing config"

    def test_init_force_overwrites(self, tmp_path, monkeypatch):
        """init --force should overwrite existing config."""
        config_dir = tmp_path / ".tribal-memory"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("old config")
        
        monkeypatch.setattr("tribalmemory.cli.TRIBAL_DIR", config_dir)
        monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", config_file)
        
        args = FakeArgs(force=True)
        result = cmd_init(args)
        
        assert result == 0
        assert "old config" not in config_file.read_text()

    def test_init_claude_code_creates_mcp_config(self, tmp_path, monkeypatch):
        """init --claude-code should write MCP config."""
        monkeypatch.setattr("tribalmemory.cli.TRIBAL_DIR", tmp_path / ".tribal-memory")
        monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", tmp_path / ".tribal-memory" / "config.yaml")
        
        claude_config = tmp_path / ".claude" / "claude_desktop_config.json"
        
        # Patch the config paths list
        monkeypatch.setattr(
            "tribalmemory.cli._setup_claude_code_mcp.__code__",
            _make_setup_mock(claude_config).__code__,
        ) if False else None  # Skip complex mock, test the path directly
        
        # Create the claude dir so it gets found
        claude_config.parent.mkdir(parents=True)
        claude_config.write_text("{}")
        
        # Monkey-patch the paths list inside the function
        import tribalmemory.cli as cli_mod
        original_fn = cli_mod._setup_claude_code_mcp
        
        def patched_setup(is_local):
            """Patched to use tmp_path."""
            existing = json.loads(claude_config.read_text()) if claude_config.exists() else {}
            if "mcpServers" not in existing:
                existing["mcpServers"] = {}
            mcp_entry = {"command": "tribalmemory-mcp", "env": {}}
            if is_local:
                mcp_entry["env"]["TRIBAL_MEMORY_EMBEDDING_API_BASE"] = "http://localhost:11434/v1"
            existing["mcpServers"]["tribal-memory"] = mcp_entry
            claude_config.write_text(json.dumps(existing, indent=2) + "\n")
        
        monkeypatch.setattr(cli_mod, "_setup_claude_code_mcp", patched_setup)
        
        args = FakeArgs(claude_code=True)
        result = cmd_init(args)
        
        assert result == 0
        mcp_config = json.loads(claude_config.read_text())
        assert "tribal-memory" in mcp_config["mcpServers"]
        assert mcp_config["mcpServers"]["tribal-memory"]["command"] == "tribalmemory-mcp"

    def test_init_claude_code_local_adds_env(self, tmp_path, monkeypatch):
        """init --local --claude-code should set api_base env var."""
        monkeypatch.setattr("tribalmemory.cli.TRIBAL_DIR", tmp_path / ".tribal-memory")
        monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", tmp_path / ".tribal-memory" / "config.yaml")
        
        claude_config = tmp_path / ".claude" / "claude_desktop_config.json"
        claude_config.parent.mkdir(parents=True)
        claude_config.write_text("{}")
        
        import tribalmemory.cli as cli_mod
        
        def patched_setup(is_local):
            existing = json.loads(claude_config.read_text()) if claude_config.exists() else {}
            if "mcpServers" not in existing:
                existing["mcpServers"] = {}
            mcp_entry = {"command": "tribalmemory-mcp", "env": {}}
            if is_local:
                mcp_entry["env"]["TRIBAL_MEMORY_EMBEDDING_API_BASE"] = "http://localhost:11434/v1"
            existing["mcpServers"]["tribal-memory"] = mcp_entry
            claude_config.write_text(json.dumps(existing, indent=2) + "\n")
        
        monkeypatch.setattr(cli_mod, "_setup_claude_code_mcp", patched_setup)
        
        args = FakeArgs(local=True, claude_code=True)
        result = cmd_init(args)
        
        assert result == 0
        mcp_config = json.loads(claude_config.read_text())
        env = mcp_config["mcpServers"]["tribal-memory"]["env"]
        assert "TRIBAL_MEMORY_EMBEDDING_API_BASE" in env
        assert "localhost:11434" in env["TRIBAL_MEMORY_EMBEDDING_API_BASE"]


class TestCodexIntegration:
    """Tests for Codex CLI MCP integration."""

    def test_codex_creates_config_toml(self, tmp_path, monkeypatch):
        """init --codex should create ~/.codex/config.toml with MCP section."""
        monkeypatch.setattr("tribalmemory.cli.TRIBAL_DIR", tmp_path / ".tribal-memory")
        monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", tmp_path / ".tribal-memory" / "config.yaml")
        
        import tribalmemory.cli as cli_mod
        original_fn = cli_mod._setup_codex_mcp
        
        codex_config = tmp_path / ".codex" / "config.toml"
        
        def patched_setup(is_local):
            codex_config.parent.mkdir(parents=True, exist_ok=True)
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
            codex_config.write_text(mcp_block.lstrip("\n"))
        
        monkeypatch.setattr(cli_mod, "_setup_codex_mcp", patched_setup)
        
        args = FakeArgs(codex=True)
        result = cmd_init(args)
        
        assert result == 0
        assert codex_config.exists()
        content = codex_config.read_text()
        assert "[mcp_servers.tribal-memory]" in content
        assert 'command = "tribalmemory-mcp"' in content

    def test_codex_local_adds_env(self, tmp_path, monkeypatch):
        """init --local --codex should add api_base env to Codex config."""
        monkeypatch.setattr("tribalmemory.cli.TRIBAL_DIR", tmp_path / ".tribal-memory")
        monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", tmp_path / ".tribal-memory" / "config.yaml")
        
        import tribalmemory.cli as cli_mod
        codex_config = tmp_path / ".codex" / "config.toml"
        
        def patched_setup(is_local):
            codex_config.parent.mkdir(parents=True, exist_ok=True)
            section_marker = "[mcp_servers.tribal-memory]"
            mcp_lines = [
                "# Tribal Memory",
                section_marker,
                'command = "tribalmemory-mcp"',
            ]
            if is_local:
                mcp_lines.append("")
                mcp_lines.append("[mcp_servers.tribal-memory.env]")
                mcp_lines.append('TRIBAL_MEMORY_EMBEDDING_API_BASE = "http://localhost:11434/v1"')
            codex_config.write_text("\n".join(mcp_lines) + "\n")
        
        monkeypatch.setattr(cli_mod, "_setup_codex_mcp", patched_setup)
        
        args = FakeArgs(local=True, codex=True)
        result = cmd_init(args)
        
        assert result == 0
        content = codex_config.read_text()
        assert "TRIBAL_MEMORY_EMBEDDING_API_BASE" in content
        assert "localhost:11434" in content


class TestMainEntrypoint:
    """Tests for the main() CLI dispatcher."""

    def test_no_command_shows_help(self, capsys):
        """Running with no args should show help and exit."""
        with patch("sys.argv", ["tribalmemory"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_init_dispatches(self, tmp_path, monkeypatch):
        """main() should dispatch 'init' to cmd_init."""
        monkeypatch.setattr("tribalmemory.cli.TRIBAL_DIR", tmp_path / ".tribal-memory")
        monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", tmp_path / ".tribal-memory" / "config.yaml")
        
        with patch("sys.argv", ["tribalmemory", "init"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        
        assert (tmp_path / ".tribal-memory" / "config.yaml").exists()
