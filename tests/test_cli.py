"""Tests for the tribalmemory CLI (init, serve, mcp commands).

TDD: RED → GREEN → REFACTOR
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from tribalmemory.cli import (
    cmd_init, main, _resolve_mcp_command, load_env_file,
    AUTO_CAPTURE_INSTRUCTIONS, CLAUDE_INSTRUCTIONS_FILE,
    CODEX_INSTRUCTIONS_FILE, ENV_FILE,
)


class FakeArgs:
    """Fake argparse namespace."""
    def __init__(self, **kwargs):
        self.fastembed = kwargs.get("fastembed", False)
        self.openai = kwargs.get("openai", False)
        self.ollama = kwargs.get("ollama", False)
        self.local = kwargs.get("local", False)
        self.claude_code = kwargs.get("claude_code", False)
        self.claude_desktop = kwargs.get("claude_desktop", False)
        self.codex = kwargs.get("codex", False)
        self.instance_id = kwargs.get("instance_id", None)
        self.force = kwargs.get("force", False)
        self.auto_capture = kwargs.get("auto_capture", False)


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Set up isolated CLI environment using tmp_path as home."""
    tribal_dir = tmp_path / ".tribal-memory"
    config_file = tribal_dir / "config.yaml"
    env_file = tribal_dir / ".env"
    monkeypatch.setattr("tribalmemory.cli.TRIBAL_DIR", tribal_dir)
    monkeypatch.setattr("tribalmemory.cli.CONFIG_FILE", config_file)
    monkeypatch.setattr("tribalmemory.cli.ENV_FILE", env_file)
    # Patch Path.home() so _setup_claude_code_mcp and
    # _setup_codex_mcp write into tmp_path
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    return tmp_path


class TestInitCommand:
    """Tests for `tribalmemory init`."""

    def test_init_default_uses_fastembed(self, cli_env):
        """init with no flags should generate FastEmbed config."""
        result = cmd_init(FakeArgs())

        assert result == 0
        config = (cli_env / ".tribal-memory" / "config.yaml")
        assert config.exists()
        content = config.read_text()
        assert "instance_id: default" in content
        assert "provider: fastembed" in content
        assert "BAAI/bge-small-en-v1.5" in content
        assert "dimensions: 384" in content

    def test_init_fastembed_explicit(self, cli_env):
        """init --fastembed should also generate FastEmbed config."""
        result = cmd_init(FakeArgs(fastembed=True))

        assert result == 0
        content = (cli_env / ".tribal-memory" / "config.yaml").read_text()
        assert "provider: fastembed" in content
        assert "BAAI/bge-small-en-v1.5" in content

    def test_init_openai_prompts_for_key(self, cli_env, monkeypatch):
        """init --openai should prompt for key and write to .env, not config."""
        monkeypatch.setattr("builtins.input", lambda _: "sk-test-key-123")
        monkeypatch.setattr("sys.stdin", type("FakeTTY", (), {"isatty": lambda s: True})())

        result = cmd_init(FakeArgs(openai=True))

        assert result == 0
        # Config should NOT contain the API key
        config = (cli_env / ".tribal-memory" / "config.yaml").read_text()
        assert "provider: openai" in config
        assert "text-embedding-3-small" in config
        assert "sk-test-key-123" not in config
        # .env should contain the key with 600 permissions
        env_path = cli_env / ".tribal-memory" / ".env"
        assert env_path.exists()
        assert "sk-test-key-123" in env_path.read_text()
        assert (env_path.stat().st_mode & 0o777) == 0o600

    def test_init_openai_uses_env_key_non_interactive(self, cli_env, monkeypatch):
        """init --openai should use OPENAI_API_KEY env var in non-interactive mode."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key-456")
        monkeypatch.setattr("sys.stdin", type("FakeNonTTY", (), {"isatty": lambda s: False})())

        result = cmd_init(FakeArgs(openai=True))

        assert result == 0
        env_path = cli_env / ".tribal-memory" / ".env"
        assert "sk-env-key-456" in env_path.read_text()

    def test_init_openai_fails_without_key_non_interactive(
        self, cli_env, monkeypatch
    ):
        """init --openai should fail when no key and no TTY."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr("sys.stdin", type("FakeNonTTY", (), {"isatty": lambda s: False})())

        with pytest.raises(SystemExit):
            cmd_init(FakeArgs(openai=True))

    def test_init_fastembed_install_declined(
        self, cli_env, monkeypatch
    ):
        """init should fail when user declines fastembed install."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        monkeypatch.setattr("builtins.input", lambda _: "n")
        monkeypatch.setattr(
            "sys.stdin",
            type("FakeTTY", (), {"isatty": lambda s: True})(),
        )
        result = cmd_init(FakeArgs())
        assert result == 1

    def test_init_fastembed_install_accepted(
        self, cli_env, monkeypatch
    ):
        """init should install fastembed when user accepts."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        monkeypatch.setattr(
            "sys.stdin",
            type("FakeTTY", (), {"isatty": lambda s: True})(),
        )
        # Mock subprocess: install succeeds, verify succeeds
        import subprocess as sp
        monkeypatch.setattr(
            sp, "check_call", lambda *a, **kw: None
        )
        monkeypatch.setattr(
            sp, "run",
            lambda *a, **kw: type("R", (), {"returncode": 0})(),
        )
        result = cmd_init(FakeArgs())
        assert result == 0
        config = (cli_env / ".tribal-memory" / "config.yaml").read_text()
        assert "provider: fastembed" in config

    def test_init_fastembed_install_non_interactive(
        self, cli_env, monkeypatch
    ):
        """init should auto-install fastembed in non-interactive mode."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        monkeypatch.setattr(
            "sys.stdin",
            type("FakeNonTTY", (), {"isatty": lambda s: False})(),
        )
        import subprocess as sp
        monkeypatch.setattr(
            sp, "check_call", lambda *a, **kw: None
        )
        monkeypatch.setattr(
            sp, "run",
            lambda *a, **kw: type("R", (), {"returncode": 0})(),
        )
        result = cmd_init(FakeArgs())
        assert result == 0

    def test_init_fastembed_install_fails(
        self, cli_env, monkeypatch
    ):
        """init should handle installation failure gracefully."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        monkeypatch.setattr(
            "sys.stdin",
            type("FakeTTY", (), {"isatty": lambda s: True})(),
        )
        import subprocess as sp
        monkeypatch.setattr(
            sp, "check_call",
            lambda *a, **kw: (_ for _ in ()).throw(
                sp.CalledProcessError(1, "pip")
            ),
        )
        result = cmd_init(FakeArgs())
        assert result == 1

    def test_init_fastembed_uv_environment_install(
        self, cli_env, monkeypatch
    ):
        """init should use uv pip install in uv tool environments."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        monkeypatch.setattr(
            "sys.stdin",
            type("FakeTTY", (), {"isatty": lambda s: True})(),
        )

        # Mock _is_uv_environment to return True
        monkeypatch.setattr(
            "tribalmemory.cli._is_uv_environment", lambda: True
        )
        # Mock shutil.which to find uv
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/uv" if cmd == "uv" else None)

        # Track which commands were called
        install_cmds = []
        import subprocess as sp

        def mock_check_call(cmd, **kw):
            install_cmds.append(cmd)

        monkeypatch.setattr(sp, "check_call", mock_check_call)
        monkeypatch.setattr(
            sp, "run",
            lambda *a, **kw: type("R", (), {"returncode": 0})(),
        )

        result = cmd_init(FakeArgs())
        assert result == 0
        # Should have used uv pip install with correct command structure
        assert len(install_cmds) == 1, (
            f"Expected exactly 1 install command, got: {install_cmds}"
        )
        uv_cmd = install_cmds[0]
        assert uv_cmd[0] == "/usr/bin/uv"
        assert uv_cmd[1:3] == ["pip", "install"]
        assert "--python" in uv_cmd
        assert "fastembed" in uv_cmd

    def test_init_fastembed_uv_install_fails_no_pip_fallback(
        self, cli_env, monkeypatch
    ):
        """init should NOT fall back to pip when uv install fails in uv env."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        monkeypatch.setattr(
            "sys.stdin",
            type("FakeTTY", (), {"isatty": lambda s: True})(),
        )
        monkeypatch.setattr(
            "tribalmemory.cli._is_uv_environment", lambda: True
        )
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/uv" if cmd == "uv" else None)

        import subprocess as sp
        call_count = [0]

        def mock_check_call(cmd, **kw):
            call_count[0] += 1
            raise sp.CalledProcessError(1, "uv")

        monkeypatch.setattr(sp, "check_call", mock_check_call)

        result = cmd_init(FakeArgs())
        assert result == 1
        # Should only have tried uv, NOT pip
        assert call_count[0] == 1

    def test_init_fastembed_uv_env_no_uv_binary(
        self, cli_env, monkeypatch
    ):
        """init should fail gracefully when in uv env but uv not on PATH."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        monkeypatch.setattr(
            "sys.stdin",
            type("FakeTTY", (), {"isatty": lambda s: True})(),
        )
        monkeypatch.setattr(
            "tribalmemory.cli._is_uv_environment", lambda: True
        )
        # uv not found on PATH
        monkeypatch.setattr("shutil.which", lambda cmd: None)

        result = cmd_init(FakeArgs())
        assert result == 1

    def test_init_ollama(self, cli_env):
        """init --ollama should generate Ollama config."""
        result = cmd_init(FakeArgs(ollama=True))

        assert result == 0
        content = (cli_env / ".tribal-memory" / "config.yaml").read_text()
        assert "localhost:11434" in content
        assert "nomic-embed-text" in content
        assert "768" in content

    def test_init_local_is_ollama_alias(self, cli_env):
        """init --local (deprecated) should behave like --ollama."""
        result = cmd_init(FakeArgs(local=True))

        assert result == 0
        content = (cli_env / ".tribal-memory" / "config.yaml").read_text()
        assert "localhost:11434" in content
        assert "nomic-embed-text" in content
        assert "768" in content

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

    def test_init_claude_code_ollama_adds_env(self, cli_env):
        """init --ollama --claude-code should set api_base env."""
        result = cmd_init(FakeArgs(ollama=True, claude_code=True))

        assert result == 0
        claude_config = cli_env / ".claude.json"
        mcp = json.loads(claude_config.read_text())
        env = mcp["mcpServers"]["tribal-memory"]["env"]
        assert "TRIBAL_MEMORY_EMBEDDING_API_BASE" in env
        assert "localhost:11434" in env["TRIBAL_MEMORY_EMBEDDING_API_BASE"]

    def test_init_claude_code_does_not_touch_desktop_config(self, cli_env):
        """init --claude-code should only create CLI config, not Desktop."""
        result = cmd_init(FakeArgs(claude_code=True))

        assert result == 0
        # CLI config should exist
        cli_config = cli_env / ".claude.json"
        assert cli_config.exists()
        cli_mcp = json.loads(cli_config.read_text())
        assert "tribal-memory" in cli_mcp["mcpServers"]
        # Desktop config should NOT be created by --claude-code
        desktop_config = cli_env / ".claude" / "claude_desktop_config.json"
        assert not desktop_config.exists()

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

    def test_init_claude_desktop_creates_config(self, cli_env):
        """init --claude-desktop should create Desktop config with absolute path."""
        result = cmd_init(FakeArgs(claude_desktop=True))

        assert result == 0
        desktop_config = cli_env / ".claude" / "claude_desktop_config.json"
        assert desktop_config.exists()
        mcp = json.loads(desktop_config.read_text())
        assert "tribal-memory" in mcp["mcpServers"]
        cmd = mcp["mcpServers"]["tribal-memory"]["command"]
        assert cmd.endswith("tribalmemory-mcp")

    def test_init_claude_desktop_ollama_adds_env(self, cli_env):
        """init --ollama --claude-desktop should set api_base env."""
        result = cmd_init(FakeArgs(ollama=True, claude_desktop=True))

        assert result == 0
        desktop_config = cli_env / ".claude" / "claude_desktop_config.json"
        mcp = json.loads(desktop_config.read_text())
        env = mcp["mcpServers"]["tribal-memory"]["env"]
        assert "TRIBAL_MEMORY_EMBEDDING_API_BASE" in env
        assert "localhost:11434" in env["TRIBAL_MEMORY_EMBEDDING_API_BASE"]

    def test_init_claude_desktop_preserves_existing_entries(self, cli_env):
        """init --claude-desktop should not clobber existing MCP entries."""
        desktop_dir = cli_env / ".claude"
        desktop_dir.mkdir(parents=True, exist_ok=True)
        desktop_config = desktop_dir / "claude_desktop_config.json"
        desktop_config.write_text(json.dumps({"mcpServers": {"other": {"command": "other-cmd"}}}) + "\n")

        result = cmd_init(FakeArgs(claude_desktop=True))

        assert result == 0
        desktop_mcp = json.loads(desktop_config.read_text())
        assert "tribal-memory" in desktop_mcp["mcpServers"]
        assert "other" in desktop_mcp["mcpServers"]  # preserved

    def test_init_claude_desktop_does_not_touch_cli_config(self, cli_env):
        """init --claude-desktop should not create CLI config."""
        result = cmd_init(FakeArgs(claude_desktop=True))

        assert result == 0
        cli_config = cli_env / ".claude.json"
        assert not cli_config.exists()

    def test_init_both_claude_flags(self, cli_env):
        """init --claude-code --claude-desktop should configure both."""
        result = cmd_init(FakeArgs(claude_code=True, claude_desktop=True))

        assert result == 0
        cli_config = cli_env / ".claude.json"
        assert cli_config.exists()
        desktop_config = cli_env / ".claude" / "claude_desktop_config.json"
        assert desktop_config.exists()


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
        """init --claude-code should write the resolved full path to CLI config."""
        fake_path = "/home/test/.local/bin/tribalmemory-mcp"
        with patch("tribalmemory.cli.shutil.which", return_value=fake_path):
            result = cmd_init(FakeArgs(claude_code=True))

        assert result == 0

        # CLI config should have full path
        cli_config = cli_env / ".claude.json"
        mcp = json.loads(cli_config.read_text())
        assert mcp["mcpServers"]["tribal-memory"]["command"] == fake_path

    def test_init_claude_desktop_uses_full_path(self, cli_env):
        """init --claude-desktop should write the resolved full path."""
        fake_path = "/home/test/.local/bin/tribalmemory-mcp"
        with patch("tribalmemory.cli.shutil.which", return_value=fake_path):
            result = cmd_init(FakeArgs(claude_desktop=True))

        assert result == 0

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

    def test_codex_ollama_adds_env(self, cli_env):
        """init --ollama --codex should add api_base env."""
        result = cmd_init(FakeArgs(ollama=True, codex=True))

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


class TestAutoCapture:
    """Tests for --auto-capture flag."""

    def test_auto_capture_creates_claude_instructions(self, cli_env):
        """--auto-capture should write memory instructions to CLAUDE.md."""
        result = cmd_init(FakeArgs(auto_capture=True))

        assert result == 0
        claude_md = cli_env / CLAUDE_INSTRUCTIONS_FILE
        assert claude_md.exists()
        content = claude_md.read_text()
        assert "tribal_remember" in content
        assert "tribal_recall" in content

    def test_auto_capture_appends_to_existing_claude_md(self, cli_env):
        """--auto-capture should append to existing CLAUDE.md, not overwrite."""
        claude_dir = cli_env / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        claude_md = claude_dir / "CLAUDE.md"
        claude_md.write_text("# Existing instructions\n\nDo stuff.\n")

        result = cmd_init(FakeArgs(auto_capture=True))

        assert result == 0
        content = claude_md.read_text()
        assert "# Existing instructions" in content  # preserved
        assert "Do stuff." in content  # preserved
        assert "tribal_remember" in content  # appended

    def test_auto_capture_skips_if_already_present(self, cli_env):
        """--auto-capture should not duplicate if instructions already exist."""
        claude_dir = cli_env / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        claude_md = claude_dir / "CLAUDE.md"
        claude_md.write_text(AUTO_CAPTURE_INSTRUCTIONS)

        result = cmd_init(FakeArgs(auto_capture=True))

        assert result == 0
        content = claude_md.read_text()
        # Should appear exactly once
        assert content.count("tribal_remember") == AUTO_CAPTURE_INSTRUCTIONS.count(
            "tribal_remember"
        )

    def test_no_auto_capture_skips_claude_md(self, cli_env):
        """Without --auto-capture, CLAUDE.md should not be created."""
        result = cmd_init(FakeArgs())

        assert result == 0
        claude_md = cli_env / ".claude" / "CLAUDE.md"
        assert not claude_md.exists()

    def test_auto_capture_claude_only_skips_codex(self, cli_env):
        """--auto-capture --claude-code (no --codex) should only write CLAUDE.md."""
        result = cmd_init(FakeArgs(auto_capture=True, claude_code=True))

        assert result == 0
        claude_md = cli_env / CLAUDE_INSTRUCTIONS_FILE
        codex_md = cli_env / CODEX_INSTRUCTIONS_FILE
        assert claude_md.exists()
        assert not codex_md.exists()

    def test_auto_capture_with_codex_writes_agents_md(self, cli_env):
        """--auto-capture --codex should write to ~/.codex/AGENTS.md."""
        result = cmd_init(FakeArgs(auto_capture=True, codex=True))

        assert result == 0
        agents_md = cli_env / CODEX_INSTRUCTIONS_FILE
        assert agents_md.exists()
        content = agents_md.read_text()
        assert "tribal_remember" in content
        assert "tribal_recall" in content

    def test_auto_capture_with_both_writes_both_files(self, cli_env):
        """--auto-capture --claude-code --codex should write both instruction files."""
        result = cmd_init(FakeArgs(auto_capture=True, claude_code=True, codex=True))

        assert result == 0
        claude_md = cli_env / CLAUDE_INSTRUCTIONS_FILE
        codex_md = cli_env / CODEX_INSTRUCTIONS_FILE
        assert claude_md.exists()
        assert codex_md.exists()
        assert "tribal_remember" in claude_md.read_text()
        assert "tribal_remember" in codex_md.read_text()

    def test_auto_capture_bare_writes_both_files(self, cli_env):
        """--auto-capture alone (no --claude-code/--codex) should write both."""
        result = cmd_init(FakeArgs(auto_capture=True))

        assert result == 0
        claude_md = cli_env / CLAUDE_INSTRUCTIONS_FILE
        codex_md = cli_env / CODEX_INSTRUCTIONS_FILE
        assert claude_md.exists()
        assert codex_md.exists()

    def test_auto_capture_codex_appends_to_existing(self, cli_env):
        """--auto-capture should append to existing Codex AGENTS.md."""
        codex_dir = cli_env / ".codex"
        codex_dir.mkdir(parents=True, exist_ok=True)
        agents_md = codex_dir / "AGENTS.md"
        agents_md.write_text("# My Agent Rules\n\nBe helpful.\n")

        result = cmd_init(FakeArgs(auto_capture=True, codex=True))

        assert result == 0
        content = agents_md.read_text()
        assert "# My Agent Rules" in content  # preserved
        assert "Be helpful." in content  # preserved
        assert "tribal_remember" in content  # appended

    def test_auto_capture_codex_idempotent(self, cli_env):
        """--auto-capture should not duplicate in Codex AGENTS.md."""
        result1 = cmd_init(FakeArgs(auto_capture=True, codex=True))
        assert result1 == 0
        # Force re-run (config already exists)
        result2 = cmd_init(FakeArgs(auto_capture=True, codex=True, force=True))
        assert result2 == 0

        agents_md = cli_env / CODEX_INSTRUCTIONS_FILE
        content = agents_md.read_text()
        assert content.count("tribal_remember") == AUTO_CAPTURE_INSTRUCTIONS.count(
            "tribal_remember"
        )

    def test_auto_capture_sets_config_flag(self, cli_env):
        """--auto-capture should add auto_capture: true to config.yaml."""
        result = cmd_init(FakeArgs(auto_capture=True))

        assert result == 0
        config = (cli_env / ".tribal-memory" / "config.yaml").read_text()
        assert "auto_capture: true" in config

    def test_no_auto_capture_omits_config_flag(self, cli_env):
        """Without --auto-capture, config should not have auto_capture: true."""
        result = cmd_init(FakeArgs())

        assert result == 0
        config = (cli_env / ".tribal-memory" / "config.yaml").read_text()
        assert "auto_capture: true" not in config


class TestEnvFile:
    """Tests for .env file handling."""

    def test_load_env_file(self, cli_env, monkeypatch):
        """load_env_file should set env vars from .env."""
        env_path = cli_env / ".tribal-memory" / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("OPENAI_API_KEY=sk-from-env-file\n")
        monkeypatch.setattr("tribalmemory.cli.ENV_FILE", env_path)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        load_env_file()

        assert os.environ.get("OPENAI_API_KEY") == "sk-from-env-file"
        # Cleanup
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def test_load_env_file_does_not_overwrite(self, cli_env, monkeypatch):
        """load_env_file should not overwrite existing env vars."""
        env_path = cli_env / ".tribal-memory" / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("OPENAI_API_KEY=sk-from-file\n")
        monkeypatch.setattr("tribalmemory.cli.ENV_FILE", env_path)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")

        load_env_file()

        assert os.environ.get("OPENAI_API_KEY") == "sk-from-shell"

    def test_load_env_file_missing(self, cli_env, monkeypatch):
        """load_env_file should be a no-op if .env doesn't exist."""
        env_path = cli_env / ".tribal-memory" / ".env"
        monkeypatch.setattr("tribalmemory.cli.ENV_FILE", env_path)
        # Should not raise
        load_env_file()


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
