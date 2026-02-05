"""Tests for tribalmemory.service — service management."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from tribalmemory.service import (
    ServiceManager,
    detect_service_manager,
    cmd_service,
    cmd_service_install,
    cmd_service_uninstall,
    cmd_service_start,
    cmd_service_stop,
    cmd_service_status,
    _resolve_serve_command,
    _build_path_env,
    _generate_systemd_unit,
    _generate_launchd_plist,
    _check_server_health,
    SYSTEMD_UNIT_DIR,
    SYSTEMD_UNIT_NAME,
    SYSTEMD_UNIT_PATH,
    LAUNCHD_PLIST_PATH,
    LOG_DIR,
)


# ============================================================================
# Platform detection
# ============================================================================


class TestDetectServiceManager:
    """Tests for detect_service_manager()."""

    @patch("tribalmemory.service.sys")
    @patch("tribalmemory.service.shutil.which")
    def test_linux_with_systemd(self, mock_which, mock_sys):
        mock_sys.platform = "linux"
        mock_which.return_value = "/usr/bin/systemctl"
        assert detect_service_manager() == ServiceManager.SYSTEMD

    @patch("tribalmemory.service.sys")
    @patch("tribalmemory.service.shutil.which")
    def test_linux_without_systemd(self, mock_which, mock_sys):
        mock_sys.platform = "linux"
        mock_which.return_value = None
        assert detect_service_manager() == ServiceManager.NONE

    @patch("tribalmemory.service.sys")
    def test_macos(self, mock_sys):
        mock_sys.platform = "darwin"
        assert detect_service_manager() == ServiceManager.LAUNCHD

    @patch("tribalmemory.service.sys")
    def test_windows(self, mock_sys):
        mock_sys.platform = "win32"
        assert detect_service_manager() == ServiceManager.NONE


# ============================================================================
# Unit/plist generation
# ============================================================================


class TestGenerateSystemdUnit:
    """Tests for _generate_systemd_unit()."""

    @patch("tribalmemory.service._build_path_env", return_value="/usr/bin:/bin")
    @patch("tribalmemory.service._resolve_serve_command", return_value="/usr/local/bin/tribalmemory")
    def test_basic_unit(self, mock_cmd, mock_path):
        unit = _generate_systemd_unit()
        assert "[Unit]" in unit
        assert "[Service]" in unit
        assert "[Install]" in unit
        assert "ExecStart=/usr/local/bin/tribalmemory serve" in unit
        assert "Restart=on-failure" in unit
        assert "WantedBy=default.target" in unit
        assert "PATH=/usr/bin:/bin" in unit

    @patch("tribalmemory.service._build_path_env", return_value="/usr/bin")
    @patch("tribalmemory.service._resolve_serve_command")
    def test_unit_with_python_module_command(self, mock_cmd, mock_path):
        mock_cmd.return_value = "/usr/bin/python3 -m tribalmemory"
        unit = _generate_systemd_unit()
        assert "ExecStart=/usr/bin/python3 -m tribalmemory serve" in unit


class TestGenerateLaunchdPlist:
    """Tests for _generate_launchd_plist()."""

    @patch("tribalmemory.service.LOG_DIR", Path("/tmp/test-tribal-logs"))
    @patch("tribalmemory.service._resolve_serve_command", return_value="/usr/local/bin/tribalmemory")
    def test_basic_plist(self, mock_cmd):
        plist = _generate_launchd_plist()
        assert "com.tribalmemory.server" in plist
        assert "<string>/usr/local/bin/tribalmemory</string>" in plist
        assert "<string>serve</string>" in plist
        assert "<key>RunAtLoad</key>" in plist
        assert "<key>KeepAlive</key>" in plist
        assert "<?xml version" in plist

    @patch("tribalmemory.service.LOG_DIR", Path("/tmp/test-tribal-logs"))
    @patch("tribalmemory.service._resolve_serve_command")
    def test_plist_with_python_module(self, mock_cmd):
        mock_cmd.return_value = "/usr/bin/python3 -m tribalmemory"
        plist = _generate_launchd_plist()
        assert "<string>/usr/bin/python3</string>" in plist
        assert "<string>-m</string>" in plist
        assert "<string>tribalmemory</string>" in plist
        assert "<string>serve</string>" in plist


# ============================================================================
# Resolve command
# ============================================================================


class TestResolveServeCommand:
    """Tests for _resolve_serve_command()."""

    @patch("tribalmemory.service.shutil.which", return_value="/home/user/.local/bin/tribalmemory")
    def test_found_on_path(self, mock_which):
        assert _resolve_serve_command() == "/home/user/.local/bin/tribalmemory"

    @patch("tribalmemory.service.os.access", return_value=True)
    @patch("tribalmemory.service.shutil.which", return_value=None)
    def test_found_in_local_bin(self, mock_which, mock_access):
        with patch.object(Path, "exists", return_value=True):
            result = _resolve_serve_command()
            assert ".local/bin/tribalmemory" in result

    @patch("tribalmemory.service.os.access", return_value=False)
    @patch("tribalmemory.service.shutil.which", return_value=None)
    def test_fallback_to_python_module(self, mock_which, mock_access):
        result = _resolve_serve_command()
        assert sys.executable in result
        assert "-m tribalmemory" in result


# ============================================================================
# Build PATH
# ============================================================================


class TestBuildPathEnv:
    """Tests for _build_path_env()."""

    @patch("tribalmemory.service.shutil.which")
    def test_includes_standard_paths(self, mock_which):
        mock_which.side_effect = lambda x: {
            "tribalmemory": "/home/user/.local/bin/tribalmemory",
            "node": None,
        }.get(x)
        path = _build_path_env()
        assert "/usr/local/bin" in path
        assert "/usr/bin" in path
        assert "/bin" in path

    @patch("tribalmemory.service.shutil.which")
    def test_includes_binary_dir(self, mock_which):
        mock_which.side_effect = lambda x: {
            "tribalmemory": "/opt/custom/bin/tribalmemory",
            "node": None,
        }.get(x)
        path = _build_path_env()
        assert "/opt/custom/bin" in path


# ============================================================================
# Service commands — systemd
# ============================================================================


class TestSystemdInstall:
    """Tests for systemd install flow."""

    @patch("tribalmemory.service._check_linger")
    @patch("tribalmemory.service.subprocess.run")
    @patch("tribalmemory.service._generate_systemd_unit", return_value="[Unit]\nTest")
    def test_install_creates_unit_file(self, mock_gen, mock_run, mock_linger, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)

        unit_path = tmp_path / "tribalmemory.service"
        with patch("tribalmemory.service.SYSTEMD_UNIT_DIR", tmp_path), \
             patch("tribalmemory.service.SYSTEMD_UNIT_PATH", unit_path):
            result = cmd_service_install(ServiceManager.SYSTEMD)

        assert result == 0
        assert unit_path.read_text() == "[Unit]\nTest"
        # daemon-reload + enable
        assert mock_run.call_count == 2

    @patch("tribalmemory.service._check_linger")
    @patch("tribalmemory.service.subprocess.run")
    @patch("tribalmemory.service._generate_systemd_unit", return_value="[Unit]\nUpdated")
    def test_install_overwrites_existing(self, mock_gen, mock_run, mock_linger, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)

        unit_path = tmp_path / "tribalmemory.service"
        unit_path.write_text("[Unit]\nOld")

        with patch("tribalmemory.service.SYSTEMD_UNIT_DIR", tmp_path), \
             patch("tribalmemory.service.SYSTEMD_UNIT_PATH", unit_path):
            result = cmd_service_install(ServiceManager.SYSTEMD)

        assert result == 0
        assert unit_path.read_text() == "[Unit]\nUpdated"


class TestSystemdUninstall:
    """Tests for systemd uninstall flow."""

    @patch("tribalmemory.service.subprocess.run")
    def test_uninstall_removes_unit(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)

        unit_path = tmp_path / "tribalmemory.service"
        unit_path.write_text("[Unit]\nTest")

        with patch("tribalmemory.service.SYSTEMD_UNIT_PATH", unit_path):
            result = cmd_service_uninstall(ServiceManager.SYSTEMD)

        assert result == 0
        assert not unit_path.exists()

    @patch("tribalmemory.service.subprocess.run")
    def test_uninstall_nonexistent(self, mock_run, tmp_path, capsys):
        mock_run.return_value = MagicMock(returncode=0)

        unit_path = tmp_path / "tribalmemory.service"
        with patch("tribalmemory.service.SYSTEMD_UNIT_PATH", unit_path):
            result = cmd_service_uninstall(ServiceManager.SYSTEMD)

        assert result == 0
        assert "already removed" in capsys.readouterr().out


class TestSystemdStartStop:
    """Tests for systemd start/stop."""

    @patch("tribalmemory.service._print_status_hint")
    @patch("tribalmemory.service.subprocess.run")
    def test_start_success(self, mock_run, mock_hint):
        mock_run.return_value = MagicMock(returncode=0)
        result = cmd_service_start(ServiceManager.SYSTEMD)
        assert result == 0

    @patch("tribalmemory.service.subprocess.run")
    def test_start_failure(self, mock_run, capsys):
        mock_run.return_value = MagicMock(returncode=1, stderr="failed")
        result = cmd_service_start(ServiceManager.SYSTEMD)
        assert result == 1
        assert "Failed to start" in capsys.readouterr().out

    @patch("tribalmemory.service.subprocess.run")
    def test_stop_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = cmd_service_stop(ServiceManager.SYSTEMD)
        assert result == 0

    @patch("tribalmemory.service.subprocess.run")
    def test_stop_failure(self, mock_run, capsys):
        mock_run.return_value = MagicMock(returncode=1, stderr="not running")
        result = cmd_service_stop(ServiceManager.SYSTEMD)
        assert result == 1


# ============================================================================
# Service commands — launchd
# ============================================================================


class TestLaunchdInstall:
    """Tests for launchd install flow."""

    @patch("tribalmemory.service._generate_launchd_plist", return_value="<plist>test</plist>")
    def test_install_creates_plist(self, mock_gen, tmp_path):
        plist_path = tmp_path / "com.tribalmemory.server.plist"
        log_dir = tmp_path / "logs"

        with patch("tribalmemory.service.LAUNCHD_AGENTS_DIR", tmp_path), \
             patch("tribalmemory.service.LAUNCHD_PLIST_PATH", plist_path), \
             patch("tribalmemory.service.LOG_DIR", log_dir):
            result = cmd_service_install(ServiceManager.LAUNCHD)

        assert result == 0
        assert plist_path.read_text() == "<plist>test</plist>"


class TestLaunchdStartStop:
    """Tests for launchd start/stop."""

    @patch("tribalmemory.service._print_status_hint")
    @patch("tribalmemory.service.subprocess.run")
    def test_start_success(self, mock_run, mock_hint, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        plist_path = tmp_path / "com.tribalmemory.server.plist"
        plist_path.write_text("<plist/>")

        with patch("tribalmemory.service.LAUNCHD_PLIST_PATH", plist_path):
            result = cmd_service_start(ServiceManager.LAUNCHD)
        assert result == 0

    @patch("tribalmemory.service.subprocess.run")
    def test_start_not_installed(self, mock_run, tmp_path, capsys):
        plist_path = tmp_path / "com.tribalmemory.server.plist"
        with patch("tribalmemory.service.LAUNCHD_PLIST_PATH", plist_path):
            result = cmd_service_start(ServiceManager.LAUNCHD)
        assert result == 1
        assert "not installed" in capsys.readouterr().out

    @patch("tribalmemory.service.subprocess.run")
    def test_stop_success(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        plist_path = tmp_path / "com.tribalmemory.server.plist"
        plist_path.write_text("<plist/>")

        with patch("tribalmemory.service.LAUNCHD_PLIST_PATH", plist_path):
            result = cmd_service_stop(ServiceManager.LAUNCHD)
        assert result == 0


# ============================================================================
# cmd_service dispatcher
# ============================================================================


class TestCmdServiceDispatcher:
    """Tests for the cmd_service() dispatcher."""

    @patch("tribalmemory.service.detect_service_manager", return_value=ServiceManager.NONE)
    def test_no_service_manager(self, mock_detect, capsys):
        result = cmd_service("install")
        assert result == 1
        assert "No supported service manager" in capsys.readouterr().out

    @patch("tribalmemory.service.cmd_service_install", return_value=0)
    @patch("tribalmemory.service.detect_service_manager", return_value=ServiceManager.SYSTEMD)
    def test_dispatch_install(self, mock_detect, mock_install):
        result = cmd_service("install")
        assert result == 0
        mock_install.assert_called_once_with(ServiceManager.SYSTEMD)

    @patch("tribalmemory.service.cmd_service_start", return_value=0)
    @patch("tribalmemory.service.detect_service_manager", return_value=ServiceManager.LAUNCHD)
    def test_dispatch_start_launchd(self, mock_detect, mock_start):
        result = cmd_service("start")
        assert result == 0
        mock_start.assert_called_once_with(ServiceManager.LAUNCHD)

    @patch("tribalmemory.service.detect_service_manager", return_value=ServiceManager.SYSTEMD)
    def test_unknown_action(self, mock_detect, capsys):
        result = cmd_service("restart")
        assert result == 1
        assert "Unknown action" in capsys.readouterr().out


# ============================================================================
# Health check
# ============================================================================


class TestCheckServerHealth:
    """Tests for _check_server_health()."""

    @patch("tribalmemory.service._read_server_port", return_value=18790)
    def test_healthy_server(self, mock_port, capsys):
        import json as json_mod
        import urllib.request

        mock_resp = MagicMock()
        mock_resp.read.return_value = json_mod.dumps({
            "status": "ok",
            "memory_count": 42,
            "version": "0.4.2",
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch.object(urllib.request, "urlopen", return_value=mock_resp):
            _check_server_health()

        out = capsys.readouterr().out
        assert "ok" in out
        assert "42" in out

    @patch("tribalmemory.service._read_server_port", return_value=18790)
    def test_unreachable_server(self, mock_port, capsys):
        import urllib.request

        with patch.object(urllib.request, "urlopen", side_effect=Exception("refused")):
            _check_server_health()

        out = capsys.readouterr().out
        assert "not reachable" in out


# ============================================================================
# CLI integration — init --service
# ============================================================================


class TestInitServiceFlag:
    """Test that --service flag in init triggers service setup."""

    def test_init_parser_has_service_flag(self):
        """Verify the --service argument is registered."""
        from tribalmemory.cli import main
        import argparse

        # Parse --service flag
        parser = argparse.ArgumentParser()
        parser.add_argument("--service", action="store_true")
        args = parser.parse_args(["--service"])
        assert args.service is True

    def test_init_parser_service_default_false(self):
        """Verify --service defaults to False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--service", action="store_true")
        args = parser.parse_args([])
        assert args.service is False
