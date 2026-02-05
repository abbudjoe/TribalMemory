"""Tribal Memory service management — systemd (Linux) and launchd (macOS).

Manages the TribalMemory HTTP server as a background system service:
    tribalmemory service install   # Create and enable the service
    tribalmemory service start     # Start the service
    tribalmemory service stop      # Stop the service
    tribalmemory service status    # Show service status
    tribalmemory service uninstall # Remove the service
"""

import os
import shutil
import subprocess
import sys
import textwrap
from enum import Enum
from pathlib import Path
from typing import Optional

from .cli import TRIBAL_DIR, CONFIG_FILE


class ServiceManager(Enum):
    SYSTEMD = "systemd"
    LAUNCHD = "launchd"
    NONE = "none"


# systemd user unit path
SYSTEMD_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
SYSTEMD_UNIT_NAME = "tribalmemory.service"
SYSTEMD_UNIT_PATH = SYSTEMD_UNIT_DIR / SYSTEMD_UNIT_NAME

# launchd plist path
LAUNCHD_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
LAUNCHD_PLIST_NAME = "com.tribalmemory.server.plist"
LAUNCHD_PLIST_PATH = LAUNCHD_AGENTS_DIR / LAUNCHD_PLIST_NAME

# Log file location
LOG_DIR = TRIBAL_DIR / "logs"
LOG_FILE = LOG_DIR / "server.log"


def detect_service_manager() -> ServiceManager:
    """Detect the available service manager on this platform."""
    if sys.platform == "darwin":
        # macOS always has launchd
        return ServiceManager.LAUNCHD

    if sys.platform.startswith("linux"):
        # Check for systemd
        if shutil.which("systemctl"):
            return ServiceManager.SYSTEMD

    return ServiceManager.NONE


def _resolve_serve_command() -> str:
    """Resolve the full path to the tribalmemory binary."""
    resolved = shutil.which("tribalmemory")
    if resolved:
        return resolved

    # Check common install locations
    candidates = [
        Path.home() / ".local" / "bin" / "tribalmemory",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    # Fallback to sys.executable -m tribalmemory
    return f"{sys.executable} -m tribalmemory"


def _build_path_env() -> str:
    """Build a PATH env string that includes common binary locations."""
    paths = []

    # Include the directory containing the tribalmemory binary
    binary = shutil.which("tribalmemory")
    if binary:
        paths.append(str(Path(binary).parent))

    # Common locations
    paths.extend([
        str(Path.home() / ".local" / "bin"),
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    ])

    # nvm node path (if present)
    nvm_dir = os.environ.get("NVM_DIR", str(Path.home() / ".nvm"))
    nvm_path = Path(nvm_dir)
    if nvm_path.exists():
        # Find current node version
        current = nvm_path / "alias" / "default"
        if current.exists():
            try:
                version = current.read_text().strip()
                node_bin = nvm_path / "versions" / "node" / f"v{version}" / "bin"
                if node_bin.exists():
                    paths.append(str(node_bin))
            except OSError:
                pass
        # Also check via currently active node
        node = shutil.which("node")
        if node:
            paths.append(str(Path(node).parent))

    return ":".join(dict.fromkeys(paths))  # deduplicate, preserve order


def _read_server_port() -> int:
    """Read the server port from config, defaulting to 18790."""
    if not CONFIG_FILE.exists():
        return 18790
    try:
        import yaml
        config = yaml.safe_load(CONFIG_FILE.read_text())
        return config.get("server", {}).get("port", 18790)
    except Exception:
        return 18790


def _generate_systemd_unit() -> str:
    """Generate the systemd user unit file content."""
    serve_cmd = _resolve_serve_command()
    path_env = _build_path_env()

    # If the command contains a space (e.g. "python -m tribalmemory"),
    # we need to handle it differently
    if " " in serve_cmd:
        exec_start = f"{serve_cmd} serve"
    else:
        exec_start = f"{serve_cmd} serve"

    return textwrap.dedent(f"""\
        [Unit]
        Description=Tribal Memory Server
        After=network.target

        [Service]
        Type=simple
        ExecStart={exec_start}
        Restart=on-failure
        RestartSec=5
        Environment=PATH={path_env}

        [Install]
        WantedBy=default.target
    """)


def _generate_launchd_plist() -> str:
    """Generate the launchd plist file content."""
    serve_cmd = _resolve_serve_command()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Split command for ProgramArguments
    if " " in serve_cmd:
        parts = serve_cmd.split() + ["serve"]
    else:
        parts = [serve_cmd, "serve"]

    args_xml = "\n".join(f"        <string>{p}</string>" for p in parts)

    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>com.tribalmemory.server</string>
            <key>ProgramArguments</key>
            <array>
        {args_xml}
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardOutPath</key>
            <string>{LOG_FILE}</string>
            <key>StandardErrorPath</key>
            <string>{LOG_FILE}</string>
        </dict>
        </plist>
    """)


def cmd_service_install(manager: ServiceManager) -> int:
    """Install the service unit/plist."""
    if manager == ServiceManager.SYSTEMD:
        SYSTEMD_UNIT_DIR.mkdir(parents=True, exist_ok=True)
        unit_content = _generate_systemd_unit()
        SYSTEMD_UNIT_PATH.write_text(unit_content)
        print(f"✅ Systemd unit written: {SYSTEMD_UNIT_PATH}")

        # Reload daemon
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)

        # Enable (starts on boot)
        subprocess.run(
            ["systemctl", "--user", "enable", SYSTEMD_UNIT_NAME],
            check=True,
        )
        print(f"✅ Service enabled (starts on boot)")

        # Check linger
        _check_linger()

        print()
        print("🚀 Start it now:")
        print("   tribalmemory service start")
        return 0

    elif manager == ServiceManager.LAUNCHD:
        LAUNCHD_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        plist_content = _generate_launchd_plist()
        LAUNCHD_PLIST_PATH.write_text(plist_content)
        print(f"✅ LaunchAgent plist written: {LAUNCHD_PLIST_PATH}")
        print()
        print("🚀 Start it now:")
        print("   tribalmemory service start")
        return 0

    return 1


def cmd_service_uninstall(manager: ServiceManager) -> int:
    """Remove the service unit/plist."""
    if manager == ServiceManager.SYSTEMD:
        # Stop first if running
        subprocess.run(
            ["systemctl", "--user", "stop", SYSTEMD_UNIT_NAME],
            capture_output=True,
        )
        subprocess.run(
            ["systemctl", "--user", "disable", SYSTEMD_UNIT_NAME],
            capture_output=True,
        )
        if SYSTEMD_UNIT_PATH.exists():
            SYSTEMD_UNIT_PATH.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            print(f"✅ Service removed: {SYSTEMD_UNIT_PATH}")
        else:
            print("⚠️  Service unit not found (already removed?)")
        return 0

    elif manager == ServiceManager.LAUNCHD:
        # Unload first if loaded
        if LAUNCHD_PLIST_PATH.exists():
            subprocess.run(
                ["launchctl", "unload", str(LAUNCHD_PLIST_PATH)],
                capture_output=True,
            )
            LAUNCHD_PLIST_PATH.unlink()
            print(f"✅ Service removed: {LAUNCHD_PLIST_PATH}")
        else:
            print("⚠️  LaunchAgent plist not found (already removed?)")
        return 0

    return 1


def cmd_service_start(manager: ServiceManager) -> int:
    """Start the service."""
    if manager == ServiceManager.SYSTEMD:
        result = subprocess.run(
            ["systemctl", "--user", "start", SYSTEMD_UNIT_NAME],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"❌ Failed to start: {result.stderr.strip()}")
            return 1
        print("✅ Tribal Memory server started")
        _print_status_hint()
        return 0

    elif manager == ServiceManager.LAUNCHD:
        if not LAUNCHD_PLIST_PATH.exists():
            print("❌ Service not installed. Run: tribalmemory service install")
            return 1
        result = subprocess.run(
            ["launchctl", "load", str(LAUNCHD_PLIST_PATH)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"❌ Failed to start: {result.stderr.strip()}")
            return 1
        print("✅ Tribal Memory server started")
        _print_status_hint()
        return 0

    return 1


def cmd_service_stop(manager: ServiceManager) -> int:
    """Stop the service."""
    if manager == ServiceManager.SYSTEMD:
        result = subprocess.run(
            ["systemctl", "--user", "stop", SYSTEMD_UNIT_NAME],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"❌ Failed to stop: {result.stderr.strip()}")
            return 1
        print("✅ Tribal Memory server stopped")
        return 0

    elif manager == ServiceManager.LAUNCHD:
        if not LAUNCHD_PLIST_PATH.exists():
            print("❌ Service not installed.")
            return 1
        result = subprocess.run(
            ["launchctl", "unload", str(LAUNCHD_PLIST_PATH)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"❌ Failed to stop: {result.stderr.strip()}")
            return 1
        print("✅ Tribal Memory server stopped")
        return 0

    return 1


def cmd_service_status(manager: ServiceManager) -> int:
    """Show service status."""
    if manager == ServiceManager.SYSTEMD:
        result = subprocess.run(
            ["systemctl", "--user", "status", SYSTEMD_UNIT_NAME],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Also check HTTP health
        _check_server_health()
        return 0

    elif manager == ServiceManager.LAUNCHD:
        result = subprocess.run(
            ["launchctl", "list", "com.tribalmemory.server"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("Service: not running")
        else:
            print(result.stdout)

        _check_server_health()
        return 0

    return 1


def _check_server_health() -> None:
    """Check if the server is reachable and print health info."""
    import json as json_mod
    import urllib.request

    port = _read_server_port()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json_mod.loads(resp.read())
            status = data.get("status", "unknown")
            count = data.get("memory_count", "?")
            version = data.get("version", "?")
            print(f"🧠 Server: {status} | {count} memories | v{version}")
    except Exception:
        print(f"🔴 Server not reachable at http://127.0.0.1:{port}")


def _check_linger() -> None:
    """Check if loginctl linger is enabled for the current user."""
    user = os.environ.get("USER", "")
    if not user:
        return

    linger_path = Path(f"/var/lib/systemd/linger/{user}")
    if not linger_path.exists():
        print()
        print("⚠️  Linger not enabled — service won't start on boot without login.")
        print(f"   Enable with: sudo loginctl enable-linger {user}")


def _print_status_hint() -> None:
    """Print a hint to check status."""
    print("   Check: tribalmemory service status")


def cmd_service(action: str) -> int:
    """Main entry point for the service subcommand."""
    manager = detect_service_manager()

    if manager == ServiceManager.NONE:
        print("❌ No supported service manager found.")
        print()
        if sys.platform == "win32":
            print("   Windows: Run tribalmemory as a scheduled task or use NSSM.")
        else:
            print("   Supported: systemd (Linux) and launchd (macOS).")
        print("   Alternatively, run: tribalmemory serve")
        return 1

    actions = {
        "install": cmd_service_install,
        "uninstall": cmd_service_uninstall,
        "start": cmd_service_start,
        "stop": cmd_service_stop,
        "status": cmd_service_status,
    }

    handler = actions.get(action)
    if not handler:
        print(f"❌ Unknown action: {action}")
        print(f"   Available: {', '.join(actions.keys())}")
        return 1

    return handler(manager)
