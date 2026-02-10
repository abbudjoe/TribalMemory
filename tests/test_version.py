"""Tests for version string consistency."""

import re
import pytest


def test_version_format():
    """Ensure version follows semantic versioning."""
    from tribalmemory import __version__
    
    # Should match semver pattern (e.g., "0.9.0", "1.2.3", "2.0.0-beta")
    semver_pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$'
    assert re.match(semver_pattern, __version__), \
        f"Version '{__version__}' does not match semver pattern"


def test_version_not_stale():
    """Ensure version is not one of the known stale values."""
    from tribalmemory import __version__
    
    stale_versions = ["0.7.1", "0.1.0"]
    assert __version__ not in stale_versions, \
        f"Version '{__version__}' is stale (matches one of {stale_versions})"


def test_server_uses_package_version():
    """Ensure server app uses the package version, not hardcoded."""
    from tribalmemory import __version__
    from tribalmemory.server.app import create_app
    
    app = create_app()
    
    # FastAPI app should use package version
    assert app.version == __version__, \
        f"FastAPI app version '{app.version}' doesn't match package version '{__version__}'"


@pytest.mark.asyncio
async def test_root_endpoint_version():
    """Ensure root endpoint returns correct version."""
    from tribalmemory import __version__
    from tribalmemory.server.app import create_app
    from httpx import AsyncClient, ASGITransport
    
    app = create_app()
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert data["version"] == __version__, \
            f"Root endpoint version '{data['version']}' doesn't match package version '{__version__}'"
