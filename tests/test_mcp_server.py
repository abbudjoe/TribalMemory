"""Tests for MCP server integration."""

import asyncio
import inspect
import pytest

mcp = pytest.importorskip("mcp")

from tribalmemory.mcp import server as mcp_server


def _extract_tool_names(server):
    """Best-effort extraction of tool names across FastMCP versions."""
    if hasattr(server, "list_tools"):
        list_tools = server.list_tools
        if inspect.iscoroutinefunction(list_tools):
            try:
                tools = asyncio.run(list_tools())
            except RuntimeError:
                tools = asyncio.get_event_loop().run_until_complete(list_tools())
        else:
            tools = list_tools()
        if isinstance(tools, dict):
            return set(tools.keys())
        if isinstance(tools, list):
            names = set()
            for tool in tools:
                if isinstance(tool, dict) and "name" in tool:
                    names.add(tool["name"])
                else:
                    name = getattr(tool, "name", None)
                    if name:
                        names.add(name)
            return names or None
    for attr in ("tools", "_tools", "tool_registry", "_tool_registry"):
        if hasattr(server, attr):
            tools = getattr(server, attr)
            if isinstance(tools, dict):
                return set(tools.keys())
            if isinstance(tools, list):
                names = set()
                for tool in tools:
                    if isinstance(tool, dict) and "name" in tool:
                        names.add(tool["name"])
                    else:
                        name = getattr(tool, "name", None)
                        if name:
                            names.add(name)
                return names or None
    return None


@pytest.mark.asyncio
async def test_get_memory_service_uses_factory(monkeypatch, tmp_path):
    class DummyService:
        pass

    dummy = DummyService()

    def fake_create_memory_service(instance_id, db_path, openai_api_key, **kwargs):
        assert instance_id == "mcp-claude-code"
        assert db_path == str(tmp_path / "db")
        assert openai_api_key == "sk-test"
        return dummy

    class DummyDb:
        path = str(tmp_path / "db")

    class DummyEmbedding:
        api_key = "sk-test"
        api_base = None
        model = "test"
        dimensions = 1536

    class DummyConfig:
        db = DummyDb()
        embedding = DummyEmbedding()

    monkeypatch.setattr(mcp_server, "_memory_service", None)
    monkeypatch.setattr(mcp_server, "create_memory_service", fake_create_memory_service)
    monkeypatch.setattr(mcp_server.TribalMemoryConfig, "from_env", classmethod(lambda cls: DummyConfig()))

    service = await mcp_server.get_memory_service()
    assert service is dummy

    # Subsequent calls should return cached instance
    service2 = await mcp_server.get_memory_service()
    assert service2 is dummy


def test_create_server_registers_tools():
    server = mcp_server.create_server()
    names = _extract_tool_names(server)
    if names is None:
        pytest.skip("Unable to introspect FastMCP tools")

    expected = {
        "tribal_remember",
        "tribal_recall",
        "tribal_correct",
        "tribal_forget",
        "tribal_stats",
    }
    assert expected.issubset(names)
