"""End-to-end MCP workflow integration tests.

Tests the complete MCP tool workflow through FastMCP's call_tool interface
with a real TribalMemoryService backed by MockEmbeddingService + MockVectorStore.
No subprocess or stdio — exercises the full stack from tool call to service layer.

Issue #9: https://github.com/abbudjoe/TribalMemory/issues/9
"""

import asyncio
import json
from typing import Any, Sequence

import pytest

mcp = pytest.importorskip("mcp")

from tribalmemory.mcp.server import create_server, get_memory_service
import tribalmemory.mcp.server as mcp_server
from tribalmemory.testing.mocks import MockEmbeddingService, MockVectorStore
from tribalmemory.services.memory import TribalMemoryService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(result: Any) -> dict:
    """Extract JSON from a FastMCP call_tool result.

    FastMCP.call_tool may return:
    - A dict directly
    - A Sequence[ContentBlock]
    - A tuple of (Sequence[ContentBlock], dict)
    """
    if isinstance(result, dict):
        return result
    # Handle tuple return: (content_blocks, metadata)
    if isinstance(result, tuple):
        result = result[0]
    # Sequence of content blocks — grab text from first
    for block in result:
        text = getattr(block, "text", None)
        if not text and isinstance(block, dict):
            text = block.get("text")
        if text:
            return json.loads(text)
    raise ValueError(f"Could not parse result: {result}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp_service():
    """Create a TribalMemoryService with mock backends.

    Sets model/dimensions attributes on the mock embedding
    service so export/import tools can read them via getattr.
    """
    embedding = MockEmbeddingService(embedding_dim=64)
    # Export/import tools use getattr(emb, "dimensions")
    embedding.dimensions = 64
    embedding.model = "mock-test-model"
    store = MockVectorStore(embedding)
    service = TribalMemoryService(
        instance_id="mcp-test",
        embedding_service=embedding,
        vector_store=store,
    )
    return service


@pytest.fixture
def mcp_app(mcp_service, monkeypatch):
    """Create a FastMCP server with the service pre-injected."""
    # Inject our test service so tools skip real initialization
    monkeypatch.setattr(mcp_server, "_memory_service", mcp_service)
    server = create_server()
    yield server
    # Cleanup: monkeypatch auto-undoes, but reset explicitly
    # to prevent test pollution if fixture scope changes
    monkeypatch.setattr(mcp_server, "_memory_service", None)


# ---------------------------------------------------------------------------
# Test: Full CRUD Workflow
# ---------------------------------------------------------------------------

class TestMCPCrudWorkflow:
    """Full lifecycle: remember → recall → correct → recall → forget → stats."""

    @pytest.mark.asyncio
    async def test_full_crud_lifecycle(self, mcp_app):
        """Issue #9 acceptance criteria: full CRUD workflow test."""
        # Step 1: Remember a memory
        result = _parse(await mcp_app.call_tool("tribal_remember", {
            "content": "The project deadline is March 15th",
            "source_type": "user_explicit",
            "tags": ["deadlines", "project"],
        }))
        assert result["success"] is True
        assert result["memory_id"] is not None
        memory_id = result["memory_id"]

        # Step 2: Recall the memory
        # MockEmbeddingService uses hash-based embeddings, so
        # min_relevance=0 bypasses similarity thresholds.
        # Real semantic matching is tested elsewhere.
        result = _parse(await mcp_app.call_tool("tribal_recall", {
            "query": "The project deadline is March 15th",
            "limit": 5,
            "min_relevance": 0.0,
        }))
        assert result["count"] >= 1
        found = [r for r in result["results"] if r["memory_id"] == memory_id]
        assert len(found) == 1
        assert "March 15th" in found[0]["content"]

        # Step 3: Correct the memory
        result = _parse(await mcp_app.call_tool("tribal_correct", {
            "original_id": memory_id,
            "corrected_content": "The project deadline is March 22nd (extended)",
            "context": "Deadline was pushed back a week",
        }))
        assert result["success"] is True
        assert result["memory_id"] is not None
        corrected_id = result["memory_id"]
        assert corrected_id != memory_id

        # Step 4: Recall should return the corrected version
        result = _parse(await mcp_app.call_tool("tribal_recall", {
            "query": "project deadline March",
            "limit": 10,
            "min_relevance": 0.0,
        }))
        assert result["count"] >= 1
        contents = [r["content"] for r in result["results"]]
        assert any("March 22nd" in c for c in contents)

        # Step 5: Forget the corrected memory
        result = _parse(await mcp_app.call_tool("tribal_forget", {
            "memory_id": corrected_id,
        }))
        assert result["success"] is True

        # Step 6: Stats should reflect operations
        result = _parse(await mcp_app.call_tool("tribal_stats", {}))
        assert isinstance(result, dict)
        assert "total_memories" in result


# ---------------------------------------------------------------------------
# Test: Deduplication Behavior
# ---------------------------------------------------------------------------

class TestMCPDeduplication:
    """Verify deduplication prevents storing near-identical memories."""

    @pytest.mark.asyncio
    async def test_duplicate_rejected(self, mcp_app):
        """Storing identical content should be rejected as duplicate."""
        # Store original
        result = _parse(await mcp_app.call_tool("tribal_remember", {
            "content": "Python uses indentation for blocks",
            "source_type": "auto_capture",
        }))
        assert result["success"] is True
        original_id = result["memory_id"]

        # Try to store the same content again
        result = _parse(await mcp_app.call_tool("tribal_remember", {
            "content": "Python uses indentation for blocks",
            "source_type": "auto_capture",
        }))
        # Should be rejected as duplicate
        assert result["success"] is False
        assert result["duplicate_of"] == original_id

    @pytest.mark.asyncio
    async def test_skip_dedup_allows_duplicate(self, mcp_app):
        """skip_dedup=True should store even if duplicate exists."""
        content = "Duplicate allowed content"

        # Store original
        result = _parse(await mcp_app.call_tool("tribal_remember", {
            "content": content,
            "source_type": "auto_capture",
        }))
        assert result["success"] is True

        # Store again with skip_dedup
        result = _parse(await mcp_app.call_tool("tribal_remember", {
            "content": content,
            "source_type": "auto_capture",
            "skip_dedup": True,
        }))
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Test: Error Handling
# ---------------------------------------------------------------------------

class TestMCPErrorHandling:
    """Verify graceful error responses for invalid inputs."""

    @pytest.mark.asyncio
    async def test_remember_empty_content(self, mcp_app):
        """Empty content should return error, not crash."""
        result = _parse(await mcp_app.call_tool(
            "tribal_remember", {"content": ""},
        ))
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_remember_whitespace_content(self, mcp_app):
        """Whitespace-only content should return error."""
        result = _parse(await mcp_app.call_tool(
            "tribal_remember", {"content": "   \n\t  "},
        ))
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_recall_empty_query(self, mcp_app):
        """Empty query should return empty results, not crash."""
        result = _parse(await mcp_app.call_tool(
            "tribal_recall", {"query": ""},
        ))
        assert result["count"] == 0
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_correct_empty_original_id(self, mcp_app):
        """Empty original_id should return error."""
        result = _parse(await mcp_app.call_tool(
            "tribal_correct",
            {"original_id": "", "corrected_content": "New"},
        ))
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_correct_empty_content(self, mcp_app):
        """Empty corrected_content should return error."""
        result = _parse(await mcp_app.call_tool(
            "tribal_correct",
            {"original_id": "some-id", "corrected_content": ""},
        ))
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_forget_empty_id(self, mcp_app):
        """Empty memory_id should return error."""
        result = _parse(await mcp_app.call_tool(
            "tribal_forget", {"memory_id": ""},
        ))
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_forget_nonexistent_id(self, mcp_app):
        """Forgetting a non-existent ID should handle gracefully."""
        result = _parse(await mcp_app.call_tool("tribal_forget", {
            "memory_id": "nonexistent-uuid-12345",
        }))
        # Should not crash — result depends on impl
        assert "success" in result

    @pytest.mark.asyncio
    async def test_recall_clamped_limit(self, mcp_app):
        """Limits outside 1-50 range should be clamped, not error."""
        # Store a memory first
        await mcp_app.call_tool("tribal_remember", {
            "content": "Test memory for limit clamping",
            "source_type": "auto_capture",
        })

        # Request with limit=0 (should clamp to 1)
        result = _parse(await mcp_app.call_tool("tribal_recall", {
            "query": "test memory",
            "limit": 0,
        }))
        assert "results" in result

        # Request with limit=100 (should clamp to 50)
        result = _parse(await mcp_app.call_tool("tribal_recall", {
            "query": "test memory",
            "limit": 100,
        }))
        assert "results" in result


# ---------------------------------------------------------------------------
# Test: Tag Filtering
# ---------------------------------------------------------------------------

class TestMCPTagFiltering:
    """Verify tag-based filtering in recall."""

    @pytest.mark.asyncio
    async def test_recall_with_tag_filter(self, mcp_app):
        """Recall with tags should only return matching memories."""
        # Store memories with different tags
        await mcp_app.call_tool("tribal_remember", {
            "content": "Python is a great language",
            "tags": ["programming", "python"],
        })
        await mcp_app.call_tool("tribal_remember", {
            "content": "The weather is nice today",
            "tags": ["weather", "personal"],
        })

        # Recall with tag filter
        result = _parse(await mcp_app.call_tool("tribal_recall", {
            "query": "something interesting",
            "tags": ["programming"],
            "min_relevance": 0.0,
        }))
        # Should only return the programming memory
        for r in result["results"]:
            assert "programming" in r["tags"]
            assert "weather" not in r["tags"]


# ---------------------------------------------------------------------------
# Test: Stats
# ---------------------------------------------------------------------------

class TestMCPStats:
    """Verify stats tool returns expected structure."""

    @pytest.mark.asyncio
    async def test_stats_empty(self, mcp_app):
        """Stats on empty service should return valid structure."""
        result = _parse(await mcp_app.call_tool("tribal_stats", {}))
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_stats_after_operations(self, mcp_app):
        """Stats should reflect stored memories."""
        # Store a few memories
        for i in range(3):
            content = f"Memory {i}: unique content topic {i}"
            await mcp_app.call_tool("tribal_remember", {
                "content": content,
                "source_type": "user_explicit",
                "tags": ["test"],
                "skip_dedup": True,
            })

        result = _parse(await mcp_app.call_tool("tribal_stats", {}))
        assert isinstance(result, dict)
        # Should reflect that memories were stored
        if "total_memories" in result:
            assert result["total_memories"] >= 3


# ---------------------------------------------------------------------------
# Test: Concurrent Requests
# ---------------------------------------------------------------------------

class TestMCPConcurrency:
    """Verify concurrent tool calls don't corrupt state."""

    @pytest.mark.asyncio
    async def test_concurrent_remembers(self, mcp_app):
        """Multiple concurrent remember calls should all succeed or dedup properly."""
        tasks = []
        for i in range(5):
            content = f"Concurrent memory {i}: topic {i}"
            tasks.append(mcp_app.call_tool("tribal_remember", {
                "content": content,
                "source_type": "auto_capture",
                "skip_dedup": True,
            }))

        results = await asyncio.gather(*tasks)
        parsed = [_parse(r) for r in results]

        # All should succeed (skip_dedup=True)
        successes = [r for r in parsed if r["success"]]
        assert len(successes) == 5

    @pytest.mark.asyncio
    async def test_concurrent_recall_and_remember(self, mcp_app):
        """Concurrent reads and writes should not crash."""
        # Seed some data
        await mcp_app.call_tool("tribal_remember", {
            "content": "Base memory for concurrency test",
            "source_type": "auto_capture",
        })

        tasks = []
        # Mix of reads and writes
        for i in range(3):
            tasks.append(mcp_app.call_tool("tribal_recall", {
                "query": "concurrency test",
            }))
            tasks.append(mcp_app.call_tool("tribal_remember", {
                "content": f"Concurrent write {i} during reads",
                "source_type": "auto_capture",
                "skip_dedup": True,
            }))

        # Should complete without errors
        results = await asyncio.gather(*tasks)
        assert len(results) == 6


# -------------------------------------------------------------------
# Test: Export / Import Workflow
# -------------------------------------------------------------------

class TestMCPExportImport:
    """Verify export and import MCP tool integration."""

    @pytest.mark.asyncio
    async def test_export_empty(self, mcp_app):
        """Export with no memories returns valid bundle."""
        result = _parse(await mcp_app.call_tool(
            "tribal_export", {},
        ))
        assert result["success"] is True
        assert result["memory_count"] == 0

    @pytest.mark.asyncio
    async def test_export_after_remember(self, mcp_app):
        """Exported bundle should contain stored memories."""
        await mcp_app.call_tool("tribal_remember", {
            "content": "Export test memory",
            "source_type": "user_explicit",
            "tags": ["export"],
        })

        result = _parse(await mcp_app.call_tool(
            "tribal_export", {},
        ))
        assert result["success"] is True
        assert result["memory_count"] >= 1
        assert len(result["entries"]) >= 1

    @pytest.mark.asyncio
    async def test_export_with_tag_filter(self, mcp_app):
        """Export filtered by tag returns only matching."""
        await mcp_app.call_tool("tribal_remember", {
            "content": "Tagged memory for export",
            "tags": ["keep"],
        })
        await mcp_app.call_tool("tribal_remember", {
            "content": "Other memory to exclude",
            "tags": ["skip"],
        })

        result = _parse(await mcp_app.call_tool(
            "tribal_export", {"tags": ["keep"]},
        ))
        assert result["success"] is True
        for entry in result["entries"]:
            assert "keep" in entry.get("tags", [])

    @pytest.mark.asyncio
    async def test_export_to_file(self, mcp_app, tmp_path):
        """Export to file path should write JSON."""
        await mcp_app.call_tool("tribal_remember", {
            "content": "File export test",
            "source_type": "auto_capture",
        })

        out = str(tmp_path / "export.json")
        result = _parse(await mcp_app.call_tool(
            "tribal_export", {"output_path": out},
        ))
        assert result["success"] is True
        assert result["output_path"] == out

        with open(out) as f:
            data = json.load(f)
        assert "manifest" in data
        assert "entries" in data

    @pytest.mark.asyncio
    async def test_round_trip_export_import(
        self, mcp_app, tmp_path,
    ):
        """Export then import should preserve memories."""
        # Store a memory
        remember = _parse(await mcp_app.call_tool(
            "tribal_remember", {
                "content": "Round trip memory",
                "source_type": "user_explicit",
                "tags": ["roundtrip"],
            },
        ))
        assert remember["success"] is True

        # Export to file
        out = str(tmp_path / "bundle.json")
        exported = _parse(await mcp_app.call_tool(
            "tribal_export", {"output_path": out},
        ))
        assert exported["success"] is True

        # Import via file (dry run first)
        dry = _parse(await mcp_app.call_tool(
            "tribal_import", {
                "input_path": out,
                "dry_run": True,
            },
        ))
        assert dry["success"] is True
        assert dry["dry_run"] is True
        assert dry["total"] >= 1

        # Import for real (skip conflicts)
        result = _parse(await mcp_app.call_tool(
            "tribal_import", {
                "input_path": out,
                "conflict_resolution": "skip",
            },
        ))
        assert result["success"] is True
        assert result["total"] >= 1

    @pytest.mark.asyncio
    async def test_import_invalid_bundle(self, mcp_app):
        """Invalid bundle JSON should return error."""
        result = _parse(await mcp_app.call_tool(
            "tribal_import", {
                "bundle_json": "not valid json{{{",
            },
        ))
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_import_no_source(self, mcp_app):
        """Import with no path or JSON should error."""
        result = _parse(await mcp_app.call_tool(
            "tribal_import", {},
        ))
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_import_invalid_conflict_resolution(
        self, mcp_app, tmp_path,
    ):
        """Invalid conflict_resolution should error."""
        # Write a minimal bundle file
        p = tmp_path / "bad.json"
        p.write_text('{"manifest":{}, "entries":[]}')
        result = _parse(await mcp_app.call_tool(
            "tribal_import", {
                "input_path": str(p),
                "conflict_resolution": "invalid",
            },
        ))
        assert result["success"] is False
        assert "error" in result
