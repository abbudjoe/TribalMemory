"""Tests for graph visualization API routes."""

import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tribalmemory.server.graph_routes import router
from tribalmemory.server.routes import router as main_router
from tribalmemory.server import app as app_module
from tribalmemory.services import create_memory_service


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    """Reset module state after each test."""
    yield
    app_module._memory_service = None
    app_module._session_store = None
    app_module._instance_id = None


@pytest.fixture
def client_with_graph(tmp_path) -> TestClient:
    """Test client with a real memory service + graph data."""
    service = create_memory_service(
        instance_id="test-graph",
        db_path=str(tmp_path / "db"),
    )

    app_module._memory_service = service
    app_module._session_store = None
    app_module._instance_id = "test-graph"

    app = FastAPI()
    app.include_router(main_router)
    app.include_router(router)

    # Seed graph data directly
    from tribalmemory.services.graph_store import (
        Entity, Relationship,
    )
    graph = service.graph_store

    entities = [
        (Entity("Python", "technology"), "mem-1"),
        (Entity("FastAPI", "technology"), "mem-1"),
        (Entity("Joe", "person"), "mem-1"),
        (Entity("Redis", "service"), "mem-2"),
        (Entity("Python", "technology"), "mem-2"),
    ]
    for entity, mem_id in entities:
        graph.add_entity(entity, memory_id=mem_id)

    relationships = [
        (Relationship("FastAPI", "Python", "uses"), "mem-1"),
        (Relationship("Joe", "Python", "knows"), "mem-1"),
        (
            Relationship("Redis", "Python", "connects_to"),
            "mem-2",
        ),
    ]
    for rel, mem_id in relationships:
        graph.add_relationship(rel, memory_id=mem_id)

    return TestClient(app)


class TestGraphStats:
    """Test /v1/graph/stats endpoint."""

    def test_stats_returns_counts(
        self, client_with_graph,
    ) -> None:
        """Stats should return entity and relationship counts."""
        resp = client_with_graph.get("/v1/graph/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_count"] >= 4
        assert data["relationship_count"] >= 3
        assert "technology" in data["entity_types"]
        assert "uses" in data["relationship_types"]


class TestEntityList:
    """Test /v1/graph/entities endpoint."""

    def test_list_entities(
        self, client_with_graph,
    ) -> None:
        """Should return paginated entity list."""
        resp = client_with_graph.get("/v1/graph/entities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 4
        assert len(data["entities"]) >= 4
        assert data["offset"] == 0
        assert data["limit"] == 50

    def test_list_entities_with_type_filter(
        self, client_with_graph,
    ) -> None:
        """Should filter by entity type."""
        resp = client_with_graph.get(
            "/v1/graph/entities?entity_type=person"
        )
        data = resp.json()
        assert all(
            e["entity_type"] == "person"
            for e in data["entities"]
        )

    def test_list_entities_with_search(
        self, client_with_graph,
    ) -> None:
        """Should search by name substring."""
        resp = client_with_graph.get(
            "/v1/graph/entities?search=Pyth"
        )
        data = resp.json()
        assert any(
            e["name"] == "Python" for e in data["entities"]
        )

    def test_list_entities_pagination(
        self, client_with_graph,
    ) -> None:
        """Should respect offset and limit."""
        resp = client_with_graph.get(
            "/v1/graph/entities?offset=0&limit=2"
        )
        data = resp.json()
        assert len(data["entities"]) <= 2
        assert data["offset"] == 0
        assert data["limit"] == 2


class TestNeighborhood:
    """Test /v1/graph/neighborhood endpoint."""

    def test_neighborhood_returns_nodes_and_edges(
        self, client_with_graph,
    ) -> None:
        """Should return neighborhood with nodes + edges."""
        resp = client_with_graph.get(
            "/v1/graph/neighborhood/Python"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["focal"] == "Python"
        assert len(data["nodes"]) >= 1
        assert data["hops"] == 1

        # Python should be in nodes
        names = [n["name"] for n in data["nodes"]]
        assert "Python" in names

    def test_neighborhood_includes_connected(
        self, client_with_graph,
    ) -> None:
        """Connected entities should appear."""
        resp = client_with_graph.get(
            "/v1/graph/neighborhood/Python"
        )
        data = resp.json()
        names = [n["name"] for n in data["nodes"]]
        # FastAPI uses Python, so it should be a neighbor
        assert "FastAPI" in names

    def test_neighborhood_not_found(
        self, client_with_graph,
    ) -> None:
        """Non-existent entity returns 404."""
        resp = client_with_graph.get(
            "/v1/graph/neighborhood/NonExistent"
        )
        assert resp.status_code == 404

    def test_neighborhood_multi_hop(
        self, client_with_graph,
    ) -> None:
        """2-hop traversal reaches further."""
        resp = client_with_graph.get(
            "/v1/graph/neighborhood/Joe?hops=2"
        )
        data = resp.json()
        assert data["hops"] == 2
        names = [n["name"] for n in data["nodes"]]
        # Joe -> Python -> FastAPI (2 hops)
        assert "Joe" in names

    def test_neighborhood_has_edges(
        self, client_with_graph,
    ) -> None:
        """Edges between neighborhood nodes are returned."""
        resp = client_with_graph.get(
            "/v1/graph/neighborhood/Python"
        )
        data = resp.json()
        assert len(data["edges"]) >= 1
        edge_types = [e["relation_type"] for e in data["edges"]]
        assert "uses" in edge_types or "connects_to" in edge_types


class TestNeighborhoodValidation:
    """Test neighborhood input validation."""

    def test_empty_entity_name(
        self, client_with_graph,
    ) -> None:
        """Empty entity name returns 400."""
        resp = client_with_graph.get(
            "/v1/graph/neighborhood/%20"
        )
        assert resp.status_code == 400

    def test_isolated_entity(self, tmp_path) -> None:
        """Entity with no relationships returns node."""
        from tribalmemory.services.graph_store import Entity
        service = create_memory_service(
            instance_id="test-iso",
            db_path=str(tmp_path / "iso-db"),
        )
        app_module._memory_service = service
        app_module._instance_id = "test-iso"

        graph = service.graph_store
        graph.add_entity(
            Entity("Lonely", "concept"), "mem-x",
        )

        app = FastAPI()
        app.include_router(main_router)
        app.include_router(router)
        client = TestClient(app)

        resp = client.get(
            "/v1/graph/neighborhood/Lonely"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["focal"] == "Lonely"
        assert len(data["nodes"]) == 1
        assert data["edges"] == []


    def test_isolated_entity_exact_match(
        self, tmp_path,
    ) -> None:
        """Exact-name lookup should not miss entities
        when many substring matches exist."""
        from tribalmemory.services.graph_store import Entity
        service = create_memory_service(
            instance_id="test-exact",
            db_path=str(tmp_path / "exact-db"),
        )
        app_module._memory_service = service
        app_module._instance_id = "test-exact"

        graph = service.graph_store
        # Add target entity (isolated)
        graph.add_entity(
            Entity("AI", "concept"), "mem-x",
        )
        # Add many substring matches to verify
        # exact lookup isn't fooled by pagination
        for i in range(50):
            graph.add_entity(
                Entity(f"AI System {i}", "technology"),
                f"mem-{i}",
            )

        app = FastAPI()
        app.include_router(main_router)
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/v1/graph/neighborhood/AI")
        assert resp.status_code == 200
        data = resp.json()
        assert data["focal"] == "AI"
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["name"] == "AI"


class TestGraphUI:
    """Test /graph HTML endpoint."""

    def test_graph_page_served(self) -> None:
        """Graph UI should be served at /graph."""
        from tribalmemory.server.app import create_app
        app = create_app()
        client = TestClient(app)
        resp = client.get("/graph")
        assert resp.status_code == 200
        assert "cytoscape" in resp.text.lower()
        assert "Knowledge Graph" in resp.text


class TestStaticAssets:
    """Verify static assets are packaged correctly."""

    def test_static_files_exist(self) -> None:
        """Static files must exist alongside server code."""
        import importlib.resources as resources
        import tribalmemory.server as srv
        srv_dir = (
            resources.files(srv) / "static"
        )
        assert srv_dir.is_dir(), (
            "server/static/ directory missing — "
            "check pyproject.toml package-data"
        )
        graph_html = srv_dir / "graph.html"
        cytoscape_js = srv_dir / "cytoscape.min.js"
        assert graph_html.is_file(), (
            "graph.html missing from static/"
        )
        assert cytoscape_js.is_file(), (
            "cytoscape.min.js missing from static/"
        )
