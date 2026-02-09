"""Graph visualization API routes.

Provides endpoints for exploring the knowledge graph:
- Entity listing with pagination
- Neighborhood exploration (N-hop traversal)
- Relationship queries

All sync GraphStore calls are wrapped in asyncio.to_thread
to avoid blocking the event loop.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..services import TribalMemoryService
from .routes import get_memory_service

router = APIRouter(prefix="/v1/graph", tags=["graph"])


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------


class EntityNode(BaseModel):
    """A node in the graph."""

    name: str
    entity_type: str
    memory_count: int = 0


class RelationshipEdge(BaseModel):
    """An edge in the graph."""

    source: str
    target: str
    relation_type: str


class NeighborhoodResponse(BaseModel):
    """Neighborhood around a focal entity."""

    focal: str
    nodes: list[EntityNode]
    edges: list[RelationshipEdge]
    hops: int


class EntityListResponse(BaseModel):
    """Paginated entity list."""

    entities: list[EntityNode]
    total: int
    offset: int
    limit: int


class GraphStatsResponse(BaseModel):
    """High-level graph statistics."""

    entity_count: int
    relationship_count: int
    entity_types: dict[str, int]
    relationship_types: dict[str, int]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_graph(
    service: TribalMemoryService,
):
    """Extract graph store or raise 404."""
    graph = service.graph_store
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail="Graph store not available",
        )
    return graph


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@router.get("/stats", response_model=GraphStatsResponse)
async def graph_stats(
    service: TribalMemoryService = Depends(
        get_memory_service,
    ),
) -> GraphStatsResponse:
    """Get high-level graph statistics."""
    graph = _get_graph(service)
    data = await asyncio.to_thread(graph.get_graph_stats)
    return GraphStatsResponse(**data)


@router.get("/entities", response_model=EntityListResponse)
async def list_entities(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    entity_type: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    service: TribalMemoryService = Depends(
        get_memory_service,
    ),
) -> EntityListResponse:
    """List entities with pagination and optional filters."""
    graph = _get_graph(service)

    entities_raw, total = await asyncio.to_thread(
        graph.list_entities,
        offset=offset,
        limit=limit,
        entity_type=entity_type,
        search=search,
    )

    entities = [EntityNode(**e) for e in entities_raw]

    return EntityListResponse(
        entities=entities,
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/neighborhood/{entity_name}",
    response_model=NeighborhoodResponse,
)
async def neighborhood(
    entity_name: str,
    hops: int = Query(default=1, ge=1, le=3),
    service: TribalMemoryService = Depends(
        get_memory_service,
    ),
) -> NeighborhoodResponse:
    """Explore the neighborhood around an entity."""
    graph = _get_graph(service)

    connected = await asyncio.to_thread(
        graph.find_connected,
        entity_name,
        hops=hops,
        include_source=True,
    )

    if not connected:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{entity_name}' not found",
        )

    entity_names = {e.name for e in connected}

    # Get memory counts + edges in thread
    def _build_neighborhood():
        memory_counts: dict[str, int] = {}
        for e in connected:
            mems = graph.get_memories_for_entity(e.name)
            memory_counts[e.name] = len(mems)

        edges: list[RelationshipEdge] = []
        seen: set[tuple[str, str, str]] = set()
        for e in connected:
            rels = graph.get_relationships_for_entity(
                e.name,
            )
            for r in rels:
                if r.target in entity_names:
                    key = (
                        r.source,
                        r.target,
                        r.relation_type,
                    )
                    if key not in seen:
                        seen.add(key)
                        edges.append(RelationshipEdge(
                            source=r.source,
                            target=r.target,
                            relation_type=r.relation_type,
                        ))
        return memory_counts, edges

    memory_counts, edges = await asyncio.to_thread(
        _build_neighborhood,
    )

    nodes = [
        EntityNode(
            name=e.name,
            entity_type=e.entity_type,
            memory_count=memory_counts.get(e.name, 0),
        )
        for e in connected
    ]

    return NeighborhoodResponse(
        focal=entity_name,
        nodes=nodes,
        edges=edges,
        hops=hops,
    )
