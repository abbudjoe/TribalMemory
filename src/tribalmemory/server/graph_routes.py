"""Graph visualization API routes.

Provides endpoints for exploring the knowledge graph:
- Entity listing with pagination
- Neighborhood exploration (N-hop traversal)
- Relationship queries
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

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
# Routes
# ------------------------------------------------------------------


@router.get("/stats", response_model=GraphStatsResponse)
async def graph_stats(
    service=Depends(get_memory_service),
) -> GraphStatsResponse:
    """Get high-level graph statistics."""
    graph = service.graph_store
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail="Graph store not available",
        )

    with graph._lock:
        entity_count = graph._conn.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]

        rel_count = graph._conn.execute(
            "SELECT COUNT(*) FROM relationships"
        ).fetchone()[0]

        type_rows = graph._conn.execute(
            "SELECT entity_type, COUNT(*) as cnt "
            "FROM entities GROUP BY entity_type"
        ).fetchall()
        entity_types = {
            r["entity_type"]: r["cnt"] for r in type_rows
        }

        rel_type_rows = graph._conn.execute(
            "SELECT relation_type, COUNT(*) as cnt "
            "FROM relationships GROUP BY relation_type"
        ).fetchall()
        rel_types = {
            r["relation_type"]: r["cnt"] for r in rel_type_rows
        }

    return GraphStatsResponse(
        entity_count=entity_count,
        relationship_count=rel_count,
        entity_types=entity_types,
        relationship_types=rel_types,
    )


@router.get("/entities", response_model=EntityListResponse)
async def list_entities(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    entity_type: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    service=Depends(get_memory_service),
) -> EntityListResponse:
    """List entities with pagination and optional filters."""
    graph = service.graph_store
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail="Graph store not available",
        )

    with graph._lock:
        where_clauses = []
        params: list = []

        if entity_type:
            where_clauses.append("e.entity_type = ?")
            params.append(entity_type)

        if search:
            where_clauses.append("e.name LIKE ?")
            params.append(f"%{search}%")

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Total count
        total = graph._conn.execute(
            f"SELECT COUNT(*) FROM entities e {where_sql}",
            params,
        ).fetchone()[0]

        # Paginated results with memory counts
        rows = graph._conn.execute(
            f"""
            SELECT e.name, e.entity_type,
                   COUNT(em.memory_id) as mem_count
            FROM entities e
            LEFT JOIN entity_memories em
                ON e.id = em.entity_id
            {where_sql}
            GROUP BY e.id, e.name, e.entity_type
            ORDER BY mem_count DESC, e.name
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()

    entities = [
        EntityNode(
            name=r["name"],
            entity_type=r["entity_type"],
            memory_count=r["mem_count"],
        )
        for r in rows
    ]

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
    service=Depends(get_memory_service),
) -> NeighborhoodResponse:
    """Explore the neighborhood around an entity.

    Returns all nodes and edges within N hops.
    """
    graph = service.graph_store
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail="Graph store not available",
        )

    # Get connected entities
    connected = graph.find_connected(
        entity_name, hops=hops, include_source=True,
    )

    if not connected:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{entity_name}' not found",
        )

    # Collect all entity names in the neighborhood
    entity_names = {e.name for e in connected}

    # Get memory counts for each entity
    memory_counts: dict[str, int] = {}
    for e in connected:
        mems = graph.get_memories_for_entity(e.name)
        memory_counts[e.name] = len(mems)

    # Get all edges between entities in the neighborhood
    edges: list[RelationshipEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for e in connected:
        rels = graph.get_relationships_for_entity(e.name)
        for r in rels:
            if r.target in entity_names:
                key = (r.source, r.target, r.relation_type)
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append(RelationshipEdge(
                        source=r.source,
                        target=r.target,
                        relation_type=r.relation_type,
                    ))

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
