"""API route handlers."""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends

from ..interfaces import MemorySource, MemoryEntry
from ..services import TribalMemoryService
from .models import (
    RememberRequest,
    RecallRequest,
    CorrectRequest,
    StoreResponse,
    RecallResponse,
    RecallResultResponse,
    MemoryEntryResponse,
    HealthResponse,
    StatsResponse,
    ForgetResponse,
    ShutdownResponse,
    SourceType,
    ExportRequest,
    ExportResponse,
    ImportRequest,
    ImportResponse,
)

router = APIRouter(prefix="/v1", tags=["memory"])


def get_memory_service() -> TribalMemoryService:
    """Dependency injection for memory service.
    
    This is set by the app during startup.
    """
    from .app import _memory_service
    if _memory_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _memory_service


def get_instance_id() -> str:
    """Get the current instance ID."""
    from .app import _instance_id
    return _instance_id or "unknown"


def _convert_source_type(source_type: SourceType) -> MemorySource:
    """Convert API source type to internal enum."""
    return MemorySource(source_type.value)


def _entry_to_response(entry: MemoryEntry) -> MemoryEntryResponse:
    """Convert internal MemoryEntry to API response."""
    return MemoryEntryResponse(
        id=entry.id,
        content=entry.content,
        source_instance=entry.source_instance,
        source_type=SourceType(entry.source_type.value),
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        tags=entry.tags,
        context=entry.context,
        confidence=entry.confidence,
        supersedes=entry.supersedes,
    )


@router.post("/remember", response_model=StoreResponse)
async def remember(
    request: RememberRequest,
    service: TribalMemoryService = Depends(get_memory_service),
) -> StoreResponse:
    """Store a new memory."""
    try:
        result = await service.remember(
            content=request.content,
            source_type=_convert_source_type(request.source_type),
            context=request.context,
            tags=request.tags,
            skip_dedup=request.skip_dedup,
        )

        return StoreResponse(
            success=result.success,
            memory_id=result.memory_id,
            duplicate_of=result.duplicate_of,
            error=result.error,
        )
    except Exception as e:
        return StoreResponse(success=False, error=str(e))


@router.post("/recall", response_model=RecallResponse)
async def recall(
    request: RecallRequest,
    service: TribalMemoryService = Depends(get_memory_service),
) -> RecallResponse:
    """Recall relevant memories for a query."""
    try:
        start_time = time.time()

        results = await service.recall(
            query=request.query,
            limit=request.limit,
            min_relevance=request.min_relevance,
            tags=request.tags,
        )

        total_time_ms = (time.time() - start_time) * 1000

        return RecallResponse(
            results=[
                RecallResultResponse(
                    memory=_entry_to_response(r.memory),
                    similarity_score=r.similarity_score,
                    retrieval_time_ms=r.retrieval_time_ms,
                )
                for r in results
            ],
            query=request.query,
            total_time_ms=total_time_ms,
        )
    except Exception as e:
        # Return empty results with error info for consistency with other endpoints
        return RecallResponse(
            results=[],
            query=request.query,
            total_time_ms=0.0,
            error=str(e),
        )


@router.post("/correct", response_model=StoreResponse)
async def correct(
    request: CorrectRequest,
    service: TribalMemoryService = Depends(get_memory_service),
) -> StoreResponse:
    """Correct an existing memory."""
    try:
        result = await service.correct(
            original_id=request.original_id,
            corrected_content=request.corrected_content,
            context=request.context,
        )

        return StoreResponse(
            success=result.success,
            memory_id=result.memory_id,
            error=result.error,
        )
    except Exception as e:
        return StoreResponse(success=False, error=str(e))


@router.delete("/forget/{memory_id}", response_model=ForgetResponse)
async def forget(
    memory_id: str,
    service: TribalMemoryService = Depends(get_memory_service),
) -> ForgetResponse:
    """Forget (delete) a specific memory. GDPR-compliant."""
    try:
        success = await service.forget(memory_id)
        return ForgetResponse(success=success, memory_id=memory_id)
    except Exception as e:
        return ForgetResponse(success=False, memory_id=memory_id)


@router.get("/memory/{memory_id}", response_model=MemoryEntryResponse)
async def get_memory(
    memory_id: str,
    service: TribalMemoryService = Depends(get_memory_service),
) -> MemoryEntryResponse:
    """Get a specific memory by ID."""
    entry = await service.get(memory_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return _entry_to_response(entry)


@router.get("/health", response_model=HealthResponse)
async def health(
    service: TribalMemoryService = Depends(get_memory_service),
    instance_id: str = Depends(get_instance_id),
) -> HealthResponse:
    """Health check endpoint."""
    try:
        stats = await service.get_stats()
        return HealthResponse(
            status="ok",
            instance_id=instance_id,
            memory_count=stats.get("total_memories", 0),
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            instance_id=instance_id,
            memory_count=0,
        )


@router.get("/stats", response_model=StatsResponse)
async def stats(
    service: TribalMemoryService = Depends(get_memory_service),
    instance_id: str = Depends(get_instance_id),
) -> StatsResponse:
    """Get memory statistics."""
    stats_data = await service.get_stats()
    return StatsResponse(
        total_memories=stats_data.get("total_memories", 0),
        by_source_type=stats_data.get("by_source_type", {}),
        by_tag=stats_data.get("by_tag", {}),
        instance_id=instance_id,
    )


@router.post("/export", response_model=ExportResponse)
async def export_memories_route(
    request: ExportRequest,
    service: TribalMemoryService = Depends(get_memory_service),
) -> ExportResponse:
    """Export memories with optional tag/date filtering."""
    from ..portability.embedding_metadata import (
        create_embedding_metadata,
    )
    from ..services.import_export import (
        ExportFilter,
        export_memories,
        parse_iso_datetime,
    )

    # Validate dates
    parsed_from, err = parse_iso_datetime(
        request.date_from, "date_from",
    )
    if err:
        return ExportResponse(success=False, error=err)
    parsed_to, err = parse_iso_datetime(
        request.date_to, "date_to",
    )
    if err:
        return ExportResponse(success=False, error=err)

    try:
        emb = service.embedding_service
        meta = create_embedding_metadata(
            model_name=getattr(emb, "model", "unknown"),
            dimensions=getattr(emb, "dimensions", 1536),
            provider="openai",
        )

        flt = None
        if request.tags or parsed_from or parsed_to:
            flt = ExportFilter(
                tags=request.tags,
                date_from=parsed_from,
                date_to=parsed_to,
            )

        bundle = await export_memories(
            store=service.vector_store,
            embedding_metadata=meta,
            filters=flt,
        )

        return ExportResponse(
            success=True,
            memory_count=bundle.manifest.memory_count,
            bundle=bundle.to_dict(),
        )
    except Exception as e:
        return ExportResponse(success=False, error=str(e))


@router.post("/import", response_model=ImportResponse)
async def import_memories_route(
    request: ImportRequest,
    service: TribalMemoryService = Depends(get_memory_service),
) -> ImportResponse:
    """Import memories from a portable bundle."""
    from ..portability.embedding_metadata import (
        PortableBundle,
        ReembeddingStrategy,
        create_embedding_metadata,
    )
    from ..services.import_export import (
        ConflictResolution,
        import_memories,
        validate_conflict_resolution,
        validate_embedding_strategy,
    )

    # Validate enum params
    err = validate_conflict_resolution(
        request.conflict_resolution,
    )
    if err:
        return ImportResponse(success=False, error=err)
    err = validate_embedding_strategy(
        request.embedding_strategy,
    )
    if err:
        return ImportResponse(success=False, error=err)

    try:
        bundle = PortableBundle.from_dict(request.bundle)
    except Exception as e:
        return ImportResponse(
            success=False, error=f"Invalid bundle: {e}",
        )

    emb = service.embedding_service
    target_meta = create_embedding_metadata(
        model_name=getattr(emb, "model", "unknown"),
        dimensions=getattr(emb, "dimensions", 1536),
        provider="openai",
    )

    cr_map = {
        "skip": ConflictResolution.SKIP,
        "overwrite": ConflictResolution.OVERWRITE,
        "merge": ConflictResolution.MERGE,
    }
    es_map = {
        "auto": ReembeddingStrategy.AUTO,
        "keep": ReembeddingStrategy.KEEP,
        "drop": ReembeddingStrategy.DROP,
    }

    try:
        summary = await import_memories(
            bundle=bundle,
            store=service.vector_store,
            target_metadata=target_meta,
            conflict_resolution=cr_map[
                request.conflict_resolution
            ],
            embedding_strategy=es_map[
                request.embedding_strategy
            ],
        )

        return ImportResponse(
            success=True,
            total=summary.total,
            imported=summary.imported,
            skipped=summary.skipped,
            overwritten=summary.overwritten,
            errors=summary.errors,
            needs_reembedding=summary.needs_reembedding,
            error_details=summary.error_details,
        )
    except Exception as e:
        return ImportResponse(success=False, error=str(e))


@router.post("/shutdown", response_model=ShutdownResponse)
async def shutdown() -> ShutdownResponse:
    """Graceful shutdown endpoint.
    
    Security note: This endpoint is localhost-only (bound to 127.0.0.1).
    It allows local process management without authentication since only
    processes on the same machine can reach it. For production deployments
    with network exposure, use systemctl/signals instead of this endpoint.
    """
    import asyncio
    import signal
    import os

    # Schedule shutdown after response is sent
    asyncio.get_event_loop().call_later(
        0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)
    )
    return ShutdownResponse(status="shutting_down")
