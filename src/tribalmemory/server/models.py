"""Pydantic models for HTTP API request/response."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Memory source types."""
    USER_EXPLICIT = "user_explicit"
    AUTO_CAPTURE = "auto_capture"
    CORRECTION = "correction"
    CROSS_INSTANCE = "cross_instance"
    LEGACY = "legacy"
    UNKNOWN = "unknown"


# =============================================================================
# Request Models
# =============================================================================

class RememberRequest(BaseModel):
    """Request to store a new memory."""
    content: str = Field(..., description="The memory content to store")
    source_type: SourceType = Field(
        default=SourceType.AUTO_CAPTURE,
        description="How this memory was captured"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context about the capture"
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="Tags for categorization and filtering"
    )
    skip_dedup: bool = Field(
        default=False,
        description="If True, store even if similar memory exists"
    )


class RecallRequest(BaseModel):
    """Request to recall memories."""
    query: str = Field(..., description="Natural language search query")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum results")
    min_relevance: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="Filter by tags"
    )
    after: Optional[str] = Field(
        default=None,
        description="Only include memories with events on/after this date (ISO or natural language)"
    )
    before: Optional[str] = Field(
        default=None,
        description="Only include memories with events on/before this date (ISO or natural language)"
    )


class CorrectRequest(BaseModel):
    """Request to correct an existing memory."""
    original_id: str = Field(..., description="ID of memory to correct")
    corrected_content: str = Field(..., description="Corrected information")
    context: Optional[str] = Field(
        default=None,
        description="Context about the correction"
    )


# =============================================================================
# Response Models
# =============================================================================

class MemoryEntryResponse(BaseModel):
    """A single memory entry."""
    id: str
    content: str
    source_instance: str
    source_type: SourceType
    created_at: datetime
    updated_at: datetime
    tags: list[str]
    context: Optional[str]
    confidence: float
    supersedes: Optional[str]

    model_config = {"from_attributes": True}


class StoreResponse(BaseModel):
    """Response from storing a memory."""
    success: bool
    memory_id: Optional[str] = None
    duplicate_of: Optional[str] = None
    error: Optional[str] = None


class RecallResultResponse(BaseModel):
    """A single recall result with score."""
    memory: MemoryEntryResponse
    similarity_score: float
    retrieval_time_ms: float


class RecallResponse(BaseModel):
    """Response from recalling memories."""
    results: list[RecallResultResponse]
    query: str
    total_time_ms: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    instance_id: str
    memory_count: int
    version: str = "0.1.0"


class StatsResponse(BaseModel):
    """Memory statistics response."""
    total_memories: int
    by_source_type: dict[str, int]
    by_tag: dict[str, int]
    instance_id: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


class ForgetResponse(BaseModel):
    """Response from forgetting a memory."""
    success: bool
    memory_id: str


class ShutdownResponse(BaseModel):
    """Response from shutdown request."""
    status: str


# =============================================================================
# Import/Export Models (Issue #7)
# =============================================================================

class ExportRequest(BaseModel):
    """Request to export memories."""
    tags: Optional[list[str]] = Field(
        default=None,
        description="Filter: only memories matching any of these tags",
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Filter: ISO 8601 datetime lower bound (created_at)",
    )
    date_to: Optional[str] = Field(
        default=None,
        description="Filter: ISO 8601 datetime upper bound (created_at)",
    )


class ExportResponse(BaseModel):
    """Response containing the exported bundle."""
    success: bool
    memory_count: int = 0
    bundle: Optional[dict] = None
    error: Optional[str] = None


class ImportRequest(BaseModel):
    """Request to import memories from a bundle."""
    bundle: dict = Field(
        ..., description="Portable bundle (manifest + entries)",
    )
    conflict_resolution: str = Field(
        default="skip",
        description="Conflict strategy: skip | overwrite | merge",
    )
    embedding_strategy: str = Field(
        default="auto",
        description="Embedding strategy: auto | keep | drop",
    )
    dry_run: bool = Field(
        default=False,
        description="Preview changes without writing",
    )


class ImportResponse(BaseModel):
    """Response from import operation."""
    success: bool
    total: int = 0
    imported: int = 0
    skipped: int = 0
    overwritten: int = 0
    errors: int = 0
    needs_reembedding: bool = False
    dry_run: bool = False
    duration_ms: float = 0.0
    error_details: list[str] = Field(default_factory=list)
    error: Optional[str] = None

# =============================================================================
# Session Indexing Models (Issue #38)
# =============================================================================

class SessionMessageRequest(BaseModel):
    """A single message in a session transcript."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="When the message was sent")


class SessionIngestRequest(BaseModel):
    """Request to ingest session transcript."""
    session_id: str = Field(..., description="Unique session identifier")
    messages: list[SessionMessageRequest] = Field(
        ..., description="Conversation messages to index"
    )
    instance_id: Optional[str] = Field(
        default=None,
        description="Override instance ID (defaults to server's instance_id)"
    )


class SessionIngestResponse(BaseModel):
    """Response from session ingestion."""
    success: bool
    chunks_created: int = 0
    messages_processed: int = 0
    error: Optional[str] = None


class SessionSearchRequest(BaseModel):
    """Request to search session transcripts."""
    query: str = Field(..., description="Natural language search query")
    session_id: Optional[str] = Field(
        default=None,
        description="Filter to specific session (optional)"
    )
    limit: int = Field(default=5, ge=1, le=50, description="Maximum results")
    min_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )


class SessionChunkResponse(BaseModel):
    """A session transcript chunk result."""
    chunk_id: str
    session_id: str
    instance_id: str
    content: str
    similarity_score: float
    start_time: datetime
    end_time: datetime
    chunk_index: int


class SessionSearchResponse(BaseModel):
    """Response from session search with pagination."""
    results: list[SessionChunkResponse]
    query: str
    total_count: int = 0
    offset: int = 0
    limit: int = 5
    has_more: bool = False
    error: Optional[str] = None
