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
