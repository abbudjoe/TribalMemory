"""Session transcript indexing service.

Indexes conversation transcripts as chunked embeddings for contextual recall.
Supports delta-based ingestion and retention-based cleanup.
"""

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

from ..interfaces import IEmbeddingService, IVectorStore

logger = logging.getLogger(__name__)


@dataclass
class SessionMessage:
    """A single message in a conversation transcript.
    
    Attributes:
        role: Message role (user, assistant, system)
        content: Message content
        timestamp: When the message was sent
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SessionChunk:
    """A chunk of conversation transcript with embedding.
    
    Attributes:
        chunk_id: Unique identifier for this chunk
        session_id: ID of the session this chunk belongs to
        instance_id: Which agent instance processed this session
        content: The actual conversation content (multiple messages)
        embedding: Vector embedding of the content
        start_time: Timestamp of first message in chunk
        end_time: Timestamp of last message in chunk
        chunk_index: Sequential index within session (0, 1, 2...)
    """
    chunk_id: str
    session_id: str
    instance_id: str
    content: str
    embedding: list[float]
    start_time: datetime
    end_time: datetime
    chunk_index: int


class SessionStore:
    """Service for indexing and searching session transcripts.
    
    Usage:
        store = SessionStore(
            instance_id="clawdio-1",
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
        
        # Ingest a session transcript
        messages = [
            SessionMessage("user", "What is Docker?", datetime.now(timezone.utc)),
            SessionMessage("assistant", "Docker is a container platform", datetime.now(timezone.utc)),
        ]
        await store.ingest("session-123", messages)
        
        # Search across all sessions
        results = await store.search("Docker setup error")
        
        # Search within specific session
        results = await store.search("Docker", session_id="session-123")
    """
    
    # Chunking parameters
    TARGET_CHUNK_TOKENS = 400  # Target size for each chunk
    WORDS_PER_TOKEN = 0.75     # Approximate tokens per word
    OVERLAP_TOKENS = 50        # Overlap between chunks for context
    
    def __init__(
        self,
        instance_id: str,
        embedding_service: IEmbeddingService,
        vector_store: IVectorStore,
    ):
        self.instance_id = instance_id
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        
        # Track last ingested index per session for delta ingestion
        self._session_state: dict[str, int] = {}
    
    async def ingest(
        self,
        session_id: str,
        messages: list[SessionMessage],
        instance_id: Optional[str] = None,
    ) -> dict:
        """Ingest session messages with delta-based processing.
        
        Only processes new messages since last ingestion for this session.
        
        Args:
            session_id: Unique identifier for the session
            messages: List of conversation messages
            instance_id: Override instance ID (defaults to self.instance_id)
        
        Returns:
            Dict with keys: success, chunks_created, messages_processed
        """
        if not messages:
            return {
                "success": True,
                "chunks_created": 0,
                "messages_processed": 0,
            }
        
        # Allow subclasses to initialize (e.g., restore state from DB)
        await self._ensure_ready()

        # Delta ingestion: only process new messages
        last_index = self._session_state.get(session_id, 0)
        new_messages = messages[last_index:]
        
        if not new_messages:
            return {
                "success": True,
                "chunks_created": 0,
                "messages_processed": 0,
            }
        
        try:
            # Create chunks from new messages
            chunks = await self._chunk_messages(
                new_messages,
                session_id,
                instance_id or self.instance_id,
            )
            
            # Store chunks in vector store.
            # Pass total_messages so LanceDB store can persist delta state.
            for chunk in chunks:
                await self._store_chunk(chunk, total_messages=len(messages))
            
            # Update state
            self._session_state[session_id] = len(messages)
            
            return {
                "success": True,
                "chunks_created": len(chunks),
                "messages_processed": len(new_messages),
            }
        
        except Exception as e:
            logger.exception(f"Failed to ingest session {session_id}: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    async def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 5,
        offset: int = 0,
        min_relevance: float = 0.0,
    ) -> dict:
        """Search session transcripts by semantic similarity.
        
        Args:
            query: Natural language search query.
            session_id: Optional filter to specific session.
            limit: Maximum number of results per page.
            offset: Number of results to skip (for pagination).
                Negative values are clamped to 0.
            min_relevance: Minimum similarity score (0.0 to 1.0).
        
        Returns:
            Dict with:
                items: List of result dicts (chunk_id, session_id,
                    instance_id, content, similarity_score,
                    start_time, end_time, chunk_index).
                total_count: Total matching results (before
                    limit/offset).
        """
        offset = max(0, offset)

        try:
            await self._ensure_ready()

            query_embedding = await self.embedding_service.embed(query)
            
            # Fetch enough results to determine total_count and
            # serve the requested page.  We request a large pool so
            # total_count reflects all matching chunks, not just the
            # first page.  _search_chunks implementations already
            # filter by min_relevance before returning.
            pool_limit = max(offset + limit, 1000)
            all_results = await self._search_chunks(
                query_embedding,
                session_id,
                limit=pool_limit,
                min_relevance=min_relevance,
            )
            
            total_count = len(all_results)
            items = all_results[offset:offset + limit]
            
            return {"items": items, "total_count": total_count}
        
        except ValueError:
            raise
        except Exception as e:
            logger.exception(f"Failed to search sessions: {e}")
            return {"items": [], "total_count": 0}
    
    async def cleanup(self, retention_days: int = 30) -> int:
        """Delete session chunks older than retention period.
        
        Args:
            retention_days: Number of days to retain chunks
        
        Returns:
            Number of chunks deleted
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=retention_days)
            
            # Find and delete expired chunks
            deleted = await self._delete_chunks_before(cutoff_time)
            
            return deleted
        
        except Exception as e:
            logger.exception(f"Failed to cleanup sessions: {e}")
            return 0
    
    async def get_stats(self) -> dict:
        """Get statistics about indexed sessions.
        
        Returns:
            Dict with keys: total_chunks, total_sessions,
            earliest_chunk, latest_chunk
        """
        try:
            chunks = await self._get_all_chunks()
            
            if not chunks:
                return {
                    "total_chunks": 0,
                    "total_sessions": 0,
                    "earliest_chunk": None,
                    "latest_chunk": None,
                }
            
            session_ids = set()
            timestamps = []
            
            for chunk in chunks:
                session_ids.add(chunk["session_id"])
                timestamps.append(chunk["start_time"])
            
            return {
                "total_chunks": len(chunks),
                "total_sessions": len(session_ids),
                "earliest_chunk": min(timestamps) if timestamps else None,
                "latest_chunk": max(timestamps) if timestamps else None,
            }
        
        except Exception as e:
            logger.exception(f"Failed to get stats: {e}")
            return {
                "total_chunks": 0,
                "total_sessions": 0,
                "earliest_chunk": None,
                "latest_chunk": None,
            }
    
    async def _chunk_messages(
        self,
        messages: list[SessionMessage],
        session_id: str,
        instance_id: str,
    ) -> list[SessionChunk]:
        """Chunk messages into ~400 token windows with overlap.
        
        Uses a simple word-count approximation: words / 0.75 ≈ tokens.
        """
        chunks = []
        chunk_index = 0
        
        # Convert messages to text with timestamps
        message_texts = []
        for msg in messages:
            text = f"{msg.role}: {msg.content}"
            message_texts.append((text, msg.timestamp))
        
        # Estimate tokens
        target_words = int(self.TARGET_CHUNK_TOKENS * self.WORDS_PER_TOKEN)
        overlap_words = int(self.OVERLAP_TOKENS * self.WORDS_PER_TOKEN)
        
        i = 0
        while i < len(message_texts):
            chunk_messages = []
            chunk_word_count = 0
            start_time = message_texts[i][1]
            end_time = start_time
            
            # Collect messages until we reach target size
            while i < len(message_texts) and chunk_word_count < target_words:
                text, timestamp = message_texts[i]
                words = len(text.split())
                chunk_messages.append(text)
                chunk_word_count += words
                end_time = timestamp
                i += 1
            
            # Create chunk
            if chunk_messages:
                content = "\n".join(chunk_messages)
                embedding = await self.embedding_service.embed(content)
                
                chunk = SessionChunk(
                    chunk_id=str(uuid.uuid4()),
                    session_id=session_id,
                    instance_id=instance_id,
                    content=content,
                    embedding=embedding,
                    start_time=start_time,
                    end_time=end_time,
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Backtrack for overlap
            if i < len(message_texts):
                # Calculate how many messages to backtrack
                overlap_word_target = 0
                backtrack = 0
                while (backtrack < len(chunk_messages) and 
                       overlap_word_target < overlap_words):
                    backtrack += 1
                    overlap_word_target += len(chunk_messages[-backtrack].split())
                
                i -= min(backtrack, 2)  # Backtrack at most 2 messages
                i = max(i, 0)
        
        return chunks
    
    async def _ensure_ready(self) -> None:
        """Hook for subclasses to perform lazy initialization.

        Called at the start of ``ingest()`` and ``search()`` before any
        state is read.  The default implementation is a no-op.
        """

    async def _store_chunk(
        self, chunk: SessionChunk, total_messages: int = 0
    ) -> None:
        """Store a session chunk in memory.
        
        Args:
            chunk: The session chunk to store.
            total_messages: Total message count for the session (used by
                LanceDB store to persist delta ingestion state).
        """
        if not hasattr(self, '_chunks'):
            self._chunks = []
        
        self._chunks.append({
            "chunk_id": chunk.chunk_id,
            "session_id": chunk.session_id,
            "instance_id": chunk.instance_id,
            "content": chunk.content,
            "embedding": chunk.embedding,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "chunk_index": chunk.chunk_index,
        })
    
    async def _search_chunks(
        self,
        query_embedding: list[float],
        session_id: Optional[str],
        limit: int,
        min_relevance: float,
    ) -> list[dict]:
        """Search for chunks by similarity."""
        if not hasattr(self, '_chunks'):
            return []
        
        # Calculate similarities
        results = []
        for chunk in self._chunks:
            # Filter by session_id if provided
            if session_id and chunk["session_id"] != session_id:
                continue
            
            similarity = self.embedding_service.similarity(
                query_embedding,
                chunk["embedding"],
            )
            
            if similarity >= min_relevance:
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "session_id": chunk["session_id"],
                    "instance_id": chunk["instance_id"],
                    "content": chunk["content"],
                    "similarity_score": similarity,
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "chunk_index": chunk["chunk_index"],
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return results[:limit]
    
    async def _delete_chunks_before(self, cutoff_time: datetime) -> int:
        """Delete chunks older than cutoff time."""
        if not hasattr(self, '_chunks'):
            return 0
        
        initial_count = len(self._chunks)
        self._chunks = [
            chunk for chunk in self._chunks
            if chunk["end_time"] >= cutoff_time
        ]
        
        return initial_count - len(self._chunks)
    
    async def _get_all_chunks(self) -> list[dict]:
        """Get all stored chunks."""
        if not hasattr(self, '_chunks'):
            return []
        return self._chunks


class LanceDBSessionStore(SessionStore):
    """LanceDB-backed session store for persistent storage.
    
    Stores session chunks in a dedicated LanceDB table with full persistence
    across restarts. Inherits chunking and delta ingestion logic from SessionStore.
    """
    
    TABLE_NAME = "session_chunks"
    
    def __init__(
        self,
        instance_id: str,
        embedding_service: IEmbeddingService,
        vector_store: IVectorStore,
        db_path: Optional[Union[str, Path]] = None,
        db_uri: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(instance_id, embedding_service, vector_store)
        self.db_path = Path(db_path) if db_path else None
        self.db_uri = db_uri
        self.api_key = api_key or os.environ.get("LANCEDB_API_KEY")
        
        self._db = None
        self._table = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazily initialize database connection."""
        if self._initialized:
            return
        
        try:
            import lancedb
        except ImportError:
            raise ImportError("LanceDB not installed. Run: pip install lancedb")
        
        if self.db_uri:
            self._db = lancedb.connect(self.db_uri, api_key=self.api_key)
        elif self.db_path:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.db_path))
        else:
            raise ValueError("Either db_path or db_uri must be provided")
        
        # Open existing table or create new one.  We check table_names()
        # first to avoid overwriting an existing table with a potentially
        # different schema.  Schema migrations are not yet supported —
        # delete the table manually if the schema changes.
        existing_tables = (
            self._db.table_names()
            if hasattr(self._db, "table_names")
            else self._db.list_tables()
        )
        if self.TABLE_NAME in existing_tables:
            self._table = self._db.open_table(self.TABLE_NAME)
            # Restore delta ingestion state from persisted chunks
            self._restore_session_state()
        else:
            self._table = self._create_table()
        
        self._initialized = True
    
    def _create_table(self) -> "lancedb.table.Table":
        """Create the session_chunks table with the defined schema.
        
        Creates table with cosine distance metric for vector similarity search.
        This eliminates the need for manual L2-to-cosine conversion and ensures
        accurate similarity scoring with normalized embeddings.
        """
        import pyarrow as pa
        
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("instance_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), self._get_embedding_dim())),
            pa.field("chunk_index", pa.int32()),
            pa.field("created_at", pa.string()),
            pa.field("start_time", pa.string()),
            pa.field("end_time", pa.string()),
            pa.field("metadata", pa.string()),
        ])
        
        table = self._db.create_table(self.TABLE_NAME, schema=schema, mode="overwrite")
        
        # Create vector index with cosine metric for accurate similarity search
        # Note: Index creation may fail on empty tables; that's okay, it will be
        # created automatically on first search
        try:
            table.create_index(metric="cosine", num_partitions=256, num_sub_vectors=96)
            logger.info("Created vector index with cosine metric for session_chunks table")
        except Exception as e:
            logger.debug(f"Could not create index immediately (will be created on first search): {e}")
        
        return table
    
    def _get_embedding_dim(self) -> int:
        """Get the expected embedding dimension from the embedding service."""
        if hasattr(self.embedding_service, 'dimensions'):
            return self.embedding_service.dimensions
        if hasattr(self.embedding_service, 'embedding_dim'):
            return self.embedding_service.embedding_dim
        return 1536  # Default for text-embedding-3-small

    def _restore_session_state(self) -> None:
        """Restore delta ingestion state from persisted chunks.

        Reads ``message_count`` from chunk metadata to find the highest
        message index per session.  This prevents duplicate chunk creation
        when the store is restarted and ``ingest()`` is called again with
        the same (or superset) message list.
        """
        try:
            # head(n) returns the first N rows from the table. For very large
            # databases (>1M chunks), we may not restore complete state, which
            # only causes harmless re-chunking on next ingest.
            #
            # Future optimization: SELECT session_id, MAX(message_count)
            # FROM ... GROUP BY session_id
            # This would eliminate the need for Python-side max() logic.
            arrow_table = self._table.head(self._MAX_SCAN_LIMIT)
            
            # Convert to Python dict for easier iteration
            data = arrow_table.to_pydict()
            session_ids = data.get("session_id", [])
            metadatas = data.get("metadata", [])
            
            for sid, meta_str in zip(session_ids, metadatas):
                try:
                    meta = json.loads(meta_str if meta_str else "{}")
                    msg_count = meta.get("message_count", 0)
                except (json.JSONDecodeError, TypeError):
                    msg_count = 0
                # Keep the highest message_count per session
                if sid not in self._session_state or msg_count > self._session_state[sid]:
                    self._session_state[sid] = msg_count
            
            if self._session_state:
                logger.info(
                    "Restored session state for %d sessions from LanceDB",
                    len(self._session_state),
                )
        except Exception as e:
            logger.warning("Failed to restore session state: %s", e)

    async def _ensure_ready(self) -> None:
        """Initialize LanceDB connection and restore session state."""
        await self._ensure_initialized()

    async def _store_chunk(
        self, chunk: SessionChunk, total_messages: int = 0
    ) -> None:
        """Store a session chunk in LanceDB."""
        await self._ensure_initialized()
        
        # Store message_count in metadata so we can restore delta
        # ingestion state on restart (see _restore_session_state).
        meta = {
            "message_count": total_messages,
        }
        row = {
            "id": chunk.chunk_id,
            "session_id": chunk.session_id,
            "instance_id": chunk.instance_id,
            "content": chunk.content,
            "embedding": chunk.embedding,
            "chunk_index": chunk.chunk_index,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "start_time": chunk.start_time.isoformat(),
            "end_time": chunk.end_time.isoformat(),
            "metadata": json.dumps(meta),
        }
        
        self._table.add([row])
    
    async def _search_chunks(
        self,
        query_embedding: list[float],
        session_id: Optional[str],
        limit: int,
        min_relevance: float,
    ) -> list[dict]:
        """Search for chunks by similarity using LanceDB vector search."""
        await self._ensure_initialized()
        
        # Build query with optional session filter.
        # Use cosine metric for direct similarity scoring.
        query = (
            self._table.search(query_embedding)
            .metric("cosine")
            .limit(limit * 2)
        )
        
        if session_id:
            safe_session_id = self._sanitize_id(session_id)
            query = query.where(f"session_id = '{safe_session_id}'")
        
        results = query.to_list()
        
        # Convert to format expected by callers
        recall_results = []
        for row in results:
            # With cosine metric, _distance is cosine distance in [0, 2].
            # Convert to similarity: 1 - distance.
            distance = row.get("_distance", 0)
            similarity = max(0.0, 1.0 - distance)
            
            if similarity >= min_relevance:
                recall_results.append({
                    "chunk_id": row["id"],
                    "session_id": row["session_id"],
                    "instance_id": row["instance_id"],
                    "content": row["content"],
                    "similarity_score": similarity,
                    "start_time": datetime.fromisoformat(row["start_time"]),
                    "end_time": datetime.fromisoformat(row["end_time"]),
                    "chunk_index": row["chunk_index"],
                })
        
        # Sort by similarity
        recall_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return recall_results[:limit]
    
    async def _delete_chunks_before(self, cutoff_time: datetime) -> int:
        """Delete chunks older than cutoff time.

        Uses LanceDB's filter-based deletion to avoid loading all rows
        into memory.  Falls back to ID-based deletion if the filter
        approach fails (e.g., older LanceDB version).

        All stored timestamps are UTC ISO strings.  The ``cutoff_time``
        is converted to UTC if timezone-aware, or assumed UTC if naive.
        """
        await self._ensure_initialized()

        # Normalise cutoff to UTC ISO for consistent string comparison
        if cutoff_time.tzinfo is not None:
            cutoff_time = cutoff_time.astimezone(timezone.utc)
        cutoff_iso = cutoff_time.isoformat()

        try:
            # Count rows matching the filter first (for return value)
            matching = (
                self._table.search()
                .where(f"end_time < '{cutoff_iso}'")
                .select(["id"])
                .limit(1_000_000)
                .to_list()
            )
            count = len(matching)
            if count == 0:
                return 0

            # Batch delete via filter expression
            self._table.delete(f"end_time < '{cutoff_iso}'")
            return count
        except Exception:
            logger.warning(
                "Filter-based deletion failed, falling back to row-by-row"
            )

        # Fallback: fetch IDs and delete individually
        old_chunks = (
            self._table.search()
            .where(f"end_time < '{cutoff_iso}'")
            .select(["id"])
            .limit(1_000_000)
            .to_list()
        )

        for chunk in old_chunks:
            safe_id = self._sanitize_id(chunk["id"])
            self._table.delete(f"id = '{safe_id}'")

        return len(old_chunks)
    
    # Maximum rows to scan in full-table operations. Protects against memory
    # exhaustion on very large session stores. Tune higher for production
    # workloads with >1M chunks, considering available memory.
    _MAX_SCAN_LIMIT = 1_000_000

    async def _get_all_chunks(self) -> list[dict]:
        """Get all stored chunks.

        .. warning::
            Scans up to ``_MAX_SCAN_LIMIT`` (1 M) rows. Databases with more
            chunks will be silently truncated.  Use pagination for
            production workloads with very large session stores.
        """
        await self._ensure_initialized()
        
        results = (
            self._table.search()
            .limit(self._MAX_SCAN_LIMIT)
            .to_list()
        )

        if len(results) >= self._MAX_SCAN_LIMIT:
            logger.warning(
                "Session chunk scan hit limit (%d); results may be truncated",
                self._MAX_SCAN_LIMIT,
            )
        
        return [
            {
                "chunk_id": row["id"],
                "session_id": row["session_id"],
                "instance_id": row["instance_id"],
                "content": row["content"],
                "start_time": datetime.fromisoformat(row["start_time"]),
                "end_time": datetime.fromisoformat(row["end_time"]),
                "chunk_index": row["chunk_index"],
            }
            for row in results
        ]
    
    _ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-:.]+$')

    def _sanitize_id(self, id_str: str) -> str:
        """Sanitize ID to prevent SQL injection in LanceDB filter expressions.

        Allows alphanumeric characters, hyphens, underscores, colons, and
        dots — covering UUIDs, slug-style IDs, and common session ID
        formats.  Rejects spaces, quotes, semicolons, and other SQL-
        dangerous characters.  Single quotes are escaped as an additional
        defense-in-depth layer.
        """
        if not self._ID_PATTERN.match(id_str):
            raise ValueError(f"Invalid ID format: {id_str[:20]}...")
        return id_str.replace("'", "''")


class InMemorySessionStore(SessionStore):
    """In-memory session store for testing and fallback.
    
    This is the original SessionStore implementation, preserved for testing
    and as a fallback when LanceDB is not available.
    """
    pass  # Inherits all behavior from SessionStore
