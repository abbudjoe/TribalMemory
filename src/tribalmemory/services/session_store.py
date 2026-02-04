"""Session transcript indexing service.

Indexes conversation transcripts as chunked embeddings for contextual recall.
Supports delta-based ingestion and retention-based cleanup.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

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
    timestamp: datetime = field(default_factory=datetime.utcnow)


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
            SessionMessage("user", "What is Docker?", datetime.utcnow()),
            SessionMessage("assistant", "Docker is a container platform", datetime.utcnow()),
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
            
            # Store chunks in vector store
            for chunk in chunks:
                await self._store_chunk(chunk)
            
            # Update state
            self._session_state[session_id] = len(messages)
            
            return {
                "success": True,
                "chunks_created": len(chunks),
                "messages_processed": len(new_messages),
            }
        
        except Exception as e:
            logger.error(f"Failed to ingest session {session_id}: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    async def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        max_results: int = 5,
        min_relevance: float = 0.0,
    ) -> list[dict]:
        """Search session transcripts by semantic similarity.
        
        Args:
            query: Natural language search query
            session_id: Optional filter to specific session
            max_results: Maximum number of results to return
            min_relevance: Minimum similarity score (0.0 to 1.0)
        
        Returns:
            List of dicts with keys: chunk_id, session_id, instance_id,
            content, similarity_score, start_time, end_time, chunk_index
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed(query)
            
            # Search chunks
            results = await self._search_chunks(
                query_embedding,
                session_id,
                max_results,
                min_relevance,
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Failed to search sessions: {e}")
            return []
    
    async def cleanup(self, retention_days: int = 30) -> int:
        """Delete session chunks older than retention period.
        
        Args:
            retention_days: Number of days to retain chunks
        
        Returns:
            Number of chunks deleted
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
            
            # Find and delete expired chunks
            deleted = await self._delete_chunks_before(cutoff_time)
            
            return deleted
        
        except Exception as e:
            logger.error(f"Failed to cleanup sessions: {e}")
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
            logger.error(f"Failed to get stats: {e}")
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
    
    async def _store_chunk(self, chunk: SessionChunk) -> None:
        """Store a session chunk (in-memory for now)."""
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
        max_results: int,
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
        
        return results[:max_results]
    
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
