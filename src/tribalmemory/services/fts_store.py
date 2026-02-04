"""SQLite FTS5 full-text search store for BM25 hybrid search.

Provides keyword-based BM25 search alongside LanceDB vector search.
FTS5 excels at exact-token queries (error strings, config names, IDs)
while vector search handles semantic/fuzzy queries.

The two are combined via hybrid scoring:
    finalScore = vectorWeight * vectorScore + textWeight * bm25Score
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FTSStore:
    """SQLite FTS5 store for keyword search over memories.

    Creates a FTS5 virtual table alongside the main vector store.
    Supports index, search, delete, and update operations.

    Note: All methods are synchronous. SQLite operations are typically
    sub-millisecond for the document counts we handle (<100k). If latency
    becomes an issue on slow storage, wrap calls in asyncio.to_thread().
    """

    def __init__(self, db_path: str):
        """Initialize FTS store.

        Args:
            db_path: Path to the SQLite database file. Created if missing.
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._fts_available: Optional[bool] = None
        self._ensure_initialized()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_initialized(self) -> None:
        """Create FTS5 virtual table if it doesn't exist."""
        conn = self._get_conn()
        if not self.is_available():
            logger.warning("FTS5 not available in this SQLite build")
            return
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(id, content, tags, tokenize='porter')
        """)
        # Mapping table to track which IDs are indexed (for upsert/delete)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fts_ids (
                id TEXT PRIMARY KEY
            )
        """)
        conn.commit()

    def is_available(self) -> bool:
        """Check if FTS5 is available in the current SQLite build."""
        if self._fts_available is not None:
            return self._fts_available
        try:
            conn = self._get_conn()
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_test "
                "USING fts5(test_col)"
            )
            conn.execute("DROP TABLE IF EXISTS _fts5_test")
            conn.commit()
            self._fts_available = True
        except sqlite3.OperationalError:
            self._fts_available = False
        return self._fts_available

    def index(self, memory_id: str, content: str, tags: list[str]) -> None:
        """Index a memory for full-text search.

        If the memory_id already exists, it is replaced (upsert).
        """
        if not self.is_available():
            return
        conn = self._get_conn()
        tags_text = " ".join(tags)

        # Check if exists — delete first for upsert
        existing = conn.execute(
            "SELECT id FROM fts_ids WHERE id = ?", (memory_id,)
        ).fetchone()
        if existing:
            conn.execute(
                "DELETE FROM memories_fts WHERE id = ?", (memory_id,)
            )

        conn.execute(
            "INSERT INTO memories_fts (id, content, tags) VALUES (?, ?, ?)",
            (memory_id, content, tags_text),
        )
        conn.execute(
            "INSERT OR REPLACE INTO fts_ids (id) VALUES (?)",
            (memory_id,),
        )
        conn.commit()

    def search(
        self, query: str, limit: int = 10
    ) -> list[dict]:
        """Search memories using BM25.

        Returns list of dicts with 'id' and 'rank' keys.
        BM25 rank is negative; more negative = better match.
        """
        if not self.is_available():
            return []
        conn = self._get_conn()
        # Use bm25() for ranking. FTS5 bm25() returns negative values
        # where more negative = better match.
        try:
            rows = conn.execute(
                """
                SELECT id, rank
                FROM memories_fts
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
            return [{"id": row["id"], "rank": row["rank"]} for row in rows]
        except sqlite3.OperationalError as e:
            # Malformed FTS query (unbalanced quotes, etc.)
            logger.warning(f"FTS5 search error: {e}")
            return []

    def delete(self, memory_id: str) -> None:
        """Remove a memory from the FTS index."""
        if not self.is_available():
            return
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM memories_fts WHERE id = ?", (memory_id,)
        )
        conn.execute("DELETE FROM fts_ids WHERE id = ?", (memory_id,))
        conn.commit()

    def count(self) -> int:
        """Return number of indexed documents."""
        if not self.is_available():
            return 0
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM fts_ids").fetchone()
        return row[0]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


def bm25_rank_to_score(rank: float) -> float:
    """Convert BM25 rank to a 0..1 score.

    FTS5 bm25() returns negative values where more negative = better.
    We use: score = 1 / (1 + abs(rank))
    """
    return 1.0 / (1.0 + abs(rank))


def hybrid_merge(
    vector_results: list[dict],
    bm25_results: list[dict],
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
) -> list[dict]:
    """Merge vector similarity and BM25 results with weighted scoring.

    BM25 ranks are min-max normalized to 0..1 so they're comparable
    with vector similarity scores (also 0..1). The best BM25 hit gets
    score 1.0, the worst gets a proportional score.

    Args:
        vector_results: List of {"id": str, "score": float} (0..1 cosine sim)
        bm25_results: List of {"id": str, "rank": float} (negative BM25 rank)
        vector_weight: Weight for vector similarity score
        text_weight: Weight for BM25 text score

    Returns:
        Merged list sorted by final_score descending.
        Each dict has: id, vector_score, text_score, final_score.
    """
    # Normalize weights
    total = vector_weight + text_weight
    if total > 0:
        vector_weight /= total
        text_weight /= total

    # Min-max normalize BM25 ranks to 0..1
    # BM25 ranks are negative; more negative = better match.
    # When empty, skip normalization entirely — no BM25 contribution.
    bm25_normalized: dict[str, float] = {}
    if bm25_results:
        abs_ranks = [abs(br["rank"]) for br in bm25_results]
        max_rank = max(abs_ranks)
        min_rank = min(abs_ranks)
        rank_range = max_rank - min_rank

        for br in bm25_results:
            if rank_range > 0:
                # Normalize: best rank (highest abs) → 1.0, worst → ~0
                score = (abs(br["rank"]) - min_rank) / rank_range
            else:
                # All same rank → all get 1.0
                score = 1.0
            bm25_normalized[br["id"]] = score

    # Build candidate map
    candidates: dict[str, dict] = {}

    for vr in vector_results:
        mid = vr["id"]
        candidates[mid] = {
            "id": mid,
            "vector_score": vr["score"],
            "text_score": 0.0,
        }

    for mid, text_score in bm25_normalized.items():
        if mid in candidates:
            candidates[mid]["text_score"] = text_score
        else:
            candidates[mid] = {
                "id": mid,
                "vector_score": 0.0,
                "text_score": text_score,
            }

    # Compute final scores
    for c in candidates.values():
        c["final_score"] = (
            vector_weight * c["vector_score"]
            + text_weight * c["text_score"]
        )

    # Sort by final score descending
    return sorted(
        candidates.values(), key=lambda x: x["final_score"], reverse=True
    )
