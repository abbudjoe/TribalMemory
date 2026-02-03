/**
 * Shared type definitions for memory-tribal plugin.
 *
 * Issue #16: Replace any[] with proper MemoryResult interface.
 */

/**
 * A memory search result from either the tribal server or
 * the builtin memory search fallback.
 */
export interface MemoryResult {
  id?: string;
  /** Path identifier, e.g. "tribal-memory:{uuid}" */
  path?: string;
  /** Similarity/relevance score (0.0–1.0) */
  score?: number;
  /** Memory content snippet (primary) */
  snippet?: string;
  /** Memory content (alternative to snippet) */
  text?: string;
  startLine?: number;
  endLine?: number;
  /** Source type (e.g. "user_explicit", "auto_capture") */
  source?: string;
  /** Tags for categorization */
  tags?: string[];
  /** ID of the memory this corrects/supersedes */
  supersedes?: string;
  /** The expanded query variant that produced this result */
  sourceQuery?: string;
}
