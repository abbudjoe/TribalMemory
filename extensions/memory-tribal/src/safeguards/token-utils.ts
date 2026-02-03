/**
 * Shared token counting utility for safeguard modules.
 * 
 * Uses a word-based approximation (~0.75 tokens per word) which is
 * fast and doesn't require a tokenizer dependency. Accurate enough
 * for budget enforcement where exact counts aren't critical.
 */

/**
 * Approximate token count for a text string.
 * @param text - Input text to count tokens for
 * @returns Estimated token count (minimum 0)
 */
export function countTokens(text: string): number {
  if (!text) return 0;
  const words = text.split(/\s+/).filter(w => w.length > 0);
  return Math.ceil(words.length * 0.75);
}
