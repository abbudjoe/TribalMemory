/**
 * Snippet Truncation
 * 
 * Truncates individual memory snippets to prevent overly long results.
 * Applied BEFORE token budget accounting.
 */

import { countTokens } from "./token-utils";

export interface TruncationConfig {
  maxTokensPerSnippet: number;
}

export const DEFAULT_TRUNCATION_CONFIG: TruncationConfig = {
  maxTokensPerSnippet: 100,
};

export class SnippetTruncator {
  private config: TruncationConfig;

  constructor(config: Partial<TruncationConfig> = {}) {
    this.config = { ...DEFAULT_TRUNCATION_CONFIG, ...config };
  }

  /**
   * Count tokens in text (approximate).
   * Delegates to shared token-utils for consistency across safeguard modules.
   */
  private countTokens(text: string): number {
    return countTokens(text);
  }

  /**
   * Truncate a snippet to max token limit.
   * Adds "..." suffix when truncated.
   */
  truncate(text: string): string {
    if (!text) return text;

    const tokens = this.countTokens(text);
    
    if (tokens <= this.config.maxTokensPerSnippet) {
      return text;
    }

    // Truncate: split into words, take first N words to approximate token limit
    const words = text.split(/\s+/);
    const targetWords = Math.floor(this.config.maxTokensPerSnippet / 0.75);
    const truncated = words.slice(0, targetWords).join(" ");
    
    return truncated + "...";
  }

  /**
   * Truncate all snippets in a result set.
   * Modifies results in place and returns them.
   */
  truncateResults(results: any[]): any[] {
    if (!results) return results;

    for (const result of results) {
      // Truncate snippet field if present
      if (result.snippet) {
        result.snippet = this.truncate(result.snippet);
      }
      
      // Truncate text field if present (fallback)
      if (result.text) {
        result.text = this.truncate(result.text);
      }
    }

    return results;
  }

  /**
   * Get config (for inspection/testing).
   */
  getConfig(): TruncationConfig {
    return { ...this.config };
  }
}
