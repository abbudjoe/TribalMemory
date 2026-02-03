/**
 * Smart Triggers
 *
 * Phase 2 of Issue #11: Skip memory recall for low-value queries
 * like greetings, acknowledgments, and emoji-only messages.
 *
 * Expected to save 30-50% of unnecessary recall operations.
 */

/**
 * Check if a string contains ONLY emoji (and optional whitespace).
 * Uses surrogate pair detection plus common emoji code points.
 */
function isEmojiOnly(text: string): boolean {
  // Strip whitespace
  const stripped = text.replace(/\s/g, "");
  if (stripped.length === 0) return false;
  // Remove surrogate pairs (most emoji above U+FFFF) and common BMP emoji
  const withoutEmoji = stripped
    // Surrogate pairs cover U+10000+ (emoticons, symbols, flags, etc.)
    .replace(/[\uD800-\uDBFF][\uDC00-\uDFFF]/g, "")
    // BMP emoji and symbol ranges
    .replace(/[\u2600-\u27BF\u2B50\u2B55\uFE0F\u200D\u20E3\u2700-\u27BF\u231A\u231B\u23E9-\u23FA\u25AA-\u25FE\u2614\u2615\u2934\u2935\u3030\u303D\u3297\u3299\u2764\u2763\u270A-\u270D]/g, "");
  return withoutEmoji.length === 0;
}

/** Strip punctuation and extra whitespace for keyword matching. */
function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[!?.…,;:'"]+/g, "")
    .trim()
    .replace(/\s+/g, " ");
}

export interface SmartTriggerConfig {
  /** Words/phrases that should skip recall when they are the entire query. */
  skipKeywords: string[];
  /** Queries shorter than this (after normalization) are skipped. Default: 2. */
  minQueryLength: number;
  /** Skip queries that are only emoji. Default: true. */
  skipEmojiOnly: boolean;
}

/** Default skip keywords: greetings, acks, pleasantries, farewells. */
const DEFAULT_SKIP_KEYWORDS: string[] = [
  // Greetings
  "hi", "hello", "hey", "hi there", "hey there", "howdy", "yo",
  // Acknowledgments
  "ok", "okay", "sure", "yes", "no", "yep", "nope", "yeah", "nah",
  "k", "yea", "alright",
  // Thanks / pleasantries
  "thanks", "thank you", "thx", "ty", "cool", "nice", "great",
  "awesome", "perfect", "got it", "sounds good", "makes sense",
  "understood", "noted", "right", "fair enough", "i see",
  // Farewells
  "bye", "goodbye", "see you", "see ya", "later", "gn",
  "good night", "goodnight", "cya", "ttyl",
  // Filler
  "lol", "haha", "lmao", "heh", "hmm", "ah", "oh", "ooh",
  "wow", "whoa", "ugh", "meh",
];

export const DEFAULT_SMART_TRIGGER_CONFIG: SmartTriggerConfig = {
  skipKeywords: DEFAULT_SKIP_KEYWORDS,
  minQueryLength: 2,
  skipEmojiOnly: true,
};

export interface ClassifyResult {
  skip: boolean;
  reason: string | null;
}

interface TriggerStats {
  totalChecked: number;
  totalSkipped: number;
  totalPassed: number;
  skipRate: number;
}

export class SmartTrigger {
  private config: SmartTriggerConfig;

  /** Set of normalized skip keywords for O(1) lookup. */
  private skipSet: Set<string>;

  private stats = { checked: 0, skipped: 0 };

  constructor(config: Partial<SmartTriggerConfig> = {}) {
    this.config = { ...DEFAULT_SMART_TRIGGER_CONFIG, ...config };
    // Merge custom keywords with defaults (if user provides extras)
    if (config.skipKeywords && !config.skipKeywords.length) {
      // Empty array = no keywords
      this.skipSet = new Set();
    } else {
      const keywords = config.skipKeywords
        ? [...DEFAULT_SKIP_KEYWORDS, ...config.skipKeywords]
        : DEFAULT_SKIP_KEYWORDS;
      this.skipSet = new Set(keywords.map(k => normalize(k)));
    }
  }

  /**
   * Classify a query and return whether to skip + the reason.
   */
  classify(query: string): ClassifyResult {
    const raw = query.trim();

    // Empty or very short
    if (raw.length < this.config.minQueryLength) {
      return { skip: true, reason: "too short" };
    }

    // Emoji-only check (before normalization strips emoji)
    if (this.config.skipEmojiOnly && isEmojiOnly(raw)) {
      return { skip: true, reason: "emoji-only" };
    }

    // Normalize for keyword matching
    const normalized = normalize(raw);

    // Empty after normalization (was all punctuation)
    if (normalized.length === 0) {
      return { skip: true, reason: "too short (empty after normalization)" };
    }

    // Exact keyword match
    if (this.skipSet.has(normalized)) {
      return { skip: true, reason: `keyword match: "${normalized}"` };
    }

    return { skip: false, reason: null };
  }

  /**
   * Check if a query should be skipped. Tracks stats.
   * @returns true if the query should be skipped (no recall needed).
   */
  shouldSkip(query: string): boolean {
    const result = this.classify(query);
    this.stats.checked++;
    if (result.skip) {
      this.stats.skipped++;
    }
    return result.skip;
  }

  /**
   * Get skip/pass statistics.
   */
  getStats(): TriggerStats {
    const { checked, skipped } = this.stats;
    return {
      totalChecked: checked,
      totalSkipped: skipped,
      totalPassed: checked - skipped,
      skipRate: checked > 0 ? skipped / checked : 0,
    };
  }

  /**
   * Reset statistics counters.
   */
  resetStats(): void {
    this.stats = { checked: 0, skipped: 0 };
  }

  /**
   * Get current config (for inspection/testing).
   */
  getConfig(): SmartTriggerConfig {
    return {
      ...this.config,
      skipKeywords: [...this.config.skipKeywords],
    };
  }
}
