/**
 * QueryExpander: Transform question-style queries into keyword-style for better embedding match
 */

import type { PersistenceLayer } from "../persistence";

interface ExpansionRule {
  pattern: RegExp;
  transform: (match: RegExpMatchArray, query: string) => string[];
}

export class QueryExpander {
  private rules: ExpansionRule[] = [];
  private learnedExpansions: Map<string, string[]> = new Map();
  private persistence: PersistenceLayer | null;

  constructor(persistence: PersistenceLayer | null = null) {
    this.persistence = persistence;
    this.initializeRules();
  }

  /**
   * Bootstrap synonym mappings for domain-specific query expansion.
   * 
   * These are initial seed mappings that help the system achieve reasonable
   * recall accuracy out of the box. Over time, the feedback loop will:
   * 1. Learn which mappings are actually useful (via usage tracking)
   * 2. Discover new mappings from successful retrievals
   * 3. Deprecate mappings that don't help
   * 
   * Some mappings may appear counterintuitive (e.g., "parents alive" → "father passed")
   * because they're designed to find relevant context, not literal matches.
   * The query "Are both parents alive?" needs to find records about parent deaths.
   * 
   * @see FeedbackTracker for learned refinements
   * @see PersistenceLayer.recordExpansion for persistence
   */
  private synonyms: Map<string, string[]> = new Map([
    // Medical
    ["medical care", ["doctor", "physician", "clinic", "hospital", "health"]],
    ["medical", ["doctor", "health", "clinic"]],
    ["healthcare", ["doctor", "medical", "clinic"]],
    // Relationships
    ["life partner", ["spouse", "husband", "wife", "partner", "married"]],
    ["partner", ["spouse", "husband", "wife"]],
    ["significant other", ["spouse", "partner", "husband", "wife"]],
    ["parents", ["mother", "father", "mom", "dad", "passed away", "alive"]],
    ["parents alive", ["father passed", "mother", "dad", "mom"]],
    // Food/Dining
    ["special occasion", ["favorite restaurant", "celebration", "birthday"]],
    ["reservations", ["favorite restaurant", "restaurant"]],
    ["restaurant recommendation", ["favorite restaurant", "best restaurant"]],
    ["dining", ["restaurant", "food", "eating out"]],
    ["pizza topping", ["least favorite food", "hated food", "cannot stand"]],
    ["never on pizza", ["least favorite", "hates", "olives"]],
    // Animals
    ["animals", ["pet", "cat", "dog", "pets"]],
    ["household animals", ["pet", "cat", "dog"]],
    // Work
    ["productive", ["focus", "work hours", "deep work", "concentration"]],
    ["productivity", ["focus hours", "best time", "work schedule"]],
    ["deep work", ["focus music", "concentration", "focus hours"]],
    ["focus during", ["focus music", "concentration music", "lo-fi"]],
    ["how long at", ["started at", "joined", "tenure", "since"]],
    ["tenure", ["started", "joined", "working since"]],
    // Language/Tech decisions
    ["language decision", ["adopted", "chose", "switched to", "performance"]],
    ["drove the decision", ["reason", "why", "because", "due to"]],
    ["why chose", ["adopted", "decision", "reason", "performance"]],
    // Hobbies
    ["expensive hobbies", ["collects", "collection", "hobby"]],
    ["hobbies", ["collects", "collection", "interests", "free time"]],
    // Editor/Tools
    ["code editor", ["IDE", "editor", "neovim", "vim", "vscode"]],
    ["text editor", ["IDE", "editor", "neovim"]],
    // Food restrictions
    ["food restrictions", ["allergies", "allergic", "dietary", "cannot eat"]],
    ["dietary restrictions", ["allergies", "allergic", "food restrictions"]],
  ]);

  private initializeRules(): void {
    // "What is my X?" → ["X", "my X", "X preference"]
    this.rules.push({
      pattern: /^what(?:'s| is) (?:my |the )?(.+?)\??$/i,
      transform: (match) => {
        const topic = match[1].trim();
        return [
          topic,
          `my ${topic}`,
          `${topic} preference`,
          `favorite ${topic}`,
        ];
      },
    });

    // "What X do I Y?" → ["X preference", "X I Y"]
    this.rules.push({
      pattern: /^what (.+?) do I (.+?)\??$/i,
      transform: (match) => {
        const thing = match[1].trim();
        const action = match[2].trim();
        return [
          `${thing} ${action}`,
          `${thing} preference`,
          `my ${thing}`,
          thing,
        ];
      },
    });

    // "Who is my X?" → ["my X", "X name", "X"]
    this.rules.push({
      pattern: /^who(?:'s| is) (?:my )?(.+?)\??$/i,
      transform: (match) => {
        const role = match[1].trim();
        return [
          `my ${role}`,
          `${role} name`,
          role,
        ];
      },
    });

    // "When is/do X?" → ["X date", "X time", "X schedule"]
    this.rules.push({
      pattern: /^when (?:is|do|does|did) (?:my |the )?(.+?)\??$/i,
      transform: (match) => {
        const event = match[1].trim();
        return [
          `${event} date`,
          `${event} time`,
          `${event} schedule`,
          event,
        ];
      },
    });

    // "Where is/do X?" → ["X location", "X address", "X"]
    this.rules.push({
      pattern: /^where (?:is|do|does|did) (?:my |the |I )?(.+?)\??$/i,
      transform: (match) => {
        const thing = match[1].trim();
        return [
          `${thing} location`,
          `${thing} address`,
          `${thing} place`,
          thing,
        ];
      },
    });

    // "How do I X?" → ["X instructions", "X method", "X"]
    this.rules.push({
      pattern: /^how (?:do|does|did|can) I (.+?)\??$/i,
      transform: (match) => {
        const action = match[1].trim();
        return [
          `${action} instructions`,
          `${action} method`,
          `how to ${action}`,
          action,
        ];
      },
    });

    // Imperative: "X the Y" → ["Y", "my Y", "Y preference"]
    this.rules.push({
      // Imperative: "get/find/show X" → ["X", "my X"]
      pattern:
        /^(?:get|find|show|tell|give|remind|update|set|add|send|book|order) (?:me )?(?:my |the )?(.+?)$/i,
      transform: (match) => {
        const topic = match[1].trim();
        return [
          topic,
          `my ${topic}`,
          `${topic} details`,
        ];
      },
    });
  }

  /**
   * Expand a query into multiple variants for broader search
   */
  expand(query: string): string[] {
    const variants = new Set<string>();
    variants.add(query); // Always include original

    // Apply rule-based expansions
    for (const rule of this.rules) {
      const match = query.match(rule.pattern);
      if (match) {
        const expansions = rule.transform(match, query);
        for (const exp of expansions) {
          if (exp && exp.length > 1) {
            variants.add(exp);
          }
        }
        break; // Only apply first matching rule
      }
    }

    // Apply learned expansions (in-memory)
    const normalized = query.toLowerCase().trim();
    const learned = this.learnedExpansions.get(normalized);
    if (learned) {
      for (const exp of learned) {
        variants.add(exp);
      }
    }

    // Apply learned expansions (from persistence)
    if (this.persistence) {
      const persistedExpansions = this.persistence.getLearnedExpansions(query);
      for (const exp of persistedExpansions) {
        variants.add(exp);
      }
    }

    // Extract key nouns/phrases as fallback
    const words = query.toLowerCase()
      .replace(/[^\w\s]/g, "")
      .split(/\s+/)
      .filter(w => w.length > 3 && !this.isStopWord(w));
    
    if (words.length > 0) {
      variants.add(words.join(" "));
    }

    // Apply semantic synonym expansion
    const queryLower = query.toLowerCase();
    for (const [phrase, synonyms] of this.synonyms) {
      if (queryLower.includes(phrase)) {
        for (const syn of synonyms) {
          variants.add(syn);
          // Also try replacing the phrase with synonym in original query
          variants.add(queryLower.replace(phrase, syn));
        }
      }
    }

    // Limit to 8 variants (increased from 5) to accommodate synonym expansion.
    // More variants = broader search but higher latency. 8 is a balanced default.
    // Each variant generates an embedding call, so keep this bounded.
    return [...variants].slice(0, 8);
  }

  private isStopWord(word: string): boolean {
    const stopWords = new Set([
      "what", "when", "where", "who", "how", "which", "that", "this",
      "have", "does", "would", "could", "should", "will", "been",
      "being", "about", "with", "from", "into", "your", "their",
    ]);
    return stopWords.has(word);
  }

  /**
   * Learn a new expansion from successful retrieval
   */
  learnExpansion(query: string, successfulVariant: string): void {
    const normalized = query.toLowerCase().trim();
    const existing = this.learnedExpansions.get(normalized) ?? [];
    if (!existing.includes(successfulVariant)) {
      existing.push(successfulVariant);
      this.learnedExpansions.set(normalized, existing.slice(-5)); // Keep last 5
    }

    // Persist learned expansion
    if (this.persistence) {
      this.persistence.recordExpansion(query, successfulVariant);
    }
  }

  /**
   * Get rule count for stats
   */
  ruleCount(): number {
    return this.rules.length + this.learnedExpansions.size;
  }
}
