"""Shared semantic expansion utilities for mock implementations.

Provides centralized semantic term dictionaries and helpers for tests.
"""

# Short words that are meaningful in technical contexts
SHORT_IMPORTANT_WORDS = {'pr', 'rm', 'ui', 'ux', 'ai', 'ml', 'db', 'js', 'ts'}

# Semantic expansion dictionaries by domain
TECH_TERMS = {
    'next', 'nextjs', 'react', 'tailwind', 'supabase', 'claude', 'api',
    'typescript', 'javascript', 'python', 'database', 'postgresql', 'backend',
    'frontend', 'framework', 'styling', 'css', 'app', 'router'
}

WORKFLOW_TERMS = {'pr', 'prs', 'review', 'commit', 'branch', 'merge', 'git'}

TESTING_TERMS = {'tdd', 'tests', 'testing', 'test', 'unit', 'coverage'}

FILE_TERMS = {'delete', 'deletion', 'trash', 'rm', 'remove', 'file', 'files'}

TIMESTAMP_TERMS = {'rfc', '3161', 'timestamp', 'provenance', 'blockchain'}

TIMEZONE_TERMS = {
    'timezone', 'eastern', 'mountain', 'pacific',
    'central', 'utc', 'summer', 'winter'
}


def get_word_variants(word: str) -> set[str]:
    """Get common variants of a word (pseudo-stemming).
    
    Args:
        word: Base word to expand.
        
    Returns:
        Set of word variants including original.
    """
    variants = {word}
    # Remove common suffixes
    for suffix in ['ing', 'tion', 'ation', 'ed', 'er', 'ly', 's', 'es']:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            root = word[:-len(suffix)]
            variants.add(root)
            # Also add other forms of the root
            variants.add(root + 's')
            variants.add(root + 'ing')
    # Add common suffixes to the word
    variants.add(word + 's')
    variants.add(word + 'ing')
    variants.add(word + 'tion')
    return variants


def get_expanded_terms(query_words: set[str], query_lower: str) -> set[str]:
    """Expand query words with semantic related terms.
    
    Args:
        query_words: Initial set of query words.
        query_lower: Lowercase query string for substring checks.
        
    Returns:
        Expanded set of query words.
    """
    expanded = set(query_words)
    
    # Tech/stack related
    if 'tech' in query_words or 'stack' in query_words or 'technology' in query_lower:
        expanded.update(TECH_TERMS)
    
    # Workflow/process related
    if 'workflow' in query_words or 'process' in query_words:
        expanded.update(WORKFLOW_TERMS)
    
    # Testing related
    if 'testing' in query_words or 'test' in query_words:
        expanded.update(TESTING_TERMS)
    
    # File operations
    if 'delete' in query_words or 'files' in query_words or 'file' in query_words:
        expanded.update(FILE_TERMS)
    
    # Timestamp/provenance
    if 'timestamp' in query_words or 'provenance' in query_words:
        expanded.update(TIMESTAMP_TERMS)
    
    # Timezone related
    if 'timezone' in query_words or 'time' in query_words:
        expanded.update(TIMEZONE_TERMS)
    
    return expanded
