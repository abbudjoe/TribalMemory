"""Synthetic corpus generator for scale/performance testing.

Generates realistic memory entries with varied content, tags,
and source types for benchmarking retrieval and storage.
"""

import random
from dataclasses import dataclass
from typing import Optional

from ..interfaces import MemoryEntry, MemorySource


@dataclass
class CorpusConfig:
    """Configuration for corpus generation."""
    size: int = 1000
    seed: Optional[int] = None
    min_content_words: int = 5
    max_content_words: int = 30


# Realistic memory content templates
_TEMPLATES = [
    "User prefers {preference} for {domain}",
    "Meeting with {person} scheduled for {time}",
    "{person} mentioned they like {preference}",
    "Project {project} uses {technology} for {purpose}",
    "Important: {fact} about {topic}",
    "User's {attribute} is {value}",
    "{person} works at {company} on {project}",
    "Reminder: {task} is due {time}",
    "The {tool} configuration uses {setting}",
    "Conversation about {topic} with {person}",
    "{person} prefers {preference} over {alternative}",
    "Bug in {project}: {description}",
    "Decision: use {technology} for {purpose}",
    "User asked about {topic} in the context of {domain}",
    "Note: {fact} regarding {topic}",
]

_PERSONS = [
    "Joe", "Alice", "Bob", "Charlie", "Diana", "Eve",
    "Frank", "Grace", "Hank", "Iris", "Jake", "Karen",
]
_PREFERENCES = [
    "dark mode", "TypeScript", "Python", "concise responses",
    "morning meetings", "async communication", "vim", "VS Code",
    "functional programming", "microservices", "monorepos",
    "test-driven development", "pair programming", "remote work",
]
_DOMAINS = [
    "web development", "machine learning", "DevOps", "UI design",
    "backend services", "data engineering", "mobile apps",
    "cloud infrastructure", "security", "performance optimization",
]
_PROJECTS = [
    "Wally", "TribalMemory", "OpenClaw", "Dashboard",
    "API Gateway", "Auth Service", "Analytics", "Notifications",
]
_TECHNOLOGIES = [
    "React", "FastAPI", "PostgreSQL", "Redis", "Docker",
    "Kubernetes", "LanceDB", "OpenAI", "Tailscale", "Synapse",
]
_TOPICS = [
    "embedding models", "vector search", "memory portability",
    "performance tuning", "caching strategies", "deduplication",
    "schema migrations", "API versioning", "error handling",
    "security best practices", "testing strategies", "CI/CD",
]
_TAGS_POOL = [
    "preferences", "meetings", "projects", "technical",
    "personal", "work", "urgent", "low-priority",
    "architecture", "bugs", "decisions", "reminders",
]
_SOURCES = [
    MemorySource.USER_EXPLICIT,
    MemorySource.AUTO_CAPTURE,
    MemorySource.CROSS_INSTANCE,
]


def generate_corpus(config: Optional[CorpusConfig] = None) -> list[MemoryEntry]:
    """Generate a synthetic corpus of memory entries.

    Args:
        config: Corpus generation configuration. Uses defaults if None.

    Returns:
        List of MemoryEntry objects with varied content and metadata.
    """
    config = config or CorpusConfig()
    rng = random.Random(config.seed)

    entries: list[MemoryEntry] = []
    for _ in range(config.size):
        template = rng.choice(_TEMPLATES)
        content = _fill_template(template, rng)

        tags = rng.sample(_TAGS_POOL, k=rng.randint(1, 3))
        source = rng.choice(_SOURCES)

        entry = MemoryEntry(
            content=content,
            tags=tags,
            source_type=source,
            source_instance=f"instance-{rng.randint(1, 5)}",
        )
        entries.append(entry)

    return entries


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill a template with random realistic values."""
    replacements = {
        "{preference}": rng.choice(_PREFERENCES),
        "{domain}": rng.choice(_DOMAINS),
        "{person}": rng.choice(_PERSONS),
        "{time}": rng.choice([
            "next Monday", "tomorrow", "Friday afternoon",
            "end of sprint", "Q2", "next week",
        ]),
        "{project}": rng.choice(_PROJECTS),
        "{technology}": rng.choice(_TECHNOLOGIES),
        "{purpose}": rng.choice([
            "the backend", "testing", "deployment", "monitoring",
            "data storage", "real-time updates", "authentication",
        ]),
        "{topic}": rng.choice(_TOPICS),
        "{attribute}": rng.choice([
            "timezone", "favorite language", "team", "role",
            "preferred editor", "working hours",
        ]),
        "{value}": rng.choice([
            "Mountain Time", "Python", "engineering", "senior dev",
            "VS Code", "9am-5pm", "night owl hours",
        ]),
        "{company}": rng.choice([
            "Google", "a startup", "Anthropic", "OpenAI",
            "Meta", "a consulting firm",
        ]),
        "{tool}": rng.choice([
            "Docker", "Kubernetes", "Nginx", "Redis",
            "PostgreSQL", "LanceDB",
        ]),
        "{setting}": rng.choice([
            "port 8080", "max_connections=100", "debug=false",
            "cache_ttl=3600", "workers=4",
        ]),
        "{fact}": rng.choice([
            "embeddings need normalization", "cache invalidation is hard",
            "deadline was moved", "requirements changed",
            "API rate limit is 100/min", "tests must pass before merge",
        ]),
        "{task}": rng.choice([
            "code review", "deploy to staging", "update docs",
            "run benchmarks", "fix flaky test", "merge PR",
        ]),
        "{alternative}": rng.choice(_PREFERENCES),
        "{description}": rng.choice([
            "query timeout under load", "missing error handler",
            "incorrect cache key", "race condition in startup",
            "memory leak in long sessions",
        ]),
    }

    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result
