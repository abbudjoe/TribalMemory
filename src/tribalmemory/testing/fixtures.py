"""Test data fixtures and loaders."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PreferenceTestCase:
    """A preference-based test case."""
    id: str
    stored_preference: str
    query: str
    expected_keywords: list[str]  # Response should contain these
    context: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    negative: bool = False  # If True, this is a negative test case


@dataclass
class ContextTaskTestCase:
    """A context-dependent task test case."""
    id: str
    memories: list[str]  # Memories to store first
    query: str
    expected_keywords: list[str]
    min_memories_referenced: int = 1


@dataclass
class ConsistencyTestCase:
    """A cross-session consistency test case."""
    id: str
    seed_memories: list[str]
    query_variations: list[str]  # Different ways to ask same thing
    expected_consistent: bool = True


@dataclass
class TestDataSet:
    """Complete test data set."""
    preferences: list[PreferenceTestCase]
    context_tasks: list[ContextTaskTestCase]
    consistency: list[ConsistencyTestCase]


def load_test_data(data_dir: Path) -> TestDataSet:
    """Load test data from JSON files."""
    preferences = []
    context_tasks = []
    consistency = []
    
    # Load preferences
    pref_file = data_dir / "preferences.json"
    if pref_file.exists():
        with open(pref_file) as f:
            data = json.load(f)
            preferences = [PreferenceTestCase(**p) for p in data]
    
    # Load context tasks
    ctx_file = data_dir / "context_tasks.json"
    if ctx_file.exists():
        with open(ctx_file) as f:
            data = json.load(f)
            context_tasks = [ContextTaskTestCase(**c) for c in data]
    
    # Load consistency
    cons_file = data_dir / "consistency.json"
    if cons_file.exists():
        with open(cons_file) as f:
            data = json.load(f)
            consistency = [ConsistencyTestCase(**c) for c in data]
    
    return TestDataSet(
        preferences=preferences,
        context_tasks=context_tasks,
        consistency=consistency
    )


def save_test_data(dataset: TestDataSet, data_dir: Path):
    """Save test data to JSON files."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save preferences
    with open(data_dir / "preferences.json", "w") as f:
        json.dump([
            {
                "id": p.id,
                "stored_preference": p.stored_preference,
                "query": p.query,
                "expected_keywords": p.expected_keywords,
                "context": p.context,
                "tags": p.tags,
            }
            for p in dataset.preferences
        ], f, indent=2)
    
    # Save context tasks
    with open(data_dir / "context_tasks.json", "w") as f:
        json.dump([
            {
                "id": c.id,
                "memories": c.memories,
                "query": c.query,
                "expected_keywords": c.expected_keywords,
                "min_memories_referenced": c.min_memories_referenced,
            }
            for c in dataset.context_tasks
        ], f, indent=2)
    
    # Save consistency
    with open(data_dir / "consistency.json", "w") as f:
        json.dump([
            {
                "id": c.id,
                "seed_memories": c.seed_memories,
                "query_variations": c.query_variations,
                "expected_consistent": c.expected_consistent,
            }
            for c in dataset.consistency
        ], f, indent=2)
