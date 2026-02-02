"""Memory recall evaluation harness.

Provides deterministic test generation, response parsing, and scoring utilities
for evaluating memory recall systems.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional
import random


NEGATIVE_KEYWORDS = [
    "not recorded",
    "not in memory",
    "no record",
    "unknown",
    "not sure",
]

LEVEL_CONFIGS = {
    "L1": {"question_count": 6, "include_synthesis": False, "include_adversarial": False},
    "L2": {"question_count": 8, "include_synthesis": False, "include_adversarial": False},
    "L3": {"question_count": 10, "include_synthesis": True, "include_adversarial": False},
    "L4": {"question_count": 12, "include_synthesis": True, "include_adversarial": True},
    "L5": {"question_count": 14, "include_synthesis": True, "include_adversarial": True},
    "L6": {"question_count": 16, "include_synthesis": True, "include_adversarial": True},
    "L7": {"question_count": 18, "include_synthesis": True, "include_adversarial": True},
}


DEFAULT_FACTS = {
    "personal": [
        {
            "question": "What is the favorite cuisine?",
            "answer": "Thai food, especially green curry",
            "required": ["thai", "green curry"],
            "bonus": ["spicy", "pad thai"],
            "category": "preferences",
        },
        {
            "question": "Who is the spouse?",
            "answer": "Morgan",
            "required": ["morgan"],
            "bonus": [],
            "category": "relationships",
        },
        {
            "question": "What is the pet's name?",
            "answer": "Whiskers",
            "required": ["whiskers"],
            "bonus": [],
            "category": "pets",
        },
    ],
    "engineering": [
        {
            "question": "When was v2 deployed?",
            "answer": "2025-01-15",
            "required": ["2025", "01", "15"],
            "bonus": [],
            "category": "release",
        },
        {
            "question": "What database do we use?",
            "answer": "Postgres",
            "required": ["postgres"],
            "bonus": ["postgresql"],
            "category": "infrastructure",
        },
    ],
    "codebase": [
        {
            "question": "What language is used for the web app?",
            "answer": "TypeScript",
            "required": ["typescript"],
            "bonus": ["ts"],
            "category": "tech",
        },
        {
            "question": "What framework is used for the API?",
            "answer": "FastAPI",
            "required": ["fastapi"],
            "bonus": [],
            "category": "tech",
        },
    ],
}


@dataclass
class Question:
    id: str
    type: str
    question: str
    expected: str
    required: list[str] = field(default_factory=list)
    bonus: list[str] = field(default_factory=list)
    category: Optional[str] = None
    components: Optional[list[dict[str, Any]]] = None
    confusion_target: Optional[str] = None
    adv_type: Optional[str] = None


@dataclass
class Test:
    level: str
    seed: str
    corpus: str
    questions: list[Question]
    question_count: int = 0

    def __post_init__(self):
        if not self.question_count:
            self.question_count = len(self.questions)

# Prevent pytest from collecting Test as a test case
Test.__test__ = False


@dataclass
class ScoreResult:
    correct: bool
    score: float
    hallucinated: bool = False
    needs_review: bool = False


@dataclass
class TestScore:
    total: int
    correct: int
    accuracy: float


class SeededRandom:
    """Deterministic RNG based on a string seed."""

    def __init__(self, seed: str):
        seed_int = int.from_bytes(sha256(seed.encode("utf-8")).digest()[:8], "big")
        self._rng = random.Random(seed_int)

    def random(self) -> float:
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, seq: list[Any]) -> Any:
        return self._rng.choice(seq)

    def shuffle(self, items: list[Any]) -> None:
        self._rng.shuffle(items)


# =============================================================================
# Data loading
# =============================================================================

def load_facts(corpus: str = "personal") -> list[dict[str, Any]]:
    """Load facts from dataset.json if available; otherwise use defaults."""
    dataset_path = Path(__file__).parent / "dataset.json"
    if dataset_path.exists():
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        return data.get(corpus, [])

    return DEFAULT_FACTS.get(corpus, DEFAULT_FACTS["personal"])


# =============================================================================
# Generation utilities
# =============================================================================

def shuffle(items: list[Any], rng: SeededRandom) -> list[Any]:
    """Return a deterministically shuffled copy of items."""
    copy = list(items)
    rng.shuffle(copy)
    return copy


def _make_positive_question(fact: dict[str, Any], qid: str) -> Question:
    return Question(
        id=qid,
        type="positive",
        question=fact["question"],
        expected=fact["answer"],
        required=fact.get("required", []),
        bonus=fact.get("bonus", []),
        category=fact.get("category"),
    )


def _make_negative_question(qid: str) -> Question:
    return Question(
        id=qid,
        type="negative",
        question="What is the favorite color?",
        expected="not recorded",
        required=NEGATIVE_KEYWORDS.copy(),
    )


def _make_synthesis_question(fact_a: dict[str, Any], fact_b: dict[str, Any], qid: str) -> Question:
    return Question(
        id=qid,
        type="synthesis",
        question=f"{fact_a['question']} Also, {fact_b['question'].lower()}",
        expected=f"{fact_a['answer']} {fact_b['answer']}",
        required=[],
        components=[
            {"required": fact_a.get("required", [])},
            {"required": fact_b.get("required", [])},
        ],
    )


def _make_adversarial_question(correct_name: str, qid: str) -> Question:
    confusion_target = "Max"
    return Question(
        id=qid,
        type="adversarial",
        question=f"Is the pet named {confusion_target}?",
        expected=f"No, the pet is named {correct_name}",
        required=[correct_name.lower()],
        confusion_target=confusion_target,
        adv_type="name_swap",
    )


def generate_test(level: str, seed: str = "default", corpus: str = "personal") -> Test:
    if level not in LEVEL_CONFIGS:
        raise ValueError(f"Unknown level: {level}")

    config = LEVEL_CONFIGS[level]
    rng = SeededRandom(seed)

    facts = load_facts(corpus)
    facts_shuffled = shuffle(facts, rng)

    questions: list[Question] = []

    # Always include at least one positive and one negative
    if facts_shuffled:
        questions.append(_make_positive_question(facts_shuffled[0], f"{level}-pos-1"))
    questions.append(_make_negative_question(f"{level}-neg-1"))

    # Fill remaining slots alternating positive/negative
    idx = 1
    while len(questions) < config["question_count"]:
        if len(questions) % 2 == 0 and facts_shuffled:
            fact = facts_shuffled[idx % len(facts_shuffled)]
            idx += 1
            questions.append(_make_positive_question(fact, f"{level}-pos-{idx}"))
        else:
            questions.append(_make_negative_question(f"{level}-neg-{len(questions)}"))

    # Optionally add synthesis and adversarial questions by replacing last entries
    if config["include_synthesis"] and len(facts_shuffled) >= 2:
        synth = _make_synthesis_question(facts_shuffled[0], facts_shuffled[1], f"{level}-syn-1")
        questions[-1] = synth

    if config["include_adversarial"] and facts_shuffled:
        adv = _make_adversarial_question(facts_shuffled[0]["answer"].split()[0], f"{level}-adv-1")
        questions[-2] = adv

    return Test(level=level, seed=seed, corpus=corpus, questions=questions)


def generate_prompt(test: Test) -> str:
    lines = [
        "Memory Recall Evaluation",
        "Answer each question based on memory.",
        "If the answer is not available, respond with: Not recorded.",
        "",
    ]

    for i, q in enumerate(test.questions, 1):
        lines.append(f"Q{i}: {q.question}")

    return "\n".join(lines)


# =============================================================================
# Parsing utilities
# =============================================================================

def parse_responses(text: str) -> dict[int, str]:
    if not text:
        return {}

    responses: dict[int, str] = {}
    current_q: Optional[int] = None

    pattern = re.compile(r"^(?:Q(?:uestion)?\s*)?(\d+)\s*:\s*(.*)$", re.IGNORECASE)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            q_num = int(match.group(1))
            answer = match.group(2).strip()
            responses[q_num] = answer
            current_q = q_num
        else:
            if current_q is not None:
                responses[current_q] = (responses[current_q] + "\n" + line).strip()

    return responses


# =============================================================================
# Scoring utilities
# =============================================================================

def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def check_negation(text: str, keyword: str, window: int = 3) -> bool:
    """Return True if keyword appears without negation in a small window."""
    normalized = _normalize_text(text).replace("'", "")
    keyword_norm = _normalize_text(keyword).replace("'", "")

    if not normalized or not keyword_norm:
        return False

    tokens = re.findall(r"\b\w+\b", normalized)
    key_tokens = re.findall(r"\b\w+\b", keyword_norm)

    if not key_tokens:
        return False

    negations = {
        "not", "no", "never", "dont", "cant", "cannot", "isnt", "wasnt", "arent", "wont",
    }

    matched = False
    for i in range(len(tokens) - len(key_tokens) + 1):
        if tokens[i:i + len(key_tokens)] == key_tokens:
            matched = True
            start = max(0, i - window)
            context = set(tokens[start:i])
            if context.intersection(negations):
                continue
            return True

    return False if matched else False


def contains_date_pattern(text: str) -> bool:
    if not text:
        return False

    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b",
        r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b",
    ]

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def score_response(question: Question, response: Optional[str]) -> ScoreResult:
    text = _normalize_text(response)

    if not text:
        return ScoreResult(correct=False, score=0.0)

    if question.type == "negative":
        for keyword in NEGATIVE_KEYWORDS:
            if keyword in text:
                return ScoreResult(correct=True, score=1.0)
        return ScoreResult(correct=False, score=0.0, hallucinated=True)

    if question.type == "synthesis":
        components = question.components or []
        if not components:
            return ScoreResult(correct=False, score=0.0)

        matched = 0
        for comp in components:
            required = comp.get("required", [])
            if required and all(check_negation(text, r) for r in required):
                matched += 1
        score = matched / len(components) if components else 0.0
        return ScoreResult(correct=score >= 0.5, score=score)

    if question.type == "adversarial":
        if question.confusion_target and check_negation(text, question.confusion_target):
            return ScoreResult(correct=False, score=0.0, needs_review=True)
        return ScoreResult(correct=True, score=1.0)

    # Positive (default)
    required = question.required or []
    if required:
        matched = sum(1 for r in required if check_negation(text, r))
        score = matched / len(required)
        return ScoreResult(correct=matched > 0, score=score)

    return ScoreResult(correct=False, score=0.0)


def score_test(test: Test, responses: dict[int, str]) -> TestScore:
    total = test.question_count
    correct = 0

    for idx, question in enumerate(test.questions, 1):
        response = responses.get(idx, "")
        result = score_response(question, response)
        if result.correct:
            correct += 1

    accuracy = correct / total if total else 0.0
    return TestScore(total=total, correct=correct, accuracy=accuracy)


# Backwards-compatible alias (if needed)
Score = ScoreResult
