#!/usr/bin/env python3
"""
Memory Recall Evaluation Harness v2

Comprehensive testing across 7 difficulty levels with multiple question types,
robust scoring, and statistical aggregation.

Usage:
    python harness.py generate <level> [--seed=<seed>] [--corpus=<corpus>]
    python harness.py score <test-file>          # Responses from stdin
    python harness.py validate                    # Run scorer validation
    python harness.py prompt <level> [options]   # Generate test + prompt only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "corpus"
RESULTS_DIR = BASE_DIR / "results"
VALIDATION_FILE = BASE_DIR / "validation-set.json"

# Facts files by corpus type
FACTS_FILES = {
    "personal": BASE_DIR / "facts-v2.json",
    "engineering": BASE_DIR / "facts-engineering.json",
    "eng": BASE_DIR / "facts-engineering.json",
    "codebase": BASE_DIR / "facts-codebase.json",
    "code": BASE_DIR / "facts-codebase.json",
}


# =============================================================================
# Configuration
# =============================================================================

# Test configuration for each level
# Rationale for counts:
# - 30 questions per level gives statistical significance while keeping tests manageable
# - 24:6 positive:negative ratio (80:20) reflects typical usage patterns
# - L5-L7 have fewer questions because they're more complex to generate/validate
LEVEL_CONFIGS = {
    "L1": {
        "count": 30, "positive": 24, "negative": 6,
        "phrasings": ["direct"], "embedded": True,
        "description": "Comprehension baseline - memory embedded in prompt"
    },
    "L2": {
        "count": 30, "positive": 24, "negative": 6,
        "phrasings": ["direct"], "file_read": True,
        "description": "File I/O baseline - agent reads file directly"
    },
    "L3": {
        "count": 30, "positive": 24, "negative": 6,
        "phrasings": ["direct", "indirect"], "memory_search": True,
        "description": "Memory recall - agent uses search tools"
    },
    "L4": {
        "count": 30, "positive": 24, "negative": 6,
        "phrasings": ["direct", "indirect"], "noisy": True,
        "description": "Noisy retrieval - multiple distractor files"
    },
    "L5": {
        "count": 20, "positive": 15, "negative": 2, "synthesis": 3,
        "phrasings": ["direct", "indirect", "inference"], "cross_ref": True,
        "description": "Synthesis - questions requiring multiple facts"
    },
    "L6": {
        "count": 20, "positive": 14, "negative": 2, "temporal_count": 4,
        "phrasings": ["direct", "temporal"], "temporal": True,
        "description": "Temporal - tracking corrections/changes"
    },
    "L7": {
        "count": 25, "positive": 15, "negative": 5, "adversarial_count": 5,
        "phrasings": ["direct", "indirect", "negation", "inference"], "adversarial": True,
        "description": "Adversarial - trap questions designed to confuse"
    },
}

# Keywords indicating "not recorded"
NEGATIVE_KEYWORDS = [
    "not recorded", "don't have", "no mention", "not found",
    "no information", "not specified", "not mentioned",
    "isn't recorded", "not in memory", "no record"
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Question:
    """A single test question."""
    id: str
    type: str  # positive, negative, synthesis, temporal, adversarial
    question: str
    expected: str
    required: list[str] = field(default_factory=list)
    bonus: list[str] = field(default_factory=list)
    category: str | None = None
    components: list[dict] | None = None  # For synthesis questions
    confusion_target: str | None = None  # For adversarial questions
    adv_type: str | None = None


@dataclass
class Test:
    """A complete test with questions."""
    level: str
    seed: str
    corpus: str
    generated_at: str
    config: dict
    question_count: int
    questions: list[Question]


@dataclass
class ScoreResult:
    """Result of scoring a single response."""
    correct: bool
    score: float
    reason: str
    completeness: float | None = None
    hallucinated: bool | None = None
    needs_review: bool = False
    component_scores: list[float] | None = None


@dataclass
class TestResults:
    """Complete results from scoring a test."""
    level: str
    seed: str
    timestamp: str
    total: int
    correct: int
    accuracy: str
    hallucination_rate: str | None = None
    scores: dict = field(default_factory=dict)
    details: list[dict] = field(default_factory=list)


# =============================================================================
# Seeded Random
# =============================================================================

class SeededRandom:
    """Deterministic random number generator using SHA-256 hash chain."""
    
    def __init__(self, seed: str | int):
        self.seed = str(seed).encode()
        self.counter = 0
        self._regenerate()
    
    def _regenerate(self) -> None:
        """Generate new entropy by hashing seed + counter."""
        data = self.seed + self.counter.to_bytes(8, "big")
        self.hash = hashlib.sha256(data).digest()
        self.offset = 0
        self.counter += 1
    
    def random(self) -> float:
        """Return a random float in [0, 1)."""
        # If we've exhausted this hash block, regenerate
        if self.offset + 4 > len(self.hash):
            self._regenerate()
        
        value = int.from_bytes(self.hash[self.offset:self.offset + 4], "big")
        self.offset += 4
        return value / 0xFFFFFFFF


def shuffle(items: list, rng: SeededRandom | None = None) -> list:
    """Fisher-Yates shuffle with optional seeded RNG."""
    result = items.copy()
    random_fn = rng.random if rng else __import__("random").random
    
    for i in range(len(result) - 1, 0, -1):
        j = int(random_fn() * (i + 1))
        result[i], result[j] = result[j], result[i]
    
    return result


# =============================================================================
# Data Loading
# =============================================================================

def load_facts(corpus: str = "personal") -> dict:
    """Load facts from the appropriate JSON file."""
    facts_file = FACTS_FILES.get(corpus, FACTS_FILES["personal"])
    
    if not facts_file.exists():
        raise FileNotFoundError(f"Facts file not found: {facts_file}")
    
    try:
        return json.loads(facts_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {facts_file}: {e}")


def load_corpus(filename: str) -> str:
    """Load a corpus file."""
    corpus_file = CORPUS_DIR / filename
    
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
    
    return corpus_file.read_text(encoding="utf-8")


# =============================================================================
# Question Generation
# =============================================================================

def select_question(fact: dict, phrasings: list[str]) -> str:
    """Select a question phrasing for a fact."""
    available = list(fact.get("questions", {}).keys())
    preferred = [p for p in phrasings if p in available]
    
    if not preferred:
        return fact["questions"].get("direct", "")
    
    import random
    return fact["questions"][random.choice(preferred)]


def generate_test(level: str, seed: str | int | None = None, corpus: str = "personal") -> Test:
    """Generate a test for a specific level."""
    if level not in LEVEL_CONFIGS:
        raise ValueError(f"Unknown level: {level}. Valid levels: {list(LEVEL_CONFIGS.keys())}")
    
    config = LEVEL_CONFIGS[level]
    data = load_facts(corpus)
    
    if seed is None:
        seed = datetime.now().isoformat()
    seed = str(seed)
    
    rng = SeededRandom(seed)
    questions: list[Question] = []
    
    # Add positive questions
    shuffled_facts = shuffle(data.get("facts", []), rng)
    for i, fact in enumerate(shuffled_facts[:config["positive"]]):
        questions.append(Question(
            id=f"pos_{fact['id']}",
            type="positive",
            category=fact.get("category"),
            question=select_question(fact, config["phrasings"]),
            expected=fact["fact"],
            required=fact["required"],
            bonus=fact.get("bonus", [])
        ))
    
    # Add negative questions
    shuffled_negatives = shuffle(data.get("negatives", []), rng)
    for neg in shuffled_negatives[:config["negative"]]:
        questions.append(Question(
            id=neg["id"],
            type="negative",
            question=neg["question"],
            expected="not recorded",
            required=NEGATIVE_KEYWORDS.copy()
        ))
    
    # Add synthesis questions for L5
    if config.get("synthesis"):
        shuffled_synthesis = shuffle(data.get("synthesis", []), rng)
        for syn in shuffled_synthesis[:config["synthesis"]]:
            questions.append(Question(
                id=f"syn_{syn['id']}",
                type="synthesis",
                question=syn["question"],
                expected=syn["expected"],
                components=syn.get("components")
            ))
    
    # Add temporal questions for L6
    if config.get("temporal_count"):
        temporal_facts = [f for f in data.get("facts", []) if f.get("category") == "temporal"]
        shuffled_temporal = shuffle(temporal_facts, rng)
        for fact in shuffled_temporal[:config["temporal_count"]]:
            q_text = fact["questions"].get("temporal") or fact["questions"].get("direct", "")
            questions.append(Question(
                id=f"temp_{fact['id']}",
                type="temporal",
                category="temporal",
                question=q_text,
                expected=fact["fact"],
                required=fact["required"],
                bonus=fact.get("bonus", [])
            ))
    
    # Add adversarial questions for L7
    if config.get("adversarial_count"):
        shuffled_adv = shuffle(data.get("adversarial", []), rng)
        for adv in shuffled_adv[:config["adversarial_count"]]:
            questions.append(Question(
                id=adv["id"],
                type="adversarial",
                question=adv["question"],
                expected=adv["expected"],
                confusion_target=adv.get("confusion_target"),
                adv_type=adv.get("type")
            ))
    
    # Shuffle final question order
    final_questions = shuffle(questions, rng)
    
    return Test(
        level=level,
        seed=seed,
        corpus=corpus,
        generated_at=datetime.now().isoformat(),
        config=config,
        question_count=len(final_questions),
        questions=final_questions
    )


# =============================================================================
# Prompt Generation
# =============================================================================

def generate_prompt(test: Test) -> str:
    """Generate the evaluation prompt for a test."""
    data = load_facts(test.corpus)
    lines = [
        "# Memory Recall Evaluation",
        "",
        "You are being tested on memory recall accuracy.",
        "Answer each question based ONLY on the information available to you.",
        'If information is not available, say "Not recorded" or "I don\'t have that information."',
        "Do NOT guess or infer — only report what is explicitly recorded.",
        ""
    ]
    
    config = test.config
    
    if config.get("embedded"):
        lines.extend([
            "## MEMORY FILE CONTENTS",
            "```",
            load_corpus(data["targetFile"]),
            "```",
            ""
        ])
    elif config.get("file_read"):
        lines.extend([
            "## INSTRUCTIONS",
            f"Read the memory file at: {CORPUS_DIR / data['targetFile']}",
            "Then answer the questions based on its contents.",
            ""
        ])
    elif config.get("memory_search"):
        lines.extend([
            "## INSTRUCTIONS",
            "Use memory_search and memory_get to find information in the memory files.",
            f"Memory directory: {CORPUS_DIR}",
            "Answer questions based on what you find.",
            ""
        ])
    elif config.get("noisy"):
        lines.extend([
            "## INSTRUCTIONS",
            "Search the memory files to find information about Person A.",
            f"Memory directory: {CORPUS_DIR}",
            "Note: There are multiple people in the memory files. Only answer about Person A.",
            ""
        ])
    
    lines.extend([
        "## QUESTIONS",
        "Answer each question on its own line, prefixed with the question number.",
        "Format: Q1: [your answer]",
        ""
    ])
    
    for i, q in enumerate(test.questions, 1):
        lines.append(f"Q{i}: {q.question}")
    
    lines.extend([
        "",
        "Answer all questions now, one per line. Be precise and factual."
    ])
    
    return "\n".join(lines)


# =============================================================================
# Response Parsing
# =============================================================================

def parse_responses(text: str) -> dict[int, str]:
    """Parse responses from agent output (robust to multiple formats)."""
    responses: dict[int, str] = {}
    
    # Patterns to match: "Q1:", "1:", "1.", "Question 1:", "1)", etc.
    patterns = [
        re.compile(r"^Q(\d+)[:\s]+(.+)$", re.IGNORECASE),
        re.compile(r"^(\d+)[:.)\s]+(.+)$"),
        re.compile(r"^Question\s+(\d+)[:\s]+(.+)$", re.IGNORECASE)
    ]
    
    current_num: int | None = None
    current_answer: list[str] = []
    
    for line in text.split("\n"):
        matched = False
        
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                # Save previous answer if exists
                if current_num is not None:
                    responses[current_num] = " ".join(current_answer).strip()
                
                current_num = int(match.group(1))
                current_answer = [match.group(2).strip()]
                matched = True
                break
        
        # Handle multi-line answers
        if not matched and current_num is not None and line.strip():
            current_answer.append(line.strip())
    
    # Save last answer
    if current_num is not None:
        responses[current_num] = " ".join(current_answer).strip()
    
    return responses


# =============================================================================
# Scoring
# =============================================================================

def contains_date_pattern(text: str) -> bool:
    """Check if response contains a valid date pattern."""
    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",                    # 2025-01-10
        r"\d{1,2}/\d{1,2}/\d{2,4}",              # 1/10/2025
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}",  # January 10
        r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",        # 10 January
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower, re.IGNORECASE) for p in date_patterns)


def check_negation(text: str, keyword: str, window: int = 20) -> bool:
    """
    Check if keyword appears WITHOUT negation in preceding text.
    
    Returns True if keyword is found and NOT preceded by negation words.
    """
    text_lower = text.lower()
    kw_lower = keyword.lower()
    
    idx = text_lower.find(kw_lower)
    if idx == -1:
        return False
    
    # Check for negation words within window before keyword
    prefix = text_lower[max(0, idx - window):idx]
    negation_words = ["not", "isn't", "wasn't", "no ", "never", "don't", "doesn't"]
    
    for neg in negation_words:
        if neg in prefix:
            return False  # Negation found - keyword doesn't count
    
    return True


def score_response(question: Question, response: str) -> ScoreResult:
    """Score a single response against expected answer."""
    resp = (response or "").lower()
    
    if question.type in ("positive", "temporal"):
        # Check required keywords (any one = pass)
        # Use negation-aware matching
        has_required = any(
            check_negation(resp, kw) for kw in question.required
        )
        
        # For temporal questions, accept date patterns
        is_temporal = question.type == "temporal" or question.category == "temporal"
        if not has_required and is_temporal and contains_date_pattern(resp):
            has_required = True
        
        bonus_count = sum(1 for kw in question.bonus if kw.lower() in resp)
        bonus_total = len(question.bonus)
        completeness = (bonus_count / bonus_total) if bonus_total > 0 else 1.0
        
        return ScoreResult(
            correct=has_required,
            score=1.0 if has_required else 0.0,
            completeness=completeness,
            reason="Required keyword found" if has_required else "Missing required keywords"
        )
    
    if question.type == "negative":
        # Should say "not recorded" or similar
        is_negative = any(kw.lower() in resp for kw in question.required)
        is_refusal = any(phrase in resp for phrase in ["shouldn't", "cannot", "can't speculate"])
        
        return ScoreResult(
            correct=is_negative or is_refusal,
            score=1.0 if (is_negative or is_refusal) else 0.0,
            hallucinated=not is_negative and not is_refusal and len(resp) > 10,
            reason="Correctly said not recorded" if is_negative else (
                "Refusal (acceptable)" if is_refusal else "Hallucinated or wrong"
            )
        )
    
    if question.type == "synthesis":
        # Score each component
        components = question.components or []
        component_scores = []
        
        for comp in components:
            has_required = any(kw.lower() in resp for kw in comp.get("required", []))
            component_scores.append(1.0 if has_required else 0.0)
        
        score = sum(component_scores) / len(component_scores) if component_scores else 0.0
        found = sum(1 for s in component_scores if s > 0)
        
        return ScoreResult(
            correct=score >= 0.5,
            score=score,
            component_scores=component_scores,
            reason=f"{found}/{len(component_scores)} components found"
        )
    
    if question.type == "adversarial":
        # These need manual review - check for corrections or "not recorded"
        correction_phrases = ["not ", "isn't", "actually", "incorrect"]
        has_correction = any(phrase in resp for phrase in correction_phrases)
        says_not_recorded = any(kw.lower() in resp for kw in NEGATIVE_KEYWORDS)
        
        return ScoreResult(
            correct=has_correction or says_not_recorded,
            score=1.0 if (has_correction or says_not_recorded) else 0.0,
            needs_review=True,
            reason="Caught the confusion" if has_correction else (
                "Said not recorded" if says_not_recorded else "May have been confused"
            )
        )
    
    return ScoreResult(correct=False, score=0.0, reason="Unknown question type")


def score_test(test: Test, responses: dict[int, str]) -> TestResults:
    """Score all responses for a test."""
    results = TestResults(
        level=test.level,
        seed=test.seed,
        timestamp=datetime.now().isoformat(),
        total=test.question_count,
        correct=0,
        accuracy="0.0",
        scores={
            "positive": {"correct": 0, "total": 0, "completeness": []},
            "negative": {"correct": 0, "total": 0, "hallucinated": 0},
            "synthesis": {"correct": 0, "total": 0, "avg_score": 0.0},
            "temporal": {"correct": 0, "total": 0},
            "adversarial": {"correct": 0, "total": 0, "needs_review": []}
        }
    )
    
    for i, q in enumerate(test.questions, 1):
        response = responses.get(i, "")
        scored = score_response(q, response)
        
        detail = {
            "question_num": i,
            "id": q.id,
            "type": q.type,
            "question": q.question,
            "expected": q.expected,
            "response": response,
            "correct": scored.correct,
            "score": scored.score,
            "reason": scored.reason
        }
        if scored.completeness is not None:
            detail["completeness"] = scored.completeness
        if scored.hallucinated is not None:
            detail["hallucinated"] = scored.hallucinated
        if scored.component_scores:
            detail["component_scores"] = scored.component_scores
        
        results.details.append(detail)
        
        if scored.correct:
            results.correct += 1
        
        # Type-specific aggregation
        type_stats = results.scores.get(q.type)
        if type_stats:
            type_stats["total"] += 1
            if scored.correct:
                type_stats["correct"] += 1
            
            if q.type == "positive" and scored.completeness is not None:
                type_stats["completeness"].append(scored.completeness)
            if q.type == "negative" and scored.hallucinated:
                type_stats["hallucinated"] += 1
            if q.type == "adversarial" and scored.needs_review:
                type_stats["needs_review"].append(i)
    
    # Calculate aggregates
    if results.total > 0:
        pct = results.correct / results.total * 100
        results.accuracy = f"{pct:.1f}"
    else:
        results.accuracy = "0.0"
    
    pos_comp = results.scores["positive"]["completeness"]
    if pos_comp:
        avg = sum(pos_comp) / len(pos_comp) * 100
        results.scores["positive"]["avg_completeness"] = f"{avg:.1f}"
    
    neg_total = results.scores["negative"]["total"]
    if neg_total > 0:
        hall = results.scores["negative"]["hallucinated"]
        results.hallucination_rate = f"{(hall / neg_total * 100):.1f}"
    
    return results


# =============================================================================
# Output Formatting
# =============================================================================

def format_results(results: TestResults) -> str:
    """Format results for display."""
    lines = [
        f"# Memory Recall Evaluation — Level {results.level}",
        "",
        f"## Overall: {results.accuracy}% ({results.correct}/{results.total})",
        "",
        "## Breakdown by Type",
        ""
    ]
    
    for type_name, stats in results.scores.items():
        if stats["total"] > 0:
            pct = (stats["correct"] / stats["total"] * 100)
            lines.append(f"- **{type_name}:** {stats['correct']}/{stats['total']} ({pct:.1f}%)")
            
            if type_name == "positive" and stats.get("avg_completeness"):
                lines.append(f"  - Completeness: {stats['avg_completeness']}%")
            if type_name == "negative" and stats.get("hallucinated", 0) > 0:
                lines.append(f"  - Hallucinations: {stats['hallucinated']}")
    
    if results.hallucination_rate:
        lines.extend([
            "",
            f"**Hallucination Rate:** {results.hallucination_rate}%"
        ])
    
    lines.extend([
        "",
        "## Details",
        ""
    ])
    
    for d in results.details:
        icon = "✅" if d["correct"] else "❌"
        lines.append(f'{icon} **Q{d["question_num"]}** [{d["type"]}]: {d["question"]}')
        lines.append(f'   Expected: {d["expected"]}')
        lines.append(f'   Got: {d["response"] or "(no response)"}')
        lines.append(f'   {d["reason"]}')
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Validation
# =============================================================================

def validate_scorer() -> bool:
    """Run scorer validation against known cases."""
    if VALIDATION_FILE.exists():
        validation_set = json.loads(VALIDATION_FILE.read_text(encoding="utf-8"))
    else:
        # Create default validation set
        validation_set = {
            "version": "1.0",
            "description": "Known Q&A pairs to validate scorer correctness",
            "cases": [
                {
                    "question": "What is the favorite cuisine?",
                    "response": "Thai food, especially green curry",
                    "expected": {"correct": True, "type": "positive"}
                },
                {
                    "question": "What is the favorite cuisine?",
                    "response": "Italian pasta",
                    "expected": {"correct": False, "type": "positive"}
                },
                {
                    "question": "What is the favorite color?",
                    "response": "Not recorded in memory",
                    "expected": {"correct": True, "type": "negative"}
                },
                {
                    "question": "What is the favorite color?",
                    "response": "Blue",
                    "expected": {"correct": False, "type": "negative", "hallucinated": True}
                },
                {
                    "question": "What is the favorite color?",
                    "response": "I don't have that information",
                    "expected": {"correct": True, "type": "negative"}
                },
            ]
        }
        VALIDATION_FILE.write_text(json.dumps(validation_set, indent=2), encoding="utf-8")
        print(f"Created validation set at {VALIDATION_FILE}")
    
    passed = 0
    failed = 0
    
    print("Running scorer validation...\n")
    
    # Keywords for validation (simplified)
    keywords_by_type = {
        "positive": ["thai", "morgan", "whiskers"],
        "negative": NEGATIVE_KEYWORDS
    }
    
    for case in validation_set["cases"]:
        q_type = case["expected"]["type"]
        mock_question = Question(
            id="validation",
            type=q_type,
            question=case["question"],
            expected="",
            required=keywords_by_type.get(q_type, [])
        )
        
        result = score_response(mock_question, case["response"])
        expected_correct = case["expected"]["correct"]
        
        resp_text = case["response"]
        response_preview = resp_text[:40] + "..." if len(resp_text) > 40 else resp_text
        
        if result.correct == expected_correct:
            print(f'✅ PASS: "{response_preview}" → {result.correct}')
            passed += 1
        else:
            print(f'❌ FAIL: "{response_preview}" → got {result.correct}, '
                  f'expected {expected_correct}')
            failed += 1
    
    print(f"\nValidation: {passed}/{passed + failed} passed")
    return failed == 0


# =============================================================================
# Serialization Helpers
# =============================================================================

def test_to_dict(test: Test) -> dict:
    """Convert Test to JSON-serializable dict."""
    return {
        "level": test.level,
        "seed": test.seed,
        "corpus": test.corpus,
        "generatedAt": test.generated_at,
        "config": test.config,
        "questionCount": test.question_count,
        "questions": [
            {k: v for k, v in asdict(q).items() if v is not None}
            for q in test.questions
        ]
    }


def dict_to_test(data: dict) -> Test:
    """Convert dict back to Test object."""
    questions = [
        Question(
            id=q["id"],
            type=q["type"],
            question=q["question"],
            expected=q["expected"],
            required=q.get("required", []),
            bonus=q.get("bonus", []),
            category=q.get("category"),
            components=q.get("components"),
            confusion_target=q.get("confusion_target") or q.get("confusionTarget"),
            adv_type=q.get("adv_type") or q.get("advType")
        )
        for q in data["questions"]
    ]
    
    return Test(
        level=data["level"],
        seed=str(data["seed"]),
        corpus=data.get("corpus", "personal"),
        generated_at=data.get("generatedAt", data.get("generated_at", "")),
        config=data["config"],
        question_count=data.get("questionCount", data.get("question_count", len(questions))),
        questions=questions
    )


def results_to_dict(results: TestResults) -> dict:
    """Convert TestResults to JSON-serializable dict."""
    return {
        "level": results.level,
        "seed": results.seed,
        "timestamp": results.timestamp,
        "total": results.total,
        "correct": results.correct,
        "accuracy": results.accuracy,
        "hallucinationRate": results.hallucination_rate,
        "scores": results.scores,
        "details": results.details
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Memory Recall Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Levels:
  L1  Baseline (Comprehension) - Memory embedded in prompt, establishes ceiling
  L2  Baseline (File I/O)      - Agent reads file directly, no retrieval needed
  L3  Memory Recall (Search)   - Agent uses search tools, ACTUAL memory testing starts here
  L4  Noisy Retrieval          - Multiple distractor files present
  L5  Synthesis                - Questions requiring multiple facts
  L6  Temporal                 - Questions about corrections/changes
  L7  Adversarial              - Trap questions designed to confuse

Examples:
  python harness.py generate L1
  python harness.py generate L3 --seed=12345 --corpus=engineering
  cat responses.txt | python harness.py score results/test-L1-xxx.json
  python harness.py validate
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a test")
    gen_parser.add_argument(
        "level", default="L1", nargs="?", help="Test level (L1-L7)"
    )
    gen_parser.add_argument("--seed", help="Random seed for reproducibility")
    gen_parser.add_argument(
        "--corpus", default="personal", help="Corpus: personal/engineering/codebase"
    )
    
    # Prompt command (same as generate but outputs prompt only)
    prompt_parser = subparsers.add_parser(
        "prompt", help="Generate test and output prompt"
    )
    prompt_parser.add_argument(
        "level", default="L1", nargs="?", help="Test level (L1-L7)"
    )
    prompt_parser.add_argument("--seed", help="Random seed for reproducibility")
    prompt_parser.add_argument(
        "--corpus", default="personal", help="Corpus: personal/engineering/codebase"
    )
    
    # Score command
    score_parser = subparsers.add_parser("score", help="Score responses from stdin")
    score_parser.add_argument("test_file", help="Path to test JSON file")
    
    # Validate command
    subparsers.add_parser("validate", help="Run scorer validation")
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.command in ("generate", "prompt"):
        test = generate_test(args.level, seed=args.seed, corpus=args.corpus)
        
        corpus_label = "ENG-" if test.corpus in ("engineering", "eng") else ""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        test_file = RESULTS_DIR / f"test-{corpus_label}{test.level}-{timestamp}.json"
        
        test_file.write_text(json.dumps(test_to_dict(test), indent=2), encoding="utf-8")
        
        print(f"Generated {test.level} test → {test_file}", file=sys.stderr)
        print(f"Seed: {test.seed}", file=sys.stderr)
        print(f"Questions: {test.question_count}", file=sys.stderr)
        print("", file=sys.stderr)
        
        print(generate_prompt(test))
    
    elif args.command == "score":
        test_path = Path(args.test_file)
        if not test_path.exists():
            print(f"Error: Test file not found: {test_path}", file=sys.stderr)
            sys.exit(1)
        
        test = dict_to_test(json.loads(test_path.read_text(encoding="utf-8")))
        
        # Read responses from stdin
        input_text = sys.stdin.read()
        responses = parse_responses(input_text)
        
        results = score_test(test, responses)
        
        # Save results
        results_name = test_path.name.replace("test-", "results-")
        results_name = results_name.replace(".json", "-scored.json")
        results_file = test_path.parent / results_name
        results_json = json.dumps(results_to_dict(results), indent=2)
        results_file.write_text(results_json, encoding="utf-8")
        print(f"Saved results to {results_file}", file=sys.stderr)
        
        print(format_results(results))
    
    elif args.command == "validate":
        success = validate_scorer()
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
