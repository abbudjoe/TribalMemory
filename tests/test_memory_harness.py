"""
Pytest tests for the memory recall evaluation harness.

Tests question generation, response parsing, scoring logic,
and the full evaluation pipeline.
"""

import json
import pytest
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "eval" / "memory-test"))

from harness import (
    Question,
    Test,
    ScoreResult,
    SeededRandom,
    shuffle,
    generate_test,
    generate_prompt,
    parse_responses,
    score_response,
    score_test,
    check_negation,
    contains_date_pattern,
    load_facts,
    LEVEL_CONFIGS,
    NEGATIVE_KEYWORDS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def validation_set():
    """Load the validation set for scorer testing."""
    validation_file = Path(__file__).parent.parent / "eval" / "memory-test" / "validation-set.json"
    if validation_file.exists():
        return json.loads(validation_file.read_text(encoding="utf-8"))
    return None


@pytest.fixture
def sample_positive_question():
    """Create a sample positive question for testing."""
    return Question(
        id="test_pos_1",
        type="positive",
        question="What is the favorite cuisine?",
        expected="Thai food",
        required=["thai", "green curry"],
        bonus=["spicy", "pad thai"],
        category="preferences"
    )


@pytest.fixture
def sample_negative_question():
    """Create a sample negative question for testing."""
    return Question(
        id="test_neg_1",
        type="negative",
        question="What is the favorite color?",
        expected="not recorded",
        required=NEGATIVE_KEYWORDS.copy()
    )


@pytest.fixture
def sample_synthesis_question():
    """Create a sample synthesis question for testing."""
    return Question(
        id="test_syn_1",
        type="synthesis",
        question="Where does the spouse work and what is their role?",
        expected="Morgan works at Cloudflare as a software architect",
        components=[
            {"required": ["morgan", "cloudflare"]},
            {"required": ["architect", "software"]}
        ]
    )


@pytest.fixture
def sample_adversarial_question():
    """Create a sample adversarial question for testing."""
    return Question(
        id="test_adv_1",
        type="adversarial",
        question="Is the pet named Max?",
        expected="No, the pet is named Whiskers",
        confusion_target="Max",
        adv_type="name_swap"
    )


# =============================================================================
# SeededRandom Tests
# =============================================================================

class TestSeededRandom:
    """Tests for deterministic random number generation."""
    
    def test_reproducibility(self):
        """Same seed should produce same sequence."""
        rng1 = SeededRandom("test_seed")
        rng2 = SeededRandom("test_seed")
        
        values1 = [rng1.random() for _ in range(10)]
        values2 = [rng2.random() for _ in range(10)]
        
        assert values1 == values2
    
    def test_different_seeds(self):
        """Different seeds should produce different sequences."""
        rng1 = SeededRandom("seed_a")
        rng2 = SeededRandom("seed_b")
        
        values1 = [rng1.random() for _ in range(10)]
        values2 = [rng2.random() for _ in range(10)]
        
        assert values1 != values2
    
    def test_range(self):
        """Random values should be in [0, 1)."""
        rng = SeededRandom("range_test")
        
        for _ in range(100):
            value = rng.random()
            assert 0 <= value < 1


class TestShuffle:
    """Tests for seeded shuffle function."""
    
    def test_shuffle_reproducibility(self):
        """Same seed should produce same shuffle."""
        items = [1, 2, 3, 4, 5]
        
        result1 = shuffle(items, SeededRandom("shuffle_test"))
        result2 = shuffle(items, SeededRandom("shuffle_test"))
        
        assert result1 == result2
    
    def test_shuffle_preserves_elements(self):
        """Shuffle should preserve all elements."""
        items = [1, 2, 3, 4, 5]
        result = shuffle(items, SeededRandom("preserve_test"))
        
        assert sorted(result) == sorted(items)
    
    def test_shuffle_does_not_modify_original(self):
        """Shuffle should not modify the original list."""
        items = [1, 2, 3, 4, 5]
        original = items.copy()
        shuffle(items, SeededRandom("modify_test"))
        
        assert items == original


# =============================================================================
# Question Generation Tests
# =============================================================================

class TestGenerateTest:
    """Tests for test generation."""
    
    @pytest.mark.parametrize("level", ["L1", "L2", "L3", "L4", "L5", "L6", "L7"])
    def test_generate_all_levels(self, level):
        """Should be able to generate tests for all levels."""
        test = generate_test(level, seed="test_seed")
        
        assert test.level == level
        assert test.seed == "test_seed"
        assert test.question_count > 0
        assert len(test.questions) == test.question_count
    
    def test_generate_with_seed_reproducibility(self):
        """Same seed should produce identical tests."""
        test1 = generate_test("L1", seed="repro_test")
        test2 = generate_test("L1", seed="repro_test")
        
        assert test1.question_count == test2.question_count
        assert [q.id for q in test1.questions] == [q.id for q in test2.questions]
    
    def test_generate_question_types(self):
        """Generated tests should have expected question types."""
        test = generate_test("L1", seed="type_test")
        
        types = {q.type for q in test.questions}
        assert "positive" in types
        assert "negative" in types
    
    def test_invalid_level_raises_error(self):
        """Invalid level should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown level"):
            generate_test("L99")
    
    @pytest.mark.parametrize("corpus", ["personal", "engineering", "codebase"])
    def test_generate_different_corpora(self, corpus):
        """Should be able to generate tests for different corpora."""
        test = generate_test("L1", seed="corpus_test", corpus=corpus)
        
        assert test.corpus == corpus
        assert test.question_count > 0


class TestGeneratePrompt:
    """Tests for prompt generation."""
    
    def test_prompt_contains_questions(self):
        """Generated prompt should contain all questions."""
        test = generate_test("L1", seed="prompt_test")
        prompt = generate_prompt(test)
        
        for i, q in enumerate(test.questions, 1):
            assert f"Q{i}:" in prompt
    
    def test_prompt_contains_instructions(self):
        """Prompt should contain standard instructions."""
        test = generate_test("L1", seed="instruction_test")
        prompt = generate_prompt(test)
        
        assert "Memory Recall Evaluation" in prompt
        assert "Not recorded" in prompt or "not recorded" in prompt


# =============================================================================
# Response Parsing Tests
# =============================================================================

class TestParseResponses:
    """Tests for response parsing."""
    
    def test_parse_standard_format(self):
        """Parse Q1: format responses."""
        text = """Q1: Thai food
Q2: Not recorded
Q3: Morgan at Cloudflare"""
        
        responses = parse_responses(text)
        
        assert responses[1] == "Thai food"
        assert responses[2] == "Not recorded"
        assert responses[3] == "Morgan at Cloudflare"
    
    def test_parse_numbered_format(self):
        """Parse 1: format responses."""
        text = """1: Thai food
2: Not recorded"""
        
        responses = parse_responses(text)
        
        assert responses[1] == "Thai food"
        assert responses[2] == "Not recorded"
    
    def test_parse_question_format(self):
        """Parse Question 1: format responses."""
        text = """Question 1: Thai food
Question 2: Not recorded"""
        
        responses = parse_responses(text)
        
        assert responses[1] == "Thai food"
        assert responses[2] == "Not recorded"
    
    def test_parse_multiline_response(self):
        """Parse multi-line responses."""
        text = """Q1: Thai food with green curry
and spicy noodles
Q2: Not recorded"""
        
        responses = parse_responses(text)
        
        assert "Thai food" in responses[1]
        assert "spicy noodles" in responses[1]
    
    def test_parse_empty_input(self):
        """Handle empty input gracefully."""
        responses = parse_responses("")
        assert responses == {}
    
    def test_parse_malformed_input(self):
        """Handle malformed input without crashing."""
        text = "This is not a proper response format at all"
        responses = parse_responses(text)
        # Should return empty dict since no patterns match
        assert isinstance(responses, dict)


# =============================================================================
# Scoring Tests
# =============================================================================

class TestScoreResponse:
    """Tests for response scoring."""
    
    def test_score_positive_correct(self, sample_positive_question):
        """Correct positive response should score 1.0."""
        result = score_response(sample_positive_question, "Thai food, especially green curry")
        
        assert result.correct is True
        assert result.score == 1.0
    
    def test_score_positive_incorrect(self, sample_positive_question):
        """Incorrect positive response should score 0.0."""
        result = score_response(sample_positive_question, "Italian pasta")
        
        assert result.correct is False
        assert result.score == 0.0
    
    def test_score_negative_correct(self, sample_negative_question):
        """Correct negative response should score 1.0."""
        result = score_response(sample_negative_question, "Not recorded in memory")
        
        assert result.correct is True
        assert result.score == 1.0
    
    def test_score_negative_hallucination(self, sample_negative_question):
        """Hallucinated negative response should be flagged."""
        result = score_response(sample_negative_question, "Blue is the favorite color")
        
        assert result.correct is False
        assert result.hallucinated is True
    
    def test_score_synthesis_partial(self, sample_synthesis_question):
        """Partial synthesis response should have partial score."""
        result = score_response(sample_synthesis_question, "Morgan works at Cloudflare")
        
        # Should get 1/2 components (missing architect/software)
        assert result.score == 0.5
        assert result.correct is True  # >=0.5 is passing
    
    def test_score_synthesis_complete(self, sample_synthesis_question):
        """Complete synthesis response should score 1.0."""
        result = score_response(
            sample_synthesis_question,
            "Morgan is a software architect at Cloudflare"
        )
        
        assert result.score == 1.0
        assert result.correct is True
    
    def test_score_adversarial_caught(self, sample_adversarial_question):
        """Adversarial response that catches confusion should pass."""
        result = score_response(
            sample_adversarial_question,
            "No, the pet is not named Max. The pet is actually named Whiskers."
        )
        
        assert result.correct is True
    
    def test_score_adversarial_fooled(self, sample_adversarial_question):
        """Adversarial response that falls for confusion should fail."""
        result = score_response(
            sample_adversarial_question,
            "Yes, the pet is named Max"
        )
        
        assert result.correct is False
        assert result.needs_review is True


class TestNegationDetection:
    """Tests for negation-aware keyword matching."""
    
    def test_simple_match(self):
        """Keyword without negation should match."""
        assert check_negation("I like Thai food", "thai") is True
    
    def test_negation_blocks_match(self):
        """Keyword with negation should not match."""
        assert check_negation("I don't like Thai food", "thai") is False
        assert check_negation("It's not Thai food", "thai") is False
    
    def test_negation_distance(self):
        """Negation far from keyword should not block."""
        # "not" is far from "thai", outside the window
        text = "I do not eat meat but I do love Thai food"
        assert check_negation(text, "thai") is True
    
    def test_no_match_returns_false(self):
        """Missing keyword should return False."""
        assert check_negation("I like Italian food", "thai") is False


class TestDatePatternDetection:
    """Tests for date pattern detection."""
    
    @pytest.mark.parametrize("text", [
        "The date is 2025-01-15",
        "It happened on 1/15/2025",
        "January 15th was the day",
        "On 15 January they met",
    ])
    def test_valid_date_patterns(self, text):
        """Should detect various date formats."""
        assert contains_date_pattern(text) is True
    
    def test_no_date_pattern(self):
        """Should return False when no date present."""
        assert contains_date_pattern("No dates here") is False


class TestValidationSet:
    """Tests using the validation set to verify scorer correctness."""
    
    def test_validation_cases(self, validation_set):
        """All validation cases should pass."""
        if validation_set is None:
            pytest.skip("Validation set file not found")
        
        # Keywords for validation (simplified)
        keywords_by_type = {
            "positive": ["thai", "morgan", "whiskers"],
            "negative": NEGATIVE_KEYWORDS
        }
        
        passed = 0
        failed = 0
        failures = []
        
        for case in validation_set["cases"]:
            q_type = case["expected"]["type"]
            question = Question(
                id="validation",
                type=q_type,
                question=case["question"],
                expected="",
                required=keywords_by_type.get(q_type, [])
            )
            
            result = score_response(question, case["response"])
            expected_correct = case["expected"]["correct"]
            
            if result.correct == expected_correct:
                passed += 1
            else:
                failed += 1
                failures.append({
                    "question": case["question"],
                    "response": case["response"],
                    "expected": expected_correct,
                    "got": result.correct
                })
        
        # Report failures
        for f in failures:
            print(f"FAIL: {f['response'][:40]}... expected {f['expected']}, got {f['got']}")
        
        assert failed == 0, f"Validation failed: {passed}/{passed + failed} passed"


# =============================================================================
# Full Pipeline Tests
# =============================================================================

class TestFullPipeline:
    """Integration tests for the full evaluation pipeline."""
    
    def test_generate_and_score_l1(self):
        """Should be able to generate and score a complete L1 test."""
        test = generate_test("L1", seed="pipeline_test")
        prompt = generate_prompt(test)
        
        # Simulate perfect responses
        responses = {}
        for i, q in enumerate(test.questions, 1):
            if q.type == "negative":
                responses[i] = "Not recorded in memory"
            else:
                # Use the first required keyword as the response
                responses[i] = q.required[0] if q.required else q.expected
        
        results = score_test(test, responses)
        
        assert results.total == test.question_count
        assert results.correct > 0
        assert float(results.accuracy) > 0
    
    def test_empty_responses(self):
        """Should handle empty responses gracefully."""
        test = generate_test("L1", seed="empty_test")
        results = score_test(test, {})
        
        assert results.correct == 0
        assert results.total == test.question_count


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_score_empty_response(self, sample_positive_question):
        """Empty response should not crash."""
        result = score_response(sample_positive_question, "")
        assert result.correct is False
    
    def test_score_none_response(self, sample_positive_question):
        """None response should not crash."""
        result = score_response(sample_positive_question, None)
        assert result.correct is False
    
    def test_unicode_in_response(self, sample_positive_question):
        """Unicode characters should be handled."""
        result = score_response(sample_positive_question, "Thai food 🍜")
        assert result.correct is True
    
    def test_very_long_response(self, sample_positive_question):
        """Very long responses should be handled."""
        long_response = "Thai " + "food " * 10000
        result = score_response(sample_positive_question, long_response)
        assert result.correct is True


# =============================================================================
# Benchmark Marker
# =============================================================================

@pytest.mark.benchmark
class TestBenchmarks:
    """Performance benchmarks for the harness."""
    
    def test_generation_speed(self):
        """Test generation should be fast."""
        import time
        
        start = time.time()
        for i in range(10):
            generate_test("L1", seed=f"bench_{i}")
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"Generation too slow: {elapsed:.2f}s for 10 tests"
    
    def test_scoring_speed(self):
        """Scoring should be fast."""
        import time
        
        test = generate_test("L1", seed="score_bench")
        responses = {i: "Test response" for i in range(1, test.question_count + 1)}
        
        start = time.time()
        for _ in range(100):
            score_test(test, responses)
        elapsed = time.time() - start
        
        assert elapsed < 2.0, f"Scoring too slow: {elapsed:.2f}s for 100 scores"
