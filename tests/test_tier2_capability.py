"""Tier 2: Capability Tests

These tests measure improvement over baseline.
Success threshold: >30% improvement in aggregate.
"""

import pytest
from pathlib import Path

from src.tribalmemory.interfaces import MemorySource


class TestPreferencePrediction:
    """T2.1: Preference Prediction Tests"""
    
    @pytest.mark.tier2
    async def test_t2_1_1_basic_preference_recall(self, memory_service, test_data):
        """T2.1.1: Basic preference recall."""
        if not test_data.preferences:
            pytest.skip("No preference test data loaded")
        
        passed = 0
        total = 0
        
        for pref in test_data.preferences:
            # Skip negative test cases - they test absence of hallucination, not recall
            if pref.negative:
                continue
            
            total += 1
            
            # Store the preference
            result = await memory_service.remember(
                pref.stored_preference,
                tags=pref.tags
            )
            if not result.success:
                continue
            
            # Query for it
            results = await memory_service.recall(pref.query, limit=3)
            
            if results:
                # Check if any expected keyword is in recalled content
                recalled_content = " ".join(r.memory.content for r in results)
                if any(kw.lower() in recalled_content.lower() for kw in pref.expected_keywords):
                    passed += 1
        
        accuracy = passed / total if total > 0 else 0
        assert accuracy >= 0.9, f"Preference recall accuracy too low: {accuracy:.1%} ({passed}/{total})"
    
    @pytest.mark.tier2
    @pytest.mark.edge_case
    async def test_t2_1_2_contradictory_preferences(self, memory_service):
        """T2.1.2: Handling contradictory preferences over time."""
        # Store old preference
        await memory_service.remember(
            "Joe prefers dark mode",
            context="Week 1 preference"
        )
        
        # Store newer preference (correction)
        result = await memory_service.remember(
            "Joe now prefers light mode",
            source_type=MemorySource.CORRECTION,
            context="Week 2 preference"
        )
        
        # Query should return the newer preference or note contradiction
        results = await memory_service.recall("Does Joe prefer dark or light mode?")
        
        # At minimum, we should get results
        assert len(results) >= 1, "Should return at least one memory"
        
        # The most recent should be ranked higher or both returned
        contents = [r.memory.content for r in results]
        assert any("light" in c.lower() for c in contents), "Recent preference should be found"


class TestCrossSessionConsistency:
    """T2.2: Cross-Session Consistency Tests"""
    
    @pytest.mark.tier2
    async def test_t2_2_1_same_question_consistency(self, memory_service, test_data):
        """T2.2.1: Same question returns consistent answers."""
        if not test_data.consistency:
            pytest.skip("No consistency test data loaded")
        
        case = test_data.consistency[0]  # Test first case
        
        # Seed memories
        for memory in case.seed_memories:
            await memory_service.remember(memory)
        
        # Query multiple times with same question
        results_list = []
        for _ in range(5):
            results = await memory_service.recall(case.query_variations[0], limit=3)
            results_list.append(results)
        
        # All should return same memories (by ID)
        first_ids = set(r.memory.id for r in results_list[0])
        for results in results_list[1:]:
            result_ids = set(r.memory.id for r in results)
            overlap = len(first_ids & result_ids) / len(first_ids) if first_ids else 0
            assert overlap >= 0.8, f"Inconsistent results across queries: {overlap:.1%} overlap"
    
    @pytest.mark.tier2
    @pytest.mark.edge_case
    async def test_t2_2_2_paraphrased_questions(self, memory_service, test_data):
        """T2.2.2: Paraphrased questions return consistent answers."""
        if not test_data.consistency:
            pytest.skip("No consistency test data loaded")
        
        case = test_data.consistency[0]
        
        # Seed memories
        for memory in case.seed_memories:
            await memory_service.remember(memory)
        
        # Query with each variation
        all_results = []
        for query in case.query_variations:
            results = await memory_service.recall(query, limit=3)
            all_results.append(set(r.memory.id for r in results))
        
        # Check pairwise consistency
        if len(all_results) >= 2:
            first = all_results[0]
            for other in all_results[1:]:
                if first and other:
                    overlap = len(first & other) / max(len(first), len(other))
                    assert overlap >= 0.5, f"Paraphrase consistency too low: {overlap:.1%}"


class TestErrorCorrectionRetention:
    """T2.3: Error Correction Retention Tests"""
    
    @pytest.mark.tier2
    async def test_t2_3_1_basic_correction_retention(self, memory_service):
        """T2.3.1: Basic correction is retained and recalled."""
        # Store incorrect info
        original = await memory_service.remember("Joe's timezone is Eastern")
        
        # Store correction
        await memory_service.correct(
            original.memory_id,
            "Correction: Joe's timezone is Mountain",
            context="User provided correction"
        )
        
        # Query should return the correction
        results = await memory_service.recall("What's Joe's timezone?")
        
        assert len(results) >= 1
        # The correction should be in results
        contents = " ".join(r.memory.content for r in results)
        assert "Mountain" in contents, "Correction should be recalled"
    
    @pytest.mark.tier2
    @pytest.mark.edge_case
    async def test_t2_3_2_correction_chain(self, memory_service):
        """T2.3.2: Chain of corrections handled properly."""
        # Original
        r1 = await memory_service.remember("Timezone is Eastern")
        
        # First correction
        r2 = await memory_service.correct(
            r1.memory_id,
            "Correction: timezone is Mountain"
        )
        
        # Second correction
        r3 = await memory_service.correct(
            r2.memory_id,
            "Correction: actually Pacific for summer"
        )
        
        # Query should prefer most recent
        results = await memory_service.recall("What's Joe's timezone?")
        
        # Most recent correction should be findable
        contents = " ".join(r.memory.content for r in results)
        assert "Pacific" in contents or "summer" in contents, "Latest correction should be found"


class TestContextDependentTasks:
    """T2.4: Context-Dependent Task Tests"""
    
    @pytest.mark.tier2
    async def test_t2_4_1_historical_reference(self, memory_service, test_data):
        """T2.4.1: Historical references are retrievable."""
        if not test_data.context_tasks:
            pytest.skip("No context task test data loaded")
        
        case = test_data.context_tasks[0]
        
        # Store all memories for this case
        for memory in case.memories:
            await memory_service.remember(memory)
        
        # Query
        results = await memory_service.recall(case.query, limit=5)
        
        # Check how many expected keywords appear
        all_content = " ".join(r.memory.content for r in results)
        found_keywords = sum(1 for kw in case.expected_keywords if kw.lower() in all_content.lower())
        
        assert found_keywords >= len(case.expected_keywords) // 2, \
            f"Only found {found_keywords}/{len(case.expected_keywords)} expected keywords"
    
    @pytest.mark.tier2
    @pytest.mark.edge_case
    async def test_t2_4_3_multi_memory_synthesis(self, memory_service):
        """T2.4.3: Multiple memories synthesized for complex query."""
        # Store separate facts
        await memory_service.remember("Wally uses Next.js 14")
        await memory_service.remember("We prefer Tailwind for styling")
        await memory_service.remember("Supabase for backend")
        await memory_service.remember("Claude API for AI features")
        
        # Query that requires synthesis
        results = await memory_service.recall("Summarize Wally's tech stack", limit=5)
        
        # Should retrieve multiple relevant memories
        assert len(results) >= 2, "Should retrieve multiple memories for synthesis"
        
        # Check coverage
        all_content = " ".join(r.memory.content for r in results)
        expected = ["Next.js", "Tailwind", "Supabase"]
        found = sum(1 for term in expected if term in all_content)
        
        assert found >= 2, f"Multi-memory synthesis incomplete: found {found}/3 components"


class TestRetrievalPrecision:
    """T2.5: Retrieval Precision Tests"""
    
    @pytest.mark.tier2
    async def test_t2_5_1_relevance_rating(self, memory_service):
        """T2.5.1: Retrieved memories are relevant to query."""
        # Store diverse memories
        memories = [
            "Joe prefers TypeScript for web projects",
            "The weather today is sunny",
            "Next.js 14 uses App Router",
            "Cats are better than dogs",
            "Supabase provides PostgreSQL database",
        ]
        
        for m in memories:
            await memory_service.remember(m)
        
        # Query about tech stack
        results = await memory_service.recall("What tech stack does Joe use?", limit=3)
        
        # Relevant: TypeScript, Next.js, Supabase
        # Not relevant: weather, cats
        relevant_terms = ["TypeScript", "Next.js", "Supabase"]
        irrelevant_terms = ["weather", "sunny", "Cats", "dogs"]
        
        all_content = " ".join(r.memory.content for r in results)
        
        relevant_count = sum(1 for t in relevant_terms if t in all_content)
        irrelevant_count = sum(1 for t in irrelevant_terms if t in all_content)
        
        # Precision: relevant / (relevant + irrelevant) in results
        precision = relevant_count / (relevant_count + irrelevant_count) if (relevant_count + irrelevant_count) > 0 else 0
        
        assert precision >= 0.6, f"Precision too low: {precision:.1%}"
    
    @pytest.mark.tier2
    @pytest.mark.edge_case
    async def test_t2_5_4_forgotten_memory_not_returned(self, memory_service):
        """T2.5.4: Forgotten memories are not returned."""
        # Store and then forget
        result = await memory_service.remember("Secret information to forget")
        assert result.success
        
        # Forget it
        forgotten = await memory_service.forget(result.memory_id)
        assert forgotten
        
        # Query should not return it
        results = await memory_service.recall("Secret information")
        
        # The forgotten memory should not appear
        for r in results:
            assert r.memory.id != result.memory_id, "Forgotten memory was returned"
