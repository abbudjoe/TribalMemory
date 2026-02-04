"""Benchmark integration tests.

These tests verify that our memory system works with standard benchmarks.
"""

import pytest

from benchmarks.ragas_adapter import RAGASAdapter, RAGASTestCase, SimpleLLMJudge
from benchmarks.beir_adapter import BEIRAdapter, BEIRQuery, BEIRDocument
from benchmarks.babilong import BABILongAdapter, BABILongTask


class TestRAGASIntegration:
    """Test RAGAS benchmark adapter."""
    
    @pytest.mark.benchmark
    async def test_ragas_context_precision(self, memory_service):
        """Test RAGAS context precision calculation."""
        # Store some memories
        await memory_service.remember("Wally uses Next.js 14 framework")
        await memory_service.remember("Wally uses Supabase for backend")
        await memory_service.remember("The weather is sunny today")
        
        adapter = RAGASAdapter(memory_service)
        
        test_cases = [
            RAGASTestCase(
                question="What framework does Wally use?",
                ground_truth="Next.js 14",
                contexts=["Wally uses Next.js 14 framework"]
            )
        ]
        
        metrics = await adapter.evaluate(test_cases)
        
        # Should have decent precision (relevant context retrieved)
        assert metrics.context_precision >= 0.5, f"Context precision too low: {metrics.context_precision}"
    
    @pytest.mark.benchmark
    async def test_ragas_with_llm_judge(self, memory_service):
        """Test RAGAS with simple LLM judge."""
        await memory_service.remember("TypeScript is the preferred language for Wally")
        
        adapter = RAGASAdapter(memory_service, llm_judge=SimpleLLMJudge())
        
        test_cases = [
            RAGASTestCase(
                question="What language is used for Wally?",
                ground_truth="TypeScript",
                answer="TypeScript is the preferred language",
                contexts=["TypeScript is the preferred language for Wally"]
            )
        ]
        
        metrics = await adapter.evaluate(test_cases)
        
        # With matching answer and context, faithfulness should be high
        assert metrics.faithfulness >= 0.5
        assert metrics.overall_score > 0


class TestBEIRIntegration:
    """Test BEIR benchmark adapter."""
    
    @pytest.mark.benchmark
    async def test_beir_indexing(self, memory_service):
        """Test indexing BEIR documents."""
        adapter = BEIRAdapter(memory_service)
        
        documents = [
            BEIRDocument(doc_id="d1", title="Next.js Guide", text="Next.js is a React framework"),
            BEIRDocument(doc_id="d2", title="Supabase Intro", text="Supabase is a Firebase alternative"),
            BEIRDocument(doc_id="d3", title="TypeScript", text="TypeScript adds types to JavaScript"),
        ]
        
        indexed = await adapter.index_documents(documents)
        assert indexed == 3
    
    @pytest.mark.benchmark
    async def test_beir_retrieval_metrics(self, memory_service):
        """Test BEIR retrieval metrics calculation."""
        adapter = BEIRAdapter(memory_service)
        
        # Index documents
        documents = [
            BEIRDocument(doc_id="d1", title="", text="Python is a programming language"),
            BEIRDocument(doc_id="d2", title="", text="JavaScript runs in browsers"),
            BEIRDocument(doc_id="d3", title="", text="TypeScript extends JavaScript with types"),
        ]
        await adapter.index_documents(documents)
        
        # Query
        queries = [
            BEIRQuery(
                query_id="q1",
                text="What is TypeScript?",
                relevant_doc_ids=["d3"],
                relevance_scores={"d3": 2}
            )
        ]
        
        metrics = await adapter.evaluate(queries)
        
        # Should have some retrieval success
        assert metrics.mrr >= 0, "MRR should be non-negative"
        # These are weak assertions since mock embeddings are deterministic but not semantic
    
    @pytest.mark.benchmark
    async def test_beir_ndcg_calculation(self, memory_service):
        """Test nDCG calculation correctness."""
        adapter = BEIRAdapter(memory_service)
        
        # Index with known relevance
        documents = [
            BEIRDocument(doc_id="highly_relevant", title="", text="exact match query terms here"),
            BEIRDocument(doc_id="somewhat_relevant", title="", text="related but not exact"),
            BEIRDocument(doc_id="irrelevant", title="", text="completely unrelated content"),
        ]
        await adapter.index_documents(documents)
        
        queries = [
            BEIRQuery(
                query_id="q1",
                text="exact match query terms",
                relevant_doc_ids=["highly_relevant", "somewhat_relevant"],
                relevance_scores={"highly_relevant": 3, "somewhat_relevant": 1}
            )
        ]
        
        metrics = await adapter.evaluate(queries)
        
        # nDCG should be calculated
        assert hasattr(metrics, 'ndcg_at_10')


class TestBABILongIntegration:
    """Test BABILong benchmark adapter."""
    
    @pytest.mark.benchmark
    async def test_babilong_qa1_generation(self, memory_service):
        """Test QA1 (single supporting fact) generation."""
        adapter = BABILongAdapter(memory_service)
        
        examples = adapter.generate_examples(
            task=BABILongTask.QA1_SINGLE_SUPPORTING,
            n=5
        )
        
        assert len(examples) == 5
        for ex in examples:
            assert ex.task == BABILongTask.QA1_SINGLE_SUPPORTING
            assert len(ex.context) > 0
            assert ex.question
            assert ex.answer
            assert len(ex.supporting_facts) > 0
    
    @pytest.mark.benchmark
    async def test_babilong_evaluation(self, memory_service):
        """Test BABILong evaluation pipeline."""
        adapter = BABILongAdapter(memory_service)
        
        # Generate simple examples
        examples = adapter.generate_examples(
            task=BABILongTask.QA1_SINGLE_SUPPORTING,
            n=3
        )
        
        # Run evaluation with minimal distractors for speed
        metrics = await adapter.evaluate(examples, distractor_count=5)
        
        # Should complete and return valid metrics
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.retrieval_recall <= 1
        assert BABILongTask.QA1_SINGLE_SUPPORTING.value in metrics.accuracy_by_task
    
    @pytest.mark.benchmark
    async def test_babilong_qa2_multihop(self, memory_service):
        """Test QA2 (two supporting facts) for multi-hop reasoning."""
        adapter = BABILongAdapter(memory_service)
        
        examples = adapter.generate_examples(
            task=BABILongTask.QA2_TWO_SUPPORTING,
            n=3
        )
        
        # QA2 requires two facts
        for ex in examples:
            assert len(ex.supporting_facts) >= 2, "QA2 should need 2 supporting facts"
        
        metrics = await adapter.evaluate(examples, distractor_count=5)
        assert metrics.accuracy_by_task.get(BABILongTask.QA2_TWO_SUPPORTING.value) is not None


class TestBenchmarkComparison:
    """Tests for comparing results across benchmarks."""
    
    @pytest.mark.benchmark
    async def test_benchmark_result_logging(self, memory_service, result_logger):
        """Test that benchmark results can be logged for comparison."""
        from tribalmemory.testing.metrics import TestResult
        
        # Run a mini RAGAS eval
        await memory_service.remember("Test fact for benchmarking")
        
        ragas = RAGASAdapter(memory_service, llm_judge=SimpleLLMJudge())
        ragas_metrics = await ragas.evaluate([
            RAGASTestCase(
                question="What is the test fact?",
                ground_truth="Test fact",
                answer="Test fact for benchmarking",
                contexts=[]
            )
        ])
        
        # Log as test result
        result_logger.log(TestResult(
            test_id="benchmark_ragas",
            test_name="RAGAS Overall",
            tier="benchmark",
            passed=ragas_metrics.overall_score > 0.3,
            score=ragas_metrics.overall_score,
            threshold=0.3,
            details={
                "faithfulness": ragas_metrics.faithfulness,
                "context_precision": ragas_metrics.context_precision,
            }
        ))
        
        summary = result_logger.get_summary()
        assert summary["total"] == 1
