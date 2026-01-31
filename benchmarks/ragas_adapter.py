"""RAGAS (RAG Assessment) benchmark adapter.

RAGAS provides standardized metrics for evaluating RAG pipelines:
- Faithfulness: Is the answer grounded in retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are retrieved contexts relevant?
- Context Recall: Are all necessary contexts retrieved?

Reference: https://github.com/explodinggradients/ragas
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol
import math


@dataclass
class RAGASMetrics:
    """RAGAS evaluation metrics."""
    faithfulness: float = 0.0          # 0-1, higher is better
    answer_relevancy: float = 0.0       # 0-1, higher is better
    context_precision: float = 0.0      # 0-1, higher is better
    context_recall: float = 0.0         # 0-1, higher is better
    
    # Aggregate score
    @property
    def overall_score(self) -> float:
        """Harmonic mean of all metrics."""
        metrics = [self.faithfulness, self.answer_relevancy, 
                   self.context_precision, self.context_recall]
        non_zero = [m for m in metrics if m > 0]
        if not non_zero:
            return 0.0
        return len(non_zero) / sum(1/m for m in non_zero)


@dataclass
class RAGASTestCase:
    """A single RAGAS test case."""
    question: str
    ground_truth: str  # Expected answer
    contexts: list[str] = field(default_factory=list)  # Retrieved contexts
    answer: Optional[str] = None  # Generated answer


class LLMJudge(Protocol):
    """Protocol for LLM-based evaluation."""
    
    async def evaluate_faithfulness(
        self, 
        answer: str, 
        contexts: list[str]
    ) -> float:
        """Score how well the answer is grounded in contexts."""
        ...
    
    async def evaluate_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """Score how relevant the answer is to the question."""
        ...


class RAGASAdapter:
    """Adapter for running RAGAS benchmarks against Tribal Memory.
    
    Usage:
        adapter = RAGASAdapter(memory_service, llm_judge)
        results = await adapter.evaluate(test_cases)
    """
    
    def __init__(
        self,
        memory_service,  # IMemoryService
        llm_judge: Optional[LLMJudge] = None
    ):
        self.memory_service = memory_service
        self.llm_judge = llm_judge
    
    async def evaluate(
        self,
        test_cases: list[RAGASTestCase],
        top_k: int = 5
    ) -> RAGASMetrics:
        """Run RAGAS evaluation on test cases."""
        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []
        recall_scores = []
        
        for case in test_cases:
            # Retrieve contexts using our memory service
            results = await self.memory_service.recall(case.question, limit=top_k)
            retrieved_contexts = [r.memory.content for r in results]
            
            # Context Precision: How many retrieved contexts are relevant?
            precision = self._calculate_context_precision(
                retrieved_contexts, 
                case.ground_truth,
                case.contexts  # Ground truth contexts if available
            )
            precision_scores.append(precision)
            
            # Context Recall: Did we retrieve necessary contexts?
            recall = self._calculate_context_recall(
                retrieved_contexts,
                case.contexts  # Required contexts
            )
            recall_scores.append(recall)
            
            # Faithfulness and Answer Relevancy require LLM judge
            if self.llm_judge and case.answer:
                faith = await self.llm_judge.evaluate_faithfulness(
                    case.answer, 
                    retrieved_contexts
                )
                faithfulness_scores.append(faith)
                
                rel = await self.llm_judge.evaluate_relevancy(
                    case.question,
                    case.answer
                )
                relevancy_scores.append(rel)
        
        return RAGASMetrics(
            faithfulness=self._mean(faithfulness_scores),
            answer_relevancy=self._mean(relevancy_scores),
            context_precision=self._mean(precision_scores),
            context_recall=self._mean(recall_scores)
        )
    
    def _calculate_context_precision(
        self,
        retrieved: list[str],
        ground_truth_answer: str,
        ground_truth_contexts: list[str]
    ) -> float:
        """Calculate precision of retrieved contexts.
        
        If ground truth contexts provided, use exact matching.
        Otherwise, use keyword overlap with answer.
        """
        if not retrieved:
            return 0.0
        
        if ground_truth_contexts:
            # Exact context matching
            relevant = sum(
                1 for r in retrieved 
                if any(self._similar(r, gt) for gt in ground_truth_contexts)
            )
            return relevant / len(retrieved)
        else:
            # Keyword overlap heuristic
            answer_words = set(ground_truth_answer.lower().split())
            relevant = 0
            for ctx in retrieved:
                ctx_words = set(ctx.lower().split())
                overlap = len(answer_words & ctx_words) / max(len(answer_words), 1)
                if overlap > 0.2:  # 20% word overlap threshold
                    relevant += 1
            return relevant / len(retrieved)
    
    def _calculate_context_recall(
        self,
        retrieved: list[str],
        required_contexts: list[str]
    ) -> float:
        """Calculate recall - did we get all required contexts?"""
        if not required_contexts:
            return 1.0  # No required contexts means perfect recall
        
        found = sum(
            1 for req in required_contexts
            if any(self._similar(req, ret) for ret in retrieved)
        )
        return found / len(required_contexts)
    
    def _similar(self, a: str, b: str, threshold: float = 0.8) -> bool:
        """Check if two strings are similar (word overlap)."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b)
        jaccard = overlap / len(words_a | words_b)
        return jaccard >= threshold
    
    @staticmethod
    def _mean(scores: list[float]) -> float:
        """Calculate mean, handling empty list."""
        return sum(scores) / len(scores) if scores else 0.0


class SimpleLLMJudge:
    """Simple heuristic-based judge for testing without LLM calls.
    
    For production, replace with actual LLM calls.
    """
    
    async def evaluate_faithfulness(
        self,
        answer: str,
        contexts: list[str]
    ) -> float:
        """Heuristic: Check if answer words appear in contexts."""
        answer_words = set(answer.lower().split())
        context_words = set()
        for ctx in contexts:
            context_words.update(ctx.lower().split())
        
        if not answer_words:
            return 0.0
        
        grounded = len(answer_words & context_words)
        return grounded / len(answer_words)
    
    async def evaluate_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """Heuristic: Check word overlap between question and answer."""
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who'}
        q_words -= stopwords
        a_words -= stopwords
        
        if not q_words:
            return 0.5  # Can't evaluate
        
        overlap = len(q_words & a_words)
        return min(1.0, overlap / len(q_words) + 0.3)  # Bias toward relevance
