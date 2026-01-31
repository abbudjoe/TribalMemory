"""BEIR (Benchmarking IR) adapter.

BEIR provides standardized evaluation for information retrieval systems
across diverse domains. Key metrics:
- nDCG@k: Normalized Discounted Cumulative Gain
- MAP: Mean Average Precision
- Recall@k: Recall at k documents
- Precision@k: Precision at k documents

Reference: https://github.com/beir-cellar/beir
Datasets: https://huggingface.co/datasets/BeIR
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class BEIRDataset(Enum):
    """Available BEIR datasets (subset relevant to memory systems)."""
    MSMARCO = "msmarco"              # Web passages
    NATURAL_QUESTIONS = "nq"         # Google search questions
    HOTPOTQA = "hotpotqa"            # Multi-hop reasoning
    FEVER = "fever"                  # Fact verification
    SCIFACT = "scifact"              # Scientific claims
    FIQA = "fiqa"                    # Financial QA
    ARGUANA = "arguana"              # Argument retrieval
    QUORA = "quora"                  # Duplicate questions
    

@dataclass
class BEIRMetrics:
    """Standard BEIR evaluation metrics."""
    ndcg_at_1: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    
    map_at_10: float = 0.0           # Mean Average Precision
    
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_100: float = 0.0
    
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    
    mrr: float = 0.0                 # Mean Reciprocal Rank
    
    @property
    def summary(self) -> dict:
        """Key metrics summary."""
        return {
            "nDCG@10": self.ndcg_at_10,
            "MAP@10": self.map_at_10,
            "Recall@10": self.recall_at_10,
            "MRR": self.mrr,
        }


@dataclass
class BEIRQuery:
    """A BEIR benchmark query."""
    query_id: str
    text: str
    relevant_doc_ids: list[str]  # Ground truth relevant documents
    relevance_scores: dict[str, int] = field(default_factory=dict)  # doc_id -> relevance (0-3)


@dataclass 
class BEIRDocument:
    """A BEIR benchmark document."""
    doc_id: str
    title: str
    text: str


class BEIRAdapter:
    """Adapter for running BEIR benchmarks against Tribal Memory.
    
    Usage:
        adapter = BEIRAdapter(memory_service)
        
        # Load documents into memory
        await adapter.index_documents(documents)
        
        # Run evaluation
        metrics = await adapter.evaluate(queries)
    """
    
    def __init__(self, memory_service):
        self.memory_service = memory_service
        self._doc_id_map: dict[str, str] = {}  # memory_id -> doc_id
    
    async def index_documents(self, documents: list[BEIRDocument]) -> int:
        """Index BEIR documents into memory service."""
        indexed = 0
        for doc in documents:
            # Combine title and text
            content = f"{doc.title}\n\n{doc.text}" if doc.title else doc.text
            
            result = await self.memory_service.remember(
                content,
                tags=[f"beir_doc:{doc.doc_id}"]
            )
            
            if result.success:
                self._doc_id_map[result.memory_id] = doc.doc_id
                indexed += 1
        
        return indexed
    
    async def evaluate(
        self,
        queries: list[BEIRQuery],
        k_values: list[int] = [1, 3, 5, 10, 100]
    ) -> BEIRMetrics:
        """Run BEIR evaluation on queries."""
        all_results = {}  # query_id -> [(doc_id, rank)]
        
        max_k = max(k_values)
        
        for query in queries:
            # Retrieve using memory service
            results = await self.memory_service.recall(query.text, limit=max_k)
            
            # Map back to BEIR doc IDs
            ranked_docs = []
            for r in results:
                doc_id = self._doc_id_map.get(r.memory.id)
                if doc_id:
                    ranked_docs.append(doc_id)
            
            all_results[query.query_id] = ranked_docs
        
        # Calculate metrics
        return self._calculate_metrics(queries, all_results, k_values)
    
    def _calculate_metrics(
        self,
        queries: list[BEIRQuery],
        results: dict[str, list[str]],
        k_values: list[int]
    ) -> BEIRMetrics:
        """Calculate all BEIR metrics."""
        metrics = BEIRMetrics()
        
        ndcg_scores = {k: [] for k in k_values}
        recall_scores = {k: [] for k in k_values}
        precision_scores = {k: [] for k in k_values}
        ap_scores = []
        rr_scores = []
        
        for query in queries:
            retrieved = results.get(query.query_id, [])
            relevant = set(query.relevant_doc_ids)
            relevance = query.relevance_scores or {d: 1 for d in relevant}
            
            # Calculate metrics for each k
            for k in k_values:
                top_k = retrieved[:k]
                
                # nDCG@k
                ndcg = self._ndcg_at_k(top_k, relevance, k)
                ndcg_scores[k].append(ndcg)
                
                # Recall@k
                recall = len(set(top_k) & relevant) / max(len(relevant), 1)
                recall_scores[k].append(recall)
                
                # Precision@k
                precision = len(set(top_k) & relevant) / k
                precision_scores[k].append(precision)
            
            # MAP (using k=10)
            ap = self._average_precision(retrieved[:10], relevant)
            ap_scores.append(ap)
            
            # MRR
            rr = self._reciprocal_rank(retrieved, relevant)
            rr_scores.append(rr)
        
        # Aggregate
        def mean(lst): return sum(lst) / len(lst) if lst else 0.0
        
        metrics.ndcg_at_1 = mean(ndcg_scores.get(1, []))
        metrics.ndcg_at_3 = mean(ndcg_scores.get(3, []))
        metrics.ndcg_at_5 = mean(ndcg_scores.get(5, []))
        metrics.ndcg_at_10 = mean(ndcg_scores.get(10, []))
        
        metrics.recall_at_1 = mean(recall_scores.get(1, []))
        metrics.recall_at_3 = mean(recall_scores.get(3, []))
        metrics.recall_at_5 = mean(recall_scores.get(5, []))
        metrics.recall_at_10 = mean(recall_scores.get(10, []))
        metrics.recall_at_100 = mean(recall_scores.get(100, []))
        
        metrics.precision_at_1 = mean(precision_scores.get(1, []))
        metrics.precision_at_3 = mean(precision_scores.get(3, []))
        metrics.precision_at_5 = mean(precision_scores.get(5, []))
        metrics.precision_at_10 = mean(precision_scores.get(10, []))
        
        metrics.map_at_10 = mean(ap_scores)
        metrics.mrr = mean(rr_scores)
        
        return metrics
    
    def _ndcg_at_k(
        self,
        retrieved: list[str],
        relevance: dict[str, int],
        k: int
    ) -> float:
        """Calculate nDCG@k."""
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            rel = relevance.get(doc_id, 0)
            dcg += (2**rel - 1) / math.log2(i + 2)  # i+2 because positions are 1-indexed
        
        # IDCG (ideal DCG)
        ideal_rels = sorted(relevance.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += (2**rel - 1) / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _average_precision(
        self,
        retrieved: list[str],
        relevant: set[str]
    ) -> float:
        """Calculate Average Precision."""
        if not relevant:
            return 0.0
        
        hits = 0
        sum_precision = 0.0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precision += precision_at_i
        
        return sum_precision / len(relevant)
    
    def _reciprocal_rank(
        self,
        retrieved: list[str],
        relevant: set[str]
    ) -> float:
        """Calculate Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0


async def load_beir_dataset(
    dataset: BEIRDataset,
    split: str = "test",
    max_docs: Optional[int] = None,
    max_queries: Optional[int] = None
) -> tuple[list[BEIRDocument], list[BEIRQuery]]:
    """Load a BEIR dataset from HuggingFace.
    
    Requires: pip install datasets
    
    Returns: (documents, queries)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")
    
    # Load corpus
    corpus = load_dataset(f"BeIR/{dataset.value}", "corpus", split="corpus")
    documents = []
    for i, item in enumerate(corpus):
        if max_docs and i >= max_docs:
            break
        documents.append(BEIRDocument(
            doc_id=item["_id"],
            title=item.get("title", ""),
            text=item["text"]
        ))
    
    # Load queries
    queries_ds = load_dataset(f"BeIR/{dataset.value}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{dataset.value}", "qrels", split=split)
    
    # Build relevance mapping
    relevance_map = {}  # query_id -> {doc_id: score}
    for item in qrels:
        qid = item["query-id"]
        if qid not in relevance_map:
            relevance_map[qid] = {}
        relevance_map[qid][item["corpus-id"]] = item["score"]
    
    queries = []
    for i, item in enumerate(queries_ds):
        if max_queries and i >= max_queries:
            break
        qid = item["_id"]
        if qid in relevance_map:
            rels = relevance_map[qid]
            queries.append(BEIRQuery(
                query_id=qid,
                text=item["text"],
                relevant_doc_ids=[d for d, s in rels.items() if s > 0],
                relevance_scores=rels
            ))
    
    return documents, queries
