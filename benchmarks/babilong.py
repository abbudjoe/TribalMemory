"""BABILong benchmark adapter.

BABILong extends the bAbI tasks to test long-range memory and reasoning.
Key task types:
- Single Supporting Fact (qa1)
- Two Supporting Facts (qa2)
- Three Supporting Facts (qa3)
- Two Arg Relations (qa4)
- Three Arg Relations (qa5)
- Yes/No Questions (qa6)
- Counting (qa7)
- Lists/Sets (qa8)
- Simple Negation (qa9)
- Indefinite Knowledge (qa10)

Reference: https://github.com/booydar/babilong
Paper: "BABILong: Testing the Limits of LLMs and Transformers on Long Sequences"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random


class BABILongTask(Enum):
    """BABILong task types."""
    QA1_SINGLE_SUPPORTING = "qa1"
    QA2_TWO_SUPPORTING = "qa2"
    QA3_THREE_SUPPORTING = "qa3"
    QA4_TWO_ARG_RELATIONS = "qa4"
    QA5_THREE_ARG_RELATIONS = "qa5"
    QA6_YES_NO = "qa6"
    QA7_COUNTING = "qa7"
    QA8_LISTS_SETS = "qa8"
    QA9_SIMPLE_NEGATION = "qa9"
    QA10_INDEFINITE = "qa10"


@dataclass
class BABILongExample:
    """A single BABILong test example."""
    task: BABILongTask
    context: list[str]           # List of facts/statements
    question: str
    answer: str
    supporting_facts: list[int]  # Indices of facts needed to answer


@dataclass
class BABILongMetrics:
    """BABILong evaluation metrics."""
    accuracy: float = 0.0
    accuracy_by_task: dict[str, float] = field(default_factory=dict)
    retrieval_recall: float = 0.0  # Did we retrieve supporting facts?
    
    # Breakdown by context length
    accuracy_by_length: dict[str, float] = field(default_factory=dict)


class BABILongAdapter:
    """Adapter for running BABILong benchmarks against Tribal Memory.
    
    Tests long-range memory and multi-hop reasoning capabilities.
    
    Usage:
        adapter = BABILongAdapter(memory_service)
        
        # Generate or load examples
        examples = adapter.generate_examples(task=BABILongTask.QA1_SINGLE_SUPPORTING, n=100)
        
        # Run evaluation
        metrics = await adapter.evaluate(examples)
    """
    
    def __init__(self, memory_service):
        self.memory_service = memory_service
    
    async def evaluate(
        self,
        examples: list[BABILongExample],
        distractor_count: int = 100
    ) -> BABILongMetrics:
        """Run BABILong evaluation.
        
        For each example:
        1. Store all context facts as memories (with distractors)
        2. Query with the question
        3. Check if supporting facts are retrieved
        4. Check if answer can be derived
        """
        correct = 0
        retrieval_hits = 0
        by_task: dict[str, list[bool]] = {}
        by_length: dict[str, list[bool]] = {}
        
        for example in examples:
            # Clear previous memories (in real usage, would use separate namespace)
            # For testing, we work with what we have
            
            # Store context facts
            memory_ids = []
            for fact in example.context:
                result = await self.memory_service.remember(
                    fact,
                    tags=[f"babilong:{example.task.value}"]
                )
                if result.success:
                    memory_ids.append(result.memory_id)
            
            # Add distractors
            distractors = self._generate_distractors(distractor_count)
            for d in distractors:
                await self.memory_service.remember(d, tags=["babilong:distractor"])
            
            # Query
            results = await self.memory_service.recall(
                example.question,
                limit=10
            )
            
            # Check retrieval recall (did we get supporting facts?)
            retrieved_contents = {r.memory.content for r in results}
            supporting_contents = {example.context[i] for i in example.supporting_facts}
            retrieved_supporting = len(retrieved_contents & supporting_contents)
            
            if retrieved_supporting == len(supporting_contents):
                retrieval_hits += 1
            
            # Check if answer is derivable
            # (Simplified: check if answer appears in retrieved content)
            all_retrieved = " ".join(retrieved_contents).lower()
            is_correct = example.answer.lower() in all_retrieved
            
            if is_correct:
                correct += 1
            
            # Track by task
            task_name = example.task.value
            if task_name not in by_task:
                by_task[task_name] = []
            by_task[task_name].append(is_correct)
            
            # Track by context length
            length_bucket = self._length_bucket(len(example.context) + distractor_count)
            if length_bucket not in by_length:
                by_length[length_bucket] = []
            by_length[length_bucket].append(is_correct)
        
        total = len(examples)
        return BABILongMetrics(
            accuracy=correct / total if total > 0 else 0.0,
            accuracy_by_task={k: sum(v)/len(v) for k, v in by_task.items()},
            retrieval_recall=retrieval_hits / total if total > 0 else 0.0,
            accuracy_by_length={k: sum(v)/len(v) for k, v in by_length.items()}
        )
    
    def generate_examples(
        self,
        task: BABILongTask,
        n: int = 100
    ) -> list[BABILongExample]:
        """Generate synthetic BABILong examples.
        
        For real evaluation, load from the official dataset.
        These are simplified templates for testing.
        """
        examples = []
        
        generators = {
            BABILongTask.QA1_SINGLE_SUPPORTING: self._gen_qa1,
            BABILongTask.QA2_TWO_SUPPORTING: self._gen_qa2,
            BABILongTask.QA6_YES_NO: self._gen_qa6,
            BABILongTask.QA7_COUNTING: self._gen_qa7,
        }
        
        generator = generators.get(task, self._gen_qa1)
        
        for _ in range(n):
            examples.append(generator(task))
        
        return examples
    
    def _gen_qa1(self, task: BABILongTask) -> BABILongExample:
        """Generate QA1: Single Supporting Fact."""
        names = ["Mary", "John", "Sandra", "Daniel", "Emily"]
        locations = ["garden", "kitchen", "office", "bedroom", "bathroom"]
        
        name = random.choice(names)
        loc1 = random.choice(locations)
        loc2 = random.choice([l for l in locations if l != loc1])
        
        context = [
            f"{name} went to the {loc1}.",
            f"{name} moved to the {loc2}.",
        ]
        
        # Add some noise
        other_name = random.choice([n for n in names if n != name])
        noise_loc = random.choice(locations)
        context.insert(1, f"{other_name} went to the {noise_loc}.")
        
        return BABILongExample(
            task=task,
            context=context,
            question=f"Where is {name}?",
            answer=loc2,
            supporting_facts=[1]  # The second fact about this person
        )
    
    def _gen_qa2(self, task: BABILongTask) -> BABILongExample:
        """Generate QA2: Two Supporting Facts."""
        names = ["Mary", "John", "Sandra", "Daniel"]
        objects = ["football", "apple", "milk", "book"]
        locations = ["garden", "kitchen", "office", "bedroom"]
        
        name = random.choice(names)
        obj = random.choice(objects)
        loc = random.choice(locations)
        
        context = [
            f"{name} picked up the {obj}.",
            f"{name} went to the {loc}.",
        ]
        
        return BABILongExample(
            task=task,
            context=context,
            question=f"Where is the {obj}?",
            answer=loc,
            supporting_facts=[0, 1]
        )
    
    def _gen_qa6(self, task: BABILongTask) -> BABILongExample:
        """Generate QA6: Yes/No Questions."""
        names = ["Mary", "John", "Sandra", "Daniel"]
        locations = ["garden", "kitchen", "office", "bedroom"]
        
        name = random.choice(names)
        loc = random.choice(locations)
        ask_loc = random.choice(locations)
        
        context = [f"{name} is in the {loc}."]
        answer = "yes" if loc == ask_loc else "no"
        
        return BABILongExample(
            task=task,
            context=context,
            question=f"Is {name} in the {ask_loc}?",
            answer=answer,
            supporting_facts=[0]
        )
    
    def _gen_qa7(self, task: BABILongTask) -> BABILongExample:
        """Generate QA7: Counting."""
        names = ["Mary", "John", "Sandra", "Daniel"]
        objects = ["apples", "oranges", "footballs", "books"]
        
        name = random.choice(names)
        obj = random.choice(objects)
        
        # Generate pick up / drop events
        count = 0
        context = []
        for _ in range(random.randint(2, 5)):
            action = random.choice(["picked up", "dropped"])
            n = random.randint(1, 3)
            
            if action == "picked up":
                count += n
            else:
                count = max(0, count - n)
            
            context.append(f"{name} {action} {n} {obj}.")
        
        return BABILongExample(
            task=task,
            context=context,
            question=f"How many {obj} is {name} carrying?",
            answer=str(count),
            supporting_facts=list(range(len(context)))
        )
    
    def _generate_distractors(self, count: int) -> list[str]:
        """Generate distractor facts."""
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        locations = ["park", "school", "library", "store", "cafe"]
        actions = ["walked to", "ran to", "visited", "left"]
        
        distractors = []
        for _ in range(count):
            name = random.choice(names)
            action = random.choice(actions)
            loc = random.choice(locations)
            distractors.append(f"{name} {action} the {loc}.")
        
        return distractors
    
    def _length_bucket(self, length: int) -> str:
        """Categorize context length into buckets."""
        if length < 50:
            return "short (<50)"
        elif length < 200:
            return "medium (50-200)"
        elif length < 500:
            return "long (200-500)"
        else:
            return "very_long (500+)"


async def load_babilong_dataset(
    task: BABILongTask,
    split: str = "test",
    context_length: int = 1000
) -> list[BABILongExample]:
    """Load official BABILong dataset.
    
    Requires: pip install datasets
    
    Args:
        task: Which BABILong task
        split: train/test
        context_length: Target context length (128, 512, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")
    
    # BABILong uses specific context length configurations
    length_map = {
        128: "0k", 512: "0k", 1000: "1k", 2000: "2k",
        4000: "4k", 8000: "8k", 16000: "16k", 32000: "32k",
        64000: "64k", 128000: "128k"
    }
    
    length_key = length_map.get(context_length, "1k")
    
    # Load from HuggingFace
    ds = load_dataset(
        "booydar/babilong",
        f"{task.value}_{length_key}",
        split=split
    )
    
    examples = []
    for item in ds:
        examples.append(BABILongExample(
            task=task,
            context=item["context"].split("\n"),
            question=item["question"],
            answer=item["answer"],
            supporting_facts=item.get("supporting_facts", [])
        ))
    
    return examples
