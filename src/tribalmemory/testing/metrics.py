"""Metrics collection and analysis for testing."""

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    operation: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    metadata: dict = field(default_factory=dict)


class LatencyTracker:
    """Track and analyze operation latencies."""
    
    def __init__(self):
        self._measurements: list[LatencyMeasurement] = []
        self._start_times: dict[str, float] = {}
    
    def start(self, operation: str) -> str:
        """Start timing an operation. Returns operation key."""
        key = f"{operation}_{time.perf_counter_ns()}"
        self._start_times[key] = time.perf_counter()
        return key
    
    def stop(self, key: str, success: bool = True, **metadata) -> float:
        """Stop timing and record measurement. Returns duration in ms."""
        if key not in self._start_times:
            raise ValueError(f"Unknown operation key: {key}")
        
        duration_ms = (time.perf_counter() - self._start_times[key]) * 1000
        operation = key.rsplit('_', 1)[0]
        
        self._measurements.append(LatencyMeasurement(
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata
        ))
        
        del self._start_times[key]
        return duration_ms
    
    def record(self, operation: str, duration_ms: float, success: bool = True, **metadata):
        """Record a measurement directly."""
        self._measurements.append(LatencyMeasurement(
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata
        ))
    
    def get_stats(self, operation: Optional[str] = None) -> dict:
        """Get statistics for an operation or all operations."""
        measurements = self._measurements
        if operation:
            measurements = [m for m in measurements if m.operation == operation]
        
        if not measurements:
            return {"count": 0}
        
        durations = [m.duration_ms for m in measurements]
        successful = [m for m in measurements if m.success]
        
        sorted_durations = sorted(durations)
        
        return {
            "count": len(measurements),
            "success_rate": len(successful) / len(measurements),
            "mean_ms": statistics.mean(durations),
            "median_ms": statistics.median(durations),
            "stdev_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
            "min_ms": min(durations),
            "max_ms": max(durations),
            "p95_ms": sorted_durations[int(len(sorted_durations) * 0.95)] if len(sorted_durations) > 1 else sorted_durations[0],
            "p99_ms": sorted_durations[int(len(sorted_durations) * 0.99)] if len(sorted_durations) > 1 else sorted_durations[0],
        }
    
    def clear(self):
        """Clear all measurements."""
        self._measurements.clear()
        self._start_times.clear()


class SimilarityCalculator:
    """Calculate and compare embedding similarities."""
    
    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            raise ValueError("Vectors must have same dimension")
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    @staticmethod
    def euclidean_distance(a: list[float], b: list[float]) -> float:
        """Calculate Euclidean distance between two vectors."""
        if len(a) != len(b):
            raise ValueError("Vectors must have same dimension")
        
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    @staticmethod
    def batch_similarities(
        query: list[float],
        candidates: list[list[float]]
    ) -> list[float]:
        """Calculate similarities between query and multiple candidates."""
        return [
            SimilarityCalculator.cosine_similarity(query, c)
            for c in candidates
        ]


@dataclass
class TestResult:
    """Result of a single test case."""
    test_id: str
    test_name: str
    tier: str
    passed: bool
    score: Optional[float] = None
    threshold: Optional[float] = None
    duration_ms: float = 0
    error: Optional[str] = None
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TestResultLogger:
    """Log and persist test results for analysis."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("test-results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results: list[TestResult] = []
        self._run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    def log(self, result: TestResult):
        """Log a test result."""
        self._results.append(result)
    
    def get_summary(self) -> dict:
        """Get summary of all test results."""
        if not self._results:
            return {"total": 0}
        
        by_tier = {}
        for r in self._results:
            if r.tier not in by_tier:
                by_tier[r.tier] = {"passed": 0, "failed": 0}
            if r.passed:
                by_tier[r.tier]["passed"] += 1
            else:
                by_tier[r.tier]["failed"] += 1
        
        passed = sum(1 for r in self._results if r.passed)
        failed = len(self._results) - passed
        
        return {
            "run_id": self._run_id,
            "total": len(self._results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self._results),
            "by_tier": by_tier,
            "failed_tests": [
                {"id": r.test_id, "name": r.test_name, "error": r.error}
                for r in self._results if not r.passed
            ]
        }
    
    def save(self, filename: Optional[str] = None):
        """Save results to JSON file."""
        filename = filename or f"results_{self._run_id}.json"
        filepath = self.output_dir / filename
        
        data = {
            "run_id": self._run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": self.get_summary(),
            "results": [
                {
                    "test_id": r.test_id,
                    "test_name": r.test_name,
                    "tier": r.tier,
                    "passed": r.passed,
                    "score": r.score,
                    "threshold": r.threshold,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self._results
            ]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def compare_to_baseline(self, baseline_path: Path) -> dict:
        """Compare current results to a baseline."""
        with open(baseline_path) as f:
            baseline = json.load(f)
        
        current_by_id = {r.test_id: r for r in self._results}
        baseline_by_id = {r["test_id"]: r for r in baseline["results"]}
        
        regressions = []
        improvements = []
        new_tests = []
        
        for test_id, result in current_by_id.items():
            if test_id not in baseline_by_id:
                new_tests.append(test_id)
                continue
            
            baseline_result = baseline_by_id[test_id]
            
            if result.passed and not baseline_result["passed"]:
                improvements.append(test_id)
            elif not result.passed and baseline_result["passed"]:
                regressions.append(test_id)
        
        return {
            "regressions": regressions,
            "improvements": improvements,
            "new_tests": new_tests,
            "baseline_pass_rate": baseline["summary"]["pass_rate"],
            "current_pass_rate": self.get_summary()["pass_rate"],
        }
    
    def clear(self):
        """Clear all results."""
        self._results.clear()
