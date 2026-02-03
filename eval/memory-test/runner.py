#!/usr/bin/env python3
"""
Automated Memory Recall Test Runner

Orchestrates running tests across multiple levels with multiple runs,
aggregates statistics, and produces a comprehensive report.

Usage:
    python runner.py quick       - Quick test (L1 x 3)
    python runner.py standard    - Standard test (L1-L4 x 3)
    python runner.py full        - Full test (L1-L7 x 5)
    python runner.py level <L>   - Single level x 5
    python runner.py report      - Generate report from existing results
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"


# =============================================================================
# Test Configurations
# =============================================================================

CONFIGS = {
    "quick": {
        "levels": ["L1"],
        "runs_per_level": 3,
        "description": "Quick validation (comprehension only)"
    },
    "standard": {
        "levels": ["L1", "L2", "L3", "L4"],
        "runs_per_level": 3,
        "description": "Standard test (levels 1-4)"
    },
    "full": {
        "levels": ["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
        "runs_per_level": 5,
        "description": "Full test suite (all levels)"
    }
}

# Expected performance benchmarks
# These establish minimum and target accuracy for each level
EXPECTED = {
    "L1": {"min": 95, "target": 100, "description": "Comprehension ceiling"},
    "L2": {"min": 90, "target": 98, "description": "File reading"},
    "L3": {"min": 80, "target": 92, "description": "Memory search"},
    "L4": {"min": 70, "target": 85, "description": "Noisy retrieval"},
    "L5": {"min": 65, "target": 80, "description": "Cross-reference synthesis"},
    "L6": {"min": 65, "target": 80, "description": "Temporal reasoning"},
    "L7": {"min": 55, "target": 75, "description": "Adversarial robustness"},
}


# =============================================================================
# Statistics
# =============================================================================

def mean(values: list[float]) -> float:
    """Calculate arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def std_dev(values: list[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    squared_diffs = [(v - m) ** 2 for v in values]
    return math.sqrt(mean(squared_diffs))


# =============================================================================
# Results Loading and Aggregation
# =============================================================================

def load_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Load all scored result files from the results directory."""
    if not results_dir.exists():
        return []
    
    results = []
    for f in sorted(results_dir.iterdir()):
        if f.name.startswith("results-") and f.name.endswith("-scored.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                results.append(data)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: Could not load {f}: {e}", file=sys.stderr)
    
    return results


def aggregate_results(results: list[dict]) -> dict[str, dict]:
    """Aggregate results by level with statistics."""
    by_level: dict[str, dict] = {}
    
    for r in results:
        level = r.get("level", "unknown")
        
        if level not in by_level:
            by_level[level] = {
                "level": level,
                "runs": 0,
                "accuracies": [],
                "positive_accuracies": [],
                "negative_accuracies": [],
                "hallucination_rates": [],
                "details": []
            }
        
        level_data = by_level[level]
        level_data["runs"] += 1
        
        # Handle both string and float accuracy values
        accuracy = r.get("accuracy", 0)
        if isinstance(accuracy, str):
            accuracy = float(accuracy)
        level_data["accuracies"].append(accuracy)
        
        # Type-specific stats
        scores = r.get("scores", {})
        
        positive = scores.get("positive", {})
        if positive.get("total", 0) > 0:
            pos_acc = positive["correct"] / positive["total"] * 100
            level_data["positive_accuracies"].append(pos_acc)
        
        negative = scores.get("negative", {})
        if negative.get("total", 0) > 0:
            neg_acc = negative["correct"] / negative["total"] * 100
            level_data["negative_accuracies"].append(neg_acc)
            
            hall_rate = r.get("hallucinationRate") or r.get("hallucination_rate")
            if hall_rate:
                if isinstance(hall_rate, str):
                    hall_rate = float(hall_rate)
                level_data["hallucination_rates"].append(hall_rate)
        
        level_data["details"].append({
            "seed": r.get("seed"),
            "accuracy": accuracy,
            "timestamp": r.get("timestamp")
        })
    
    # Calculate statistics for each level
    for level_data in by_level.values():
        accs = level_data["accuracies"]
        level_data["mean"] = mean(accs)
        level_data["std_dev"] = std_dev(accs)
        level_data["min"] = min(accs) if accs else 0
        level_data["max"] = max(accs) if accs else 0
        level_data["positive_recall"] = mean(level_data["positive_accuracies"])
        level_data["negative_recall"] = mean(level_data["negative_accuracies"])
        level_data["hallucination_rate"] = mean(level_data["hallucination_rates"])
    
    return by_level


# =============================================================================
# Report Generation
# =============================================================================

def format_report(aggregate: dict[str, dict]) -> str:
    """Generate a markdown report from aggregated results."""
    lines = [
        "# Memory Recall Evaluation Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary by Level",
        "",
        "| Level | Runs | Mean ± StdDev | Range | Positive | Negative | Halluc. | Status |",
        "|-------|------|---------------|-------|----------|----------|---------|--------|"
    ]
    
    ordered_levels = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    
    for level_name in ordered_levels:
        level = aggregate.get(level_name)
        if not level:
            continue
        
        exp = EXPECTED.get(level_name, {"min": 0, "target": 100})
        
        if level["mean"] >= exp["target"]:
            status = "✅"
        elif level["mean"] >= exp["min"]:
            status = "⚠️"
        else:
            status = "❌"
        
        lines.append(
            f"| {level_name} | {level['runs']} | "
            f"{level['mean']:.1f}% ± {level['std_dev']:.1f} | "
            f"{level['min']:.0f}-{level['max']:.0f}% | "
            f"{level['positive_recall']:.1f}% | "
            f"{level['negative_recall']:.1f}% | "
            f"{level['hallucination_rate']:.1f}% | {status} |"
        )
    
    lines.extend([
        "",
        "## Benchmarks",
        "",
        "| Level | Description | Min | Target |",
        "|-------|-------------|-----|--------|"
    ])
    
    for level, exp in EXPECTED.items():
        lines.append(f"| {level} | {exp['description']} | {exp['min']}% | {exp['target']}% |")
    
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- ✅ **Pass**: Met target benchmark",
        "- ⚠️ **Marginal**: Above minimum but below target",
        "- ❌ **Fail**: Below minimum acceptable threshold",
        "",
        "## Individual Run Details",
        ""
    ])
    
    for level_name in ordered_levels:
        level = aggregate.get(level_name)
        if not level:
            continue
        
        lines.append(f"### {level_name}")
        lines.append("")
        
        for run in level["details"]:
            acc = run["accuracy"]
            if isinstance(acc, float):
                acc = f"{acc:.1f}"
            lines.append(f"- {run['timestamp']}: {acc}% (seed: {run['seed']})")
        
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Test Plan Generation
# =============================================================================

def generate_test_plan(config_name: str) -> tuple[dict, list[dict]]:
    """Generate a test plan from a configuration."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    
    config = CONFIGS[config_name]
    plan = []
    
    for level in config["levels"]:
        for run in range(config["runs_per_level"]):
            plan.append({"level": level, "run": run + 1})
    
    return config, plan


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Memory Recall Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  quick      Run quick test (L1 × 3)
  standard   Run standard test (L1-L4 × 3)
  full       Run full test (L1-L7 × 5)
  level <L>  Run single level × 5
  report     Generate report from existing results

The runner generates test plans. Tests are executed via sessions_spawn
and scored using the harness.

Example workflow:
  1. python runner.py standard          # Generate plan
  2. Run tests via sessions_spawn       # Execute each test
  3. python runner.py report            # Aggregate results
        """
    )
    
    parser.add_argument("command", nargs="?", default="help",
                       help="Command: quick, standard, full, level, report")
    parser.add_argument("level", nargs="?", default="L1",
                       help="Level for 'level' command")
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.command in ("quick", "standard", "full"):
        config, plan = generate_test_plan(args.command)
        
        print(f"\n📊 Memory Recall Evaluation — {config['description']}")
        print(f"   Levels: {', '.join(config['levels'])}")
        print(f"   Runs per level: {config['runs_per_level']}")
        print(f"   Total tests: {len(plan)}")
        print("")
        
        # Save the test plan
        plan_file = RESULTS_DIR / f"plan-{args.command}-{int(datetime.now().timestamp())}.json"
        plan_data = {
            "config": config,
            "plan": plan,
            "started": datetime.now().isoformat()
        }
        plan_file.write_text(json.dumps(plan_data, indent=2), encoding="utf-8")
        
        print(f"Test plan saved to: {plan_file}")
        print("")
        print("To run tests manually:")
        for item in plan[:3]:
            print(f"  python harness.py generate {item['level']}")
        if len(plan) > 3:
            print(f"  ... ({len(plan) - 3} more)")
    
    elif args.command == "level":
        level = args.level
        print(f"\n📊 Testing Level {level} × 5 runs")
        
        plan = [{"level": level, "run": i + 1} for i in range(5)]
        
        plan_file = RESULTS_DIR / f"plan-{level}-{int(datetime.now().timestamp())}.json"
        plan_data = {
            "plan": plan,
            "started": datetime.now().isoformat()
        }
        plan_file.write_text(json.dumps(plan_data, indent=2), encoding="utf-8")
        
        print(f"Plan saved: {plan_file}")
    
    elif args.command == "report":
        results = load_results()
        
        if not results:
            print("No results found in results directory", file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(results)} result files")
        
        aggregate = aggregate_results(results)
        report = format_report(aggregate)
        
        report_file = RESULTS_DIR / f"report-{int(datetime.now().timestamp())}.md"
        report_file.write_text(report, encoding="utf-8")
        
        print(f"Report saved to: {report_file}")
        print("")
        print(report)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
