# Tier 2 Memory Retrieval Test Harness

A comprehensive test suite for evaluating memory search accuracy against a known corpus. Supports both OpenClaw local memory (baseline) and Tribal Memory.

## Overview

This harness tests whether memory systems can correctly retrieve relevant information when asked natural language queries. It measures:

- **Overall accuracy** (% of queries finding expected content)
- **Category breakdown** (personal, work, health, family, etc.)
- **Difficulty analysis** (easy keyword matches vs. semantic understanding)
- **Failure patterns** (no results, wrong chunks, low scores)

## Directory Structure

```
tier2/
├── README.md                    # This file
├── lib-test-common.sh           # Shared test library
├── run-tests.sh                 # Baseline test runner (OpenClaw memory)
├── run-tests-tribal.sh          # Tribal Memory test runner
├── seed-tribal.sh               # Corpus seeder for Tribal Memory
├── dataset.json                 # Query dataset with expected results
├── results.json                 # Baseline results (generated)
├── results-tribal.json          # Tribal Memory results (generated)
├── RESULTS-COMPARISON-*.md      # Comparison reports
└── corpus/
    └── USER.md                  # Test corpus (~200 facts about "Alex Chen")
```

## Prerequisites

### Required Tools

```bash
# All tests require:
jq --version    # JSON processor
bc --version    # Calculator for score math

# Baseline tests require:
openclaw --version

# Tribal Memory tests require:
curl --version
# Tribal Memory server running on localhost:18790
```

### Installation

```bash
# Clone the repo (if not already)
git clone https://github.com/abbudjoe/tribal-memory.git
cd tribal-memory/tests/tier2

# Make scripts executable
chmod +x run-tests.sh run-tests-tribal.sh seed-tribal.sh lib-test-common.sh
```

## Quick Start

### Running Baseline Tests (OpenClaw Local Memory)

```bash
# 1. Setup test agent workspace
mkdir -p ~/clawd/test-baseline-workspace/memory
cp corpus/USER.md ~/clawd/test-baseline-workspace/memory/

# 2. Create agent (if not exists) and index
openclaw memory index --agent test-baseline

# 3. Run tests
./run-tests.sh

# 4. Results saved to results.json
```

### Running Tribal Memory Tests

```bash
# 1. Ensure Tribal Memory server is running
curl http://localhost:18790/v1/health

# 2. Seed the corpus into Tribal Memory
./seed-tribal.sh

# 3. Run tests
./run-tests-tribal.sh

# 4. Results saved to results-tribal.json
```

## Test Dataset

### Source: `dataset.json`

65 queries across 8 categories, manually curated with expected answers.

| Category | Count | Examples |
|----------|-------|----------|
| family | 10 | Wife's name, daughter's birthday, parents' location |
| health | 7 | Allergies, medications, exercise routine |
| work | 11 | Job title, manager, meeting schedule |
| preferences | 12 | Coffee order, favorite restaurant, hobbies |
| personal | 4 | Birthday, car, languages spoken |
| locations | 3 | Home address, office, gym |
| technical | 3 | Editor, programming languages, keyboard |
| imperative | 15 | Action commands requiring context |

### Difficulty Levels

- **Easy** (15): Direct keyword matches ("What is my wife's name?")
- **Medium** (23): Requires finding specific facts ("What days do I go into the office?")
- **Hard** (27): Semantic understanding needed ("What should I bring to a seafood restaurant?")

### Corpus: `corpus/USER.md`

A synthetic user profile for "Alex Chen" containing ~200 facts organized into sections:
- Personal Information
- Family & Relationships
- Food Preferences
- Health & Wellness
- Work & Career
- Technical Setup
- Locations & Addresses
- And more...

**Note:** This is synthetic test data, not real user information.

## Test Data Generation

### Creating New Test Queries

Edit `dataset.json` to add queries:

```json
{
  "id": "q066",
  "query": "Your natural language query",
  "expected": ["substring1", "substring2"],
  "category": "personal|work|health|family|preferences|locations|technical|imperative",
  "difficulty": "easy|medium|hard",
  "expected_behavior": "execute"
}
```

### Expanding the Corpus

Edit `corpus/USER.md` to add facts. Follow the existing format:
- Use markdown headers (`##`) for sections
- Use bullet points (`-`) for individual facts
- Use `**Bold:**` for fact labels

After modifying, re-run seeding:
```bash
# For Tribal Memory
./seed-tribal.sh

# For baseline
cp corpus/USER.md ~/clawd/test-baseline-workspace/memory/
openclaw memory index --agent test-baseline
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT` | `test-baseline` | Agent name for baseline tests |
| `DATASET` | `./dataset.json` | Path to query dataset |
| `RESULTS` | `./results.json` | Output file for results |
| `TOP_K` | `5` | Number of results to check |
| `TRIBAL_URL` | `http://localhost:18790` | Tribal Memory server URL |

### Example

```bash
# Test a different agent
AGENT=my-custom-agent ./run-tests.sh

# Use custom dataset
DATASET=./my-queries.json ./run-tests.sh

# Different Tribal Memory server
TRIBAL_URL=http://tribal.example.com:18790 ./run-tests-tribal.sh
```

## Results Format

```json
{
  "summary": {
    "total": 65,
    "passed": 53,
    "failed": 12,
    "score": 0.815
  },
  "by_category": {
    "family": { "passed": 10, "total": 10, "score": 1.0 },
    "preferences": { "passed": 11, "total": 12, "score": 0.91 }
  },
  "by_difficulty": {
    "easy": { "passed": 13, "total": 15, "score": 0.86 },
    "medium": { "passed": 22, "total": 23, "score": 0.95 },
    "hard": { "passed": 18, "total": 27, "score": 0.66 }
  },
  "failure_analysis": {
    "no_results": ["q041", "imp003", "imp015"],
    "wrong_chunk": ["q017", "imp005", "imp010"],
    "low_score": ["q010", "q015", "imp001"]
  },
  "queries": [/* detailed per-query results */]
}
```

## Scoring Rules

A query **PASSES** if:
1. Memory search returns at least one result
2. Any result contains at least one expected substring (case-insensitive)
3. At least one result has similarity score ≥ 0.5

A query **FAILS** with reason:
- `no_results`: Search returned empty results
- `wrong_chunk`: Results returned but none contain expected substrings
- `low_score`: Correct content found but similarity scores < 0.5

## Interpreting Results

| Score | Rating | Interpretation |
|-------|--------|----------------|
| 90%+ | Excellent | Memory system working great |
| 80-89% | Good | Minor retrieval gaps |
| 60-79% | Fair | Significant issues to address |
| <60% | Poor | Major problems with indexing/retrieval |

### Common Failure Patterns

1. **High no_results**: Corpus not indexed properly
2. **High wrong_chunk**: Chunking too coarse, information buried
3. **High low_score**: Embedding model struggling with query style
4. **Hard queries failing**: Need better semantic understanding

## Large Files Policy

The following files are committed to the repo for reproducibility:
- `dataset.json` (~14KB) - Query definitions
- `corpus/USER.md` (~8KB) - Test corpus
- `results.json` / `results-tribal.json` (~260KB each) - Latest test results

**Not committed:**
- Intermediate/temporary files
- Multiple historical result snapshots (keep latest only)

To reduce repo size, consider adding to `.gitignore`:
```
results-*.json
!results.json
!results-tribal.json
```

## Shared Test Library

Common logic is extracted to `lib-test-common.sh`:
- Color definitions
- Dependency checks
- Result formatting
- Summary generation

Both `run-tests.sh` and `run-tests-tribal.sh` source this library.

## Future Improvements

- [ ] CI automation (GitHub Actions)
- [ ] Shellcheck linting
- [ ] Python/TypeScript test runner for better maintainability
- [ ] Parallel query execution
- [ ] Historical result tracking

## License

Part of the tribal-memory project.
