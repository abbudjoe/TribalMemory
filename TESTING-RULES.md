# Tribal Memory Testing Rules

**MANDATORY** — Read before running ANY baseline or evaluation tests.

---

## Core Principles

1. **Never test against real user data**
2. **Always use anonymized, synthetic test cases**
3. **Isolate test environment from production**

---

## Test Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Main Clawdio (Orchestrator)                            │
│  - Spawns test sessions                                 │
│  - Sends test queries                                   │
│  - Scores responses                                     │
│  - Records results                                      │
└─────────────────────┬───────────────────────────────────┘
                      │ sessions_spawn(agentId: "test-baseline")
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Test Agent: "test-baseline"                            │
│  - Isolated workspace: ~/clawd/test-baseline-workspace/ │
│  - MEMORY.md = MEMORY_PERSON_A.md (anonymized)          │
│  - memory/*.md = synthetic test data                    │
│  - No access to real user data                          │
└─────────────────────────────────────────────────────────┘
```

---

## Test Data Requirements

### Allowed ✅
- Fictional personas (Person A, Person B, Person C)
- Synthetic preferences, facts, decisions
- Randomized generated content
- Anonymized project names (e.g., "Project Alpha" not "Wally")

### Forbidden ❌
- Real MEMORY.md from production workspace
- Real user names, preferences, or personal info
- Real project details, credentials, or infrastructure
- Any data that could identify actual users

---

## Test Corpus Location

```
~/clawd/eval/memory-test/corpus/
├── MEMORY_PERSON_A.md    # Primary test persona (~100 facts)
├── MEMORY_PERSON_B.md    # Distractor persona
├── MEMORY_PERSON_C.md    # Distractor persona
├── MEMORY_ENGINEERING.md # Technical depth corpus
├── MEMORY_CODEBASE.md    # Code-related corpus
└── PROJECT_NOTES.md      # Synthetic project info
```

---

## Test Workspace Setup

The test agent workspace must contain ONLY:

```
~/clawd/test-baseline-workspace/
├── MEMORY.md             # Copy of MEMORY_PERSON_A.md
├── memory/
│   └── *.md              # Synthetic daily logs if needed
├── AGENTS.md             # Minimal agent instructions
├── SOUL.md               # Test agent persona
└── USER.md               # Fictional user profile
```

---

## Running Tests

### Step 1: Verify Test Environment
Before ANY test run:
- [ ] Test agent configured in gateway config
- [ ] Test workspace contains ONLY anonymized data
- [ ] No symlinks or references to real data

### Step 2: Spawn Test Session
```javascript
sessions_spawn({
  agentId: "test-baseline",
  task: "...",
  label: "baseline-test-run-YYYY-MM-DD"
})
```

### Step 3: Execute Queries
Send test queries via `sessions_send` to the spawned session.

### Step 4: Score Results
Compare responses against ground truth from test corpus.

### Step 5: Record Results
Store in `tests/baseline/results/` with:
- Timestamp
- Test corpus version
- Query/response pairs
- Scores

---

## Result Storage Rules

1. Results go in `tests/baseline/results/` (gitignored)
2. Never include real user data in result files
3. Use anonymized IDs (test-001, query-042, etc.)
4. Include reproducibility info (seed, corpus hash, timestamp)

---

## Pre-Test Checklist

Copy this checklist before each test run:

```markdown
## Pre-Test Verification

- [ ] Using `test-baseline` agent (not main Clawdio)
- [ ] Test workspace has ONLY synthetic data
- [ ] MEMORY.md is MEMORY_PERSON_A.md (not real)
- [ ] No real user data in test queries
- [ ] Results directory is gitignored
- [ ] Test seed documented: _______________
```

---

## Why These Rules Matter

1. **Privacy:** Real preferences shouldn't appear in test logs or results
2. **Reproducibility:** Anonymized data can be shared and re-run
3. **Isolation:** Tests shouldn't corrupt production memory
4. **Clean baseline:** Synthetic data is controlled; real data has noise

---

## Violations

If you realize you've tested against real data:
1. STOP immediately
2. Delete result files
3. Document the violation in memory/YYYY-MM-DD.md
4. Re-run with proper test environment

---

*Created: 2026-01-31*
*Location: ~/clawd/projects/tribal-memory/TESTING-RULES.md*
