# Tribal Memory - Success Criteria

**Status:** ⏳ Pending Approval  
**Approver:** Joe  
**Created:** 2026-01-28  
**Last Updated:** 2026-01-31

---

## Overview

Evidence-based evaluation criteria for the Tribal Memory project. No "vibes-based" success declarations.

**Core Principle:** Define success BEFORE building, measure baseline BEFORE deployment, run controlled comparisons AFTER.

---

## Tier 1: Functional (Must Pass)

**Gate:** ALL Tier 1 criteria must pass. Any failure blocks deployment.

| # | Criterion | Metric | Threshold | Test Method | Status |
|---|-----------|--------|-----------|-------------|--------|
| 1.1 | Write-Read Integrity | Exact query recall accuracy | 100% | Automated test battery | ✅ Passing |
| 1.2 | Cross-Instance Propagation | Memory available to all instances | <5 seconds | Multi-instance test | 🔲 Needs Clawdio-1 |
| 1.3 | Deduplication | Duplicate detection rate | >90% | Duplicate injection test | ✅ Passing |
| 1.4 | Provenance Tracking | Source attribution accuracy | 100% | Provenance query test | ✅ Passing |
| 1.5 | No Performance Regression | Response latency vs baseline | <20% increase | Latency benchmark | ✅ Passing |

---

## Tier 2: Capability (Primary Evaluation)

**Gate:** If aggregate improvement <10% at Week 12 review, trigger project review.

| # | Criterion | Metric | Baseline | Success Threshold | Status |
|---|-----------|--------|----------|-------------------|--------|
| 2.1 | Preference Prediction | Accuracy on held-out tests | TBD Week 4 | >30% improvement | 🔲 Needs baseline |
| 2.2 | Cross-Session Consistency | Response similarity score | TBD Week 4 | >0.9 score | 🔲 Needs baseline |
| 2.3 | Error Correction Retention | Corrections remembered after 7d | TBD Week 4 | >90% retention | 🔲 Needs baseline |
| 2.4 | Context-Dependent Task Success | Tasks requiring history | 0% (impossible without memory) | >75% success | 🔲 Needs baseline |
| 2.5 | Retrieval Precision | % relevant memories returned | N/A | >80% precision | ✅ Test passing |

---

## Tier 3: Emergence (Stretch Goals)

**Note:** Tier 3 is exploratory. No kill decision based solely on Tier 3.

| # | Criterion | Metric | Success Indicator | Status |
|---|-----------|--------|-------------------|--------|
| 3.1 | Cross-Instance Synthesis | Combined knowledge quality | Joe rates 4+/5 | 🔲 Needs Clawdio-1 |
| 3.2 | Longitudinal Learning Curve | Improvement over 12 weeks | Positive slope | 🔲 Needs time |
| 3.3 | Anticipatory Assistance | Unprompted context surfacing | Measurable increase vs baseline | 🔲 Needs baseline |

---

## Kill Criteria

| Condition | Action |
|-----------|--------|
| Tier 1 failures persist >2 weeks | Stop, diagnose, fix or kill |
| Tier 2 shows <10% improvement at Week 12 | Project review meeting |
| Memory system causes >50% latency increase | Immediate investigation |
| Joe satisfaction drops vs baseline | Immediate investigation |
| Context overload detected | Reduce injection, re-evaluate |

---

## Evaluation Schedule

| Checkpoint | When | Purpose |
|------------|------|---------|
| **Baseline measurement** | Week 4 | Establish Tier 2 baselines |
| **Tier 1 validation** | Week 5-6 | All functional tests must pass |
| **Weekly check-ins** | Weeks 5-16 | Progress review |
| **Week 12 Review** | Week 16 | Kill/continue decision |

---

## Approval

- [ ] **Joe approves these success criteria**

Once approved, these criteria are locked. Changes require explicit re-approval.

---

*Extracted from PRD.md for standalone review*
