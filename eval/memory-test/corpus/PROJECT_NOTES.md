# Project Notes — Hyperion

*Technical notes and decisions for the Hyperion distributed cache project*

---

## Overview

Hyperion is a distributed cache layer designed for sub-millisecond read latency at scale. Primary use case is session storage and feature flags.

## Architecture Decisions

### Storage Engine
- Chose **LMDB** over RocksDB for embedded storage
- Reasoning: Better read performance, simpler crash recovery
- Date: 2024-07-20

### Consensus Protocol  
- Using **Raft** via the tikv/raft-rs library
- Considered Paxos, rejected for complexity
- Date: 2024-08-01

### Wire Protocol
- Custom binary protocol over TCP
- Rejected gRPC for latency overhead (ironic given later service mesh decision)
- Benchmarked at 45μs p99 on loopback
- Date: 2024-08-15

## Milestones

- [x] Proof of concept — September 2024
- [x] Internal alpha — November 2024  
- [x] Load testing — January 2025
- [ ] Beta launch — March 2025
- [ ] GA launch — April 2025

## Team

- **Tech lead:** Person A (the one in MEMORY_PERSON_A.md)
- **Core contributors:** Maya, Jordan (from Person A's direct reports)
- **SRE partner:** Kim from Platform team

## Known Issues

1. Memory fragmentation under sustained writes — investigating jemalloc
2. Raft leader election takes ~500ms — acceptable but could improve
3. Cross-region replication not yet implemented

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Read latency p50 | <1ms | 0.3ms |
| Read latency p99 | <5ms | 2.1ms |
| Write latency p50 | <10ms | 8ms |
| Throughput | 100k ops/s | 120k ops/s |

---

*Project context for cross-reference testing*
