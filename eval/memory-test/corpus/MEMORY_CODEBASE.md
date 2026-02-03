# Codebase Memory — Hyperion Project

*File locations, function signatures, exact values, PR history, test coverage*

---

## File Structure

```
hyperion/
├── Cargo.toml                    # Workspace root, MSRV 1.72
├── proto/
│   ├── cache.proto               # Service definitions
│   └── internal.proto            # Raft messages
├── core/
│   ├── src/
│   │   ├── lib.rs                # Re-exports
│   │   ├── cache/
│   │   │   ├── mod.rs
│   │   │   ├── entry.rs          # CacheEntry struct
│   │   │   ├── store.rs          # CacheStore trait
│   │   │   ├── lru.rs            # LRU eviction
│   │   │   └── metrics.rs        # Prometheus metrics
│   │   ├── raft/
│   │   │   ├── mod.rs
│   │   │   ├── node.rs           # RaftNode impl
│   │   │   ├── log.rs            # Raft log storage
│   │   │   └── state_machine.rs  # Apply commands
│   │   └── config.rs             # Configuration
│   └── Cargo.toml
├── server/
│   ├── src/
│   │   ├── main.rs               # Entry point, 127 lines
│   │   ├── grpc.rs               # tonic handlers
│   │   ├── admin.rs              # Admin API (port 9090)
│   │   └── health.rs             # Health checks
│   └── Cargo.toml
├── client/
│   ├── src/
│   │   ├── lib.rs
│   │   ├── connection.rs         # Connection pooling
│   │   └── retry.rs              # Retry logic
│   └── Cargo.toml
├── nexctl/                        # CLI tool (Node.js)
│   ├── package.json              # Node 18 required
│   ├── src/
│   │   ├── index.ts
│   │   ├── commands/
│   │   │   ├── get.ts
│   │   │   ├── set.ts
│   │   │   └── admin.ts
│   │   └── config.ts
│   └── tsconfig.json
└── tests/
    ├── integration/
    │   ├── basic_ops.rs          # 45 tests
    │   ├── raft_election.rs      # 12 tests
    │   └── failure_modes.rs      # 23 tests
    └── benchmark/
        └── throughput.rs
```

---

## Exact Function Signatures

### core/src/cache/entry.rs

```rust
// Line 15-24
pub struct CacheEntry {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub ttl_ms: u64,
    pub created_at: u64,
    pub version: u64,
    pub flags: EntryFlags,
}

// Line 34 - The is_expired check
impl CacheEntry {
    pub fn is_expired(&self, now_mono: u64) -> bool {
        now_mono > self.created_at + self.ttl_ms
    }
    
    // Line 42 - Size calculation for eviction
    pub fn size_bytes(&self) -> usize {
        self.key.len() + self.value.len() + 32 // 32 bytes for metadata
    }
    
    // Line 48 - HMAC validation
    pub fn verify_hmac(&self, key: &[u8]) -> bool {
        // Uses ring::hmac with SHA-256
        let tag = hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, key), &self.value);
        // Tag stored in last 32 bytes of flags
        constant_time_eq(&tag.as_ref()[..], &self.flags.hmac_tag)
    }
}
```

### core/src/cache/store.rs

```rust
// Line 8 - The trait definition
pub trait CacheStore: Send + Sync {
    fn get(&self, key: &[u8]) -> Option<&CacheEntry>;
    fn set(&mut self, entry: CacheEntry) -> Result<(), StoreError>;
    fn delete(&mut self, key: &[u8]) -> bool;
    fn evict(&mut self, count: usize) -> usize;
    fn len(&self) -> usize;
    fn size_bytes(&self) -> usize;
}

// Line 45 - HashMapStore implementation
pub struct HashMapStore {
    entries: hashbrown::HashMap<Vec<u8>, CacheEntry>,
    max_size_bytes: usize,
    current_size_bytes: usize,
    eviction_queue: VecDeque<Vec<u8>>,  // For LRU tracking
}
```

### core/src/raft/node.rs

```rust
// Line 23
pub struct RaftNode {
    id: u64,
    peers: Vec<String>,
    state: RaftState,
    log: RaftLog,
    state_machine: Box<dyn StateMachine>,
    last_heartbeat: Instant,
    election_timeout: Duration,    // Default: 1500ms
    heartbeat_interval: Duration,  // Default: 150ms
}

// Line 89 - The tick function (called every 50ms)
impl RaftNode {
    pub fn tick(&mut self) -> Vec<RaftMessage> {
        let now = Instant::now();
        match self.state {
            RaftState::Follower => {
                if now.duration_since(self.last_heartbeat) > self.election_timeout {
                    self.start_election()
                }
            }
            RaftState::Leader => {
                if now.duration_since(self.last_heartbeat) > self.heartbeat_interval {
                    self.send_heartbeats()
                }
            }
            // ...
        }
    }
}
```

### server/src/main.rs

```rust
// Line 1-15
use hyperion_core::{cache::HashMapStore, raft::RaftNode};
use tonic::transport::Server;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::init();
    
    let config = Config::from_env()?;
    
    // Line 23 - DB pool warmup (added after cold start incident)
    let pool = db::create_pool(&config.cockroach_dsn).await?;
    pool.acquire().await?;  // Warmup connection
    
    // Line 28 - Initialize cache store
    let store = HashMapStore::new(config.cache_size_bytes);
    
    // ...
}
```

---

## Exact Configuration Values

### Default Port Assignments
| Service | Port | Configurable Via |
|---------|------|------------------|
| gRPC API | 7000 | `HYPERION_GRPC_PORT` |
| Admin API | 9090 | `HYPERION_ADMIN_PORT` |
| Raft internal | 7001 | `HYPERION_RAFT_PORT` |
| Metrics | 9091 | `HYPERION_METRICS_PORT` |

### Exact Timeouts (milliseconds)
| Timeout | Value | Constant Name |
|---------|-------|---------------|
| Raft heartbeat | 150 | `RAFT_HEARTBEAT_MS` |
| Raft election | 1500 | `RAFT_ELECTION_TIMEOUT_MS` |
| Raft leader lease | 1000 | `RAFT_LEASE_MS` |
| Client request | 100 | `DEFAULT_TIMEOUT_MS` |
| DB query | 5000 | `DB_QUERY_TIMEOUT_MS` |
| Health check | 500 | `HEALTH_CHECK_TIMEOUT_MS` |

### Exact Limits
| Limit | Value | Why |
|-------|-------|-----|
| Max key size | 1KB (1024 bytes) | Index memory |
| Max value size | 512KB (524288 bytes) | DoS prevention |
| Max batch size | 100 keys | Memory pressure |
| Max connections per client | 10 | Fair sharing |
| Eviction batch | 1000 entries | Performance |
| Max TTL | 7 days (604800000 ms) | Prevent stale |

---

## PR and Commit History

### PR #847 — Memory Leak Fix (2024-11-22)
**Title:** "fix: return references instead of clones in cache hot path"
**Author:** @maya-dev
**Reviewers:** @james-wright, @principal-eng (me)
**Files changed:**
- `core/src/cache/store.rs` (+12, -8)
- `core/src/cache/entry.rs` (+4, -2)
- `tests/integration/basic_ops.rs` (+25, -0)

**Key change:**
```rust
// Before (line 52 of store.rs)
fn get(&self, key: &[u8]) -> Option<CacheEntry> {
    self.entries.get(key).cloned()
}

// After
fn get(&self, key: &[u8]) -> Option<&CacheEntry> {
    self.entries.get(key)
}
```

### PR #812 — Jitter Implementation (2024-10-18)
**Title:** "fix: add jitter to retry backoff to prevent thundering herd"
**Author:** @principal-eng (me)
**Reviewers:** @rebecca-chen
**Linked issue:** #809 (2024-10-15 incident postmortem)
**Files changed:**
- `client/src/retry.rs` (+34, -12)
- `docs/postmortems/2024-10-15-thundering-herd.md` (+89, -0)

### PR #901 — Split Brain Prevention (2024-12-05)
**Title:** "fix: increase election timeout and add leader lease"
**Author:** @principal-eng (me)
**Reviewers:** @james-wright
**Linked issue:** #898 (split brain incident)
**Files changed:**
- `core/src/raft/node.rs` (+45, -12)
- `core/src/config.rs` (+8, -2)

### PR #923 — Cold Start Fix (2025-01-10)
**Title:** "fix: eagerly warm up DB pool on startup"
**Author:** @jordan-dev
**Files changed:**
- `server/src/main.rs` (+3, -0)

---

## Test Coverage Details

### tests/integration/basic_ops.rs (45 tests)
```rust
#[test] fn test_get_nonexistent_key()      // Line 12
#[test] fn test_set_and_get()              // Line 23
#[test] fn test_set_overwrites()           // Line 35
#[test] fn test_delete_existing()          // Line 48
#[test] fn test_delete_nonexistent()       // Line 59
#[test] fn test_ttl_expiration()           // Line 67 - uses tokio::time::advance
#[test] fn test_batch_get_partial()        // Line 89
#[test] fn test_batch_get_all_miss()       // Line 112
#[test] fn test_max_value_size()           // Line 125 - expects StoreError::ValueTooLarge
#[test] fn test_max_key_size()             // Line 138
// ... 35 more tests
```

### tests/integration/raft_election.rs (12 tests)
```rust
#[test] fn test_initial_leader_election()  // Line 15
#[test] fn test_leader_stepdown()          // Line 45
#[test] fn test_network_partition()        // Line 78 - the split brain test
#[test] fn test_rejoin_after_partition()   // Line 123
// ... 8 more tests
```

**Current coverage:** 84.3% (measured by cargo-tarpaulin)
**Uncovered:** Mostly error handling paths in raft/log.rs

---

## Exact Error Messages

### StoreError variants (core/src/cache/store.rs line 12)
```rust
pub enum StoreError {
    KeyTooLarge { size: usize, max: usize },
    // "Key size {size} exceeds maximum {max} bytes"
    
    ValueTooLarge { size: usize, max: usize },
    // "Value size {size} exceeds maximum {max} bytes"
    
    StoreFull { current: usize, max: usize },
    // "Store is full: {current}/{max} bytes"
    
    HmacVerificationFailed,
    // "HMAC verification failed for cache entry"
    
    SerializationError(String),
    // "Failed to serialize entry: {0}"
}
```

### RaftError variants (core/src/raft/node.rs line 12)
```rust
pub enum RaftError {
    NotLeader { leader_hint: Option<String> },
    // "Not the leader. Try: {leader_hint}"
    
    ElectionInProgress,
    // "Election in progress, retry after 2 seconds"
    
    LogCompactionFailed { reason: String },
    // "Log compaction failed: {reason}"
    
    PeerUnreachable { peer: String },
    // "Peer {peer} unreachable"
}
```

---

## Database Schema

### CockroachDB Tables

```sql
-- Table: cache_metadata (for persistence across restarts)
CREATE TABLE cache_metadata (
    node_id UUID PRIMARY KEY,
    last_snapshot_index BIGINT NOT NULL DEFAULT 0,
    last_snapshot_term BIGINT NOT NULL DEFAULT 0,
    cluster_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Table: raft_log (persistent Raft log)
CREATE TABLE raft_log (
    node_id UUID NOT NULL,
    log_index BIGINT NOT NULL,
    term BIGINT NOT NULL,
    command_type SMALLINT NOT NULL,  -- 1=Set, 2=Delete, 3=Noop
    command_data BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (node_id, log_index)
);

-- Index for log compaction
CREATE INDEX idx_raft_log_term ON raft_log (node_id, term);

-- Table: cluster_membership
CREATE TABLE cluster_membership (
    cluster_id UUID PRIMARY KEY,
    nodes JSONB NOT NULL,  -- Array of {id, address}
    version BIGINT NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## Environment-Specific Values

### Production (prod)
```bash
HYPERION_CLUSTER_NODES=hyperion-0.prod:7000,hyperion-1.prod:7000,hyperion-2.prod:7000
HYPERION_CACHE_SIZE_MB=8192       # 8GB per node
HYPERION_REPLICATION_FACTOR=3
COCKROACH_DSN=postgresql://hyperion@roach.prod:26257/hyperion_prod?sslmode=verify-full
LOG_LEVEL=info
```

### Staging (staging)
```bash
HYPERION_CLUSTER_NODES=hyperion-0.staging:7000
HYPERION_CACHE_SIZE_MB=1024       # 1GB (single node)
HYPERION_REPLICATION_FACTOR=1
COCKROACH_DSN=postgresql://hyperion@roach.staging:26257/hyperion_staging?sslmode=require
LOG_LEVEL=debug
```

### Development (dev)
```bash
HYPERION_CLUSTER_NODES=localhost:7000
HYPERION_CACHE_SIZE_MB=256        # 256MB
HYPERION_REPLICATION_FACTOR=1
COCKROACH_DSN=postgresql://root@localhost:26257/hyperion_dev?sslmode=disable
LOG_LEVEL=trace
```

---

## Metric Names (Prometheus)

```
# Counter: total requests by method
hyperion_requests_total{method="get|set|delete|batch_get"}

# Histogram: request latency in seconds
hyperion_request_duration_seconds{method="...", status="ok|error"}

# Gauge: current cache size in bytes
hyperion_cache_size_bytes

# Gauge: number of entries
hyperion_cache_entries_total

# Counter: evictions
hyperion_evictions_total{reason="ttl|lru|manual"}

# Gauge: Raft state (0=follower, 1=candidate, 2=leader)
hyperion_raft_state

# Counter: Raft elections
hyperion_raft_elections_total{outcome="won|lost|timeout"}

# Histogram: Raft replication latency
hyperion_raft_replication_seconds
```

---

## Known Tech Debt

1. **TODO in core/src/raft/log.rs:145** — "Implement log compaction snapshot transfer"
2. **FIXME in server/src/grpc.rs:89** — "Remove unwrap, handle error properly"
3. **HACK in client/src/retry.rs:34** — "Hardcoded max delay, should be configurable"
4. **Stale comment in store.rs:78** — References old API, needs update

---

*~100 codebase-specific facts for deep recall testing*
