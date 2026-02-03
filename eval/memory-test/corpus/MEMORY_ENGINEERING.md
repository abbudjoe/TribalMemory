# Engineering Memory — Hyperion Project

*Technical notes, decisions, debugging history, and implementation details*

---

## Architecture Overview

### System Design
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   Gateway   │────▶│  Hyperion   │
│   (gRPC)    │     │  (Envoy)    │     │   Cluster   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Auth Svc   │     │ CockroachDB │
                    │  (JWT/mTLS) │     │  (persist)  │
                    └─────────────┘     └─────────────┘
```

### Why Separate Auth Service?
- **Decision date:** 2024-07-15
- **Decision:** Auth is a separate microservice, not embedded in Gateway
- **Rationale:** 
  1. Auth needs to scale independently (token validation is CPU-bound)
  2. Allows A/B testing auth strategies without Gateway redeploy
  3. Security isolation — auth service has minimal attack surface
  4. Can swap JWT for mTLS without touching Gateway
- **Trade-off accepted:** Extra network hop adds ~2ms latency
- **Who decided:** Architecture review board (Rebecca, James, me)

### Why gRPC over REST for Internal Services?
- **Decision date:** 2025-01-05
- **Rejected:** REST with JSON
- **Chosen:** gRPC with Protocol Buffers
- **Rationale:**
  1. Binary encoding: 10x smaller payloads than JSON
  2. Code generation: Type-safe clients in Rust/Go/Python
  3. Streaming: Need bidirectional streaming for cache invalidation
  4. Latency: Measured 3ms improvement in p99 over REST
- **Downside:** Harder to debug (not curl-friendly), requires protoc toolchain

---

## Code Patterns

### The `CacheEntry` Struct
```rust
// core/src/cache/entry.rs
pub struct CacheEntry {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub ttl_ms: u64,
    pub created_at: u64,
    pub version: u64,        // For optimistic locking
    pub flags: EntryFlags,   // Compression, encryption bits
}

impl CacheEntry {
    /// Check if entry is expired. Uses monotonic clock to avoid 
    /// issues with system time changes.
    pub fn is_expired(&self, now_mono: u64) -> bool {
        now_mono > self.created_at + self.ttl_ms
    }
}
```

**Why `Vec<u8>` for key/value instead of String?**
- Binary keys are common (hashed session IDs)
- Avoids UTF-8 validation overhead
- Allows storing serialized protobufs directly

### The Retry Pattern with Exponential Backoff
```rust
// common/src/retry.rs
pub async fn retry_with_backoff<F, T, E>(
    mut f: F,
    max_attempts: u32,
    base_delay_ms: u64,
) -> Result<T, E>
where
    F: FnMut() -> Future<Output = Result<T, E>>,
{
    let mut attempt = 0;
    loop {
        match f().await {
            Ok(v) => return Ok(v),
            Err(e) if attempt >= max_attempts => return Err(e),
            Err(_) => {
                let delay = base_delay_ms * 2u64.pow(attempt);
                let jitter = rand::thread_rng().gen_range(0..delay/4);
                tokio::time::sleep(Duration::from_millis(delay + jitter)).await;
                attempt += 1;
            }
        }
    }
}
```

**Why jitter?**
- Prevents thundering herd when multiple clients retry simultaneously
- Without jitter, all retries align and cause load spikes
- Learned this the hard way in the 2024-10-15 incident

### Singleton Pattern for Config
```rust
// We use once_cell, NOT lazy_static
use once_cell::sync::Lazy;

static CONFIG: Lazy<Config> = Lazy::new(|| {
    Config::from_env().expect("Failed to load config")
});

// Access: CONFIG.cache_size_mb
```

**Why once_cell over lazy_static?**
- lazy_static requires a macro, once_cell is just a type
- once_cell allows non-static lazy initialization too
- Better error messages on initialization failure

---

## API Specifications

### Cache Get Endpoint
```protobuf
// proto/cache.proto
service CacheService {
    rpc Get(GetRequest) returns (GetResponse);
    rpc Set(SetRequest) returns (SetResponse);
    rpc Delete(DeleteRequest) returns (DeleteResponse);
    rpc BatchGet(BatchGetRequest) returns (stream GetResponse);
}

message GetRequest {
    bytes key = 1;
    bool allow_stale = 2;      // Return expired entry if fresh unavailable
    uint32 timeout_ms = 3;     // Max wait time, default 100ms
}

message GetResponse {
    bytes value = 1;
    uint64 version = 2;
    bool is_stale = 3;         // True if allow_stale returned expired data
    CacheStatus status = 4;    // HIT, MISS, ERROR
}
```

### Rate Limits
| Endpoint | Limit | Window | Scope |
|----------|-------|--------|-------|
| Get | 10,000 | 1s | per client IP |
| Set | 1,000 | 1s | per client IP |
| BatchGet | 100 | 1s | per client IP |
| Admin/* | 10 | 1s | per API key |

**Exceeded limit returns:** `StatusCode::RESOURCE_EXHAUSTED` with `retry-after` header

---

## Configuration Details

### Production Redis Settings
```yaml
# redis/production.conf
maxmemory 8gb
maxmemory-policy allkeys-lru
tcp-keepalive 300
timeout 0
tcp-backlog 511

# CRITICAL: These settings for Hyperion compatibility
notify-keyspace-events Ex    # For TTL expiration notifications
appendonly yes               # Durability over performance
appendfsync everysec         # Balance between safety and speed
```

**Why `appendfsync everysec` instead of `always`?**
- `always` gave us 200ms write latency spikes
- `everysec` worst case loses 1 second of data on crash
- Acceptable trade-off: Hyperion has its own WAL for critical data

### Environment Variables
```bash
# Required
HYPERION_CLUSTER_NODES=node1:7000,node2:7000,node3:7000
HYPERION_REPLICATION_FACTOR=3
COCKROACH_DSN=postgresql://hyperion@roach:26257/hyperion?sslmode=verify-full

# Optional with defaults
HYPERION_CACHE_SIZE_MB=4096          # Default: 4GB
HYPERION_EVICTION_BATCH_SIZE=1000    # Default: 1000
HYPERION_RAFT_HEARTBEAT_MS=150       # Default: 150ms
HYPERION_RAFT_ELECTION_TIMEOUT_MS=1500  # Default: 10x heartbeat
LOG_LEVEL=info                        # Default: info
OTEL_EXPORTER_OTLP_ENDPOINT=         # Optional: OpenTelemetry collector
```

### Feature Flags
```json
// feature_flags.json (synced from LaunchDarkly)
{
    "enable_compression": true,        // LZ4 for values > 1KB
    "enable_batching": true,           // Batch small writes
    "new_eviction_algorithm": false,   // Testing LRU-K variant
    "shadow_traffic_percent": 5,       // Mirror to new cluster
    "enable_client_caching": true,     // Client-side TTL hints
    "max_value_size_kb": 512           // Reject larger values
}
```

---

## Debugging History

### 2024-10-15: The Thundering Herd Incident
**Symptoms:**
- 3am alert: p99 latency spiked from 5ms to 2000ms
- Cache hit rate dropped from 95% to 20%
- CockroachDB connections maxed out

**Root cause:**
- Scheduled job expired 50K cache entries simultaneously
- All services retried at the same time (no jitter)
- Retry storms overwhelmed the database

**Fix:**
1. Added jitter to retry logic (see retry pattern above)
2. Staggered TTLs: `ttl = base_ttl + rand(0, base_ttl * 0.1)`
3. Added circuit breaker with 50% failure threshold

**Postmortem:** `docs/postmortems/2024-10-15-thundering-herd.md`

### 2024-11-22: Memory Leak in v2.3.0
**Symptoms:**
- RSS grew linearly: 4GB → 12GB over 6 hours
- Eventually OOM-killed

**Investigation:**
```bash
# Heap profiling revealed:
heaptrack ./hyperion-server
# 800MB in: core::cache::entry::CacheEntry::clone
# Clone was being called in hot path instead of borrow
```

**Root cause:**
```rust
// BAD: Cloning on every get
fn get(&self, key: &[u8]) -> Option<CacheEntry> {
    self.entries.get(key).cloned()  // <-- Allocates!
}

// GOOD: Return reference with lifetime
fn get(&self, key: &[u8]) -> Option<&CacheEntry> {
    self.entries.get(key)
}
```

**Fix:** PR #847 — Return references instead of clones in hot path
**Lesson:** Always profile before assuming. The leak looked like a cache bug but was a Rust ownership issue.

### 2024-12-03: Split Brain During Network Partition
**Symptoms:**
- Two nodes both thought they were Raft leader
- Conflicting writes corrupted 200 cache entries

**Root cause:**
- Network partition lasted exactly 1.5 seconds
- Election timeout was 1.5 seconds (too tight)
- Both sides elected new leaders simultaneously

**Fix:**
- Increased election timeout to 10x heartbeat (150ms → 1500ms)
- Added leader lease mechanism
- Implemented read-index for linearizable reads

**Config change:**
```yaml
raft:
  heartbeat_interval_ms: 150
  election_timeout_ms: 1500    # Was 1500, now formula: 10x heartbeat
  lease_duration_ms: 1000      # New: leader lease
```

### 2025-01-08: ETIMEOUT on Cold Starts
**Symptoms:**
- First request after deploy always timed out
- Subsequent requests fine

**Root cause:**
- Lazy initialization of CockroachDB connection pool
- First request triggered pool creation (500ms)
- Request timeout was 100ms

**Fix:**
```rust
// main.rs - Eagerly initialize pool at startup
#[tokio::main]
async fn main() {
    let pool = db::create_pool().await.expect("DB pool failed");
    pool.acquire().await.expect("DB warmup failed");  // <-- Warmup connection
    
    start_server(pool).await;
}
```

---

## Dependencies & Constraints

### Version Pinning Rationale
```toml
# Cargo.toml
[dependencies]
tokio = "=1.35.1"      # 1.36 has regression in io_uring, see rust-lang/tokio#6298
rustls = "=0.21.10"    # 0.22 requires MSRV 1.73, we're on 1.72
prost = "0.12"         # Latest stable, no issues
tonic = "0.10"         # Must match prost major version

[target.'cfg(target_os = "linux")'.dependencies]
io-uring = "0.6"       # Only on Linux, disabled on macOS
```

### Why Node 18 (not 20)?
- **Date:** 2024-09-01
- **Reason:** Our nexctl CLI uses native modules
- `better-sqlite3` doesn't have prebuilds for Node 20 + musl (Alpine)
- Building from source adds 3 minutes to Docker build
- Pinned until better-sqlite3 releases Node 20 musl prebuilds

### Rust MSRV (Minimum Supported Rust Version)
- **Current:** 1.72.0
- **Reason:** Ubuntu 22.04 LTS ships rustc 1.72
- **Constraint:** Can't use features from 1.73+ (like `async fn` in traits)
- **Review date:** 2025-04-01 (when Ubuntu 24.04 is baseline)

---

## Performance Benchmarks

### Baseline Numbers (as of 2025-01-15)
| Operation | p50 | p99 | p999 | Throughput |
|-----------|-----|-----|------|------------|
| Get (hit) | 0.3ms | 2.1ms | 8ms | 120k ops/s |
| Get (miss) | 0.5ms | 3ms | 12ms | 100k ops/s |
| Set | 1.2ms | 8ms | 25ms | 45k ops/s |
| Delete | 0.8ms | 5ms | 15ms | 60k ops/s |
| BatchGet (100 keys) | 5ms | 20ms | 50ms | 5k batch/s |

**Test conditions:** 3-node cluster, 8-core VMs, 32GB RAM, NVMe SSD

### Optimization History
1. **2024-08-10:** Switched from `std::HashMap` to `hashbrown` — 15% throughput increase
2. **2024-09-20:** Enabled io_uring on Linux — 20% latency reduction
3. **2024-11-05:** Added client-side connection pooling — 30% p99 improvement
4. **2025-01-10:** LZ4 compression for values >1KB — 40% bandwidth reduction

---

## Security Considerations

### Threat Model
1. **Untrusted clients:** Assume all cache clients are potentially malicious
2. **Network:** Internal network is NOT trusted (zero trust)
3. **Data sensitivity:** Cache may contain PII (encrypted at rest)

### Mitigations
| Threat | Mitigation |
|--------|------------|
| Cache poisoning | HMAC on all writes, verified on read |
| DoS via large values | Max value size 512KB, reject larger |
| Timing attacks | Constant-time comparison for auth tokens |
| Data exfil | mTLS required, no plaintext connections |
| Memory disclosure | Zero-fill deallocated cache entries |

### Secret Management
```bash
# Secrets are NEVER in environment variables
# Loaded from Vault at startup:
vault kv get secret/hyperion/prod/db-password
vault kv get secret/hyperion/prod/tls-cert
vault kv get secret/hyperion/prod/hmac-key
```

---

## Common Issues & Solutions

### "Connection refused" to CockroachDB
```bash
# Check if CockroachDB is actually running
kubectl get pods -l app=cockroachdb

# Check if service endpoint exists
kubectl get endpoints cockroachdb

# Common fix: DNS not resolved yet after deploy
# Wait 30s or restart the Hyperion pod
```

### "Raft leader not found" errors
```bash
# Usually means cluster is in election
# Wait 5 seconds and retry

# If persistent, check network between nodes:
kubectl exec -it hyperion-0 -- ping hyperion-1

# Force leader election (emergency only):
curl -X POST http://localhost:9090/admin/raft/step-down
```

### High memory usage
```bash
# Check cache size vs configured max
curl http://localhost:9090/metrics | grep hyperion_cache_size_bytes

# If at limit, eviction should be working
# If over limit, check for the clone() bug (see 2024-11-22 incident)

# Force eviction (emergency):
curl -X POST http://localhost:9090/admin/cache/evict?count=10000
```

### Slow batch operations
```bash
# BatchGet over 100 keys is rate-limited
# Split into multiple calls or use pagination:
for batch in $(split_keys 100); do
    hyperion-cli batch-get $batch
done
```

---

## Project History & Evolution

### v1.0 (2024-06)
- Initial release, single-node only
- HashMap-based storage, no persistence
- REST API

### v2.0 (2024-09)
- Raft consensus for multi-node
- CockroachDB for persistence
- gRPC API (REST deprecated)

### v2.3 (2024-11)
- Memory leak fix
- Client-side caching hints
- Compression support

### v3.0 (2025-02, planned)
- Cross-region replication
- Tiered storage (hot/warm/cold)
- GraphQL API for complex queries

---

*~150 engineering facts for depth testing*
