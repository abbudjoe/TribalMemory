# Import/Export Migration Guide

Tribal Memory supports portable data migration via JSON bundles.
You can export memories from one instance, transfer the file,
and import into another — even across different embedding models.

## Quick Start

### Export all memories

```python
from tribalmemory.portability.embedding_metadata import (
    create_embedding_metadata,
)
from tribalmemory.services.import_export import export_memories

meta = create_embedding_metadata(
    model_name="text-embedding-3-small",
    dimensions=1536,
    provider="openai",
)

bundle = await export_memories(
    store=vector_store,
    embedding_metadata=meta,
)

# Write to file
import json
with open("backup.json", "w") as f:
    json.dump(bundle.to_dict(), f, default=str)
```

### Import from file

```python
from tribalmemory.portability.embedding_metadata import (
    PortableBundle,
    create_embedding_metadata,
)
from tribalmemory.services.import_export import (
    import_memories,
    ConflictResolution,
)

with open("backup.json") as f:
    bundle = PortableBundle.from_dict(json.load(f))

target_meta = create_embedding_metadata(
    model_name="text-embedding-3-small",
    dimensions=1536,
    provider="openai",
)

summary = await import_memories(
    bundle=bundle,
    store=target_store,
    target_metadata=target_meta,
    conflict_resolution=ConflictResolution.SKIP,
)

print(f"Imported: {summary.imported}")
print(f"Skipped:  {summary.skipped}")
print(f"Errors:   {summary.errors}")
```

## Filtering Exports

### By tags

```python
from tribalmemory.services.import_export import ExportFilter

bundle = await export_memories(
    store=store,
    embedding_metadata=meta,
    filters=ExportFilter(tags=["preferences", "work"]),
)
```

### By date range

```python
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)
bundle = await export_memories(
    store=store,
    embedding_metadata=meta,
    filters=ExportFilter(
        date_from=now - timedelta(days=30),
        date_to=now,
    ),
)
```

### Combined filters

Tag and date filters are intersected — entries must match
at least one tag **and** fall within the date range.

## Conflict Resolution

When importing, existing memories with the same ID trigger
a conflict. Three strategies are available:

| Strategy      | Behavior                                    |
| ------------- | ------------------------------------------- |
| `SKIP`        | Keep existing, ignore incoming (default)    |
| `OVERWRITE`   | Replace existing with incoming              |
| `MERGE`       | Keep whichever has the newer `updated_at`   |

### Merging data from multiple sources

Use `MERGE` when combining data from instances that may have
diverged:

```python
summary = await import_memories(
    bundle=bundle,
    store=store,
    target_metadata=meta,
    conflict_resolution=ConflictResolution.MERGE,
)
```

## Dry-Run Mode

Preview what an import would do before committing:

```python
summary = await import_memories(
    bundle=bundle,
    store=store,
    target_metadata=meta,
    conflict_resolution=ConflictResolution.OVERWRITE,
    dry_run=True,
)

print(f"Would import:    {summary.imported}")
print(f"Would overwrite: {summary.overwritten}")
print(f"Would skip:      {summary.skipped}")

# Nothing was written — safe to inspect first
```

## Progress Tracking

For large imports, pass a callback to monitor progress:

```python
def show_progress(current: int, total: int) -> None:
    pct = current / total * 100
    print(f"\r{current}/{total} ({pct:.0f}%)", end="")

summary = await import_memories(
    bundle=bundle,
    store=store,
    target_metadata=meta,
    on_progress=show_progress,
)
print()  # newline after progress
```

## Embedding Model Mismatches

When the source and target use different embedding models,
the import can handle it automatically:

| Strategy | Behavior                                     |
| -------- | -------------------------------------------- |
| `AUTO`   | Keep if models match, drop if not (default)  |
| `KEEP`   | Always keep source embeddings                |
| `DROP`   | Always clear embeddings (must re-embed)      |

If embeddings are dropped, `summary.needs_reembedding` will
be `True`. You'll need to re-embed those entries before they
can be searched.

## Large Datasets

### Recommended limits

- **< 50k entries**: `export_memories()` works well
- **50k–100k entries**: Works but uses significant memory
- **> 100k entries**: Use streaming or batch by date range

### Streaming export

For very large stores, use the streaming API to avoid loading
everything into memory:

```python
from tribalmemory.services.import_export import (
    export_memories_streaming,
)

with open("export.jsonl", "w") as f:
    async for entry in export_memories_streaming(
        store=store,
        embedding_metadata=meta,
        batch_size=1000,
    ):
        line = json.dumps(entry_to_dict(entry), default=str)
        f.write(line + "\n")
```

### Batched export by date range

Split large exports into monthly batches:

```python
from datetime import datetime, timedelta, timezone

start = datetime(2025, 1, 1, tzinfo=timezone.utc)
end = datetime.now(timezone.utc)
current = start

while current < end:
    next_month = current + timedelta(days=30)
    bundle = await export_memories(
        store=store,
        embedding_metadata=meta,
        filters=ExportFilter(
            date_from=current,
            date_to=min(next_month, end),
        ),
    )
    filename = f"export-{current.strftime('%Y-%m')}.json"
    with open(filename, "w") as f:
        json.dump(bundle.to_dict(), f, default=str)
    current = next_month
```

## MCP Tools

If using Tribal Memory via MCP (e.g. with Claude Code):

```
# Export
tribal_export(tags=["work"], output_path="/tmp/work.json")

# Import with dry-run preview
tribal_import(
    input_path="/tmp/work.json",
    conflict_resolution="merge",
    dry_run=True,
)

# Import for real
tribal_import(
    input_path="/tmp/work.json",
    conflict_resolution="merge",
)
```

## HTTP API

```bash
# Export
curl -X POST http://localhost:18790/v1/export \
  -H "Content-Type: application/json" \
  -d '{"tags": ["preferences"]}'

# Import
curl -X POST http://localhost:18790/v1/import \
  -H "Content-Type: application/json" \
  -d @backup.json
```

## Troubleshooting

**Import skips everything:**
Default conflict resolution is `SKIP`. If re-importing the
same data, all IDs already exist. Use `OVERWRITE` or `MERGE`.

**`needs_reembedding` is True:**
Source and target use different embedding models. Entries were
imported without vectors. Run your embedding service on the
imported entries before they'll appear in searches.

**Export seems truncated:**
Default limit is 100,000 entries. Check logs for a truncation
warning. Use streaming export or date-range batching for
larger datasets.

**Invalid date format errors:**
Dates must be ISO 8601 format: `2026-01-15T10:30:00` or
`2026-01-15T10:30:00+00:00`. Partial dates like `2026-01-15`
also work.
