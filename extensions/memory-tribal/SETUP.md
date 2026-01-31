# Memory-Tribal Plugin Setup Guide

This guide documents how to set up and configure the memory-tribal plugin for OpenClaw.

## Prerequisites

- OpenClaw 2026.1.29 or later
- Node.js 20+ (for npm dependencies)
- SQLite3 (for persistence layer)

## Quick Start

```bash
# 1. Clone/copy the plugin to your workspace
cp -r extensions/memory-tribal ~/my-workspace/extensions/

# 2. Install dependencies
cd ~/my-workspace/extensions/memory-tribal
npm install

# 3. Link the plugin to OpenClaw
openclaw plugins install -l ~/my-workspace/extensions/memory-tribal

# 4. Set as active memory slot
openclaw config set plugins.slots.memory=memory-tribal

# 5. Restart gateway
openclaw gateway restart
```

## Detailed Setup

### Step 1: Plugin Structure

Your plugin directory must have this structure:

```
memory-tribal/
├── index.ts              # Main entry point
├── package.json          # MUST include openclaw.extensions
├── openclaw.plugin.json  # Plugin manifest
├── node_modules/         # Dependencies (after npm install)
└── src/
    ├── persistence.ts
    ├── tribal-client.ts
    └── learned/
        ├── query-cache.ts
        ├── query-expander.ts
        └── feedback-tracker.ts
```

### Step 2: Package.json Requirements

**Critical:** Your `package.json` MUST include the `openclaw.extensions` field:

```json
{
  "name": "memory-tribal",
  "version": "0.1.0",
  "main": "index.ts",
  "type": "module",
  "openclaw": {
    "extensions": ["./index.ts"]
  },
  "dependencies": {
    "@sinclair/typebox": "^0.32.0",
    "better-sqlite3": "^11.0.0"
  }
}
```

Without `openclaw.extensions`, the plugin installer will fail with:
```
Error: package.json missing openclaw.extensions
```

### Step 3: Install Dependencies

```bash
cd ~/my-workspace/extensions/memory-tribal
npm install
```

This creates `node_modules/` with required packages.

### Step 4: Install Plugin

Use the `-l` flag to **link** (not copy) the plugin for development:

```bash
openclaw plugins install -l /path/to/memory-tribal
```

This:
- Creates a symlink in `~/.openclaw/extensions/`
- Adds the plugin to `plugins.load.paths` in config
- Enables the plugin in `plugins.entries`

**Common errors:**

| Error | Solution |
|-------|----------|
| `plugin already exists` | Delete `~/.openclaw/extensions/memory-tribal` first |
| `package.json missing openclaw.extensions` | Add the `openclaw` field to package.json |

### Step 5: Configure Memory Slot

Memory plugins are **slot plugins** — only one can be active at a time.

Set memory-tribal as the active memory slot:

```bash
# Via CLI
openclaw config set plugins.slots.memory=memory-tribal

# Or via config.patch (programmatic)
gateway config.patch '{"plugins":{"slots":{"memory":"memory-tribal"}}}'
```

This disables the default `memory-core` plugin and activates memory-tribal.

### Step 6: Restart Gateway

```bash
openclaw gateway restart
```

Or if using the gateway tool programmatically:
```
gateway action=restart reason="Enable memory-tribal plugin"
```

### Step 7: Verify Installation

```bash
openclaw plugins list | grep memory
```

Expected output:
```
│ memory-tribal │ loaded   │ ~/...extensions/memory-tribal/index.ts │ 0.1.0 │
│ memory-core   │ disabled │ (slot taken by memory-tribal)          │       │
```

## Configuration Options

Configure via `plugins.entries.memory-tribal.config`:

```json
{
  "plugins": {
    "entries": {
      "memory-tribal": {
        "enabled": true,
        "config": {
          "tribalServerUrl": "http://localhost:18790",
          "queryCacheEnabled": true,
          "queryExpansionEnabled": true,
          "feedbackEnabled": true,
          "minCacheSuccesses": 3
        }
      }
    }
  }
}
```

| Option | Default | Description |
|--------|---------|-------------|
| `tribalServerUrl` | `http://localhost:18790` | Tribal memory HTTP server URL |
| `queryCacheEnabled` | `true` | Cache successful query→fact mappings |
| `queryExpansionEnabled` | `true` | Expand questions to keywords |
| `feedbackEnabled` | `true` | Track retrieval feedback |
| `minCacheSuccesses` | `3` | Hits before caching a mapping |

## Troubleshooting

### Plugin not appearing in `plugins list`

1. Check symlink exists: `ls -la ~/.openclaw/extensions/`
2. Verify `openclaw.plugin.json` exists in plugin root
3. Check `package.json` has `openclaw.extensions` field
4. Run `openclaw plugins doctor`

### Plugin loads but tools don't work

1. Check gateway logs: `openclaw gateway logs`
2. Verify slot is set: `openclaw config get plugins.slots.memory`
3. Ensure dependencies installed: `ls node_modules/` in plugin dir

### "invalid config" when patching

- Plugin ID must match `openclaw.plugin.json` → `id` field
- Slot plugins use `plugins.slots.<slot>`, not just `plugins.entries`

## Development Workflow

For active development, use the linked install (`-l` flag) so changes are reflected immediately:

```bash
# Initial setup
openclaw plugins install -l ./extensions/memory-tribal

# After code changes, just restart gateway
openclaw gateway restart

# Check logs for errors
openclaw gateway logs --tail 50
```

## Uninstalling

```bash
# Remove from config
openclaw plugins disable memory-tribal

# Reset memory slot to default
openclaw config set plugins.slots.memory=memory-core

# Delete symlink (optional)
rm ~/.openclaw/extensions/memory-tribal

# Restart
openclaw gateway restart
```

## Related Documentation

- [OpenClaw Plugin Guide](/plugin.md)
- [Plugin Manifest Reference](/plugins/manifest)
- [Memory Slot Plugins](/plugins/memory)
