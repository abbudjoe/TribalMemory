#!/bin/bash
# install.sh — Quick install for memory-tribal OpenClaw plugin
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/abbudjoe/TribalMemory/main/extensions/memory-tribal/install.sh | bash
#
# Or locally:
#   ./extensions/memory-tribal/install.sh

set -e

DEST="$HOME/.openclaw/extensions/memory-tribal"

echo "Installing memory-tribal plugin..."

# Create extensions directory if needed
mkdir -p "$HOME/.openclaw/extensions"

# Check if we're in the TribalMemory repo
if [ -f "extensions/memory-tribal/package.json" ]; then
    echo "  Copying from local repo..."
    cp -r extensions/memory-tribal/ "$DEST/"
else
    # Clone from GitHub
    echo "  Downloading from GitHub..."
    TMP=$(mktemp -d)
    git clone --depth 1 https://github.com/abbudjoe/TribalMemory.git "$TMP" 2>/dev/null
    cp -r "$TMP/extensions/memory-tribal/" "$DEST/"
    rm -rf "$TMP"
fi

# Install dependencies
echo "  Installing npm dependencies..."
cd "$DEST"
npm install --quiet

echo ""
echo "✅ memory-tribal installed to $DEST"
echo ""
echo "Next steps:"
echo "  1. Start Tribal Memory server: tribalmemory serve"
echo "  2. Add to openclaw.json:"
echo '     { "plugins": { "slots": { "memory": "memory-tribal" } } }'
echo "  3. Restart OpenClaw: systemctl --user restart openclaw-gateway"
