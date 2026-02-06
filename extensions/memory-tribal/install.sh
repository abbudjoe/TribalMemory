#!/bin/bash
# install.sh — Quick install for memory-tribal OpenClaw plugin
#
# ⚠️  SECURITY WARNING: This script downloads code from GitHub.
# For production use, download and inspect manually:
#
#   git clone https://github.com/abbudjoe/TribalMemory.git
#   cat TribalMemory/extensions/memory-tribal/install.sh  # Review first
#   ./TribalMemory/extensions/memory-tribal/install.sh
#
# Quick install (requires trust):
#   curl -sSL https://raw.githubusercontent.com/abbudjoe/TribalMemory/main/extensions/memory-tribal/install.sh | bash
#
# Or locally:
#   ./extensions/memory-tribal/install.sh

set -e  # Exit on error
set -u  # Exit on unset variable

DEST="${HOME}/.openclaw/extensions/memory-tribal"

echo "Installing memory-tribal plugin..."

# Create extensions directory if needed
mkdir -p "${HOME}/.openclaw/extensions"

# Check if we're in the TribalMemory repo
if [ -f "extensions/memory-tribal/package.json" ]; then
    echo "  Copying from local repo..."
    cp -r extensions/memory-tribal/ "$DEST/"
else
    # Clone from GitHub
    echo "  Downloading from GitHub..."
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT  # Cleanup on exit
    
    if ! git clone --depth 1 https://github.com/abbudjoe/TribalMemory.git "$TMP" 2>/dev/null; then
        echo "❌ Failed to clone repository. Check network and try again."
        exit 1
    fi
    
    cp -r "$TMP/extensions/memory-tribal/" "$DEST/"
fi

# Install dependencies
echo "  Installing npm dependencies..."
cd "$DEST"

if ! npm install --quiet; then
    echo "❌ npm install failed. Check Node.js version (requires 18+) and permissions."
    exit 1
fi

echo ""
echo "✅ memory-tribal installed to $DEST"
echo ""
echo "Next steps:"
echo "  1. Start Tribal Memory server: tribalmemory serve"
echo "  2. Add to openclaw.json:"
echo '     { "plugins": { "slots": { "memory": "memory-tribal" } } }'
echo "  3. Restart OpenClaw: systemctl --user restart openclaw-gateway"
