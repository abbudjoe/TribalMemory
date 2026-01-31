#!/bin/bash
# Seed Tribal Memory with test corpus
# Uses smaller semantic chunks for better retrieval

TRIBAL_URL="${TRIBAL_URL:-http://localhost:18790}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS="${1:-$SCRIPT_DIR/corpus/USER.md}"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          Tribal Memory Corpus Seeder                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Source: $CORPUS"
echo "Target: $TRIBAL_URL"
echo ""

# Check server is running
if ! curl -s "$TRIBAL_URL/v1/health" > /dev/null 2>&1; then
    echo "ERROR: Tribal Memory server not responding at $TRIBAL_URL"
    exit 1
fi

# Get current stats
BEFORE=$(curl -s "$TRIBAL_URL/v1/stats" | jq -r '.total_memories // 0')
echo "Memories before: $BEFORE"
echo ""

# Parse markdown into semantic chunks (by section + individual facts)
# Strategy: Each "## Section" header starts a new context
# Each bullet point or line within becomes a separate memory

CURRENT_SECTION=""
STORED=0
FAILED=0

while IFS= read -r line; do
    # Skip empty lines
    [[ -z "$line" ]] && continue
    
    # Detect section headers
    if [[ "$line" =~ ^##[[:space:]](.+)$ ]]; then
        CURRENT_SECTION="${BASH_REMATCH[1]}"
        continue
    fi
    
    # Skip the title
    [[ "$line" =~ ^#[[:space:]] ]] && continue
    
    # Process bullet points and content lines
    if [[ "$line" =~ ^-[[:space:]](.+)$ ]] || [[ "$line" =~ ^[[:space:]]+-[[:space:]](.+)$ ]]; then
        CONTENT="${BASH_REMATCH[1]:-$line}"
        # Clean up the line
        CONTENT=$(echo "$CONTENT" | sed 's/^- //')
        
        # Skip very short lines
        [[ ${#CONTENT} -lt 10 ]] && continue
        
        # Build the memory with context
        if [[ -n "$CURRENT_SECTION" ]]; then
            FULL_CONTENT="[$CURRENT_SECTION] $CONTENT"
        else
            FULL_CONTENT="$CONTENT"
        fi
        
        # Determine tags based on section
        TAGS="[]"
        case "$CURRENT_SECTION" in
            *Personal*) TAGS='["personal"]' ;;
            *Family*) TAGS='["family"]' ;;
            *Food*) TAGS='["preferences", "food"]' ;;
            *Health*) TAGS='["health"]' ;;
            *Work*) TAGS='["work"]' ;;
            *Technical*) TAGS='["technical"]' ;;
            *Location*|*Address*) TAGS='["locations"]' ;;
            *Transport*) TAGS='["transport"]' ;;
            *Hobbies*) TAGS='["preferences", "hobbies"]' ;;
            *Financial*) TAGS='["financial"]' ;;
            *Calendar*) TAGS='["schedule"]' ;;
            *Preferences*) TAGS='["preferences"]' ;;
            *Important*Dates*) TAGS='["dates", "family"]' ;;
            *Emergency*) TAGS='["emergency", "contacts"]' ;;
            *Recent*) TAGS='["context"]' ;;
        esac
        
        # Store the memory
        PAYLOAD=$(jq -n \
            --arg content "$FULL_CONTENT" \
            --arg context "Seeded from USER.md corpus" \
            --argjson tags "$TAGS" \
            '{content: $content, source_type: "user_explicit", context: $context, tags: $tags, skip_dedup: true}')
        
        RESPONSE=$(curl -s -X POST "$TRIBAL_URL/v1/remember" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD")
        
        if echo "$RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
            ((STORED++))
            printf "."
        else
            ((FAILED++))
            printf "x"
        fi
    fi
done < "$CORPUS"

echo ""
echo ""

# Get final stats
AFTER=$(curl -s "$TRIBAL_URL/v1/stats" | jq -r '.total_memories // 0')
echo "────────────────────────────────────────────────────────────────"
echo "Stored: $STORED memories"
echo "Failed: $FAILED"
echo "Total in Tribal Memory: $AFTER"
echo "────────────────────────────────────────────────────────────────"
