#!/bin/bash
# ============================================================
#  🧠 Tribal Memory Demo: "One Brain, Two Agents"
# ============================================================
#
#  Agent A (Claude Code) stores memories.
#  Agent B (Codex CLI) recalls them.
#  Same server. Shared brain.
#
#  Prerequisites:
#    - curl
#    - python3
#    - A running Tribal Memory server (tribalmemory serve)
#
#  Usage:
#    tribalmemory serve              # start the server
#    bash demo.sh                    # run this demo
#
#  The demo uses the default server at http://localhost:18790.
#  Override with: TRIBAL_MEMORY_SERVER=http://host:port bash demo.sh
#
# ============================================================

set -e

SERVER="${TRIBAL_MEMORY_SERVER:-http://localhost:18790}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Prerequisite checks ──
command -v curl >/dev/null 2>&1 || { echo -e "${YELLOW}⚠️  curl is required but not found${NC}"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo -e "${YELLOW}⚠️  python3 is required but not found${NC}"; exit 1; }

slow_print() {
    echo -e "$1"
    sleep 1
}

divider() {
    echo -e "${DIM}────────────────────────────────────────────${NC}"
}

# ── Helper: make API call with error handling ──
api_call() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    local result

    if [ "$method" = "POST" ]; then
        result=$(curl -sf -X POST "$SERVER$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data" 2>/dev/null) || {
            echo -e "  ${YELLOW}⚠️  API call failed: $method $endpoint${NC}"
            echo -e "  ${DIM}Is the server running at $SERVER?${NC}"
            exit 1
        }
    else
        result=$(curl -sf "$SERVER$endpoint" 2>/dev/null) || {
            echo -e "  ${YELLOW}⚠️  API call failed: $method $endpoint${NC}"
            echo -e "  ${DIM}Is the server running at $SERVER?${NC}"
            exit 1
        }
    fi
    echo "$result"
}

# ── Helper: safe JSON field extraction ──
json_field() {
    python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('$1', '?'))
except (json.JSONDecodeError, KeyError):
    print('?')
" 2>/dev/null
}

# ── Server health check ──
if ! curl -sf "$SERVER/v1/health" >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Server not reachable at $SERVER${NC}"
    echo -e "${DIM}Please start the server first:${NC}"
    echo -e "${DIM}  tribalmemory serve${NC}"
    echo ""
    echo -e "${DIM}Or set a custom URL:${NC}"
    echo -e "${DIM}  TRIBAL_MEMORY_SERVER=http://host:port bash demo.sh${NC}"
    exit 1
fi

# Header
clear
echo ""
echo -e "${BOLD}🧠 Tribal Memory — One Brain, Two Agents${NC}"
echo -e "${DIM}Your AI tools don't share a brain. Until now.${NC}"
echo ""
divider
sleep 2

# ── Agent A: Claude Code stores memories ──
echo ""
echo -e "${GREEN}${BOLD}▶ Agent A (Claude Code)${NC} ${DIM}instance: claude-code${NC}"
echo ""
sleep 1

echo -e "${DIM}  Storing: ${NC}\"The auth service uses JWT with RS256 signing\""
RESULT=$(api_call POST "/v1/remember" '{
    "content": "The auth service uses JWT with RS256 signing",
    "source_type": "auto_capture",
    "tags": ["architecture", "auth"],
    "instance_id": "claude-code"
  }')
MEM_ID=$(echo "$RESULT" | json_field memory_id)
echo -e "  ${GREEN}✅ Stored${NC} ${DIM}(id: ${MEM_ID:0:8}...)${NC}"
sleep 1

echo ""
echo -e "${DIM}  Storing: ${NC}\"Database is Postgres 16 with pgvector for embeddings\""
RESULT2=$(api_call POST "/v1/remember" '{
    "content": "Database is Postgres 16 with pgvector for embeddings",
    "source_type": "auto_capture",
    "tags": ["architecture", "database"],
    "instance_id": "claude-code"
  }')
MEM_ID2=$(echo "$RESULT2" | json_field memory_id)
echo -e "  ${GREEN}✅ Stored${NC} ${DIM}(id: ${MEM_ID2:0:8}...)${NC}"
sleep 1

echo ""
echo -e "${DIM}  Storing: ${NC}\"Frontend uses Next.js 15 with App Router\""
RESULT3=$(api_call POST "/v1/remember" '{
    "content": "Frontend uses Next.js 15 with App Router",
    "source_type": "user_explicit",
    "tags": ["architecture", "frontend"],
    "instance_id": "claude-code"
  }')
MEM_ID3=$(echo "$RESULT3" | json_field memory_id)
echo -e "  ${GREEN}✅ Stored${NC} ${DIM}(id: ${MEM_ID3:0:8}...)${NC}"

echo ""
divider
sleep 2

# ── Agent B: Codex recalls memories ──
echo ""
echo -e "${BLUE}${BOLD}▶ Agent B (Codex CLI)${NC} ${DIM}instance: codex — different agent, same brain${NC}"
echo ""
sleep 1

echo -e "${DIM}  Asking: ${NC}\"How does authentication work?\""
echo ""
RECALL=$(api_call POST "/v1/recall" '{
    "query": "How does authentication work?",
    "limit": 3
  }')

echo "$RECALL" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for r in data.get('results', []):
        m = r['memory']
        score = r['similarity_score']
        bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
        print(f'  \033[0;36m{bar}\033[0m {score:.0%}  {m[\"content\"]}')
        print(f'  \033[2m  └─ from: {m[\"source_instance\"]}  tags: {m[\"tags\"]}\033[0m')
        print()
except (json.JSONDecodeError, KeyError, TypeError) as e:
    print(f'  \033[1;33m⚠️  Failed to parse results: {e}\033[0m')
"
sleep 2

echo -e "${DIM}  Asking: ${NC}\"What database and frontend stack are we using?\""
echo ""
RECALL2=$(api_call POST "/v1/recall" '{
    "query": "What database and frontend stack are we using?",
    "limit": 3
  }')

echo "$RECALL2" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for r in data.get('results', []):
        m = r['memory']
        score = r['similarity_score']
        bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
        print(f'  \033[0;36m{bar}\033[0m {score:.0%}  {m[\"content\"]}')
        print(f'  \033[2m  └─ from: {m[\"source_instance\"]}  tags: {m[\"tags\"]}\033[0m')
        print()
except (json.JSONDecodeError, KeyError, TypeError) as e:
    print(f'  \033[1;33m⚠️  Failed to parse results: {e}\033[0m')
"

divider
echo ""
echo -e "${YELLOW}${BOLD}⚡ Claude Code stored it. Codex recalled it.${NC}"
echo -e "${YELLOW}   Same server. Shared memory. Zero config.${NC}"
echo ""
echo -e "${DIM}   pip install tribalmemory${NC}"
echo -e "${DIM}   tribalmemory init --local${NC}"
echo -e "${DIM}   tribalmemory serve${NC}"
echo ""
echo -e "${DIM}   https://github.com/abbudjoe/TribalMemory${NC}"
echo ""
