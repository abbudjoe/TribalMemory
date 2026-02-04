#!/bin/bash
# ============================================================
#  🧠 Tribal Memory Demo: "One Brain, Two Agents"
# ============================================================
#
#  Agent A (Claude Code) stores memories.
#  Agent B (Codex CLI) recalls them.
#  Same server. Shared brain.
#
# ============================================================

set -e

DEMO_CONFIG="/tmp/tribal-demo/config.yaml"
SERVER="http://localhost:18791"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

slow_print() {
    echo -e "$1"
    sleep 1
}

divider() {
    echo -e "${DIM}────────────────────────────────────────────${NC}"
}

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
RESULT=$(curl -s -X POST "$SERVER/v1/remember" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The auth service uses JWT with RS256 signing",
    "source_type": "auto_capture",
    "tags": ["architecture", "auth"],
    "instance_id": "claude-code"
  }')
MEM_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('memory_id','?'))")
echo -e "  ${GREEN}✅ Stored${NC} ${DIM}(id: ${MEM_ID:0:8}...)${NC}"
sleep 1

echo ""
echo -e "${DIM}  Storing: ${NC}\"Database is Postgres 16 with pgvector for embeddings\""
RESULT2=$(curl -s -X POST "$SERVER/v1/remember" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Database is Postgres 16 with pgvector for embeddings",
    "source_type": "auto_capture",
    "tags": ["architecture", "database"],
    "instance_id": "claude-code"
  }')
MEM_ID2=$(echo "$RESULT2" | python3 -c "import sys,json; print(json.load(sys.stdin).get('memory_id','?'))")
echo -e "  ${GREEN}✅ Stored${NC} ${DIM}(id: ${MEM_ID2:0:8}...)${NC}"
sleep 1

echo ""
echo -e "${DIM}  Storing: ${NC}\"Frontend uses Next.js 15 with App Router\""
RESULT3=$(curl -s -X POST "$SERVER/v1/remember" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Frontend uses Next.js 15 with App Router",
    "source_type": "user_explicit",
    "tags": ["architecture", "frontend"],
    "instance_id": "claude-code"
  }')
MEM_ID3=$(echo "$RESULT3" | python3 -c "import sys,json; print(json.load(sys.stdin).get('memory_id','?'))")
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
RECALL=$(curl -s -X POST "$SERVER/v1/recall" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work?",
    "limit": 3
  }')

echo "$RECALL" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for r in data.get('results', []):
    m = r['memory']
    score = r['similarity_score']
    bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
    print(f'  \033[0;36m{bar}\033[0m {score:.0%}  {m[\"content\"]}')
    print(f'  \033[2m  └─ from: {m[\"source_instance\"]}  tags: {m[\"tags\"]}\033[0m')
    print()
"
sleep 2

echo -e "${DIM}  Asking: ${NC}\"What database and frontend stack are we using?\""
echo ""
RECALL2=$(curl -s -X POST "$SERVER/v1/recall" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What database and frontend stack are we using?",
    "limit": 3
  }')

echo "$RECALL2" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for r in data.get('results', []):
    m = r['memory']
    score = r['similarity_score']
    bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
    print(f'  \033[0;36m{bar}\033[0m {score:.0%}  {m[\"content\"]}')
    print(f'  \033[2m  └─ from: {m[\"source_instance\"]}  tags: {m[\"tags\"]}\033[0m')
    print()
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
