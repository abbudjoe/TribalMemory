#!/bin/bash
#
# Tier 2 Memory Retrieval Test Runner
# Tests memory search accuracy against a known corpus
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-$SCRIPT_DIR/dataset.json}"
RESULTS="${RESULTS:-$SCRIPT_DIR/results-tribal.json}"
AGENT="${AGENT:-test-baseline}"
TOP_K="${TOP_K:-5}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check dependencies
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed"
    exit 1
fi

# Check tribal memory server is running
if ! curl -s http://localhost:18790/v1/health > /dev/null 2>&1; then
    echo "Error: Tribal Memory server not responding at http://localhost:18790"
    exit 1
fi

# Initialize results structure
init_results() {
    cat <<EOF
{
  "summary": { "total": 0, "passed": 0, "failed": 0, "score": 0 },
  "timestamp": "$(date -Iseconds)",
  "agent": "$AGENT",
  "by_category": {},
  "by_difficulty": {},
  "queries": [],
  "failure_analysis": {
    "no_results": [],
    "wrong_chunk": [],
    "low_score": [],
    "common_patterns": ""
  }
}
EOF
}

# Run a single query and check results
run_query() {
    local id="$1"
    local query="$2"
    local expected_json="$3"
    local category="$4"
    local difficulty="$5"
    local expected_behavior="$6"  # "execute" or "clarify"
    
    # Run memory search via Tribal Memory API
    local search_result
    local tribal_url="${TRIBAL_URL:-http://localhost:18790}"
    search_result=$(curl -s -X POST "$tribal_url/v1/recall" \
        -H "Content-Type: application/json" \
        -d "{\"query\": $(echo "$query" | jq -Rs .), \"limit\": 5, \"min_relevance\": 0.1}" \
        2>/dev/null || echo '{"results":[]}')
    
    # Parse results (Tribal Memory format: .results[].memory.content, .results[].similarity_score)
    local got_results
    got_results=$(echo "$search_result" | jq -c '[.results[:5] | .[] | {text: .memory.content, score: .similarity_score, file: "tribal-memory"}]' 2>/dev/null || echo '[]')
    
    # Check if any expected substring is found in any result
    local status="FAIL"
    local failure_reason="no_results"
    local result_count
    result_count=$(echo "$got_results" | jq 'length')
    
    if [ "$result_count" -gt 0 ]; then
        failure_reason="wrong_chunk"
        
        # Handle imperative queries with expected_behavior
        if [ "$expected_behavior" = "clarify" ]; then
            # For "clarify" behavior: PASS if we find ambiguous context (multiple matches)
            # Check if results contain evidence of ambiguity (both potential targets)
            local combined_text
            combined_text=$(echo "$got_results" | jq -r '.[].text' | tr '\n' ' ')
            
            # Detect ambiguity patterns - look for multiple matching entities
            local has_sarah_chen=false
            local has_sarah_martinez=false
            local has_wally=false
            local has_beacon=false
            local has_homekit=false
            local has_multiple_configs=false
            local has_multiple_repos=false
            
            echo "$combined_text" | grep -qi "Sarah Chen" && has_sarah_chen=true
            echo "$combined_text" | grep -qi "Sarah Martinez" && has_sarah_martinez=true
            echo "$combined_text" | grep -qi "Wally" && has_wally=true
            echo "$combined_text" | grep -qi "Beacon" && has_beacon=true
            echo "$combined_text" | grep -qi "HomeKit" && has_homekit=true
            echo "$combined_text" | grep -qi "config.yaml\|settings.json\|\.env" && has_multiple_configs=true
            echo "$combined_text" | grep -qi "github.com/alexchen" && has_multiple_repos=true
            
            # Check for specific ambiguity patterns based on query
            case "$id" in
                imp001) # Email Sarah - need both Sarahs
                    if [ "$has_sarah_chen" = true ] && [ "$has_sarah_martinez" = true ]; then
                        status="PASS"
                        failure_reason=""
                    fi
                    ;;
                imp002) # Deploy the app - need multiple projects
                    local project_count=0
                    [ "$has_wally" = true ] && project_count=$((project_count + 1))
                    [ "$has_beacon" = true ] && project_count=$((project_count + 1))
                    [ "$has_homekit" = true ] && project_count=$((project_count + 1))
                    if [ "$project_count" -ge 2 ]; then
                        status="PASS"
                        failure_reason=""
                    fi
                    ;;
                imp003|imp004|imp005) # Config/repo/team ambiguity
                    if [ "$has_multiple_configs" = true ] || [ "$has_multiple_repos" = true ]; then
                        status="PASS"
                        failure_reason=""
                    fi
                    # Fallback: if we retrieved project info showing multiple options
                    if [ "$has_wally" = true ] && [ "$has_beacon" = true ]; then
                        status="PASS"
                        failure_reason=""
                    fi
                    ;;
            esac
            
            # Generic fallback: if "clarify" expected and results show ambiguity keywords
            if [ "$status" = "FAIL" ]; then
                local expected_array
                expected_array=$(echo "$expected_json" | jq -r '.[]')
                local match_count=0
                while IFS= read -r expected_substr; do
                    if echo "$got_results" | jq -r '.[].text' | grep -qi "$expected_substr"; then
                        match_count=$((match_count + 1))
                    fi
                done <<< "$expected_array"
                # If we found context about the ambiguity, pass
                if [ "$match_count" -ge 1 ]; then
                    status="PASS"
                    failure_reason=""
                fi
            fi
        else
            # Standard "execute" behavior: check each expected substring against all results
            local expected_array
            expected_array=$(echo "$expected_json" | jq -r '.[]')
            
            while IFS= read -r expected_substr; do
                # Case-insensitive search in all result texts
                if echo "$got_results" | jq -r '.[].text' | grep -qi "$expected_substr"; then
                    status="PASS"
                    failure_reason=""
                    break
                fi
            done <<< "$expected_array"
        fi
        
        # Check for low scores if still failing
        if [ "$status" = "FAIL" ]; then
            local max_score
            max_score=$(echo "$got_results" | jq '[.[].score] | max // 0')
            if (( $(echo "$max_score < 0.5" | bc -l) )); then
                failure_reason="low_score"
            fi
        fi
    fi
    
    # Build query result JSON
    local query_result
    query_result=$(jq -n \
        --arg id "$id" \
        --arg query "$query" \
        --arg status "$status" \
        --argjson expected "$expected_json" \
        --argjson got "$got_results" \
        --arg failure_reason "$failure_reason" \
        --arg category "$category" \
        --arg difficulty "$difficulty" \
        --arg expected_behavior "${expected_behavior:-execute}" \
        '{
            id: $id,
            query: $query,
            status: $status,
            expected: $expected,
            got: $got,
            failure_reason: (if $failure_reason == "" then null else $failure_reason end),
            category: $category,
            difficulty: $difficulty,
            expected_behavior: $expected_behavior
        }')
    
    echo "$query_result"
}

# Main test runner
main() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     Tier 2 TRIBAL MEMORY Test Harness                     ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Agent: ${YELLOW}$AGENT${NC}"
    echo -e "Dataset: ${YELLOW}$DATASET${NC}"
    echo -e "Started: $(date)"
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    
    # Read dataset
    local total_queries
    total_queries=$(jq '.queries | length' "$DATASET")
    
    # Initialize counters
    local passed=0
    local failed=0
    local results_json="[]"
    
    # Category and difficulty tracking
    declare -A cat_passed cat_total diff_passed diff_total
    
    # Process each query
    local i=0
    while read -r query_obj; do
        i=$((i + 1))
        
        local id query expected category difficulty expected_behavior
        id=$(echo "$query_obj" | jq -r '.id')
        query=$(echo "$query_obj" | jq -r '.query')
        expected=$(echo "$query_obj" | jq -c '.expected')
        category=$(echo "$query_obj" | jq -r '.category')
        difficulty=$(echo "$query_obj" | jq -r '.difficulty')
        expected_behavior=$(echo "$query_obj" | jq -r '.expected_behavior // "execute"')
        
        # Run query
        local result
        result=$(run_query "$id" "$query" "$expected" "$category" "$difficulty" "$expected_behavior")
        
        # Extract status
        local status
        status=$(echo "$result" | jq -r '.status')
        
        # Update counters
        if [ "$status" = "PASS" ]; then
            passed=$((passed + 1))
            echo -e "[${GREEN}PASS${NC}] $id: $query"
            cat_passed[$category]=$((${cat_passed[$category]:-0} + 1))
            diff_passed[$difficulty]=$((${diff_passed[$difficulty]:-0} + 1))
        else
            failed=$((failed + 1))
            local reason
            reason=$(echo "$result" | jq -r '.failure_reason // "unknown"')
            echo -e "[${RED}FAIL${NC}] $id: $query (${YELLOW}$reason${NC})"
        fi
        
        cat_total[$category]=$((${cat_total[$category]:-0} + 1))
        diff_total[$difficulty]=$((${diff_total[$difficulty]:-0} + 1))
        
        # Append to results
        results_json=$(echo "$results_json" | jq --argjson r "$result" '. + [$r]')
        
    done < <(jq -c '.queries[]' "$DATASET")
    
    # Calculate score
    local score
    score=$(echo "scale=4; $passed / $total_queries" | bc)
    
    # Build category breakdown
    local by_category="{}"
    for cat in "${!cat_total[@]}"; do
        by_category=$(echo "$by_category" | jq \
            --arg cat "$cat" \
            --argjson passed "${cat_passed[$cat]:-0}" \
            --argjson total "${cat_total[$cat]}" \
            '. + {($cat): {passed: $passed, total: $total, score: (($passed / $total) * 100 | floor / 100)}}')
    done
    
    # Build difficulty breakdown
    local by_difficulty="{}"
    for diff in "${!diff_total[@]}"; do
        by_difficulty=$(echo "$by_difficulty" | jq \
            --arg diff "$diff" \
            --argjson passed "${diff_passed[$diff]:-0}" \
            --argjson total "${diff_total[$diff]}" \
            '. + {($diff): {passed: $passed, total: $total, score: (($passed / $total) * 100 | floor / 100)}}')
    done
    
    # Build failure analysis
    local no_results wrong_chunk low_score
    no_results=$(echo "$results_json" | jq '[.[] | select(.failure_reason == "no_results") | .id]')
    wrong_chunk=$(echo "$results_json" | jq '[.[] | select(.failure_reason == "wrong_chunk") | .id]')
    low_score=$(echo "$results_json" | jq '[.[] | select(.failure_reason == "low_score") | .id]')
    
    # Build final results JSON
    # Write results_json to temp file to avoid "Argument list too long"
    local tmp_queries=$(mktemp)
    echo "$results_json" > "$tmp_queries"
    
    local final_results
    final_results=$(jq -n \
        --argjson total "$total_queries" \
        --argjson passed "$passed" \
        --argjson failed "$failed" \
        --argjson score "$score" \
        --arg timestamp "$(date -Iseconds)" \
        --arg agent "$AGENT" \
        --argjson by_category "$by_category" \
        --argjson by_difficulty "$by_difficulty" \
        --slurpfile queries "$tmp_queries" \
        --argjson no_results "$no_results" \
        --argjson wrong_chunk "$wrong_chunk" \
        --argjson low_score "$low_score" \
        '{
            summary: {
                total: $total,
                passed: $passed,
                failed: $failed,
                score: $score
            },
            timestamp: $timestamp,
            agent: $agent,
            by_category: $by_category,
            by_difficulty: $by_difficulty,
            queries: $queries[0],
            failure_analysis: {
                no_results: $no_results,
                wrong_chunk: $wrong_chunk,
                low_score: $low_score,
                common_patterns: ""
            }
        }')
    
    rm -f "$tmp_queries"
    
    # Save results
    echo "$final_results" | jq '.' > "$RESULTS"
    
    # Print summary
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo -e "${BLUE}## Summary${NC}"
    echo ""
    
    local score_pct
    score_pct=$(echo "scale=1; $score * 100" | bc)
    
    if (( $(echo "$score >= 0.8" | bc -l) )); then
        echo -e "**Score: ${GREEN}${score_pct}%${NC}** ($passed/$total_queries passed)"
    elif (( $(echo "$score >= 0.6" | bc -l) )); then
        echo -e "**Score: ${YELLOW}${score_pct}%${NC}** ($passed/$total_queries passed)"
    else
        echo -e "**Score: ${RED}${score_pct}%${NC}** ($passed/$total_queries passed)"
    fi
    
    echo ""
    echo -e "${BLUE}### By Category${NC}"
    echo ""
    for cat in $(echo "$by_category" | jq -r 'keys[]' | sort); do
        local cat_info
        cat_info=$(echo "$by_category" | jq -r --arg c "$cat" '.[$c] | "\(.passed)/\(.total)"')
        local cat_score
        cat_score=$(echo "$by_category" | jq -r --arg c "$cat" '.[$c].score * 100 | floor')
        echo "- $cat: $cat_info (${cat_score}%)"
    done
    
    echo ""
    echo -e "${BLUE}### By Difficulty${NC}"
    echo ""
    for diff in easy medium hard; do
        if echo "$by_difficulty" | jq -e --arg d "$diff" '.[$d]' > /dev/null 2>&1; then
            local diff_info
            diff_info=$(echo "$by_difficulty" | jq -r --arg d "$diff" '.[$d] | "\(.passed)/\(.total)"')
            local diff_score
            diff_score=$(echo "$by_difficulty" | jq -r --arg d "$diff" '.[$d].score * 100 | floor')
            echo "- $diff: $diff_info (${diff_score}%)"
        fi
    done
    
    if [ "$failed" -gt 0 ]; then
        echo ""
        echo -e "${BLUE}### Failure Analysis${NC}"
        echo ""
        
        local no_results_count wrong_chunk_count low_score_count
        no_results_count=$(echo "$no_results" | jq 'length')
        wrong_chunk_count=$(echo "$wrong_chunk" | jq 'length')
        low_score_count=$(echo "$low_score" | jq 'length')
        
        [ "$no_results_count" -gt 0 ] && echo "- No results: $no_results_count queries"
        [ "$wrong_chunk_count" -gt 0 ] && echo "- Wrong chunk retrieved: $wrong_chunk_count queries"
        [ "$low_score_count" -gt 0 ] && echo "- Low similarity score: $low_score_count queries"
    fi
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "Results saved to: $RESULTS"
    echo "Completed: $(date)"
    echo ""
}

# Run main
main "$@"
