#!/bin/bash
#
# Common test library for Tier 2 Memory Retrieval Tests
# Sourced by run-tests.sh and run-tests-tribal.sh
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check common dependencies
check_common_deps() {
    if ! command -v jq &> /dev/null; then
        echo "Error: jq is required but not installed"
        exit 1
    fi
    
    if ! command -v bc &> /dev/null; then
        echo "Error: bc is required but not installed"
        exit 1
    fi
}

# Initialize results structure
init_results() {
    local agent="${1:-unknown}"
    cat <<EOF
{
  "summary": { "total": 0, "passed": 0, "failed": 0, "score": 0 },
  "timestamp": "$(date -Iseconds)",
  "agent": "$agent",
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

# Check if expected substrings are found in results
# Arguments: expected_json, got_results
# Returns: 0 if found, 1 if not found
check_expected_in_results() {
    local expected_json="$1"
    local got_results="$2"
    
    local found=false
    local expected_count
    expected_count=$(echo "$expected_json" | jq 'length')
    
    for ((i=0; i<expected_count; i++)); do
        local expected_str
        expected_str=$(echo "$expected_json" | jq -r ".[$i]" | tr '[:upper:]' '[:lower:]')
        
        # Check each result for this expected string
        local result_count
        result_count=$(echo "$got_results" | jq 'length')
        
        for ((j=0; j<result_count; j++)); do
            local result_text
            result_text=$(echo "$got_results" | jq -r ".[$j].text" | tr '[:upper:]' '[:lower:]')
            
            if [[ "$result_text" == *"$expected_str"* ]]; then
                found=true
                break 2
            fi
        done
    done
    
    if [ "$found" = true ]; then
        return 0
    else
        return 1
    fi
}

# Check similarity scores
# Returns: "pass" if any score >= 0.5, "low_score" otherwise
check_similarity_scores() {
    local got_results="$1"
    local threshold="${2:-0.5}"
    
    local max_score
    max_score=$(echo "$got_results" | jq '[.[].score // 0] | max // 0')
    
    if (( $(echo "$max_score >= $threshold" | bc -l) )); then
        echo "pass"
    else
        echo "low_score"
    fi
}

# Format a single query result for JSON output
format_query_result() {
    local id="$1"
    local query="$2"
    local status="$3"
    local expected_json="$4"
    local got_results="$5"
    local failure_reason="$6"
    local category="$7"
    local difficulty="$8"
    local expected_behavior="$9"
    
    jq -n \
        --arg id "$id" \
        --arg query "$query" \
        --arg status "$status" \
        --argjson expected "$expected_json" \
        --argjson got "$got_results" \
        --arg failure_reason "$failure_reason" \
        --arg category "$category" \
        --arg difficulty "$difficulty" \
        --arg expected_behavior "$expected_behavior" \
        '{
            id: $id,
            query: $query,
            status: $status,
            expected: $expected,
            got: $got,
            failure_reason: $failure_reason,
            category: $category,
            difficulty: $difficulty,
            expected_behavior: $expected_behavior
        }'
}

# Print test result line
print_result() {
    local status="$1"
    local id="$2"
    local query="$3"
    local failure_reason="$4"
    
    if [ "$status" = "PASS" ]; then
        echo -e "[${GREEN}PASS${NC}] $id: $query"
    else
        echo -e "[${RED}FAIL${NC}] $id: $query (${YELLOW}$failure_reason${NC})"
    fi
}

# Generate final summary
generate_summary() {
    local results_json="$1"
    local total_queries="$2"
    local passed="$3"
    local failed="$4"
    local agent="$5"
    local results_file="$6"
    
    # Calculate score
    local score
    if [ "$total_queries" -gt 0 ]; then
        score=$(echo "scale=4; $passed / $total_queries" | bc)
    else
        score=0
    fi
    
    # Calculate category stats
    local by_category
    by_category=$(echo "$results_json" | jq -c '
        group_by(.category) | 
        map({
            key: .[0].category,
            value: {
                passed: [.[] | select(.status == "PASS")] | length,
                total: length,
                score: (([.[] | select(.status == "PASS")] | length) / length)
            }
        }) | from_entries
    ')
    
    # Calculate difficulty stats
    local by_difficulty
    by_difficulty=$(echo "$results_json" | jq -c '
        group_by(.difficulty) | 
        map({
            key: .[0].difficulty,
            value: {
                passed: [.[] | select(.status == "PASS")] | length,
                total: length,
                score: (([.[] | select(.status == "PASS")] | length) / length)
            }
        }) | from_entries
    ')
    
    # Collect failure IDs
    local no_results wrong_chunk low_score
    no_results=$(echo "$results_json" | jq '[.[] | select(.failure_reason == "no_results") | .id]')
    wrong_chunk=$(echo "$results_json" | jq '[.[] | select(.failure_reason == "wrong_chunk") | .id]')
    low_score=$(echo "$results_json" | jq '[.[] | select(.failure_reason == "low_score") | .id]')
    
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
        --arg agent "$agent" \
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
    echo "$final_results" | jq '.' > "$results_file"
    
    # Print summary
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo -e "${BLUE}## Summary${NC}"
    echo ""
    
    local score_pct
    score_pct=$(echo "scale=4; $score * 100" | bc)
    
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
    echo "$by_category" | jq -r 'to_entries | sort_by(.key) | .[] | "- \(.key): \(.value.passed)/\(.value.total) (\(.value.score * 100 | floor)%)"'
    
    echo ""
    echo -e "${BLUE}### By Difficulty${NC}"
    echo ""
    echo "$by_difficulty" | jq -r 'to_entries | sort_by(.key) | .[] | "- \(.key): \(.value.passed)/\(.value.total) (\(.value.score * 100 | floor)%)"'
    
    echo ""
    echo -e "${BLUE}### Failure Analysis${NC}"
    echo ""
    echo "- No results: $(echo "$no_results" | jq 'length') queries"
    echo "- Wrong chunk retrieved: $(echo "$wrong_chunk" | jq 'length') queries"
    echo "- Low similarity score: $(echo "$low_score" | jq 'length') queries"
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "Results saved to: $results_file"
    echo "Completed: $(date)"
}
