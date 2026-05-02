#!/usr/bin/env bash
# RAG Pipeline — candidate count ablation
# Tests HYBRID_CANDIDATE_COUNT at 10, 20, 30.
# For each: restarts app, warms up, checks retrieval quality (section 2),
# then measures warm /search latency from container logs.
#
# Usage: ./examples/ablation.sh [API_BASE_URL]
# Requires: sudo access for docker restart (or add user to docker group)
#
# Runtime: ~10 minutes total (each candidate count needs 3–4 warm requests)

set -uo pipefail
BASE="${1:-http://localhost:8001}"
ENV_FILE="$(dirname "$0")/../.env"
DOCKER_CMD="${DOCKER_CMD:-docker}"

B='\033[1m'; C='\033[0;36m'; G='\033[0;32m'; Y='\033[1;33m'; NC='\033[0m'

section() { printf "\n${B}${C}%s${NC}\n" "$1"; }

wait_healthy() {
  printf "  Waiting for /health"
  for _ in $(seq 1 30); do
    s=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/health" 2>/dev/null || echo "000")
    [ "$s" = "200" ] && { printf " ${G}✓${NC}\n"; return 0; }
    printf "."; sleep 2
  done
  printf "\n  App not healthy after 60s\n"; return 1
}

warm_once() {
  printf "  Warming (one search to load models)..."
  curl -s --get --data-urlencode "q=EU energy policy sanctions" "$BASE/search?limit=3" > /dev/null
  printf " ${G}done${NC}\n"
}

measure_rerank_ms() {
  # Refresh sudo credentials so log grab inside $() subshell doesn't need a TTY
  $DOCKER_CMD info > /dev/null 2>&1 || true
  local since
  since=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  curl -s --get --data-urlencode "q=G7 price ceiling Russian crude oil" "$BASE/search?limit=5" > /dev/null
  sleep 2
  $DOCKER_CMD compose logs app --since="$since" --tail=20 2>/dev/null \
    | grep "app.api.search search retrieve_ms" | tail -1 \
    | grep -oE "rerank_ms=[0-9]+" | cut -d= -f2 || echo "?"
}

hit_at_1() {
  # Each probe has its own anchor keyword — avoids false positives from a shared bag.
  # Format: "query|||anchor_that_appears_only_in_the_correct_chunk"
  hits=0
  local probes=(
    "G7 price ceiling Russian crude oil|||price cap"
    "shadow fleet tankers oil cap circumvent|||shadow fleet"
    "Sanktionen Drittländer Russland|||sanktionen"
    "partnerskapsavtale energi forsvar|||partnerskapsavtale"
  )
  for probe in "${probes[@]}"; do
    q="${probe%%|||*}"
    anchor="${probe##*|||}"
    resp=$(curl -s --get --data-urlencode "q=$q" "$BASE/search?limit=3")
    rank=$(printf '%s' "$resp" | python3 -c "
import json, sys
results = json.load(sys.stdin).get('results', [])
anchor = sys.argv[1].lower()
for i, r in enumerate(results, 1):
    c = (r.get('text','') + r.get('title','')).lower()
    if anchor in c:
        print(i); sys.exit(0)
print(0)
" "$anchor" 2>/dev/null || echo 0)
    [ "$rank" = "1" ] && hits=$((hits+1))
  done
  echo "$hits/4"
}

# ── Table header ──────────────────────────────────────────────────────────────
printf "\n${B}RAG Pipeline — Candidate Count Ablation${NC}\n"
printf "ENV_FILE: %s\n" "$ENV_FILE"
printf "\n${B}%-10s  %-14s  %-20s  %s${NC}\n" \
  "count" "rerank_ms" "Hit@1 (4 probes)" "verdict"
printf "%-10s  %-14s  %-20s  %s\n" \
  "----------" "--------------" "--------------------" "-------"

for COUNT in 10 20 30; do
  section "Testing HYBRID_CANDIDATE_COUNT=$COUNT"

  # Update .env
  if grep -q "^HYBRID_CANDIDATE_COUNT=" "$ENV_FILE"; then
    sed -i "s/^HYBRID_CANDIDATE_COUNT=.*/HYBRID_CANDIDATE_COUNT=$COUNT/" "$ENV_FILE"
  else
    echo "HYBRID_CANDIDATE_COUNT=$COUNT" >> "$ENV_FILE"
  fi
  printf "  .env updated → HYBRID_CANDIDATE_COUNT=%d\n" "$COUNT"

  # Recreate container to pick up new .env (restart reuses old env)
  printf "  Recreating app container..."
  $DOCKER_CMD compose up -d --no-deps --force-recreate app > /dev/null 2>&1
  printf " ${G}done${NC}\n"

  wait_healthy || continue

  # Verify the running process actually loaded the new value
  live_count=$($DOCKER_CMD compose exec -T app python3 -c \
    "from app.config import get_settings; print(get_settings().hybrid_candidate_count)" 2>/dev/null || echo "?")
  if [ "$live_count" != "$COUNT" ]; then
    printf "  ${Y}WARNING: live config=%s, expected %d — env may not have applied${NC}\n" "$live_count" "$COUNT"
  else
    printf "  Live config verified: hybrid_candidate_count=%s\n" "$live_count"
  fi

  warm_once

  # Measure rerank_ms (two probes, take the second — first may include GC noise)
  printf "  Measuring rerank_ms (probe 1)..."
  curl -s --get --data-urlencode "q=EU energy security sanctions" "$BASE/search?limit=5" > /dev/null
  printf " done\n"
  printf "  Measuring rerank_ms (probe 2)..."
  rerank=$(measure_rerank_ms)
  printf " %sms\n" "$rerank"

  # Quality spot-check (4 key queries)
  printf "  Checking retrieval quality..."
  quality=$(hit_at_1)
  printf " %s\n" "$quality"

  # Verdict
  if [ "$rerank" != "?" ] && [ "$quality" = "4/4" ]; then
    verdict="${G}✓ viable${NC}"
  elif [ "$quality" != "4/4" ]; then
    verdict="${Y}~ quality drop${NC}"
  else
    verdict="${Y}~ no log data${NC}"
  fi

  printf "\n${B}%-10s  %-14s  %-20s  " "$COUNT" "${rerank}ms" "$quality"
  printf "${verdict}${NC}\n"
done

# ── Restore baseline (30) if nothing degraded, otherwise leave last tested ───
printf "\n${B}Ablation complete.${NC}\n"
printf "Review the table above. If count=10 shows 4/4 quality, update .env permanently:\n"
printf "  sed -i 's/^HYBRID_CANDIDATE_COUNT=.*/HYBRID_CANDIDATE_COUNT=10/' %s\n" "$ENV_FILE"
printf "Then apply: %s compose up -d --no-deps --force-recreate app\n\n" "$DOCKER_CMD"
