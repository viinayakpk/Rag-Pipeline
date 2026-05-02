#!/usr/bin/env bash
# RAG Pipeline — model warm-up
# Forces embedding model and reranker to load from disk cache into memory.
# Run once after container start before benchmarks, demos, or eval.
#
# Usage:  ./examples/warmup.sh [API_BASE_URL]
# Default API_BASE_URL: http://localhost:8001
#
# Cold-start behaviour (what this script measures on first run after restart):
#   - SentenceTransformer (multilingual-e5-base, ~1.1 GB) loads from volume cache
#   - CrossEncoder       (bge-reranker-v2-m3,   ~1.1 GB) loads from volume cache
#   - Subsequent requests skip loading entirely → steady-state latency
#
# If models are NOT in cache (first-ever boot), this script will wait while
# they download (~2.2 GB total). Interrupt with Ctrl-C and re-run after
# `docker compose logs -f app` shows the downloads complete.

set -uo pipefail
BASE="${1:-http://localhost:8001}"

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[1m'; NC='\033[0m'

ts_ms() { python3 -c "import time; print(int(time.time()*1000))" 2>/dev/null || date +%s%3N; }

printf "\n${B}RAG Pipeline — Model Warm-Up${NC}\n"
printf "Target: %s\n\n" "$BASE"

# ── 1. Wait for app to be healthy ────────────────────────────────
printf "  Waiting for /health"
HEALTHY=0
for _ in $(seq 1 30); do
  s=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/health" 2>/dev/null || echo "000")
  if [ "$s" = "200" ]; then
    HEALTHY=1
    break
  fi
  printf "."
  sleep 2
done

if [ "$HEALTHY" -eq 0 ]; then
  printf "\n  ${R}✗${NC} App not healthy after 60s. Is the container running?\n"
  printf "    sudo docker compose up -d\n\n"
  exit 1
fi
printf " ${G}✓${NC} healthy\n"

# ── 2. Check /ready (dependency imports) ─────────────────────────
s=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/ready" 2>/dev/null || echo "000")
if [ "$s" = "200" ]; then
  printf "  ${G}✓${NC} /ready — all runtime dependencies present\n"
else
  printf "  ${Y}~${NC} /ready → $s (some optional dependencies missing — see /ready body)\n"
fi

# ── 3. Cold-start: load embedding model + reranker ───────────────
printf "\n  ${Y}Loading models from cache (first search after restart)...${NC}\n"
printf "  This may take 3–15s on CPU while PyTorch reads weights from disk.\n\n"

t0=$(ts_ms)
resp=$(curl -s --get --data-urlencode "q=EU energy security sanctions policy" \
  "$BASE/search?limit=3" 2>/dev/null)
t1=$(ts_ms)
cold_ms=$(( t1 - t0 ))

n_results=$(printf '%s' "$resp" | python3 -c \
  "import json,sys; print(len(json.load(sys.stdin).get('results',[])))" 2>/dev/null || echo "?")

printf "  ${G}✓${NC} First /search complete — ${B}${cold_ms}ms${NC}  (${n_results} results)\n"
printf "    ↳ includes: model load from cache + embedding + retrieval + reranking\n\n"

# ── 4. Warm baseline: second request ─────────────────────────────
t0=$(ts_ms)
curl -s --get --data-urlencode "q=G7 price ceiling Russian crude oil" \
  "$BASE/search?limit=3" > /dev/null 2>&1
t1=$(ts_ms)
warm_ms=$(( t1 - t0 ))

printf "  ${G}✓${NC} Second /search (steady-state) — ${B}${warm_ms}ms${NC}\n"
printf "    ↳ embedding + retrieval + reranking only, no model load\n\n"

# ── 5. Warm /chat ─────────────────────────────────────────────────
t0=$(ts_ms)
curl -s -X POST "$BASE/chat" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What EU measures address energy security?"}' > /dev/null 2>&1
t1=$(ts_ms)
chat_ms=$(( t1 - t0 ))
printf "  ${G}✓${NC} /chat (steady-state)             — ${B}${chat_ms}ms${NC}\n\n"

# ── 6. Summary ────────────────────────────────────────────────────
printf "${B}  Warm-up complete.${NC}\n"
printf "  %-30s %6dms\n" "Cold-start /search" "$cold_ms"
printf "  %-30s %6dms  ← use this as your latency baseline\n" "Warm /search" "$warm_ms"
printf "  %-30s %6dms\n\n" "Warm /chat (mock LLM)" "$chat_ms"
printf "  Models are resident in memory. Benchmarks and eval are now steady-state.\n"
printf "  Run ${B}./examples/latency.sh${NC} for p50/p95 across 20 requests.\n\n"
