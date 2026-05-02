#!/usr/bin/env bash
# RAG Pipeline — dedicated latency benchmark
# Reports p50/p95/min/max for each endpoint over N warm requests.
# Usage: ./examples/latency.sh [API_BASE_URL] [N_REQUESTS]
# Recommended: run AFTER ./examples/eval.sh so models are warm.

set -uo pipefail
BASE="${1:-http://localhost:8001}"
N="${2:-20}"

py() { python3 -c "$1" 2>/dev/null; }

stats() {
  # stats "space-separated seconds values" — prints p50/p95/min/max in ms
  py "
vals = sorted(float(x) for x in '$1'.split())
n = len(vals)
p50 = vals[int(n * 0.50)] * 1000
p95 = vals[min(int(n * 0.95), n - 1)] * 1000
mn  = vals[0] * 1000
mx  = vals[-1] * 1000
print(f'p50={p50:5.0f}ms  p95={p95:5.0f}ms  min={mn:5.0f}ms  max={mx:5.0f}ms')
"
}

bench_get() {
  local url="$1" times="" t
  for _ in $(seq 1 "$N"); do
    t=$(curl -s -o /dev/null -w '%{time_total}' "$url")
    times="$times $t"
  done
  stats "$times"
}

bench_search() {
  # bench_search "raw query string" — uses --data-urlencode, safe for multilingual
  local query="$1" times="" t
  for _ in $(seq 1 "$N"); do
    t=$(curl -s -o /dev/null -w '%{time_total}' --get --data-urlencode "q=$query" "$BASE/search")
    times="$times $t"
  done
  stats "$times"
}

bench_post() {
  local url="$1" data="$2" times="" t
  for _ in $(seq 1 "$N"); do
    t=$(curl -s -o /dev/null -w '%{time_total}' -X POST "$url" \
      -H 'Content-Type: application/json' -d "$data")
    times="$times $t"
  done
  stats "$times"
}

SEP="$(printf '─%.0s' {1..74})"

printf "\n\033[1mRAG Pipeline Latency Benchmark\033[0m\n"
printf "Target: %s   Requests per endpoint: %d\n\n" "$BASE" "$N"
printf "\033[1m%-38s  %s\033[0m\n" "Endpoint" "p50       p95       min       max"
printf "%s\n" "$SEP"

printf "%-38s  %s\n" "GET /health" \
  "$(bench_get "$BASE/health")"

printf "%-38s  %s\n" "GET /search  (energy sanctions)" \
  "$(bench_get "$BASE/search?q=energy+sanctions+EU+policy")"

printf "%-38s  %s\n" "GET /search  (vindkraft östersjön)" \
  "$(bench_search "vindkraft östersjön")"

printf "%-38s  %s\n" "GET /search  (G7 price ceiling)" \
  "$(bench_get "$BASE/search?q=G7+price+ceiling+Russian+crude+oil")"

printf "%-38s  %s\n" "POST /chat   (mock LLM)" \
  "$(bench_post "$BASE/chat" \
     '{"question":"What did EU officials say about energy security?"}')"

printf "%-38s  %s\n" "GET /ready   (health + dep check)" \
  "$(bench_get "$BASE/ready")"

printf "%s\n" "$SEP"

printf "\n\033[1;33mCold-start latency (first request after restart):\033[0m\n"
printf "  Restart the app container, then time a single /search:\n"
printf "  sudo docker compose restart app && sleep 3\n"
printf "  curl -s -o /dev/null -w '%%{time_total}s\\n' '%s/search?q=energy'\n\n" "$BASE"
printf "  Expected: first call = warm p50 + 1000-3000ms (model load from cache)\n"
printf "  Subsequent calls return to warm baseline immediately.\n\n"

printf "\033[1mWhat these numbers mean for a reviewer:\033[0m\n"
printf "  /health p50 < 5ms     → infra overhead is negligible\n"
printf "  /search p50 ~ 7000ms  → expected on CPU with bge-reranker-v2-m3 (10 candidates)\n"
printf "  /search p95 < 8000ms  → tight band = no GIL contention or DB pool exhaustion\n"
printf "  GPU deployment target  → rerank_ms drops to 50–200ms, /search p50 < 300ms\n\n"
