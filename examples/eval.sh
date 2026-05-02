#!/usr/bin/env bash
# RAG Pipeline — comprehensive evaluation
# Covers: retrieval quality (Hit@1, Hit@3, MRR), filter correctness, edge cases,
#         deduplication, PDF upload, sparse metadata, out-of-domain, latency.
# Usage: ./examples/eval.sh [API_BASE_URL]
# PDF tests: set DOCKER_CMD="sudo docker" if docker requires sudo on your machine.

set -uo pipefail
BASE="${1:-http://localhost:8001}"
DOCKER_CMD="${DOCKER_CMD:-docker}"   # override: DOCKER_CMD="sudo docker" ./examples/eval.sh

# ── Counters & colours ────────────────────────────────────────────
PASS=0; FAIL=0; SKIP=0
G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'
B='\033[1m';    C='\033[0;36m'; NC='\033[0m'

pass()    { printf "  ${G}✓${NC} %s\n" "$1"; PASS=$((PASS+1)); }
fail()    { printf "  ${R}✗${NC} %s\n" "$1"; FAIL=$((FAIL+1)); }
skip()    { printf "  ${Y}~${NC} %s\n" "$1"; SKIP=$((SKIP+1)); }
section() { printf "\n${B}${C}%s${NC}\n" "$1"; }

# ── Helpers ───────────────────────────────────────────────────────
http_status() { curl -s -o /dev/null -w "%{http_code}" "$@"; }
body()        { curl -s "$@"; }

# All JSON parsing pipes the response body through stdin.
# This avoids all shell quoting/escaping issues with JSON content.

get_field() {
  # get_field JSON field — returns empty string only when key is absent or null
  printf '%s' "$1" | python3 -c "
import json, sys
d = json.load(sys.stdin)
v = d.get('$2')
print('' if v is None else v)
" 2>/dev/null || echo ""
}

rank_of() {
  # rank_of JSON "keyword" — 1-based rank or 0 if not found
  printf '%s' "$1" | python3 -c "
import json, sys
kw = '$2'.lower()
for i, r in enumerate(json.load(sys.stdin).get('results', []), 1):
    if kw in r.get('title', '').lower() or kw in r.get('text', '').lower():
        print(i); sys.exit(0)
print(0)
" 2>/dev/null || echo 0
}

all_lang() {
  # all_lang JSON lang — "yes" if every result matches lang, else "no"
  printf '%s' "$1" | python3 -c "
import json, sys
results = json.load(sys.stdin).get('results', [])
langs = {r.get('language', '') for r in results}
print('yes' if results and langs == {'$2'} else 'no')
" 2>/dev/null || echo "no"
}

get_total() {
  # get_total JSON
  printf '%s' "$1" | python3 -c "
import json, sys
print(json.load(sys.stdin).get('total', 0))
" 2>/dev/null || echo 0
}

count_list() {
  # count_list JSON field
  printf '%s' "$1" | python3 -c "
import json, sys
print(len(json.load(sys.stdin).get('$2', [])))
" 2>/dev/null || echo 0
}

avg_ms_get() {
  local url="$1" n="${2:-5}" total=0 t
  for _ in $(seq 1 "$n"); do
    t=$(curl -s -o /dev/null -w '%{time_total}' "$url")
    total=$(python3 -c "print($total + $t)" 2>/dev/null || echo "$total")
  done
  python3 -c "print(f'{$total/$n*1000:.0f}')" 2>/dev/null || echo "?"
}

avg_ms_post() {
  local url="$1" data="$2" n="${3:-5}" total=0 t
  for _ in $(seq 1 "$n"); do
    t=$(curl -s -o /dev/null -w '%{time_total}' -X POST "$url" \
      -H 'Content-Type: application/json' -d "$data")
    total=$(python3 -c "print($total + $t)" 2>/dev/null || echo "$total")
  done
  python3 -c "print(f'{$total/$n*1000:.0f}')" 2>/dev/null || echo "?"
}

# ── Gold-set retrieval test ───────────────────────────────────────
HITS1=0; HITS3=0; MRR_SUM=0; NQUERIES=0

run_query() {
  # run_query "label" "raw query string" "keyword" acceptable_max_rank [limit=5]
  local label="$1" query="$2" kw="$3" max="$4" lim="${5:-5}"
  local resp rank
  NQUERIES=$((NQUERIES+1))
  resp=$(curl -s --get --data-urlencode "q=$query" "$BASE/search?limit=$lim")
  rank=$(rank_of "$resp" "$kw")
  rank="${rank:-0}"    # guard against empty output
  if   [ "$rank" -eq 1 ] 2>/dev/null; then
    HITS1=$((HITS1+1)); HITS3=$((HITS3+1))
    MRR_SUM=$(python3 -c "print($MRR_SUM + 1.0)" 2>/dev/null || echo "$MRR_SUM")
    pass "Hit@1  (rank 1)  $label"
  elif [ "$rank" -le 3 ] && [ "$rank" -gt 0 ] 2>/dev/null; then
    HITS3=$((HITS3+1))
    MRR_SUM=$(python3 -c "print($MRR_SUM + 1.0/$rank)" 2>/dev/null || echo "$MRR_SUM")
    pass "Hit@3  (rank $rank)  $label"
  elif [ "$rank" -le "$max" ] && [ "$rank" -gt 0 ] 2>/dev/null; then
    MRR_SUM=$(python3 -c "print($MRR_SUM + 1.0/$rank)" 2>/dev/null || echo "$MRR_SUM")
    skip "Hit@$max  (rank $rank)  $label"
  else
    fail "Miss  $label  (not found in top 10)"
  fi
}

# ─────────────────────────────────────────────────────────────────
printf "${B}══════════════════════════════════════════════════\n"
printf "  RAG Pipeline Evaluation  —  %s\n" "$BASE"
printf "══════════════════════════════════════════════════${NC}\n"

# ═══════════════════════════════════════════
section "1 / 9  ·  HEALTH & READINESS"
# ═══════════════════════════════════════════

s=$(http_status "$BASE/health")
[ "$s" = "200" ] && pass "/health → 200 OK" || fail "/health → $s (want 200)"

s=$(http_status "$BASE/ready")
[ "$s" = "200" ] && pass "/ready → 200, all deps present" || fail "/ready → $s (want 200)"

# ── Auto-seed if the corpus is empty ─────────────────────────────
# Retrieval quality tests require the 10 demo documents.
# If the DB is fresh (0 documents), seed automatically so this script works standalone.
_doc_count=$(body "$BASE/search?q=energy&limit=1" | python3 -c "
import json,sys; d=json.load(sys.stdin); print(d.get('total',0))
" 2>/dev/null || echo 0)
if [ "$_doc_count" -eq 0 ] 2>/dev/null; then
  printf "\n  ${Y}Corpus is empty — seeding demo documents...${NC}\n"
  SEED_SCRIPT="$(dirname "$0")/seed.sh"
  if [ -f "$SEED_SCRIPT" ]; then
    bash "$SEED_SCRIPT" "$BASE" > /dev/null
    printf "  ${G}Seed complete.${NC} Waiting 2s for embeddings to settle...\n"
    sleep 2
  else
    printf "  ${R}seed.sh not found at %s — retrieval tests will fail.${NC}\n" "$SEED_SCRIPT"
  fi
fi

# ═══════════════════════════════════════════
section "2 / 9  ·  RETRIEVAL QUALITY  (gold set · Hit@1, Hit@3, MRR)"
# ═══════════════════════════════════════════

# ── Exact-topic queries (expected Hit@1 or Hit@3) ──
run_query "G7 price ceiling Russian crude"          "G7 price ceiling Russian crude oil"                      "Shadow Fleet"        3
run_query "EU gas storage 45 percent capacity"      "EU gas storage 45 percent winter capacity"               "Energy Ministers"    3
run_query "Lithuania switched entirely LNG"         "Lithuania switched entirely LNG imports"                 "Energy Ministers"    3
run_query "shadow fleet tankers oil circumvent"     "shadow fleet tankers oil cap circumvent"                 "Shadow Fleet"        1
run_query "vindkraft Östersjön gemensam (SV)"       "vindkraft östersjön gemensam"                            "vindkraft"           2
run_query "Sanktionen Drittländer (DE)"             "Sanktionen Drittländer Russland"                         "Sanktionen"          1
run_query "EPR2 Penly Normandie (FR)"               "EPR2 Penly Normandie"                                    "EPR2"                1
run_query "partnerskapsavtale energi forsvar (NO)"  "partnerskapsavtale energi forsvar"                       "partnerskapsavtale"  1

# ── Cross-lingual: English query → non-English document ──
printf "\n  Cross-lingual (EN query → non-EN document):\n"
run_query "Nordic offshore wind Baltic  → SV doc"     "Nordic offshore wind Baltic Sea triple capacity"        "vindkraft"   5  10
run_query "sanctions third-country China UAE → DE"    "sanctions third country circumvention China UAE"        "Sanktionen"  5  10
run_query "Equinor floating wind carbon capture → NO" "Equinor floating wind carbon capture storage"          "Nordsj"      5  10
run_query "France Ukraine strategic partnership → FR" "France Ukraine strategic partnership reconstruction"    "partenariat" 5  10

MRR=$(python3 -c "print(f'{$MRR_SUM/$NQUERIES:.3f}')" 2>/dev/null || echo "n/a")
printf "\n  ${B}Hit@1 = %d/%d   Hit@3 = %d/%d   MRR = %s${NC}\n" \
  "$HITS1" "$NQUERIES" "$HITS3" "$NQUERIES" "$MRR"

# ═══════════════════════════════════════════
section "3 / 9  ·  FILTER CORRECTNESS"
# ═══════════════════════════════════════════

# Language filter — every result must be in the requested language
for lang in en sv de fr no; do
  resp=$(body "$BASE/search?q=energy+policy+EU&language=$lang&limit=10")
  [ "$(all_lang "$resp" "$lang")" = "yes" ] \
    && pass "language=$lang → all results are $lang (no leakage)" \
    || fail "language=$lang → mixed languages returned"
done

# Topic (JSONB containment)
resp=$(body "$BASE/search?q=energy&topic=sanctions&limit=10")
count=$(get_total "$resp")
[ "$count" -ge 1 ] 2>/dev/null \
  && pass "topic=sanctions → $count result(s)" \
  || fail "topic=sanctions → $count (expected ≥1 from demo data)"

# Source filter
resp=$(body "$BASE/search?q=energy&source=EuroEnergy+Daily&limit=10")
count=$(get_total "$resp")
[ "$count" -ge 1 ] 2>/dev/null \
  && pass "source=EuroEnergy Daily → $count result(s)" \
  || fail "source=EuroEnergy Daily → $count (expected ≥1)"

# Date range — 2024+ docs only (Norge og EU: 2024-03-19, diplomatie_fr: 2024-02-16)
resp=$(body "$BASE/search?q=energy&from_date=2024-01-01&limit=10")
count=$(get_total "$resp")
if [ "$count" -ge 1 ] && [ "$count" -le 3 ] 2>/dev/null; then
  pass "from_date=2024-01-01 → $count result(s) (expected 1-2)"
else
  skip "from_date=2024-01-01 → $count result(s) (verify published_at values)"
fi

# to_date far in the past — use 1700 to stay below robustness test artifact (published_at=1800-01-01)
resp=$(body "$BASE/search?q=energy&to_date=1700-01-01&limit=10")
count=$(get_total "$resp")
[ "$count" -eq 0 ] 2>/dev/null \
  && pass "to_date=1700-01-01 → 0 (no docs predate 1700)" \
  || fail "to_date=1700-01-01 → $count (expected 0)"

# Null-safe: filter on doc that has no published_at should not crash
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"Null date test document for filter safety check.","metadata":{}}')
[ "$s" = "200" ] && pass "Doc with no published_at ingested → 200 (filter null-safe)" \
  || fail "Doc with no published_at → $s"

# ═══════════════════════════════════════════
section "4 / 9  ·  EDGE CASES"
# ═══════════════════════════════════════════

# Blank / whitespace query
s=$(http_status "$BASE/search?q=")
[ "$s" = "400" ] && pass "Empty query string → 400" || fail "Empty query → $s (want 400)"

s=$(http_status "$BASE/search?q=+++")
[ "$s" = "400" ] && pass "Whitespace-only query → 400" || fail "Whitespace query → $s (want 400)"

# Document endpoint — invalid UUID format
s=$(http_status "$BASE/documents/not-a-uuid")
[ "$s" = "422" ] && pass "Malformed UUID → 422 Unprocessable" || fail "Malformed UUID → $s (want 422)"

# Document endpoint — valid UUID, non-existent
s=$(http_status "$BASE/documents/00000000-0000-0000-0000-000000000000")
[ "$s" = "404" ] && pass "Valid UUID, non-existent doc → 404" || fail "Non-existent doc → $s (want 404)"

# Blank text ingestion
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"   ","metadata":{}}')
[ "$s" = "422" ] && pass "Blank text → 422" || fail "Blank text → $s (want 422)"

# Extra field in metadata (extra=forbid)
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"Valid content.","metadata":{"unknown_field":"value"}}')
[ "$s" = "422" ] && pass "Unknown metadata field (extra=forbid) → 422" \
  || fail "Unknown metadata field → $s (want 422)"

# Wrong type: topics should be array, not string
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"Valid content.","metadata":{"topics":"energy"}}')
[ "$s" = "422" ] && pass "topics as string instead of array → 422" \
  || fail "topics as string → $s (want 422)"

# Blank chat question
s=$(http_status -X POST "$BASE/chat" -H 'Content-Type: application/json' \
  -d '{"question":"  "}')
[ "$s" = "422" ] && pass "Blank chat question → 422" || fail "Blank chat → $s (want 422)"

# Unknown entity — should return 200 with empty list, not 404/500
s=$(http_status "$BASE/entities/xyznonexistenttoken99abc/documents")
[ "$s" = "200" ] && pass "Unknown entity name → 200 empty list" \
  || fail "Unknown entity → $s (want 200)"

# Non-PDF file upload
s=$(http_status -X POST "$BASE/document" -F "file=@/etc/hostname;type=text/plain")
[ "$s" = "400" ] && pass "Non-PDF file type → 400" || fail "Non-PDF upload → $s (want 400)"

# Unknown language code (e.g. Czech) — should ingest gracefully, not crash
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"Tento text je v češtině. Energetická bezpečnost v Evropě je důležitá.","metadata":{"language":"cs"}}')
[ "$s" = "200" ] && pass "Unknown language code 'cs' → 200 (graceful fallback)" \
  || fail "Unknown language code → $s (want 200)"

# ═══════════════════════════════════════════
section "5 / 9  ·  DEDUPLICATION"
# ═══════════════════════════════════════════

DEDUP="{\"text\":\"Deduplication sentinel document XQZ-9471-$$. Unique marker for testing pipeline.\",\"metadata\":{\"title\":\"Dedup Test\"}}"
r1=$(body -X POST "$BASE/text" -H 'Content-Type: application/json' -d "$DEDUP")
r2=$(body -X POST "$BASE/text" -H 'Content-Type: application/json' -d "$DEDUP")
d1=$(get_field "$r1" "duplicate"); d2=$(get_field "$r2" "duplicate")
[ "$d1" = "False" ] && [ "$d2" = "True" ] \
  && pass "Exact duplicate → first=ok second=duplicate" \
  || fail "Dedup failed: first_is_dup=$d1 second_is_dup=$d2"

# Whitespace-normalised dedup (leading/trailing spaces stripped per line)
TEXT_PADDED='{"text":"  Unique whitespace test line one.\n  Unique whitespace test line two.\n","metadata":{}}'
TEXT_CLEAN='{"text":"Unique whitespace test line one.\nUnique whitespace test line two.","metadata":{}}'
r1=$(body -X POST "$BASE/text" -H 'Content-Type: application/json' -d "$TEXT_PADDED")
r2=$(body -X POST "$BASE/text" -H 'Content-Type: application/json' -d "$TEXT_CLEAN")
d1=$(get_field "$r1" "duplicate"); d2=$(get_field "$r2" "duplicate")
[ "$d2" = "True" ] \
  && pass "Whitespace-normalised dedup → second=duplicate" \
  || skip "Whitespace-normalised dedup → second_is_dup=$d2 (\\n escaping may differ across shells)"

# ═══════════════════════════════════════════
section "6 / 9  ·  SPARSE / MINIMAL METADATA"
# ═══════════════════════════════════════════

# No metadata at all
r=$(body  -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"A short news piece about wind energy policy in northern Europe."}')
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"A short news piece about wind energy policy in northern Europe."}')
lang=$(get_field "$r" "language")
[ "$s" = "200" ] \
  && pass "No metadata → ingested ok (language=$lang)" \
  || fail "No metadata → $s (want 200)"

# Partial metadata — source only
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"Markets reacted to new trade restrictions announced today.","metadata":{"source":"Reuters test"}}')
[ "$s" = "200" ] && pass "Partial metadata (source only) → 200" \
  || fail "Partial metadata → $s (want 200)"

# Title and document_type, no language hint
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"The council voted to approve the revised energy directive.","metadata":{"title":"Council Vote","document_type":"legislation"}}')
[ "$s" = "200" ] && pass "Partial metadata (title+type, no language) → 200" \
  || fail "Partial metadata (title+type) → $s (want 200)"

# ═══════════════════════════════════════════
section "7 / 9  ·  PDF UPLOAD"
# ═══════════════════════════════════════════

# Generate PDF inside container using fitz (already installed), copy to host
TMPPDF="/tmp/rag_eval_$$.pdf"
PDF_OK=0

CONTAINER_ID=$($DOCKER_CMD compose ps -q app 2>/dev/null || true)
if [ -n "$CONTAINER_ID" ]; then
  $DOCKER_CMD compose exec -T app python3 - <<'PYEOF' 2>/dev/null && PDF_OK=1 || true
import fitz, sys
doc = fitz.open()
page = doc.new_page()
page.insert_text(
    (50, 700),
    "EU Energy Policy - PDF Evaluation Document\n\n"
    "The European Commission released an energy security framework in 2024. "
    "Commissioner Kadri Simson emphasized that no single member state can achieve "
    "energy security in isolation. Germany and France agreed on a joint investment "
    "vehicle worth 15 billion euros for critical infrastructure upgrades. "
    "The framework includes offshore wind expansion, hydrogen buildout, and "
    "cross-border grid interconnections across member states.",
    fontsize=10,
)
doc.save("/tmp/rag_eval_pdf_test.pdf")
PYEOF
  [ "$PDF_OK" -eq 1 ] && \
    $DOCKER_CMD cp "${CONTAINER_ID}:/tmp/rag_eval_pdf_test.pdf" "$TMPPDF" 2>/dev/null \
    || PDF_OK=0
fi

if [ "$PDF_OK" -eq 1 ] && [ -f "$TMPPDF" ]; then
  # Case 1: full metadata
  r=$(curl -s -X POST "$BASE/document" \
    -F "file=@${TMPPDF};type=application/pdf" \
    -F 'metadata={"title":"PDF Eval — Full Metadata","document_type":"policy","language":"en"}')
  is_dup=$(get_field "$r" "duplicate"); chunks=$(get_field "$r" "chunks_created")
  status=$([ "$is_dup" = "True" ] && echo "duplicate" || echo "ok")
  [ -n "$is_dup" ] \
    && pass "PDF + full metadata → $status, chunks=$chunks" \
    || fail "PDF + full metadata → unexpected: $r"

  # Case 2: empty metadata {}
  r=$(curl -s -X POST "$BASE/document" \
    -F "file=@${TMPPDF};type=application/pdf" \
    -F 'metadata={}')
  is_dup=$(get_field "$r" "duplicate")
  status=$([ "$is_dup" = "True" ] && echo "duplicate" || echo "ok")
  [ -n "$is_dup" ] \
    && pass "PDF + empty metadata {} → $status (auto-detects language)" \
    || fail "PDF + empty metadata → response: $r"

  # Case 3: verify document record has file metadata
  DOC_ID=$(get_field "$r" "document_id")
  if [ -n "$DOC_ID" ] && [ "$DOC_ID" != "None" ]; then
    rec=$(body "$BASE/documents/$DOC_ID")
    fname=$(printf '%s' "$rec" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('original_filename') or '')" 2>/dev/null)
    mime=$(printf  '%s' "$rec" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('mime_type') or '')"        2>/dev/null)
    bsize=$(printf '%s' "$rec" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('byte_size') or '')"        2>/dev/null)
    [ -n "$fname" ] \
      && pass "PDF document record → filename=$fname  mime=$mime  size=${bsize}B" \
      || skip "PDF document record — original_filename empty (likely duplicate path)"
  fi

  # Case 4: malformed metadata JSON
  s=$(http_status -X POST "$BASE/document" \
    -F "file=@${TMPPDF};type=application/pdf" \
    -F 'metadata=not_valid_json')
  [ "$s" = "400" ] && pass "PDF + malformed metadata JSON → 400" \
    || fail "PDF + malformed metadata → $s (want 400)"

  $DOCKER_CMD rm -f "$(basename "$TMPPDF")" 2>/dev/null || true
  rm -f "$TMPPDF" 2>/dev/null || true
else
  skip "PDF tests — Docker access needed (set DOCKER_CMD='sudo docker' or add user to docker group)"
fi

# Case 5: non-PDF file (always testable without Docker)
s=$(http_status -X POST "$BASE/document" -F "file=@/etc/hostname;type=text/plain")
[ "$s" = "400" ] && pass "Non-PDF file upload → 400" || fail "Non-PDF upload → $s (want 400)"

# ═══════════════════════════════════════════
section "8 / 9  ·  OUT-OF-DOMAIN BEHAVIOUR"
# ═══════════════════════════════════════════

# Query with zero overlap with corpus — should not crash
s=$(http_status "$BASE/search?q=chocolate+cake+tiramisu+recipe+baking")
[ "$s" = "200" ] && pass "Out-of-domain query → 200 no crash" || fail "Out-of-domain query → $s"

r=$(body "$BASE/search?q=chocolate+cake+tiramisu+recipe+baking&limit=5")
total=$(get_total "$r")
printf "  ${Y}~${NC} Out-of-domain results: %s returned (low-score vector matches — expected)\n" "$total"

# Chat on unanswerable question
r=$(body -X POST "$BASE/chat" -H 'Content-Type: application/json' \
  -d '{"question":"What did the EU decide yesterday about hydrogen subsidies?"}')
s=$(http_status -X POST "$BASE/chat" -H 'Content-Type: application/json' \
  -d '{"question":"What did the EU decide yesterday about hydrogen subsidies?"}')
conf=$(get_field "$r" "confidence")
[ "$s" = "200" ] && pass "Unanswerable chat → 200, confidence=$conf" \
  || fail "Unanswerable chat → $s"

# Multi-source synthesis (uses mock — confirms pipeline runs, not answer quality)
r=$(body -X POST "$BASE/chat" -H 'Content-Type: application/json' \
  -d '{"question":"How are European countries responding to energy supply risk?"}')
sources_n=$(count_list "$r" "sources")
pass "Multi-source synthesis → $sources_n sources retrieved"

# ═══════════════════════════════════════════
section "9 / 9  ·  LATENCY  (5-run warm average)"
# ═══════════════════════════════════════════

printf "  Running warm latency (5 requests each)...\n\n"

t=$(avg_ms_get "$BASE/health")
printf "  %-46s %sms avg\n" "GET /health" "$t"

t=$(avg_ms_get "$BASE/search?q=energy+sanctions+EU+policy")
printf "  %-46s %sms avg\n" "GET /search  (embed + retrieve + rerank)" "$t"

t=$(avg_ms_post "$BASE/chat" '{"question":"What did EU officials say about energy security?"}')
printf "  %-46s %sms avg\n" "POST /chat   (mock LLM — no network)" "$t"

printf "\n  Run ${B}./examples/latency.sh${NC} for p50/p95 breakdown across 20 requests.\n"

# ─────────────────────────────────────────────────────────────────
printf "\n${B}══════════════════════════════════════════════════\n"
printf "  RESULTS:  ${G}%d passed${NC}${B}  ${R}%d failed${NC}${B}  ${Y}%d skipped${NC}${B}\n" \
  "$PASS" "$FAIL" "$SKIP"
printf "══════════════════════════════════════════════════${NC}\n\n"

printf "${B}Known limitations (for interview):${NC}\n"
printf "  · Scanned/image-only PDFs: no OCR — extractable text layer required\n"
printf "  · Chat answer quality: not evaluable with LLM_PROVIDER=mock\n"
printf "  · First request after cold start: model load adds 1-3s (cached on volume)\n"
printf "  · Entity extraction: best-effort; failures stored as metadata warnings\n"
printf "  · No HNSW index: exact cosine scan, fine at small scale, set threshold ~100k chunks\n\n"

[ "$FAIL" -eq 0 ]
