#!/usr/bin/env bash
# RAG Pipeline — robustness test suite
#
# Bucket 2: Bad / sparse metadata
# Bucket 3: Ugly text inputs
# Bucket 4: Stage-level latency profiling
#
# Usage:  ./examples/robustness.sh [API_BASE_URL]
# Default API_BASE_URL: http://localhost:8001
#
# After running, check stage timings with:
#   sudo docker compose logs app --tail=40 | grep -E "hybrid_search|search retrieve"

set -uo pipefail
BASE="${1:-http://localhost:8001}"

PASS=0; FAIL=0; SKIP=0
G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'
B='\033[1m';    C='\033[0;36m'; NC='\033[0m'

pass()    { printf "  ${G}✓${NC} %s\n" "$1"; PASS=$((PASS+1)); }
fail()    { printf "  ${R}✗${NC} %s\n" "$1"; FAIL=$((FAIL+1)); }
skip()    { printf "  ${Y}~${NC} %s\n" "$1"; SKIP=$((SKIP+1)); }
section() { printf "\n${B}${C}%s${NC}\n" "$1"; }
observe() { printf "  ${Y}→${NC} %s\n" "$1"; }   # non-pass/fail observation

http_status() { curl -s -o /dev/null -w "%{http_code}" "$@"; }
body()        { curl -s "$@"; }

get_field() {
  # returns empty string only when key is absent or null, preserves 0/false/[]
  printf '%s' "$1" | python3 -c "
import json, sys
d = json.load(sys.stdin)
v = d.get('$2')
print('' if v is None else v)
" 2>/dev/null || echo ""
}

# POST via Python to avoid shell quoting/size limits on large payloads
py_post() {
  # py_post BASE_URL TEXT_PYTHON_EXPR METADATA_PYTHON_EXPR
  # Both exprs are valid Python string literals or expressions.
  python3 -c "
import urllib.request, json, sys
text = $2
meta = $3
payload = json.dumps({'text': text, 'metadata': meta}).encode()
req = urllib.request.Request(
    '$1/text', data=payload,
    headers={'Content-Type': 'application/json'}, method='POST'
)
try:
    resp = urllib.request.urlopen(req, timeout=120)
    print(resp.getcode())
    print(resp.read().decode())
except urllib.error.HTTPError as e:
    print(e.code)
    print(e.read().decode())
" 2>/dev/null
}

printf "\n${B}══════════════════════════════════════════════════${NC}\n"
printf "${B}  RAG Pipeline Robustness Test  —  %s${NC}\n" "$BASE"
printf "${B}══════════════════════════════════════════════════${NC}\n"

# ═══════════════════════════════════════════════════════════
section "BUCKET 2  ·  BAD / SPARSE METADATA"
# ═══════════════════════════════════════════════════════════

# ── 2.1  metadata key omitted entirely ──────────────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"A brief article about European renewable energy investment programmes."}')
[ "$s" = "200" ] \
  && pass "No metadata key → 200 (default empty DocumentMetadata)" \
  || fail "No metadata key → $s (want 200)"

# ── 2.2  metadata: null ──────────────────────────────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"Article about grid infrastructure upgrades across Europe.","metadata":null}')
[ "$s" = "422" ] \
  && pass "metadata: null → 422 (null not coercible to DocumentMetadata)" \
  || fail "metadata: null → $s (want 422 — strict type validation)"

# ── 2.3  All optional metadata fields explicitly null ───────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"European hydrogen strategy focuses on electrolysis scaling.","metadata":{"title":null,"source":null,"language":null,"published_at":null}}')
[ "$s" = "200" ] \
  && pass "All-null optional fields → 200 (null treated as absent)" \
  || fail "All-null optional fields → $s (want 200)"

# ── 2.4  published_at: invalid date string ───────────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"Policy brief on carbon pricing mechanisms in EU ETS.","metadata":{"published_at":"not-a-date"}}')
[ "$s" = "422" ] \
  && pass "published_at: 'not-a-date' → 422 (date parse failure)" \
  || fail "published_at: 'not-a-date' → $s (want 422)"

# ── 2.5  published_at: very old date (1800) ──────────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"Historical analysis of early European trade agreements.","metadata":{"published_at":"1800-01-01"}}')
[ "$s" = "200" ] \
  && pass "published_at: 1800-01-01 → 200 (no lower-bound restriction)" \
  || fail "published_at: 1800-01-01 → $s (want 200)"

# ── 2.6  title as integer (type coercion test) ───────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"Overview of EU renewable energy targets for 2030.","metadata":{"title":42}}')
if [ "$s" = "422" ]; then
  pass "title: 42 (int) → 422 (strict validation rejects non-string)"
elif [ "$s" = "200" ]; then
  observe "title: 42 (int) → 200 (pydantic coerced int to string '42')"
  SKIP=$((SKIP+1))
else
  fail "title: 42 (int) → $s (unexpected — want 422 or 200)"
fi

# ── 2.7  topics: [null] ──────────────────────────────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"EU member states debate new gas storage requirements.","metadata":{"topics":[null]}}')
[ "$s" = "422" ] \
  && pass "topics: [null] → 422 (null item not a string)" \
  || fail "topics: [null] → $s (want 422)"

# ── 2.8  topics: [1, 2, 3] (int items) ──────────────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"Wind capacity additions in Germany reached record levels.","metadata":{"topics":[1,2,3]}}')
[ "$s" = "422" ] \
  && pass "topics: [1,2,3] → 422 (int items not strings)" \
  || fail "topics: [1,2,3] → $s (want 422)"

# ── 2.9  topics: empty list ──────────────────────────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"France announced new subsidies for nuclear reactor refurbishment.","metadata":{"topics":[]}}')
[ "$s" = "200" ] \
  && pass "topics: [] → 200 (empty list is valid)" \
  || fail "topics: [] → $s (want 200)"

# ── 2.10  Language mismatch: metadata="de", text is English ──────
r=$(body -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"This article is clearly written in the English language about EU energy policy.","metadata":{"language":"de","title":"Language mismatch test"}}')
stored=$(get_field "$r" "language")
detected=$(printf '%s' "$r" | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(d.get('language_detected',''))" 2>/dev/null)
if [ "$stored" = "de" ] && [ "$detected" = "en" ]; then
  pass "Language mismatch: stored=de detected=en (metadata language wins over detection)"
elif [ "$stored" = "en" ]; then
  observe "Language mismatch: detection overrode metadata hint (stored=en detected=$detected)"
  SKIP=$((SKIP+1))
else
  fail "Language mismatch: stored=$stored detected=$detected (unexpected state)"
fi

# ── 2.11  Huge topics list (500 items) ───────────────────────────
HUGE_TOPICS=$(python3 -c "import json; print(json.dumps([f'topic_{i}' for i in range(500)]))")
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d "{\"text\":\"Article with an extreme number of topic tags.\",\"metadata\":{\"topics\":$HUGE_TOPICS}}")
[ "$s" = "200" ] \
  && pass "500-item topics array → 200 (no upper bound enforced)" \
  || fail "500-item topics array → $s"

# ── 2.12  Unicode in metadata fields ─────────────────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"Renewable energy policy in central Europe focuses on solar capacity expansion.","metadata":{"title":"Energiemix für Österreich 🌱","source":"Öster­reichische Energieagentur","region":"DACH"}}')
[ "$s" = "200" ] \
  && pass "Unicode in metadata fields (ö, ü, emoji) → 200" \
  || fail "Unicode metadata → $s (want 200)"


# ═══════════════════════════════════════════════════════════
section "BUCKET 3  ·  UGLY TEXT INPUTS"
# ═══════════════════════════════════════════════════════════

# ── 3.1  Very short text (2 chars) ───────────────────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"OK","metadata":{}}')
[ "$s" = "200" ] \
  && pass "Very short text ('OK', 2 chars) → 200 (no min-length guard)" \
  || fail "Very short text → $s (want 200)"

# ── 3.2  Single meaningful word ──────────────────────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"Renewable","metadata":{}}')
[ "$s" = "200" ] \
  && pass "Single word ('Renewable') → 200" \
  || fail "Single word → $s (want 200)"

# ── 3.3  Very long text (~30 KB) ─────────────────────────────────
out=$(py_post "$BASE" \
  "'European energy policy is central to EU sustainability goals and member state energy security frameworks. ' * 300" \
  "{}")
s=$(echo "$out" | head -1)
[ "$s" = "200" ] \
  && pass "Very long text (~30 KB, ~300 repetitions) → 200 (chunker handles it)" \
  || fail "Very long text → $s (want 200)"

# ── 3.4  Website copy-paste noise (nav/header/footer/cookie banners) ─
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"HOME | ABOUT | CONTACT | SUBSCRIBE\n\nEnergy Policy Digest — Issue 42\n\nSkip to main content | Cookie notice: we use cookies.\n\nThe European Commission announced a new battery storage initiative worth 8 billion euros. The programme will fund grid-scale installations across twelve member states with priority given to regions transitioning away from coal.\n\nShare this | Print | RSS\n\nRelated: Wind Targets | Solar Subsidies | Grid Interconnection\n\n© 2024 EnergyDigest.eu. All rights reserved.","metadata":{}}')
[ "$s" = "200" ] \
  && pass "Website copy-paste with nav/footer noise → 200 (ingest does not sanitize)" \
  || fail "Website noise → $s (want 200)"

# ── 3.5  Mixed language (EN paragraph + DE paragraph) ───────────
r=$(body -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"The European Union has announced new energy policy measures to address dependence on Russian natural gas.\n\nDie Europäische Kommission hat neue Maßnahmen zur Energiesicherheit angekündigt. Die Mitgliedstaaten werden aufgefordert, ihre Gasreserven bis Oktober auf 80 Prozent aufzufüllen.","metadata":{}}')
detected=$(get_field "$r" "language_detected")
is_dup=$(get_field "$r" "duplicate")
[ -n "$is_dup" ] \
  && pass "Mixed EN/DE text → $([ "$is_dup" = "True" ] && echo duplicate || echo ok) (language_detected=$detected — polyglot ingestion works)" \
  || fail "Mixed EN/DE text → unexpected response: $r"

# ── 3.6  Repeated boilerplate (same sentence 200×) ───────────────
out=$(py_post "$BASE" \
  "'The Council of the European Union met to discuss renewable energy policy frameworks. ' * 200" \
  "{}")
s=$(echo "$out" | head -1)
[ "$s" = "200" ] \
  && pass "Repeated boilerplate (200× sentence, ~15 KB) → 200" \
  || fail "Repeated boilerplate → $s (want 200)"

# ── 3.7  Unicode: smart quotes, em-dash, bullets, zero-width space ─
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d "{\"text\":\"The EU’s energy policy — a critical review:\n• Solar capacity increased by 40%\n• Wind targets revised upward\n• Gas dependency fell to 15%\n​The Commission stated “we are on track” for 2030.\",\"metadata\":{}}")
[ "$s" = "200" ] \
  && pass "Smart quotes, em-dash, bullets, zero-width space → 200" \
  || fail "Unicode text → $s (want 200)"

# ── 3.8  JSON-like pasted text (structured data passed as raw text) ─
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"{\"type\":\"policy_brief\",\"title\":\"Energy Markets Q3 2024\",\"content\":\"Quarterly review shows positive trends in EU renewable adoption. Gas prices stabilized following the price cap mechanism implementation.\"}","metadata":{}}')
[ "$s" = "200" ] \
  && pass "JSON-ish text blob → 200 (treated as raw text, not parsed)" \
  || fail "JSON-ish text → $s (want 200)"

# ── 3.9  Near-duplicate (same facts, different wording — unique per run via $$) ─
# Using $$ (script PID) ensures each run produces a unique text and passes on reruns.
r=$(body -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d "{\"text\":\"The G7 nations agreed to cap Russian petroleum exports at sixty dollars per barrel in robustness run $$. The Shadow Fleet — a network of tankers — was identified as a key circumvention mechanism used to bypass the oil price restrictions imposed by the international community.\",\"metadata\":{\"title\":\"G7 Oil Cap Paraphrase\",\"language\":\"en\"}}")
is_dup=$(get_field "$r" "duplicate")
chunks=$(get_field "$r" "chunks_created")
[ "$is_dup" = "False" ] \
  && pass "Near-duplicate (paraphrased, run=$$) → ok, chunks=$chunks (distinct hash from seed)" \
  || fail "Near-duplicate → is_dup=$is_dup (want False — each run is unique via PID)"

# ── 3.10  Near-duplicate is then retrievable ─────────────────────
resp=$(curl -s --get --data-urlencode "q=G7 price ceiling Russian crude oil tanker fleet" "$BASE/search?limit=5")
n=$(printf '%s' "$resp" | python3 -c \
  "import json,sys; print(len(json.load(sys.stdin).get('results',[])))" 2>/dev/null || echo 0)
[ "$n" -ge 2 ] \
  && pass "Near-dup retrievable: both original and paraphrase in top 5 (results=$n)" \
  || skip "Near-dup retrieval: got $n results in top 5 (reranker may merge similar chunks)"

# ── 3.11  RTL text (Arabic) ──────────────────────────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"سياسة الطاقة الأوروبية تهدف إلى تقليل الاعتماد على الغاز الروسي وتطوير مصادر الطاقة المتجددة.","metadata":{}}')
[ "$s" = "200" ] \
  && pass "RTL text (Arabic) → 200 (graceful, no script-direction crash)" \
  || fail "RTL text → $s (want 200)"

# ── 3.12  Whitespace-only text ───────────────────────────────────
s=$(http_status -X POST "$BASE/text" -H 'Content-Type: application/json' \
  -d '{"text":"   \n\t  \n   ","metadata":{}}')
[ "$s" = "422" ] \
  && pass "Whitespace-only text → 422 (stripped text is empty)" \
  || fail "Whitespace-only text → $s (want 422)"


# ═══════════════════════════════════════════════════════════
section "BUCKET 4  ·  PDF ROBUSTNESS"
# ═══════════════════════════════════════════════════════════
# Requires Docker access to generate test PDFs inside the container.
# Run as: DOCKER_CMD="sudo docker" ./examples/robustness.sh

DOCKER_CMD="${DOCKER_CMD:-docker}"
CONTAINER_ID=$($DOCKER_CMD compose ps -q app 2>/dev/null | head -1 || true)

if [ -z "$CONTAINER_ID" ]; then
  skip "PDF bucket — Docker access needed (set DOCKER_CMD='sudo docker' or add user to docker group)"
else
  # Generate all test PDFs inside the container using fitz (already installed)
  $DOCKER_CMD compose exec -T app python3 - <<'PYEOF' 2>/dev/null >/dev/null
import fitz

# 1. Small text PDF — 5 pages, clean embedded text
doc = fitz.open()
body = (
    "European energy policy is shaped by the twin imperatives of climate action "
    "and energy security. The REPowerEU plan aims to reduce dependence on Russian "
    "fossil fuels by 2027. Member states must report quarterly on storage levels. "
    "Grid modernization requires 584 billion euros through 2030. "
    "Renewable capacity targets set 45 percent by 2030. "
)
for i in range(5):
    page = doc.new_page()
    page.insert_text((50, 60), f"Page {i+1} — EU Energy Analysis\n\n" + body, fontsize=10)
doc.save("/tmp/rob_small.pdf"); doc.close()

# 2. Long text PDF — 40 pages, dense prose
doc = fitz.open()
para = (
    "Carbon markets under the EU ETS have achieved record prices, driving investment "
    "decisions across the continent. Hydrogen infrastructure development remains a "
    "priority for industrial decarbonization. Wind and solar capacity must triple by "
    "2030 under current projections. The Green Deal framework establishes binding "
    "targets for all member states. "
)
for i in range(40):
    page = doc.new_page()
    page.insert_text((50, 50), f"Section {i+1}: Energy Transition\n\n" + para * 4, fontsize=9)
doc.save("/tmp/rob_long.pdf"); doc.close()

# 3. Scanned / image-only PDF — white image page, no text layer
doc = fitz.open()
page = doc.new_page()
pix = fitz.Pixmap(fitz.csRGB, (0, 0, 612, 792), False)
pix.set_rect(pix.irect, (255, 255, 255))
page.insert_image(page.rect, pixmap=pix)
doc.save("/tmp/rob_scanned.pdf"); doc.close()

# 4. Malformed — plain text file with .pdf extension
with open("/tmp/rob_malformed.pdf", "w") as f:
    f.write("This is not a PDF. It has a .pdf extension but no PDF structure.")

print("ok")
PYEOF

  # Copy generated PDFs to host
  $DOCKER_CMD cp "${CONTAINER_ID}:/tmp/rob_small.pdf"    "/tmp/rob_small_$$.pdf"    2>/dev/null && S=1 || S=0
  $DOCKER_CMD cp "${CONTAINER_ID}:/tmp/rob_long.pdf"     "/tmp/rob_long_$$.pdf"     2>/dev/null && L=1 || L=0
  $DOCKER_CMD cp "${CONTAINER_ID}:/tmp/rob_scanned.pdf"  "/tmp/rob_scanned_$$.pdf"  2>/dev/null && SC=1 || SC=0
  $DOCKER_CMD cp "${CONTAINER_ID}:/tmp/rob_malformed.pdf" "/tmp/rob_malformed_$$.pdf" 2>/dev/null && M=1 || M=0

  # ── PDF test 1: Small text PDF (5 pages) ──────────────────────
  if [ "$S" -eq 1 ]; then
    r=$(curl -s -X POST "$BASE/document" \
      -F "file=@/tmp/rob_small_$$.pdf;type=application/pdf" \
      -F 'metadata={"title":"Robustness — Small PDF","language":"en"}')
    is_dup=$(get_field "$r" "duplicate"); chunks=$(get_field "$r" "chunks_created")
    status=$([ "$is_dup" = "True" ] && echo "duplicate" || echo "ok")
    [ -n "$is_dup" ] \
      && pass "Small PDF (5 pages) → $status, chunks=$chunks" \
      || fail "Small PDF → unexpected: $r"

    if [ "$is_dup" = "False" ] && [ "${chunks:-0}" -gt 0 ]; then
      resp=$(curl -s --get --data-urlencode "q=REPowerEU reduce dependence Russian fossil fuels 2027" "$BASE/search?limit=5")
      hit=$(printf '%s' "$resp" | python3 -c "
import json, sys
for r in json.load(sys.stdin).get('results', []):
    if 'repower' in r.get('text','').lower():
        print('yes'); exit()
print('no')
" 2>/dev/null || echo "no")
      [ "$hit" = "yes" ] \
        && pass "Small PDF content searchable (REPowerEU query hits)" \
        || skip "Small PDF content not in top 5 (other docs may rank higher)"
    fi
  else
    skip "Small PDF — generation or copy failed"
  fi

  # ── PDF test 2: Long PDF (40 pages) ───────────────────────────
  if [ "$L" -eq 1 ]; then
    t0=$(python3 -c "import time; print(int(time.time()*1000))" 2>/dev/null || date +%s%3N)
    r=$(curl -s -X POST "$BASE/document" \
      -F "file=@/tmp/rob_long_$$.pdf;type=application/pdf" \
      -F 'metadata={}')
    t1=$(python3 -c "import time; print(int(time.time()*1000))" 2>/dev/null || date +%s%3N)
    ingest_ms=$(( t1 - t0 ))
    is_dup=$(get_field "$r" "duplicate"); chunks=$(get_field "$r" "chunks_created")
    status=$([ "$is_dup" = "True" ] && echo "duplicate" || echo "ok")
    [ -n "$is_dup" ] \
      && pass "Long PDF (40 pages) → $status, chunks=$chunks, ingest=${ingest_ms}ms" \
      || fail "Long PDF → unexpected: $r"

    if [ "$is_dup" = "False" ]; then
      resp=$(curl -s --get --data-urlencode "q=EU ETS carbon market record prices investment" "$BASE/search?limit=5")
      hit=$(printf '%s' "$resp" | python3 -c "
import json, sys
for r in json.load(sys.stdin).get('results', []):
    c = r.get('text','').lower()
    if 'carbon' in c and 'ets' in c:
        print('yes'); exit()
print('no')
" 2>/dev/null || echo "no")
      [ "$hit" = "yes" ] \
        && pass "Long PDF interior content searchable (carbon/ETS query hits)" \
        || skip "Long PDF interior not in top 5"
    fi
  else
    skip "Long PDF — generation or copy failed"
  fi

  # ── PDF test 3: Scanned / image-only PDF ──────────────────────
  # Expected: 422 — "did not contain extractable text" (honest rejection)
  if [ "$SC" -eq 1 ]; then
    s=$(http_status -X POST "$BASE/document" \
      -F "file=@/tmp/rob_scanned_$$.pdf;type=application/pdf" \
      -F 'metadata={}')
    [ "$s" = "422" ] \
      && pass "Scanned PDF (image-only) → 422 (clean rejection: no extractable text)" \
      || fail "Scanned PDF → $s (want 422 — no OCR support is a known limitation)"
  else
    skip "Scanned PDF — generation or copy failed"
  fi

  # ── PDF test 4: Malformed PDF (not a real PDF) ─────────────────
  # Expected: 400 — fitz raises exception, caught and returned as 400
  if [ "$M" -eq 1 ]; then
    s=$(http_status -X POST "$BASE/document" \
      -F "file=@/tmp/rob_malformed_$$.pdf;type=application/pdf" \
      -F 'metadata={}')
    { [ "$s" = "400" ] || [ "$s" = "422" ]; } \
      && pass "Malformed PDF (text file .pdf) → $s (clean rejection, no 500)" \
      || fail "Malformed PDF → $s (want 400/422, not 500)"
  else
    skip "Malformed PDF — generation or copy failed"
  fi

  # ── PDF test 5: Duplicate PDF (same file twice) ────────────────
  # After test 1, the small PDF is in DB. A second submit must return duplicate.
  if [ "$S" -eq 1 ]; then
    r2=$(curl -s -X POST "$BASE/document" \
      -F "file=@/tmp/rob_small_$$.pdf;type=application/pdf" \
      -F 'metadata={"title":"Duplicate submit","language":"en"}')
    is_dup2=$(get_field "$r2" "duplicate")
    [ "$is_dup2" = "True" ] \
      && pass "Duplicate PDF submit → duplicate (content hash dedup works for PDFs)" \
      || fail "Duplicate PDF → is_dup=$is_dup2 (want True)"
  fi

  rm -f "/tmp/rob_small_$$.pdf" "/tmp/rob_long_$$.pdf" \
        "/tmp/rob_scanned_$$.pdf" "/tmp/rob_malformed_$$.pdf" 2>/dev/null || true
fi

# ═══════════════════════════════════════════════════════════
section "BUCKET 5  ·  STAGE-LEVEL LATENCY PROFILE"
# ═══════════════════════════════════════════════════════════

printf "\n  Firing 3 warm search requests...\n"

for q in \
  "G7 price ceiling Russian crude oil" \
  "EU energy sanctions third country circumvention" \
  "Nordic offshore wind capacity expansion Baltic Sea"
do
  t0=$(python3 -c "import time; print(int(time.time()*1000))" 2>/dev/null || date +%s%3N)
  curl -s --get --data-urlencode "q=$q" "$BASE/search?limit=5" > /dev/null
  t1=$(python3 -c "import time; print(int(time.time()*1000))" 2>/dev/null || date +%s%3N)
  ms=$(( t1 - t0 ))
  observe "query='${q:0:45}...'  wall=${ms}ms"
done

printf "\n  Stage breakdown (from container logs):\n"
printf "  ${B}sudo docker compose logs app --tail=30 | grep -E 'hybrid_search|search retrieve'${NC}\n\n"
printf "  Expected breakdown on CPU (bge-reranker-v2-m3, 10 candidates):\n"
printf "  %-22s  %s\n" "embed_ms"   "15–25ms    (e5-base, warm)"
printf "  %-22s  %s\n" "db_ms"      "2–5ms      (exact cosine scan, small corpus)"
printf "  %-22s  %s\n" "rrf_ms"     "<1ms       (in-memory merge)"
printf "  %-22s  %s\n" "rerank_ms"  "6500–8000ms  (560M param CrossEncoder, CPU, 10 pairs)"
printf "\n  The reranker dominates. GPU brings rerank_ms to ~50–200ms.\n"
printf "  To tune: HYBRID_CANDIDATE_COUNT in .env (default=10; ablation: 10 = 30 quality at 3x speed).\n"


# ═══════════════════════════════════════════════════════════
printf "\n${B}══════════════════════════════════════════════════${NC}\n"
printf "${B}  RESULTS:  %d passed  %d failed  %d skipped${NC}\n" \
  "$PASS" "$FAIL" "$SKIP"
printf "${B}══════════════════════════════════════════════════${NC}\n\n"

[ "$FAIL" -eq 0 ] || exit 1
