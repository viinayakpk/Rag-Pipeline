#!/usr/bin/env bash
# Seed the API with 10 demo documents (2 per language: EN, SV, DE, FR, NO).
# Usage: ./examples/seed.sh [API_BASE_URL]
# Default URL: http://localhost:8001

set -euo pipefail

BASE="${1:-http://localhost:8001}"
DATA_DIR="$(dirname "$0")/demo_data"

echo "Seeding ${BASE} with demo documents from ${DATA_DIR}..."
echo ""

for file in "$DATA_DIR"/*.json; do
  name="$(basename "$file")"
  echo -n "  POST /text  ← ${name}  ... "
  response=$(curl -s -X POST "${BASE}/text" \
    -H "Content-Type: application/json" \
    -d "@${file}")
  doc_id=$(echo "$response" | grep -o '"document_id":"[^"]*"' | cut -d'"' -f4)
  is_dup=$(printf '%s' "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); v=d.get('duplicate'); print('' if v is None else v)" 2>/dev/null || echo "")
  msg=$([ "$is_dup" = "True" ] && echo "duplicate" || echo "ok")
  chunks=$(echo "$response" | grep -o '"chunks_created":[0-9]*' | cut -d: -f2)
  echo "id=${doc_id}  chunks=${chunks}  status=${msg}"
done

echo ""
echo "Done. Try a search:"
echo "  curl '${BASE}/search?q=energy+sanctions'"
echo "  curl '${BASE}/search?q=energy&language=sv'"
echo ""
echo "Or open the docs:"
echo "  open ${BASE}/docs"
