# Ahody RAG — Knowledge Base Backend

A metadata-aware, multilingual RAG backend for internal knowledge base search with source-grounded answers. Fully implemented and evaluated.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Python 3.11 | Async, automatic Swagger docs, Pydantic validation |
| Database | PostgreSQL + pgvector | Documents, metadata, and vectors in one store; metadata filtering is a WHERE clause |
| Keyword search | PostgreSQL FTS | Built-in, language-aware configs for all 5 target languages |
| Embeddings | `intfloat/multilingual-e5-base` (local) | One model maps EN/SV/DE/NO/FR into a shared semantic space |
| Reranker | `BAAI/bge-reranker-v2-m3` (local) | Cross-encoder improves precision after broad hybrid retrieval |
| Entity extraction | spaCy (language-specific models) | Best-effort NER for 5 languages |
| LLM | `mock` (default) / Ollama / OpenRouter | mock: safe baseline, no setup; Ollama: local real answers; OpenRouter: hosted |
| Migrations | Alembic | Auditable schema changes |
| Deployment | Docker Compose | Single-command setup |

---

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Default: LLM_PROVIDER=mock — no LLM setup required.

# 2. Start (first run downloads ML models, takes 2–5 minutes)
docker compose up -d

# 3. Verify
curl http://localhost:8001/health
# → {"status":"ok","database":"ok","llm_provider":"mock"}

# 4. Open interactive API docs
# Visit http://localhost:8001/docs in a browser
```

**Host ports:** app → `localhost:8001`, Postgres → `localhost:5433`, optional Ollama → `localhost:11435`

---

## Evaluate

```bash
# 1. Run the full evaluation suite
#    eval.sh auto-seeds the demo corpus if the DB is empty — no manual seed step needed.
#    DOCKER_CMD is required only on Linux where docker requires sudo.
DOCKER_CMD="sudo docker" bash examples/eval.sh
# Expected: 48 passed, 0 failed, 0 skipped
# Hit@1 = 12/12, MRR = 1.000

# 2. Run the robustness suite (bad metadata, ugly text, PDF edge cases)
DOCKER_CMD="sudo docker" bash examples/robustness.sh
# Expected: 31 passed, 0 failed
# (24 passed without Docker access — the PDF bucket (7 tests) requires DOCKER_CMD="sudo docker")

# 3. Run the latency benchmark
bash examples/latency.sh
# Expected: /search p50 ≈ 7 000ms on CPU (bge-reranker-v2-m3, 10 candidates)
```

**Candidate-count ablation** (already done, results below):

```bash
DOCKER_CMD="sudo docker" bash examples/ablation.sh
```

---

## Performance

Measured on CPU (no GPU), warm requests, `HYBRID_CANDIDATE_COUNT=10`:

| Stage | Time |
|-------|------|
| embed (e5-base) | 15–25 ms |
| DB query + RRF | 2–5 ms |
| rerank (bge-reranker-v2-m3, 10 pairs) | 6 500–8 000 ms |
| **total /search** | **~7 000 ms** |

**Candidate-count ablation result:**

| count | rerank_ms | Hit@1 | verdict |
|-------|-----------|-------|---------|
| 10 | 7 051 ms | 4/4 | **chosen** |
| 20 | 14 737 ms | 4/4 | viable |
| 30 | 21 837 ms | 4/4 | baseline |

`HYBRID_CANDIDATE_COUNT=10` cuts warm latency by 3× with no retrieval quality loss on this corpus. The correct chunk is consistently in the top-10 RRF results; the extra tail candidates were wasted cross-encoder compute.

**GPU acceleration (optional):** CUDA is auto-detected in Python — if the host exposes a GPU to the container, rerank_ms drops to ~50–200 ms. The default compose setup is CPU-safe and requires no GPU runtime. GPU passthrough requires `nvidia-container-toolkit` installed on the host and added to the compose file; this is intentionally left as an optional override rather than a default to ensure the repo runs anywhere.

---

## Known Limitations

- **Scanned PDFs:** no OCR, a text layer is required. Image-only PDFs are rejected with a 422 error.
- **Word/PPTX files:** not supported. Convert to PDF or plain text before ingestion. Silent failure is not possible, unsupported MIME types are rejected at the upload boundary.
- **LLM answer quality:** only verifiable with a real LLM provider (`ollama` or `openrouter`). `LLM_PROVIDER=mock` is the safe default for evaluation.
- **`confidence` is a retrieval-strength heuristic, not a factual verifier.** It reflects how many relevant chunks were found and how strongly they matched the query — not whether the LLM's answer is factually grounded. An unanswerable question can return `confidence: low` even though the LLM correctly states the context is insufficient. Downstream consumers should treat `confidence` as a signal about retrieval quality, not answer correctness.
- **No HNSW index:** exact cosine scan is correct and fast at small scale; set an HNSW index threshold at approximately 100 000 chunks for production.
- **First cold-start:** model load from the Docker volume adds 2–5 s on the first request after container restart. Subsequent calls return to warm baseline immediately.
- **Entity extraction quality:** Norwegian uses a multilingual spaCy fallback (no dedicated model). English, Swedish, German, and French use dedicated models.
- **No document deletion:** `DELETE /documents/{id}` is not implemented. Documents can be re-ingested (deduplication via SHA-256 prevents duplicates) but not removed without direct database access. Re-ingesting an identical document returns HTTP 200 with `"duplicate": true` and `"chunks_created": 0` — no data is written.
- **No document listing:** `GET /documents` (list all) is not implemented. Individual documents are accessible at `GET /documents/{id}`.

---

## API Reference

### Authentication

API key auth is **off by default** (`API_KEY` is empty in `.env.example`). To enable it, set a non-empty value in `.env`:

```
API_KEY=your-secret-key
```

When `API_KEY` is set, all business endpoints (`/text`, `/document`, `/search`, `/chat`, `/documents/{id}`, `/entities/...`) require the header:

```
X-API-Key: your-secret-key
```

Missing or wrong key returns `403 Forbidden`. The monitoring endpoints `/health` and `/ready` are always open regardless of `API_KEY`.

---

### `POST /text` — Upload plain text

```bash
curl -X POST http://localhost:8001/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "EU energy ministers met in Brussels...",
    "metadata": {
      "title": "EU Ministers Debate Energy Security",
      "source": "EuroEnergy Daily",
      "author": "Jane Smith",
      "language": "en",
      "published_at": "2023-09-14",
      "region": "EU",
      "topics": ["energy", "sanctions"],
      "document_type": "news_article"
    }
  }'
```

All metadata fields are optional. Language is auto-detected when omitted. Supported metadata fields: `title`, `source`, `author`, `language`, `published_at`, `region`, `topics` (array of strings), `document_type`. Extra fields are rejected with 422.

**Response:**
```json
{
  "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "chunks_created": 3,
  "entities_extracted": 5,
  "language": "en",
  "language_detected": "en",
  "duplicate": false
}
```

- `language`: the language stored for this document (from metadata hint or auto-detection)
- `language_detected`: the raw auto-detected language, regardless of any metadata hint
- `duplicate`: `true` if the document was already in the database (SHA-256 match). On duplicate, `chunks_created` is 0 and no new data is written; HTTP status is still 200.

### `POST /document` — Upload PDF

```bash
curl -X POST http://localhost:8001/document \
  -F "file=@report.pdf" \
  -F 'metadata={"source":"Publisher","language":"en"}'
```

Response shape is identical to `POST /text` above.

### `GET /search` — Hybrid search with filters

```bash
curl "http://localhost:8001/search?q=energy+sanctions&language=en&from_date=2023-01-01&limit=5"
```

Filters: `language`, `source`, `document_type`, `region`, `topic`, `from_date`, `to_date`, `limit`

### `POST /chat` — Grounded Q&A with citations

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What did EU officials say about energy security?","filters":{"language":"en"}}'
```

All search filters are available in `filters`: `language`, `source`, `document_type`, `region`, `topic`, `from_date`, `to_date`. Omit `filters` entirely to search across all documents.

```bash
# Cross-lingual: English question, retrieve only Swedish documents
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What are Nordic countries doing on offshore wind?","filters":{"language":"sv"}}'

# Date-scoped: restrict to documents published in 2024
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What were the latest energy forecasts?","filters":{"from_date":"2024-01-01"}}'
```

The `confidence` field in the response is a **retrieval-strength heuristic** (based on the number and score of matched chunks), not a factual verifier. See [Known Limitations](#known-limitations).

### Other endpoints

```bash
curl http://localhost:8001/documents/{id}               # inspect document record
curl http://localhost:8001/entities/Germany/documents   # entity co-occurrence
curl http://localhost:8001/health                       # liveness
curl http://localhost:8001/ready                        # readiness + dep check
```

---

## Architecture

### Ingestion

```
POST /text or /document
  → SHA-256 dedup check
  → langdetect language detection (metadata hint overrides)
  → validate and normalise metadata (Pydantic, strict mode)
  → chunk (500 tokens, 60-token overlap — fits e5-base 512-token window)
  → embed with "passage: " prefix (multilingual-e5-base)
  → store: document + chunks + embeddings
  → [best-effort] spaCy NER → entities / chunk_entities tables
```

### Search

```
GET /search
  → embed query with "query: " prefix
  → vector search (exact cosine, pgvector <=>)  +  FTS (plainto_tsquery, language-aware)
  → apply metadata filters (language, date, region, topic JSONB containment)
  → merge with RRF (k=60): Σ 1/(60 + rank)
  → rerank top-N candidates with bge-reranker-v2-m3 (CrossEncoder)
  → return top-K with scores and source attribution
```

### Chat

```
POST /chat
  → run search pipeline → top-K reranked chunks
  → build grounded prompt ("answer only from context, cite sources")
  → call LLM (Ollama or OpenRouter)
  → return answer + source references + confidence
```

---

## Design Decisions

**Why pgvector instead of a dedicated vector DB?**
Metadata filters, deduplication, entity co-occurrence, and vector search all live in one system. A separate vector DB would require synchronising two stores. At this scale, exact cosine scan in pgvector is fast and removes the operational overhead.

**Why bge-reranker-v2-m3?**
CrossEncoder reranking consistently improves precision-at-K over vector-only retrieval. The 560M-parameter v2-m3 model handles all 5 target languages without separate models. The latency cost is real on CPU and is documented; a GPU deployment resolves it.

**Why `HYBRID_CANDIDATE_COUNT=10` (not 30)?**
Candidate-count ablation (see Performance section) showed that reducing from 30 to 10 cuts rerank latency by 3× with no retrieval quality change. The correct chunk is consistently in the top-10 RRF results for this corpus; the additional tail candidates were wasted compute.

**Why RRF instead of learned fusion weights?**
RRF requires no labelled training data and no weight tuning. It consistently outperforms simple score averaging and is a reliable default when ground truth is scarce.

**Why not Neo4j for entity storage?**
Entity co-occurrence at this scale is a SQL join, not a graph traversal. Adding an unfamiliar service under a 4-day constraint would reduce reliability without adding capability. The schema supports a future migration if richer graph queries are needed.

---

## Supported Languages

| Code | Language | Embedding | NER |
|---|---|---|---|
| `en` | English | multilingual-e5-base | spaCy en_core_web_sm |
| `sv` | Swedish | multilingual-e5-base | spaCy sv_core_news_sm |
| `de` | German | multilingual-e5-base | spaCy de_core_news_sm |
| `fr` | French | multilingual-e5-base | spaCy fr_core_news_sm |
| `no` | Norwegian | multilingual-e5-base | multilingual fallback |

Norwegian (`nb`/`nn`) is normalised to `no` at the API boundary. Cross-lingual retrieval is supported: an English query retrieves relevant Norwegian or Swedish documents via the shared embedding space.

---

## LLM Configuration

**Default (no setup required):**
```
LLM_PROVIDER=mock
```

**Ollama (local, no API key):**
```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2:3b
```

**OpenRouter (free tier):**
```
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

---

## Environment Variables

See `.env.example` for full reference. Key variables:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` \| `ollama` \| `openrouter` |
| `HYBRID_CANDIDATE_COUNT` | `10` | RRF candidates passed to reranker (ablation-tuned). Set to 10 in `.env`; code fallback without `.env` is 30 (~22s latency). Always copy `.env.example` → `.env`. |
| `CHAT_CONTEXT_TOP_K` | `5` | Chunks sent to LLM after reranking |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Downloaded on first use, cached in `model_cache` volume |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Downloaded on first use, cached in `model_cache` volume |
| `APP_HOST_PORT` | `8001` | Host port for the app |
| `POSTGRES_HOST_PORT` | `5433` | Host port for Postgres |

---

## About This Submission

**The assignment:** Build a RAG backend for a Swedish company's internal knowledge base — upload API, chunking, embeddings, persistent storage, `/search`, `/chat` with citations. Hybrid search, reranking, and auth were listed as optional stretch goals.

I implemented all the stretch goals. Here is the reasoning.

### The decision that shaped everything else

The spec states a core principle I agreed with immediately: *the LLM is not the source of truth — retrieved chunks are.* A grounded answer with accurate citations is verifiably correct. A fluent answer without them cannot be trusted. That principle meant the retrieval pipeline had to work well before the LLM integration mattered at all.

Hybrid search (vector + full-text with RRF fusion) and cross-encoder reranking were the most direct way to raise retrieval precision. They were stretch goals on paper, but they were the right answer to the actual problem. I built them first.

### Language scope

The brief did not specify which languages to support. After confirming that Ahody is Swedish and expanding into EU markets, I chose EN, SV, DE, NO, FR and dropped Finnish in favour of French. This choice is encoded in exactly two places: the spaCy model list and the FTS config map. If the requirement differs, both change in a few lines.

### Why the evaluation infrastructure exists

`eval.sh`, `robustness.sh`, `latency.sh`, and `ablation.sh` were not in the spec. I built them because without measurable evaluation there was no way to know whether the pipeline was actually working — not just running, but returning the right answers.

The evaluation suite produced concrete numbers:
- Hit@1 = 12/12, MRR = 1.000 on the gold retrieval set
- 31/31 robustness tests across bad metadata, ugly text, and PDF edge cases
- Rerank latency reduced from ~22s to ~7s through candidate-count ablation, with zero retrieval quality loss

These numbers are not impressive because the corpus is small. They are useful because any interviewer can reproduce them on a fresh machine in under 10 minutes. That was the goal.

### AI collaboration

Claude Code (Anthropic) was my primary implementation tool throughout this project. I made every architectural and technology decision — what to build, what to skip, which models to use, how to measure quality, what to be honest about in the README. Claude wrote and iterated on the code, caught bugs, ran test iterations, and allowed me to cover significantly more ground in four days than I could have alone.

I am transparent about this because AI-augmented development is the current reality of engineering, and because the spec explicitly evaluates AI collaboration. The judgment calls were mine; the code velocity came from the tooling.

### What I would prioritise next

- **HNSW index** at ~100k chunks for production-scale vector search (exact cosine scan is correct and fast at this scale)
- **Streaming `/chat`** responses — long LLM answers currently block the connection until completion
- **`DELETE /documents/{id}` and `GET /documents`** for full document lifecycle management
- **Answer-level confidence** — the current `confidence` field is a retrieval-strength heuristic; grounding it in the LLM's own uncertainty signals would require a separate verification step
