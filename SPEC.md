# Ahody RAG — Authoritative Build Specification

> **This is the single source of truth.**  
> `implementation.md` and `implementation.claude.md` are superseded by this document.  
> All implementation decisions, rationales, and code patterns come from here.

---

## 1. Assignment Summary (verbatim from brief)

> Ahody's customers need an internal knowledge base where they can upload source material (memos, reports, articles from sources other than the newspaper) and then ask questions against the content in natural language, with source references. Your assignment is to build the foundation.

**Must-haves:** Upload API (text + binary), Indexing (chunking + embeddings + persistent storage), Search (`GET /search`), Chat with citations (`POST /chat`), Persistence across restart.

**Stretch (optional):** Graph representation, Hybrid search, Streaming, Reranking, Auth.

**Evaluated on:** Judgment, source references, data modeling, AI collaboration, code quality, tradeoff awareness. Not evaluated on: UI, perfect error handling, CI/CD, production observability.

---

## 2. Candidate Assumptions (not stated in the brief)

These are explicit assumptions made by the candidate, not facts from the assignment:

**Language scope:** The brief does not specify which languages to support. The target languages are **EN, SV, DE, NO, FR** — a deliberate choice made after discussion with the hiring contact who confirmed Ahody is Swedish and expanding into EU markets including France. Finnish was considered and explicitly dropped in favour of French for this reason. If the actual requirement differs, only the spaCy model list and FTS config map need changing — no architectural changes required.

**Document types:** The brief mentions "memos, reports, articles." The data model accepts any document type via the `document_type` field and applies no hardcoded assumptions about content structure.

**Scale:** The brief describes a foundation, not a production system. Design decisions (chunk size, exact vector search, single Postgres instance) are calibrated for tens to hundreds of documents — the expected scale for a work sample. Each decision notes how it would change at larger scale.

---

## 3. Architecture Decisions

### 3.1 Core principle

**The LLM is not the source of truth. Retrieved chunks are.**

The retrieval pipeline must return the right evidence before the LLM generates prose. A wrong answer with citations is caught immediately. A right answer without citations cannot be trusted.

### 3.2 Decision table

| Decision | Choice | Rejected alternative | Reason |
|---|---|---|---|
| Vector DB | PostgreSQL + pgvector | Qdrant, Weaviate | Keeps documents, metadata, and vectors in one store; metadata filtering is a WHERE clause; no extra service |
| Graph layer | Postgres entity tables | Neo4j | Co-occurrence queries satisfied with SQL joins; Neo4j adds a third service and requires Cypher; optional per the brief |
| Keyword search | PostgreSQL FTS | Elasticsearch, BM25 | Already in Postgres; language-aware configs built in |
| Score fusion | RRF (k=60) | Weighted formula | RRF has no weights to tune without a labeled eval set; parameter-free; principled |
| Chunk size | **500 tokens, 60 overlap** | 512, 1024 | Leaves headroom for E5 prefix + special tokens within 512-token model limit (see §8) |
| Embeddings | `intfloat/multilingual-e5-base` | multilingual-e5-large | Covers all 5 languages in shared space; 3× smaller than large variant; sufficient for this scale |
| Reranker | `BAAI/bge-reranker-v2-m3` | ms-marco cross-encoders | Best multilingual cross-encoder available |
| Vector index | **Exact cosine (no IVFFlat)** | IVFFlat | Exact search is faster and has 100% recall at <10k vectors; IVFFlat trades recall for speed only worthwhile at 10k+ |
| NER | spaCy (lang-specific models) | GLiNER | More mature, better documented, pip-installable models; GLiNER improves multilingual quality but adds complexity |
| LLM path | **Ollama (primary), OpenRouter (optional)** | OpenAI | Ollama: zero API key, runs locally; OpenRouter: free tier, optional upgrade; both use OpenAI-compatible client |
| Model loading | **HF model cache volume** | Baked into Docker image | 1.5GB of models in image is too heavy; volume cache downloads once, reuses forever |

---

## 4. Stack

```
Language:         Python 3.11+
API:              FastAPI
Validation:       Pydantic v2
ORM:              SQLAlchemy 2 (async)
Migrations:       Alembic
Database:         PostgreSQL 16 + pgvector
Embeddings:       intfloat/multilingual-e5-base  (local, HF model cache)
Reranker:         BAAI/bge-reranker-v2-m3        (local, HF model cache)
NER:              spaCy with language models      (baked into Docker image)
LLM (primary):    Ollama + llama3.2:3b            (zero API key required)
LLM (optional):   OpenRouter free tier            (higher quality option)
Deployment:       Docker Compose
Testing:          pytest + httpx (async)
```

---

## 5. Project Structure

```
rag-kb/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, router registration
│   ├── config.py                # Settings from .env via pydantic-settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── text.py              # POST /text
│   │   ├── document.py          # POST /document
│   │   ├── search.py            # GET /search
│   │   ├── chat.py              # POST /chat
│   │   └── documents.py         # GET /documents/{id}, GET /entities/{name}/documents
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chunking.py          # RecursiveCharacterTextSplitter, 500 tokens, 60 overlap
│   │   ├── embeddings.py        # multilingual-e5-base with passage:/query: prefix
│   │   ├── ingestion.py         # Orchestrates the full ingest pipeline
│   │   ├── retrieval.py         # Vector search + FTS + RRF merge + metadata filters
│   │   ├── reranking.py         # bge-reranker-v2-m3 cross-encoder
│   │   ├── entities.py          # spaCy NER, entity upsert, co-occurrence
│   │   ├── llm.py               # LLMClient protocol + Ollama + OpenRouter + Mock
│   │   └── chat.py              # Full chat pipeline: retrieve → rerank → generate → cite
│   ├── db/
│   │   ├── __init__.py
│   │   ├── base.py              # SQLAlchemy declarative base
│   │   ├── session.py           # Async engine + session factory + get_db dependency
│   │   ├── models.py            # All ORM models
│   │   └── migrations/          # Alembic
│   │       ├── env.py
│   │       ├── script.py.mako
│   │       └── versions/
│   └── schemas/
│       ├── __init__.py
│       ├── ingest.py            # TextUploadRequest, UploadResponse
│       ├── search.py            # SearchFilters, SearchResult, SearchResponse
│       └── chat.py              # ChatRequest, ChatResponse, SourceReference
├── tests/
│   ├── conftest.py              # DB fixture, test client, mock LLM override
│   ├── test_ingest_text.py
│   ├── test_ingest_document.py
│   ├── test_search.py
│   └── test_chat.py
├── examples/
│   ├── demo_data/               # 10 synthetic documents
│   │   ├── 01_en_energy_2014.txt
│   │   ├── 02_de_gas_2014.txt
│   │   ├── 03_fr_eu_energy_2018.txt
│   │   ├── 04_sv_energy_2022.txt
│   │   ├── 05_no_nato_2023.txt
│   │   ├── 06_en_sanctions_2022.txt
│   │   ├── 07_en_climate_2015.txt
│   │   ├── 08_en_elections_2019.txt
│   │   ├── 09_en_media_memo.txt
│   │   └── 10_en_sanctions_report.txt   # uploaded as "PDF" in seed script via /text
│   ├── seed.sh                  # Uploads all demo docs
│   ├── upload_text.sh
│   ├── upload_pdf.sh
│   ├── search.sh
│   └── chat.sh
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── alembic.ini
├── .env.example
└── README.md
```

---

## 6. Docker Compose

```yaml
# docker-compose.yml

services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-ragkb}
      POSTGRES_USER: ${POSTGRES_USER:-raguser}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-ragpass}
    ports:
      - "${POSTGRES_HOST_PORT:-5433}:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-raguser}"]
      interval: 5s
      timeout: 5s
      retries: 10

  app:
    build: .
    env_file: .env
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-raguser}:${POSTGRES_PASSWORD:-ragpass}@db:5432/${POSTGRES_DB:-ragkb}
    ports:
      - "${APP_HOST_PORT:-8001}:8000"
    depends_on:
      db:
        condition: service_healthy
    volumes:
      - model_cache:/root/.cache   # HuggingFace models cached here across restarts
    # No --reload: evaluator-facing command. Use `uvicorn ... --reload` for local dev override.
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000
    # host.docker.internal resolves on Mac/Windows automatically.
    # On Linux it requires this mapping so the app container can reach Ollama on the host.
    extra_hosts:
      - "host.docker.internal:host-gateway"

  # Ollama service — activate with: docker compose --profile ollama up
  # When using this profile, set OLLAMA_BASE_URL=http://ollama:11434 in .env
  # The model is pulled automatically on first startup (~2GB download, cached in ollama_data volume).
  ollama:
    image: ollama/ollama
    profiles: [ollama]
    ports:
      - "${OLLAMA_HOST_PORT:-11435}:11434"
    volumes:
      - ollama_data:/root/.ollama
    # Pull the model on startup. The entrypoint waits for the server to be ready first.
    entrypoint: >
      /bin/sh -c "ollama serve &
      until ollama list > /dev/null 2>&1; do sleep 1; done &&
      ollama pull ${OLLAMA_MODEL:-llama3.2:3b} &&
      wait"

volumes:
  pg_data:
  model_cache:
  ollama_data:
```

**LLM setup paths — choose one:**

| Path | Command | Requirement |
|---|---|---|
| Ollama (no API key) | `docker compose --profile ollama up` | Set `OLLAMA_BASE_URL=http://ollama:11434` in `.env`. Host port default is `11435`. First run pulls ~2GB model. |
| External Ollama (already running) | `docker compose up` | Set `OLLAMA_BASE_URL=http://host.docker.internal:11434`. Works on Mac/Windows/Linux. |
| OpenRouter (free tier) | `docker compose up` | Set `LLM_PROVIDER=openrouter` + `OPENROUTER_API_KEY` in `.env`. App host port default is `8001`. |
| Mock (retrieval only) | `docker compose up` | Set `LLM_PROVIDER=mock`. No LLM call — good for testing retrieval in isolation. App host port default is `8001`. |

---

## 7. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy models are installed as pip packages (from wheel URLs in requirements.txt).
# They are small (~10-50MB each) and baked into the image — no runtime download needed.
# HuggingFace models (multilingual-e5-base, bge-reranker-v2-m3) are ~1.5GB total
# and are NOT baked in. They download on first use and cache in the model_cache volume.

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 8. Requirements

```
# requirements.txt

# API
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9

# Validation + config
pydantic==2.7.0
pydantic-settings==2.3.0

# Database
sqlalchemy[asyncio]==2.0.31
asyncpg==0.29.0
alembic==1.13.2
pgvector==0.3.0

# Embeddings + reranking (GPU auto-detected by PyTorch; works on CPU too)
sentence-transformers==3.0.1
torch==2.3.0

# PDF extraction
pymupdf==1.24.9

# Chunking (uses the model's own tokenizer for accurate token counts)
langchain-text-splitters==0.2.2
transformers==4.43.0

# NER — spaCy language models installed as pip packages
# This avoids a separate `python -m spacy download` step in the Dockerfile
spacy==3.7.5
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl
https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.7.0/fr_core_news_sm-3.7.0-py3-none-any.whl
https://github.com/explosion/spacy-models/releases/download/sv_core_news_sm-3.7.0/sv_core_news_sm-3.7.0-py3-none-any.whl
https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.7.0/xx_ent_wiki_sm-3.7.0-py3-none-any.whl

# Language detection
langdetect==1.0.9

# LLM (OpenAI-compatible client used for both Ollama and OpenRouter)
openai==1.40.0

# Testing
pytest==8.3.0
pytest-asyncio==0.23.8
httpx==0.27.0
```

> **Note on GPU:** `torch` auto-detects CUDA/MPS. No GPU-specific flags are needed; models run on CPU if no GPU is available (slower but functional). For a work sample with ~50 documents, CPU inference is acceptable.

---

## 9. Environment Variables

```bash
# .env.example

# App
APP_HOST=0.0.0.0
APP_PORT=8000
APP_HOST_PORT=8001

# Database (matches docker-compose defaults)
POSTGRES_DB=ragkb
POSTGRES_USER=raguser
POSTGRES_PASSWORD=ragpass
POSTGRES_HOST_PORT=5433
DATABASE_URL=postgresql+asyncpg://raguser:ragpass@db:5432/ragkb

# Embedding and reranking models (downloaded automatically on first use)
EMBEDDING_MODEL=intfloat/multilingual-e5-base
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# LLM — bootstrap milestone default:
# The first implemented milestone uses mock so the app boots without any provider setup.
LLM_PROVIDER=mock

# Option A: Ollama via compose profile (docker compose --profile ollama up)
# Model lives in the ollama_data volume; pulled automatically on first run.
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://ollama:11434
# OLLAMA_MODEL=llama3.2:3b
# OLLAMA_HOST_PORT=11435

# Option B: Ollama running on the host machine (docker compose up, no profile)
# host.docker.internal works on Mac/Windows; on Linux it resolves via extra_hosts in compose.
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://host.docker.internal:11434
# OLLAMA_MODEL=llama3.2:3b

# Option C: OpenRouter (free tier, better quality — sign up at openrouter.ai)
# LLM_PROVIDER=openrouter
# OPENROUTER_API_KEY=sk-or-...
# OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Option D: Mock — retrieval only, no LLM call. Useful for testing search without any LLM setup.
# Already active above.

# Retrieval tuning
DEFAULT_SEARCH_LIMIT=10
HYBRID_CANDIDATE_COUNT=30
CHAT_CONTEXT_TOP_K=5

# Optional API key auth (leave empty to disable)
API_KEY=
```

---

## 10. Database Schema

### SQL (canonical, with notes)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────────────
-- DOCUMENTS
-- ─────────────────────────────────────────────────────
CREATE TABLE documents (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title               TEXT,
    source              TEXT,
    language            TEXT,
    detected_language   TEXT,          -- always the detector's result, regardless of user input
    published_at        TIMESTAMPTZ,   -- stored null if absent — never invented
    uploaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    document_type       TEXT,
    region              TEXT,
    author              TEXT,
    topics              JSONB,         -- array of strings: ["energy", "policy"]
    original_filename   TEXT,
    mime_type           TEXT,
    byte_size           BIGINT,
    content_hash        TEXT UNIQUE,   -- SHA-256 of normalized text; prevents duplicate ingestion
    ingestion_status    TEXT NOT NULL DEFAULT 'pending',  -- pending|processing|ready|failed
    extracted_text      TEXT,
    metadata            JSONB          -- raw user metadata + inference provenance
);

-- ─────────────────────────────────────────────────────
-- CHUNKS
-- ─────────────────────────────────────────────────────
CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,          -- 0-based
    text            TEXT NOT NULL,
    token_count     INT,

    -- Denormalized from document: avoids joins on every search query
    language        TEXT,
    source          TEXT,
    title           TEXT,
    published_at    TIMESTAMPTZ,
    document_type   TEXT,
    region          TEXT,
    topics          JSONB,                 -- same structure as documents.topics

    -- Vector: 768 dimensions for multilingual-e5-base
    embedding       VECTOR(768),

    -- FTS: pre-computed, maintained by trigger below
    fts_vector      TSVECTOR,

    UNIQUE (document_id, chunk_index)
);

-- Metadata filter indexes (B-tree)
CREATE INDEX idx_chunks_language     ON chunks (language);
CREATE INDEX idx_chunks_published_at ON chunks (published_at);
CREATE INDEX idx_chunks_source       ON chunks (source);
CREATE INDEX idx_chunks_region       ON chunks (region);
CREATE INDEX idx_chunks_document_type ON chunks (document_type);
CREATE INDEX idx_chunks_document_id  ON chunks (document_id);

-- FTS index (GIN)
CREATE INDEX idx_chunks_fts ON chunks USING GIN (fts_vector);

-- VECTOR INDEX NOTE:
-- For demonstration-scale data (< ~5000 vectors), exact cosine search has:
--   - 100% recall (no approximation loss)
--   - Faster setup (no VACUUM/ANALYZE required)
--   - Comparable or better speed than IVFFlat at this scale
-- No special index is created on the embedding column.
-- At larger scale, replace with:
--   CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
-- HNSW is preferred over IVFFlat: better recall, no lists tuning, works without pre-population.

-- FTS trigger: uses language-specific PostgreSQL text search config when available
CREATE OR REPLACE FUNCTION update_chunk_fts()
RETURNS TRIGGER AS $$
DECLARE
    cfg REGCONFIG;
BEGIN
    cfg := CASE NEW.language
        WHEN 'en' THEN 'english'::REGCONFIG
        WHEN 'de' THEN 'german'::REGCONFIG
        WHEN 'fr' THEN 'french'::REGCONFIG
        WHEN 'sv' THEN 'swedish'::REGCONFIG
        WHEN 'no' THEN 'norwegian'::REGCONFIG
        ELSE 'simple'::REGCONFIG
    END;
    NEW.fts_vector := to_tsvector(cfg, COALESCE(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_chunk_fts
BEFORE INSERT OR UPDATE OF text, language ON chunks
FOR EACH ROW EXECUTE FUNCTION update_chunk_fts();

-- ─────────────────────────────────────────────────────
-- ENTITY GRAPH (Postgres-native, no Neo4j required)
-- ─────────────────────────────────────────────────────
CREATE TABLE entities (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL,
    normalized_name TEXT NOT NULL,     -- lowercase, stripped, for deduplication
    entity_type     TEXT NOT NULL,     -- PERSON | ORG | GPE | LOC | NORP | EVENT
    UNIQUE (normalized_name, entity_type)
);
CREATE INDEX idx_entities_normalized ON entities (normalized_name);

CREATE TABLE chunk_entities (
    chunk_id    UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_id   UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, entity_id)
);
CREATE INDEX idx_chunk_entities_entity   ON chunk_entities (entity_id);
CREATE INDEX idx_chunk_entities_document ON chunk_entities (document_id);

-- Entity co-occurrence query (the "graph" use case):
-- All documents mentioning both 'Germany' and 'European Commission':
--
-- SELECT DISTINCT d.id, d.title, d.published_at
-- FROM documents d
-- JOIN chunks c ON c.document_id = d.id
-- JOIN chunk_entities ce1 ON ce1.chunk_id = c.id
-- JOIN entities e1 ON e1.id = ce1.entity_id AND e1.normalized_name = 'germany'
-- JOIN chunk_entities ce2 ON ce2.chunk_id = c.id
-- JOIN entities e2 ON e2.id = ce2.entity_id AND e2.normalized_name = 'european commission'
-- ORDER BY d.published_at DESC;

-- ─────────────────────────────────────────────────────
-- CHAT LOGS (optional, useful for evaluation trail)
-- ─────────────────────────────────────────────────────
CREATE TABLE chat_logs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question            TEXT NOT NULL,
    filters             JSONB,
    answer              TEXT,
    confidence          TEXT,
    retrieved_chunk_ids JSONB,         -- list of UUIDs as strings
    sources             JSONB,
    retrieval_notes     JSONB,
    model_name          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 11. SQLAlchemy Models

```python
# app/db/models.py
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String, Text, Integer, BigInteger, DateTime, ForeignKey,
    UniqueConstraint, Index, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.asyncio import AsyncAttrs
from .base import Base


class Document(AsyncAttrs, Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[Optional[str]] = mapped_column(Text)
    source: Mapped[Optional[str]] = mapped_column(Text)
    language: Mapped[Optional[str]] = mapped_column(String(10))
    detected_language: Mapped[Optional[str]] = mapped_column(String(10))
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    document_type: Mapped[Optional[str]] = mapped_column(String(50))
    region: Mapped[Optional[str]] = mapped_column(String(100))
    author: Mapped[Optional[str]] = mapped_column(Text)
    topics: Mapped[Optional[list]] = mapped_column(JSONB)
    original_filename: Mapped[Optional[str]] = mapped_column(Text)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100))
    byte_size: Mapped[Optional[int]] = mapped_column(BigInteger)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), unique=True)
    ingestion_status: Mapped[str] = mapped_column(String(20), default="pending")
    extracted_text: Mapped[Optional[str]] = mapped_column(Text)
    metadata: Mapped[Optional[dict]] = mapped_column(JSONB)

    chunks: Mapped[list[Chunk]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(AsyncAttrs, Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    language: Mapped[Optional[str]] = mapped_column(String(10))
    source: Mapped[Optional[str]] = mapped_column(Text)
    title: Mapped[Optional[str]] = mapped_column(Text)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    document_type: Mapped[Optional[str]] = mapped_column(String(50))
    region: Mapped[Optional[str]] = mapped_column(String(100))
    topics: Mapped[Optional[list]] = mapped_column(JSONB)   # list[str], e.g. ["energy", "policy"]
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(768))

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
    chunk_entities: Mapped[list[ChunkEntity]] = relationship(back_populates="chunk", cascade="all, delete-orphan")


class Entity(AsyncAttrs, Base):
    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_name: Mapped[str] = mapped_column(Text, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(20), nullable=False)

    __table_args__ = (UniqueConstraint("normalized_name", "entity_type"),)

    chunk_entities: Mapped[list[ChunkEntity]] = relationship(back_populates="entity")


class ChunkEntity(AsyncAttrs, Base):
    __tablename__ = "chunk_entities"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")
    )

    chunk: Mapped[Chunk] = relationship(back_populates="chunk_entities")
    entity: Mapped[Entity] = relationship(back_populates="chunk_entities")


class ChatLog(AsyncAttrs, Base):
    __tablename__ = "chat_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    filters: Mapped[Optional[dict]] = mapped_column(JSONB)
    answer: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[Optional[str]] = mapped_column(String(20))
    retrieved_chunk_ids: Mapped[Optional[dict]] = mapped_column(JSONB)
    sources: Mapped[Optional[dict]] = mapped_column(JSONB)
    retrieval_notes: Mapped[Optional[dict]] = mapped_column(JSONB)
    model_name: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

---

## 12. Chunking Service

```python
# app/services/chunking.py
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from functools import lru_cache
from ..config import settings

CHUNK_SIZE = 500      # content tokens (see rationale below)
CHUNK_OVERLAP = 60    # token overlap between adjacent chunks


@lru_cache(maxsize=1)
def _get_tokenizer():
    return AutoTokenizer.from_pretrained(settings.embedding_model)


def _token_count(text: str) -> int:
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    token_count: int


def chunk_text(text: str) -> list[TextChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=_token_count,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
    )
    raw = splitter.split_text(text)
    return [
        TextChunk(text=c, chunk_index=i, token_count=_token_count(c))
        for i, c in enumerate(raw)
        if c.strip()
    ]
```

### Chunk size rationale (correct version)

**Why 500 tokens (content), not 512:**

`multilingual-e5-base` is built on XLM-RoBERTa and has a **maximum sequence length of 512 tokens total** — including all tokens in the encoded sequence.

When a chunk is embedded, the actual input to the model is:
```
[CLS] passage : Ġ {content tokens} [SEP]
  1      1   1  1                     1   = ~5 overhead tokens
```

If `chunk_size = 512` content tokens were used, the total would be ~517 tokens, exceeding the model limit. The model silently truncates, losing the last ~5 tokens of every chunk.

Setting `chunk_size = 500` gives: 500 (content) + 5 (overhead) = 505 tokens — comfortably within the 512-token limit with no truncation.

**Why 500 tokens for content, not 1024:**
500 tokens ≈ 375 words ≈ 3 paragraphs. For memos, reports, and news articles, this is:
- Large enough to preserve the context of one coherent argument or section
- Small enough that each chunk covers a focused topic, making citations precise and verifiable

With 1024-token chunks, a single chunk may span multiple topics, making retrieval less precise and making it harder for the reviewer to locate the supporting sentence in the source.

**Why 60-token overlap:**
Prevents context loss at chunk boundaries. When a sentence spans a chunk break, its first half appears at the end of one chunk and the complete sentence appears at the start of the next.

---

## 13. Embedding Service

```python
# app/services/embeddings.py
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from ..config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Auto-downloads from HuggingFace on first call; cached in HF_HOME volume
    return SentenceTransformer(settings.embedding_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """For indexing. E5 requires 'passage: ' prefix — omitting it degrades retrieval quality."""
    model = _get_model()
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(prefixed, normalize_embeddings=True, batch_size=32).tolist()


def embed_query(query: str) -> list[float]:
    """For search queries. Uses 'query: ' prefix."""
    model = _get_model()
    return model.encode(f"query: {query}", normalize_embeddings=True).tolist()
```

**Why multilingual-e5-base:**
A single model maps semantically similar content from all five target languages into a shared vector space. A Swedish query "energisäkerhet Europa" will retrieve an English chunk about "energy security in Europe" without any translation step.

**Why the prefix matters:**
E5 models are trained with contrastive learning using these exact prefixes. Without them, the model produces embeddings that are not aligned with the training distribution, resulting in degraded retrieval — particularly for cross-language queries.

---

## 14. Retrieval Service (Hybrid + Metadata Filters)

```python
# app/services/retrieval.py
import json
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from ..db.models import Chunk
from ..schemas.search import SearchFilters

RRF_K = 60


async def hybrid_search(
    query: str,
    query_embedding: list[float],
    filters: SearchFilters | None,
    db: AsyncSession,
    limit: int = 30,
) -> list[uuid.UUID]:
    """Returns ordered list of chunk IDs by RRF score."""
    where_clause, params = _build_where(filters)

    vector_ids = await _vector_search(query_embedding, where_clause, params, db, limit)
    keyword_ids = await _keyword_search(query, filters, where_clause, params, db, limit)

    return _rrf_merge(vector_ids, keyword_ids)[:limit]


def _vec_to_pg(embedding: list[float]) -> str:
    """Format a float list as a pgvector literal: '[1.0,2.0,...]'.
    Used for raw SQL binding. The explicit CAST(:emb AS vector) in the query
    ensures asyncpg does not try to infer the type from the string value.
    """
    return "[" + ",".join(repr(x) for x in embedding) + "]"


async def _vector_search(
    embedding: list[float],
    where: str,
    params: dict,
    db: AsyncSession,
    limit: int,
) -> list[uuid.UUID]:
    # Exact cosine search — no IVFFlat index, 100% recall at this scale.
    # CAST(:embedding AS vector) makes the type explicit; avoids asyncpg coercion ambiguity.
    rows = await db.execute(
        text(f"""
            SELECT id
            FROM chunks
            WHERE embedding IS NOT NULL {where}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """),
        {"embedding": _vec_to_pg(embedding), "limit": limit, **params},
    )
    return [row.id for row in rows]


async def _keyword_search(
    query: str,
    filters: SearchFilters | None,
    where: str,
    params: dict,
    db: AsyncSession,
    limit: int,
) -> list[uuid.UUID]:
    lang = filters.language if filters else None
    fts_cfg = {
        "en": "english", "de": "german", "fr": "french",
        "sv": "swedish", "no": "norwegian",
    }.get(lang or "", "simple")

    rows = await db.execute(
        text(f"""
            SELECT id
            FROM chunks
            WHERE fts_vector @@ plainto_tsquery('{fts_cfg}'::REGCONFIG, :q) {where}
            ORDER BY ts_rank_cd(fts_vector, plainto_tsquery('{fts_cfg}'::REGCONFIG, :q)) DESC
            LIMIT :limit
        """),
        {"q": query, "limit": limit, **params},
    )
    return [row.id for row in rows]


def _build_where(filters: SearchFilters | None) -> tuple[str, dict]:
    if not filters:
        return "", {}
    clauses, params = [], {}
    if filters.language:
        clauses.append("AND language = :f_lang")
        params["f_lang"] = filters.language
    if filters.source:
        clauses.append("AND source = :f_source")
        params["f_source"] = filters.source
    if filters.document_type:
        clauses.append("AND document_type = :f_dtype")
        params["f_dtype"] = filters.document_type
    if filters.region:
        clauses.append("AND region = :f_region")
        params["f_region"] = filters.region
    if filters.topic:
        # CORRECT: use CAST to JSONB for the @> containment operator
        # json.dumps produces a valid JSON array string e.g. '["energy"]'
        clauses.append("AND topics @> CAST(:f_topic AS JSONB)")
        params["f_topic"] = json.dumps([filters.topic])
    if filters.from_date:
        clauses.append("AND published_at >= :f_from")
        params["f_from"] = filters.from_date
    if filters.to_date:
        clauses.append("AND published_at <= :f_to")
        params["f_to"] = filters.to_date
    return " ".join(clauses), params


def _rrf_merge(
    vector_ids: list[uuid.UUID],
    keyword_ids: list[uuid.UUID],
) -> list[uuid.UUID]:
    scores: dict[uuid.UUID, float] = {}
    for rank, cid in enumerate(vector_ids, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    for rank, cid in enumerate(keyword_ids, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)
```

**Why RRF over a weighted formula:**
A weighted formula (`0.55 * vector + 0.25 * keyword + ...`) requires tuning weights against a labeled evaluation set. Without such data, the weights are arbitrary guesses. RRF is parameter-free: it rewards chunks that rank well in both systems, requires no calibration, and is well-established in IR literature. The single constant k=60 is a standard default, not a tuned value.

---

## 15. Reranking Service

```python
# app/services/reranking.py
from sentence_transformers import CrossEncoder
from functools import lru_cache
from ..config import settings


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = _get_reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
```

**Why two-stage retrieval:**
Embedding similarity (bi-encoder) is fast: one query embedding compared against stored chunk embeddings. Good for top-30 retrieval. The cross-encoder reranker reads the full (query, chunk) pair together — more accurate but too slow for full-corpus search. Reranking only top-30 candidates takes ~100ms; the quality improvement in the final top-5 is significant.

---

## 16. Entity Extraction Service

```python
# app/services/entities.py
from functools import lru_cache
import spacy

LANG_TO_MODEL = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "sv": "sv_core_news_sm",
    # Norwegian: no dedicated spaCy model; multilingual fallback used
    # Quality is lower than language-specific models — treat as best-effort
    "no": "xx_ent_wiki_sm",
}
FALLBACK_MODEL = "xx_ent_wiki_sm"
KEPT_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT"}
MAX_CHARS = 8000   # cap for NER performance


@lru_cache(maxsize=8)
def _load_nlp(model_name: str):
    return spacy.load(model_name)


class EntityExtractionService:
    """
    Best-effort multilingual NER. Quality varies by language:
    - EN, DE, FR, SV: good (dedicated models)
    - NO: lower quality (multilingual fallback model)
    Extraction failure never blocks ingestion.
    """

    def extract(self, text: str, lang: str) -> list[dict]:
        model_name = LANG_TO_MODEL.get(lang, FALLBACK_MODEL)
        doc = _load_nlp(model_name)(text[:MAX_CHARS])
        seen: set[tuple[str, str]] = set()
        result = []
        for ent in doc.ents:
            name = ent.text.strip()
            key = (name.lower(), ent.label_)
            if ent.label_ not in KEPT_TYPES or len(name) < 2 or key in seen:
                continue
            seen.add(key)
            result.append({"name": name, "type": ent.label_})
        return result
```

**NER quality honest statement:**
- EN, DE, FR, SV: language-specific spaCy models. Quality is good for common entity types (persons, organizations, locations).
- NO: `xx_ent_wiki_sm` multilingual model. Quality is lower. Norwegian NER is best-effort.
- Entity extraction is a boost to entity-based retrieval, not a hard dependency. Extraction failure is caught and logged; ingestion proceeds.

---

## 17. LLM Adapter

```python
# app/services/llm.py
from typing import Protocol, runtime_checkable
from openai import AsyncOpenAI
from ..config import settings

SYSTEM_PROMPT = """You are an assistant for an internal knowledge base.

Rules you must follow:
- Answer using ONLY the provided context passages.
- If the context does not contain enough information, respond: "The available documents do not contain enough information to answer this question."
- If two sources disagree, say so explicitly: identify which source says what, and do not pick a side unless the evidence clearly supports one claim.
- Reference passages by their [Source N] label when making a claim.
- Do not invent facts, dates, names, or organizations.
- Prefer saying "I don't know from these sources" over guessing."""


@runtime_checkable
class LLMClient(Protocol):
    async def generate(self, question: str, context_chunks: list[dict]) -> str: ...


def _build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Source {i+1}: {c.get('title', 'Unknown')} | {c.get('source', '')} | "
        f"{c.get('published_at', 'unknown date')} | lang:{c.get('language', '?')}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )


class OllamaClient:
    def __init__(self):
        self._client = AsyncOpenAI(
            base_url=settings.ollama_base_url + "/v1",
            api_key="ollama",   # Ollama ignores the key but the client requires a non-empty value
        )

    async def generate(self, question: str, context_chunks: list[dict]) -> str:
        resp = await self._client.chat.completions.create(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{_build_context(context_chunks)}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        return resp.choices[0].message.content


class OpenRouterClient:
    def __init__(self):
        self._client = AsyncOpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

    async def generate(self, question: str, context_chunks: list[dict]) -> str:
        resp = await self._client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{_build_context(context_chunks)}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        return resp.choices[0].message.content


class MockLLMClient:
    """Deterministic, no network call. Used in pytest and retrieval-only testing."""
    async def generate(self, question: str, context_chunks: list[dict]) -> str:
        refs = ", ".join(f"[Source {i+1}]" for i in range(len(context_chunks)))
        return f"[MOCK] Based on {refs}: test answer for '{question[:50]}'."


def get_llm_client() -> LLMClient:
    if settings.llm_provider == "mock":
        return MockLLMClient()
    if settings.llm_provider == "openrouter":
        return OpenRouterClient()
    return OllamaClient()  # default
```

---

## 18. Ingestion Pipeline

Full step-by-step for text upload (PDF adds one step before step 1):

```
0. [PDF only] Extract text via PyMuPDF. Reject if empty.
1. Hash content (SHA-256). Check for duplicate — return early if found.
2. Detect language (langdetect on first 2000 chars). Fallback: null.
3. Normalize metadata: fill defaults, preserve what's null rather than inventing.
4. Write Document row, status=processing. flush() to get id.
5. chunk_text(extracted_text) → list[TextChunk].
6. embed_passages([chunk.text for chunk in chunks]) → list[embedding] (one batch).
7. Write Chunk rows with embeddings and denormalized metadata. flush().
   [FTS trigger fires automatically on INSERT — no code needed.]
8. [Best-effort] Extract entities → upsert Entity rows → write ChunkEntity rows. flush().
   Wrapped in try/except: NER failure never propagates.
9. Update Document status=ready. commit().
10. Return UploadResponse.
```

### Metadata normalization rules

| Field | User provides | User omits |
|---|---|---|
| `language` | Use as-is | langdetect result; null if detection fails |
| `title` | Use as-is | First line of text (≤120 chars); else "Untitled document" |
| `title` (PDF) | Use as-is | Filename stem (e.g., "report_2022"); else "Untitled document" |
| `source` | Use as-is | `"user_upload"` |
| `document_type` | Use as-is | `"text_note"` for /text; `"pdf_document"` for /document |
| `published_at` | Use as-is | `null` — never invent a date |
| `topics` | Use as-is | `null` |
| `region`, `author` | Use as-is | `null` |

Inference provenance is stored: `metadata = {"provided": {...}, "inferred": {"language": "sv", "title": "First line..."}}`.

---

## 19. API Design

### POST /text

```
Request: { "text": str, "metadata"?: { title?, source?, language?, published_at?, document_type?, region?, author?, topics?, collection?, external_id? } }
Response: { "document_id": UUID, "chunks_created": int, "entities_extracted": int, "language_detected": str|null, "duplicate": bool }
Errors: 422 if text is empty
```

### POST /document

```
Request: multipart/form-data — file (PDF), metadata (JSON string, optional)
Response: { "document_id": UUID, "chunks_created": int, "entities_extracted": int, "duplicate": bool }
Errors: 400 if not PDF; 422 if PDF has no extractable text
```

### GET /search

```
Query params:
  q          (required) search text
  language   ISO 639-1 code
  source     exact match
  document_type exact match
  region     exact match
  topic      contained in topics array
  from_date  ISO date (YYYY-MM-DD)
  to_date    ISO date
  limit      int, default=10, max=50

Response:
{
  "query": str,
  "filters_applied": { ... },
  "results": [
    {
      "chunk_id": UUID,
      "document_id": UUID,
      "title": str|null,
      "source": str|null,
      "published_at": str|null,
      "language": str|null,
      "score": float,
      "text": str
    }
  ]
}
Note: empty results → empty list, not 404.
```

### POST /chat

```
Request:
{
  "question": str,
  "filters"?: { language?, source?, document_type?, region?, topic?, from_date?, to_date? }
}

Response:
{
  "answer": str,
  "confidence": "high"|"medium"|"low"|"insufficient",
  "sources": [{ "document_id", "chunk_id", "title", "source", "published_at", "language" }],
  "retrieval_notes": {
    "hybrid_candidates": int,
    "reranked_to": int,
    "metadata_filters_applied": [str],
    "entity_boost_applied": bool
  }
}

Insufficient evidence: answer = "The available documents do not contain enough information..."
Contradictory sources: LLM instructed to surface disagreement, not pick a side.
```

### GET /documents/{id}

```
Response: document metadata + list of chunk summaries (id, index, token_count, snippet)
```

### GET /entities/{name}/documents

```
Response: list of documents mentioning the named entity (via chunk_entities join)
```

### GET /health

```
Response: { "status": "ok" }
```

---

## 20. Supported Languages

| Code | Language | spaCy model | PostgreSQL FTS config | NER quality |
|---|---|---|---|---|
| `en` | English | `en_core_web_sm` | `english` | Good |
| `sv` | Swedish | `sv_core_news_sm` | `swedish` | Good |
| `de` | German | `de_core_news_sm` | `german` | Good |
| `fr` | French | `fr_core_news_sm` | `french` | Good |
| `no` | Norwegian | `xx_ent_wiki_sm` | `norwegian` | Best-effort (multilingual fallback) |
| other | Any | `xx_ent_wiki_sm` | `simple` | Best-effort |

---

## 21. Demo Data

10 synthetic documents, 5 languages, designed to demonstrate all retrieval scenarios.

| # | File | Lang | Topic | Year | Demonstrates |
|---|---|---|---|---|---|
| 1 | `01_en_energy_2014.txt` | EN | EU energy security | 2014 | Base case |
| 2 | `02_de_gas_2014.txt` | DE | Germany gas supply | 2014 | Same topic, same year, different language → cross-language retrieval |
| 3 | `03_fr_eu_energy_2018.txt` | FR | EU energy strategy | 2018 | French language coverage |
| 4 | `04_sv_energy_2022.txt` | SV | Swedish energy policy | 2022 | Same topic, 8 years later → date filtering |
| 5 | `05_no_nato_2023.txt` | NO | Norway NATO | 2023 | Norwegian coverage, different topic |
| 6 | `06_en_sanctions_2022.txt` | EN | Ukraine sanctions, energy | 2022 | Same entities as doc 1 (Germany, EC) → entity co-occurrence |
| 7 | `07_en_climate_2015.txt` | EN | EU climate negotiations | 2015 | Different subtopic within EU policy |
| 8 | `08_en_elections_2019.txt` | EN | EU elections, misinformation | 2019 | Different domain |
| 9 | `09_en_media_memo.txt` | EN | Internal media regulation | 2020 | Different document_type (memo) |
| 10 | `10_en_sanctions_report.txt` | EN | Energy sanctions detail | 2022 | Contradicts/evolves doc 6's position → contradiction surfacing |

**Demo query set:**

```bash
# 1. Date-filtered semantic search
curl "http://localhost:8001/search?q=energy+security&language=en&from_date=2014-01-01&to_date=2015-01-01"

# 2. Cross-language semantic search (German query, no filter — should find EN + DE docs)
curl "http://localhost:8001/search?q=Energiesicherheit+Europa&limit=10"

# 3. Chat with date filter
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What did EU officials say about energy security in 2014?","filters":{"from_date":"2014-01-01","to_date":"2014-12-31"}}'

# 4. Entity co-occurrence
curl "http://localhost:8001/entities/European%20Commission/documents"

# 5. Contradiction surfacing
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Was the EU position on energy sanctions consistent between 2014 and 2022?"}'
```

---

## 22. Testing Plan

```python
# tests/conftest.py
import pytest, pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.llm import MockLLMClient

@pytest_asyncio.fixture
async def client():
    # Override LLM with mock — no network call in tests
    app.dependency_overrides[get_llm_client] = lambda: MockLLMClient()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
```

| Test | Verifies |
|---|---|
| Upload text with full metadata | document + chunks in DB, metadata stored correctly |
| Upload text with no metadata | language detected, title inferred, source defaults to user_upload |
| Upload text duplicate | second upload returns duplicate=true, no new chunks |
| Upload PDF | text extracted, chunks created |
| Upload empty PDF | 422 error |
| Search returns results | hybrid search finds seeded document |
| Language filter | results restricted to requested language |
| Date filter | results restricted by published_at range |
| Topic filter | `topics @> '["energy"]'` correctly filters |
| Chat returns citations | answer includes at least one source reference |
| Chat insufficient evidence | confidence=insufficient when no matching docs |
| Entity lookup | GET /entities/{name}/documents returns relevant docs |

---

## 23. 4-Day Build Timeline

### Day 1 — Foundation

**Morning:** Git repo, Docker Compose, Postgres up and verified, `.env.example`, `GET /health` alive.  
Commits: "Initial FastAPI skeleton with Docker Compose" · "Add Postgres with pgvector"

**Midday:** SQLAlchemy models, Alembic init + first migration, `alembic upgrade head`, verify tables.  
Commit: "Add database models and Alembic migrations"

**Evening:** ChunkingService, EmbeddingService, `POST /text` endpoint, ingestion pipeline (steps 1-7).  
Commits: "Add text upload endpoint with chunking" · "Add multilingual-e5 embedding service"

---

### Day 2 — Ingestion Complete + Search

**Morning:** `POST /document` (PyMuPDF), test with a real PDF.  
Commit: "Add PDF upload endpoint with PyMuPDF extraction"

**Midday:** RetrievalService (vector + FTS + RRF + filters), `GET /search`.  
Commits: "Add hybrid retrieval with vector and FTS" · "Add metadata filters to search"

**Evening:** RerankingService, integrate into search, test that reranked order differs.  
Commit: "Add cross-encoder reranking after hybrid retrieval"

---

### Day 3 — Chat + Entities + Demo

**Morning:** LLMClient (Ollama + OpenRouter + Mock), ChatService, `POST /chat`.  
Commits: "Add LLM adapter with Ollama and OpenRouter support" · "Add chat endpoint with citations"

**Midday:** EntityExtractionService, integrate into ingestion (step 8), entity endpoints.  
Commits: "Add spaCy NER and Postgres entity tables" · "Add entity lookup endpoint"

**Evening:** Demo data (10 files), `seed.sh`, example curl scripts. Full end-to-end test.  
Commits: "Add multilingual demo dataset" · "Add example curl scripts"

---

### Day 4 — Tests + README + Polish

**Morning:** pytest suite — all test cases from §22. Fix failures.  
Commit: "Add test suite with mock LLM"

**Midday:** README.md finalized (see §24 for content). `.env.example` verified.  
Commit: "Add README with architecture, tradeoffs, and AI collaboration notes"

**Afternoon:** Full fresh `docker compose up`, run `seed.sh`, run all demo queries, fix anything broken.  
Commit: "Final end-to-end verification and polish"

**If time:** Simple `X-API-Key` middleware for auth.  
Commit: "Add optional API key authentication"

---

## 24. README Strategy (incremental commits)

The README is committed incrementally — each section is added alongside the code it describes. At Day 4 it becomes the full document below.

**Sections in commit order:**
1. Day 1 morning: Title, stack table, Quick Start placeholder, GET /health
2. Day 1 evening: POST /text section + curl example
3. Day 2: GET /search section, hybrid search architecture
4. Day 3: POST /chat section, entity lookup, demo data section
5. Day 4: Full README — tradeoffs, AI collaboration, what was left out

**The "What Was Completed" section is only written after the code is done.** It describes reality, not plans.

**AI collaboration section — honest account to include:**

> I used Claude Code as my primary implementation tool. Here is an exact account of what was delegated versus what required my direction.
>
> **Delegated:** FastAPI boilerplate, Alembic setup, Docker Compose template, spaCy model loading patterns, RRF implementation, initial test scaffold.
>
> **Directed by me — where AI needed correction:**
>
> *Chunk size:* AI initially chose 512 content tokens. I corrected this to 500 after working through the math: the E5 model's 512-token limit covers the full encoded sequence including the "passage: " prefix (~3 tokens) and special tokens [CLS]/[SEP] (~2 tokens). 512 content tokens would overflow. 500 tokens leaves correct headroom.
>
> *Vector indexing:* AI recommended an IVFFlat index. I removed it after reading pgvector docs: IVFFlat trades recall for speed and requires a minimum dataset size to be effective. At demonstration scale (hundreds of chunks), exact cosine search is faster, has 100% recall, and requires no VACUUM step.
>
> *Topics JSONB filter:* AI's initial SQL used `topics @> :filter_topic` with a plain string binding. This fails at runtime because asyncpg passes the string as TEXT, not JSONB. I corrected it to `CAST(:filter_topic AS JSONB)` with `json.dumps([filters.topic])` as the bound value.
>
> *NER quality:* AI wrote "reliable NER for all 5 languages." I softened this to "best-effort" for Norwegian, which uses a multilingual fallback model, not a dedicated Norwegian model.
>
> *Neo4j:* The original blueprint required Neo4j. I removed it and replaced it with Postgres entity tables. The co-occurrence query requirement is fully satisfiable with a SQL join, and adding an unfamiliar third service under a 4-day constraint is not sound engineering judgment.

---

## 25. Interview Prep

### "Why chunk size 512 and not 1024?"
> "I actually chose 500 tokens, not 512, and the reason matters: multilingual-e5-base has a 512-token maximum sequence length that applies to the complete encoded sequence, including the 'passage:' prefix and special tokens. If I set chunk content to 512 tokens, the total input would overflow and the model would silently truncate. 500 tokens with approximately 5 tokens of overhead stays safely within the limit. The choice over 1024 is about citation precision: 500 tokens is about 3 paragraphs, enough context to answer a question, small enough that each chunk is focused on one topic and citations are easy to verify."

### "How would you add the graph component with another week?"
> "The entity co-occurrence data is already collected — it's stored in Postgres entity tables. With more time I'd migrate that layer to Neo4j. Specifically: write entities and chunk_entities to Neo4j instead of Postgres, add entity normalization and alias resolution so 'EC' and 'European Commission' merge, build weighted CO_OCCURS_WITH edges, then evaluate whether graph-expanded search actually improves recall on the demo queries. I'd run both paths side by side and only activate graph boosting if the evaluation showed improvement — noisy entity extraction can hurt precision if used as a hard filter."

### "What happens if two documents contradict each other?"
> "The LLM is explicitly instructed in the system prompt: 'If two sources disagree, say so explicitly — identify which source says what, and do not pick a side unless the evidence clearly supports one claim.' The response shape includes sources with chunk_id and document_id, so the reviewer can read both original passages. The answer might say: 'Source 1 from 2014 states X, while Source 2 from 2022 indicates Y — the position appears to have evolved.' The evaluator can verify both claims against the actual chunks."

### "Where did you trust the AI and where did you correct it?"
> "Trusted for: Docker Compose, Alembic setup, FastAPI routing, RRF implementation, mock LLM pattern. Corrected on: chunk size math (overflow), IVFFlat over-engineering, JSONB topics filter bug, NER quality overstatement, and removal of Neo4j."

---

## 26. Acceptance Criteria

- [ ] `docker compose up` starts cleanly from a fresh clone with only `.env` configured
- [ ] `POST /text` accepts full metadata, partial metadata, and no metadata
- [ ] `POST /document` accepts a PDF, extracts text, returns chunk count
- [ ] Duplicate content returns `duplicate: true` without re-ingesting
- [ ] `docker compose restart` — all documents remain searchable (persistence)
- [ ] `GET /search?q=...` returns chunks with scores and document references
- [ ] Language filter limits results to that language only
- [ ] Date range filter limits results by `published_at`
- [ ] Topic filter (`?topic=energy`) correctly uses JSONB containment
- [ ] Multilingual query (no `language` filter) returns results across languages
- [ ] `POST /chat` returns answer with at least one source reference
- [ ] `POST /chat` with unanswerable question returns `confidence: insufficient`
- [ ] `GET /entities/{name}/documents` returns documents mentioning that entity
- [ ] `GET /documents/{id}` returns document metadata and chunk list
- [ ] `GET /health` returns 200
- [ ] `examples/seed.sh` runs without errors on a fresh system
- [ ] All demo queries return expected results after seeding
- [ ] `pytest` passes with no failures
- [ ] GitHub repo has 15+ commits, no single large dump commit
- [ ] README is accurate to the actual built state

---

## 27. What to Leave Out (and How to Explain Each)

| Omission | README/interview explanation |
|---|---|
| Neo4j | Entity co-occurrence is implemented with Postgres joins on `chunk_entities`. Neo4j would improve query expressiveness at large graph scale but adds a third service and requires Cypher. The assignment marks it optional. With more time I would evaluate whether graph traversal complexity justifies the overhead. |
| Streaming | `POST /chat` returns synchronously. Adding streaming requires `StreamingResponse` in FastAPI and the OpenAI streaming API — straightforward but not part of core retrieval correctness. |
| Fine-tuned embeddings | `multilingual-e5-base` is a strong open-source baseline. Fine-tuning requires labeled query-passage pairs for this specific domain. |
| Frontend | Out of scope per the assignment. Swagger UI at `/docs` is sufficient for evaluator testing. |
| Auth | An `X-API-Key` middleware is trivial to add. Not in baseline scope. |
| CI/CD, distributed ingestion, metrics | Out of scope per the assignment. |
