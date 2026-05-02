from __future__ import annotations

import logging
import time
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.session import get_db
from app.schemas.search import ChunkResult, SearchFilters, SearchResponse
from app.services.reranking import rerank_async
from app.services.retrieval import hybrid_search

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query"),
    language: str | None = Query(None),
    source: str | None = Query(None),
    document_type: str | None = Query(None),
    region: str | None = Query(None),
    topic: str | None = Query(None),
    from_date: date | None = Query(None),
    to_date: date | None = Query(None),
    limit: int | None = Query(None, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Search query must not be empty")
    q = q.strip()

    settings = get_settings()
    effective_limit = limit or settings.default_search_limit

    filters = SearchFilters(
        language=language,
        source=source,
        document_type=document_type,
        region=region,
        topic=topic,
        from_date=from_date,
        to_date=to_date,
    )

    t0 = time.perf_counter()
    candidates = await hybrid_search(
        db, q, filters, candidate_count=settings.hybrid_candidate_count
    )
    retrieve_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    top = await rerank_async(q, candidates, top_k=effective_limit)
    rerank_ms = (time.perf_counter() - t1) * 1000

    logger.info(
        "search retrieve_ms=%.0f rerank_ms=%.0f total_ms=%.0f candidates=%d results=%d",
        retrieve_ms, rerank_ms, retrieve_ms + rerank_ms, len(candidates), len(top),
    )

    results = [
        ChunkResult(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            text=c["text"],
            score=c.get("score", 0.0),
            title=c.get("title"),
            source=c.get("source"),
            language=c.get("language"),
            published_at=c.get("published_at"),
            chunk_index=c.get("chunk_index", 0),
            token_count=c.get("token_count"),
        )
        for c in top
    ]

    return SearchResponse(query=q, results=results, total=len(results))
