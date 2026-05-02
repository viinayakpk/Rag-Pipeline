from __future__ import annotations

import json
import logging
import time

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.search import SearchFilters
from app.services.embeddings import embed_query_async

logger = logging.getLogger(__name__)

_FTS_CONFIG_BY_LANGUAGE = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "sv": "swedish",
    "no": "norwegian",
}
_DEFAULT_FTS_CONFIG = "simple"
_ALL_FTS_CONFIGS = tuple(
    dict.fromkeys((*_FTS_CONFIG_BY_LANGUAGE.values(), _DEFAULT_FTS_CONFIG))
)


def _vec_to_pg(embedding: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in embedding) + "]"


def _get_fts_configs(language: str | None) -> tuple[str, ...]:
    if language:
        return (_FTS_CONFIG_BY_LANGUAGE.get(language.lower(), _DEFAULT_FTS_CONFIG),)
    return _ALL_FTS_CONFIGS


def _build_fts_rank_expression(configs: tuple[str, ...]) -> str:
    if any(config not in _ALL_FTS_CONFIGS for config in configs):
        raise ValueError(f"Unsupported FTS config set: {configs!r}")

    terms = [
        f"ts_rank_cd(c.fts_vector, plainto_tsquery('{config}'::REGCONFIG, :query))"
        for config in configs
    ]
    if len(terms) == 1:
        return terms[0]
    return "GREATEST(" + ", ".join(terms) + ")"


def _build_fts_match_expression(configs: tuple[str, ...]) -> str:
    if any(config not in _ALL_FTS_CONFIGS for config in configs):
        raise ValueError(f"Unsupported FTS config set: {configs!r}")

    return " OR ".join(
        f"c.fts_vector @@ plainto_tsquery('{config}'::REGCONFIG, :query)"
        for config in configs
    )


def _build_filter_sql(filters: SearchFilters | None) -> tuple[str, dict]:
    if filters is None:
        return "", {}
    clauses = ""
    params: dict = {}
    if filters.language:
        clauses += " AND c.language = :f_language"
        params["f_language"] = filters.language
    if filters.source:
        clauses += " AND c.source = :f_source"
        params["f_source"] = filters.source
    if filters.document_type:
        clauses += " AND c.document_type = :f_document_type"
        params["f_document_type"] = filters.document_type
    if filters.region:
        clauses += " AND c.region = :f_region"
        params["f_region"] = filters.region
    if filters.topic:
        clauses += " AND c.topics @> CAST(:f_topic AS JSONB)"
        params["f_topic"] = json.dumps([filters.topic])
    if filters.from_date:
        clauses += " AND c.published_at >= :f_from_date"
        params["f_from_date"] = filters.from_date
    if filters.to_date:
        clauses += " AND c.published_at <= :f_to_date"
        params["f_to_date"] = filters.to_date
    return clauses, params


def _row_to_dict(row) -> dict:
    return {
        "chunk_id": str(row[0]),
        "document_id": str(row[1]),
        "chunk_index": row[2],
        "text": row[3],
        "title": row[4],
        "source": row[5],
        "language": row[6],
        "published_at": str(row[7]) if row[7] else None,
        "token_count": row[8],
        "score": 0.0,
    }


async def hybrid_search(
    db: AsyncSession,
    query: str,
    filters: SearchFilters | None = None,
    candidate_count: int = 30,
) -> list[dict]:
    t_start = time.perf_counter()

    embedding = await embed_query_async(query)
    embed_ms = (time.perf_counter() - t_start) * 1000

    vec_str = _vec_to_pg(embedding)

    filter_sql, filter_params = _build_filter_sql(filters)
    fts_configs = _get_fts_configs(filters.language if filters else None)
    fts_rank_expression = _build_fts_rank_expression(fts_configs)
    fts_match_expression = _build_fts_match_expression(fts_configs)

    vector_sql = text(f"""
        SELECT c.id, c.document_id, c.chunk_index, c.text,
               c.title, c.source, c.language, c.published_at, c.token_count,
               ROW_NUMBER() OVER (ORDER BY c.embedding <=> CAST(:embedding AS vector)) AS rank
        FROM chunks c
        WHERE c.embedding IS NOT NULL
        {filter_sql}
        ORDER BY c.embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)

    fts_sql = text(f"""
        SELECT c.id, c.document_id, c.chunk_index, c.text,
               c.title, c.source, c.language, c.published_at, c.token_count,
               ROW_NUMBER() OVER (
                   ORDER BY {fts_rank_expression} DESC
               ) AS rank
        FROM chunks c
        WHERE ({fts_match_expression})
        {filter_sql}
        ORDER BY {fts_rank_expression} DESC
        LIMIT :limit
    """)

    vec_params = {"embedding": vec_str, "limit": candidate_count, **filter_params}
    fts_params = {"query": query, "limit": candidate_count, **filter_params}

    t_db = time.perf_counter()
    vec_result = await db.execute(vector_sql, vec_params)
    fts_result = await db.execute(fts_sql, fts_params)
    db_ms = (time.perf_counter() - t_db) * 1000

    vec_rows = vec_result.fetchall()
    fts_rows = fts_result.fetchall()

    # Reciprocal Rank Fusion (k=60): score = Σ 1/(60 + rank) across both lists.
    t_rrf = time.perf_counter()
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for row in vec_rows:
        cid = str(row[0])
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + row[9])
        if cid not in chunk_data:
            chunk_data[cid] = _row_to_dict(row)

    for row in fts_rows:
        cid = str(row[0])
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + row[9])
        if cid not in chunk_data:
            chunk_data[cid] = _row_to_dict(row)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for cid, score in ranked[:candidate_count]:
        entry = dict(chunk_data[cid])
        entry["score"] = score
        results.append(entry)
    rrf_ms = (time.perf_counter() - t_rrf) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "hybrid_search embed_ms=%.0f db_ms=%.0f rrf_ms=%.0f total_ms=%.0f "
        "vec_rows=%d fts_rows=%d candidates=%d",
        embed_ms, db_ms, rrf_ms, total_ms, len(vec_rows), len(fts_rows), len(results),
    )

    return results
