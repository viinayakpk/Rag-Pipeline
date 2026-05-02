from __future__ import annotations

import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import ChatLog
from app.schemas.chat import ChatRequest, ChatResponse, RetrievalNotes
from app.schemas.search import ChunkResult
from app.services.llm import generate_text_response, get_active_model_name
from app.services.reranking import rerank_async
from app.services.retrieval import hybrid_search

_PROMPT_TEMPLATE = """\
You are answering from retrieved documents only.

Rules:
- Use only the provided context.
- If the context is insufficient, say that clearly.
- If the context contains disagreement, surface the disagreement instead of picking a side.
- Cite sources inline using [1], [2], etc.

Context:
{context}

Question: {question}
Answer:"""


def _metadata_filters_applied(request: ChatRequest) -> list[str]:
    if not request.filters:
        return []
    return sorted(
        key for key, value in request.filters.model_dump(exclude_none=True).items() if value
    )


def _confidence_for_sources(source_count: int, top_score: float = 1.0) -> str:
    """Retrieval-strength heuristic, not truth confidence.
    Thresholds approximate bge-reranker-v2-m3 sigmoid midpoint (0.50);
    recalibrate with real queries if score distribution shifts."""
    if source_count == 0:
        return "insufficient"
    if top_score < 0.50:
        return "low"
    if source_count >= 4 and top_score >= 0.75:
        return "high"
    if source_count >= 2:
        return "medium"
    return "low"


def _cited_indices(answer: str, total: int) -> set[int]:
    """Return the 1-based [N] citation indices found in the LLM answer.
    Falls back to all indices when the model cites nothing explicitly."""
    found = {int(m) for m in re.findall(r'\[(\d+)\]', answer) if 1 <= int(m) <= total}
    return found if found else set(range(1, total + 1))


async def answer_question(db: AsyncSession, request: ChatRequest) -> ChatResponse:
    settings = get_settings()

    candidates = await hybrid_search(
        db,
        request.question,
        request.filters,
        candidate_count=settings.hybrid_candidate_count,
    )
    top_chunks = await rerank_async(
        request.question, candidates, top_k=settings.chat_context_top_k
    )
    retrieval_notes = RetrievalNotes(
        hybrid_candidates=len(candidates),
        reranked_to=len(top_chunks),
        metadata_filters_applied=_metadata_filters_applied(request),
        entity_boost_applied=False,
    )

    if not top_chunks:
        answer = "The available documents do not contain enough information to answer this question."
        confidence = "insufficient"
    else:
        context_parts = []
        for i, chunk in enumerate(top_chunks, 1):
            title = chunk.get("title") or "Untitled"
            context_parts.append(f"[{i}] {title}\n{chunk['text']}")
        context = "\n\n".join(context_parts)
        prompt = _PROMPT_TEMPLATE.format(context=context, question=request.question)
        answer = await generate_text_response(prompt)
        top_score = top_chunks[0].get("score", 0.0) if top_chunks else 0.0
        confidence = _confidence_for_sources(len(top_chunks), top_score)

    cited = _cited_indices(answer, len(top_chunks))
    cited_chunks = [c for i, c in enumerate(top_chunks, 1) if i in cited]

    log = ChatLog(
        question=request.question,
        filters=request.filters.model_dump(mode="json") if request.filters else {},
        answer=answer,
        confidence=confidence,
        retrieved_chunk_ids=[c["chunk_id"] for c in top_chunks],
        sources=cited_chunks,
        retrieval_notes=retrieval_notes.model_dump(),
        model_name=get_active_model_name(),
    )
    db.add(log)
    await db.commit()

    sources = [
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
        for c in cited_chunks
    ]

    return ChatResponse(
        question=request.question,
        answer=answer,
        confidence=confidence,
        sources=sources,
        retrieval_notes=retrieval_notes,
    )
