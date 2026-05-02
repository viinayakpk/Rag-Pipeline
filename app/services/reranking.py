from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from app.config import get_settings
from app.errors import ServiceDependencyError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_torch_device() -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ServiceDependencyError(
            "torch is required for reranking. "
            "Install project dependencies before calling search or chat."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Reranker using device=%s", device)
    return device


@lru_cache(maxsize=1)
def _get_reranker():
    try:
        from sentence_transformers import CrossEncoder
    except ModuleNotFoundError as exc:
        raise ServiceDependencyError(
            "sentence-transformers is required for reranking. "
            "Install project dependencies before calling search or chat."
        ) from exc
    return CrossEncoder(
        get_settings().reranker_model,
        device=_get_torch_device(),
    )


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    if not chunks:
        return []
    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    result = []
    for score, chunk in ranked[:top_k]:
        entry = dict(chunk)
        entry["score"] = float(score)
        result.append(entry)
    return result


async def rerank_async(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, rerank, query, chunks, top_k)
