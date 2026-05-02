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
            "torch is required for embeddings. "
            "Install project dependencies before calling search or ingestion."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Embedding model using device=%s", device)
    return device


@lru_cache(maxsize=1)
def _get_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ServiceDependencyError(
            "sentence-transformers is required for embeddings. "
            "Install project dependencies before calling search or ingestion."
        ) from exc
    return SentenceTransformer(
        get_settings().embedding_model,
        device=_get_torch_device(),
    )


def embed_passages(texts: list[str]) -> list[list[float]]:
    # E5 requires "passage: " prefix for documents at inference time.
    prefixed = [f"passage: {t}" for t in texts]
    vecs = _get_model().encode(prefixed, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def embed_query(query: str) -> list[float]:
    # E5 requires "query: " prefix for queries at inference time.
    vec = _get_model().encode(f"query: {query}", normalize_embeddings=True)
    return vec.tolist()


async def embed_passages_async(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embed_passages, texts)


async def embed_query_async(query: str) -> list[float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embed_query, query)
