from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.errors import ServiceDependencyError

CHUNK_SIZE = 500
CHUNK_OVERLAP = 60


@dataclass(slots=True)
class TextChunk:
    text: str
    chunk_index: int
    token_count: int


@lru_cache(maxsize=1)
def _get_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ServiceDependencyError(
            "transformers is required for token-aware chunking. "
            "Install project dependencies before calling ingestion."
        ) from exc
    return AutoTokenizer.from_pretrained(get_settings().embedding_model)


def _token_count(text: str) -> int:
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


def chunk_text(text: str) -> list[TextChunk]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ModuleNotFoundError as exc:
        raise ServiceDependencyError(
            "langchain-text-splitters is required for chunking. "
            "Install project dependencies before calling ingestion."
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=_token_count,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)
    return [
        TextChunk(text=chunk, chunk_index=index, token_count=_token_count(chunk))
        for index, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]
