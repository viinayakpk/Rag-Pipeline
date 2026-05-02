from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, field_validator


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"expected a string, got {type(value).__name__}")
    cleaned = value.strip()
    return cleaned or None


def _normalize_language(value: str | None) -> str | None:
    cleaned = _clean_optional_text(value)
    if cleaned is None:
        return None
    normalized = cleaned.lower()
    return {"nb": "no", "nn": "no"}.get(normalized, normalized)


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str | None = None
    source: str | None = None
    document_type: str | None = None
    region: str | None = None
    topic: str | None = None
    from_date: date | None = None
    to_date: date | None = None

    @field_validator("source", "document_type", "region", "topic", mode="before")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        return _clean_optional_text(value)

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language_code(cls, value: str | None) -> str | None:
        return _normalize_language(value)


class ChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    title: str | None = None
    source: str | None = None
    language: str | None = None
    published_at: str | None = None
    chunk_index: int
    token_count: int | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    total: int
