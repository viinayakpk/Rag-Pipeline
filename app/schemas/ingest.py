from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    source: str | None = None
    language: str | None = None
    published_at: date | None = None
    region: str | None = None
    author: str | None = None
    topics: list[str] = Field(default_factory=list)
    document_type: str | None = None

    @field_validator("title", "source", "region", "author", "document_type", mode="before")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        return _clean_optional_text(value)

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language_code(cls, value: str | None) -> str | None:
        return _normalize_language(value)

    @field_validator("topics", mode="before")
    @classmethod
    def clean_topics(cls, value) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("topics must be a JSON array of strings")

        cleaned_topics: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("topics must only contain strings")
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                cleaned_topics.append(cleaned)
        return cleaned_topics


class TextIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("text must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be empty")
        return cleaned


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    entities_extracted: int = 0
    language: str
    language_detected: str | None = None
    duplicate: bool = False
