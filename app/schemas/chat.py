from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from app.schemas.search import ChunkResult, SearchFilters


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    filters: SearchFilters | None = None

    @field_validator("question", mode="before")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("question must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        return cleaned


class RetrievalNotes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hybrid_candidates: int
    reranked_to: int
    metadata_filters_applied: list[str]
    entity_boost_applied: bool = False


class ChatResponse(BaseModel):
    question: str
    answer: str
    confidence: Literal["high", "medium", "low", "insufficient"]
    sources: list[ChunkResult]
    retrieval_notes: RetrievalNotes
