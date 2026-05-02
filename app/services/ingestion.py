from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime, time, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import ServiceDependencyError
from app.db.models import Chunk, Document
from app.schemas.ingest import DocumentMetadata
from app.services.chunking import TextChunk, chunk_text
from app.services.embeddings import embed_passages_async
from app.services.entities import extract_entities


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _normalize_text_for_hash(text_content: str) -> str:
    lines = [line.strip() for line in text_content.splitlines()]
    return "\n".join(line for line in lines if line)


def _normalize_language_code(language: str | None) -> str | None:
    if not language:
        return None
    normalized = language.strip().lower()
    return {"nb": "no", "nn": "no"}.get(normalized, normalized)


def _detect_language(text_content: str) -> str:
    try:
        from langdetect import LangDetectException, detect
    except ModuleNotFoundError as exc:
        raise ServiceDependencyError(
            "langdetect is required for language detection during ingestion. "
            "Install project dependencies before calling /text or /document."
        ) from exc

    try:
        return detect(text_content)
    except LangDetectException:
        return "en"


def _coerce_published_at(value: date | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _build_document_metadata(
    metadata: DocumentMetadata,
    detected_language: str,
    effective_language: str,
) -> dict:
    payload = metadata.model_dump(mode="json", exclude_none=True)
    payload["inference"] = {
        "detected_language": detected_language,
        "effective_language": effective_language,
    }
    return payload


async def ingest_text(
    db: AsyncSession,
    text_content: str,
    metadata: DocumentMetadata,
    *,
    original_filename: str | None = None,
    mime_type: str | None = None,
    byte_size: int | None = None,
) -> dict:
    stripped_text = text_content.strip()
    if not stripped_text:
        raise ValueError("Text content must not be empty")

    content_hash = _sha256(_normalize_text_for_hash(text_content))

    existing = await db.execute(
        text("SELECT id, language, detected_language FROM documents WHERE content_hash = :h"),
        {"h": content_hash},
    )
    row = existing.fetchone()
    if row:
        return {
            "document_id": str(row[0]),
            "chunks_created": 0,
            "entities_extracted": 0,
            "language": row[1] or "unknown",
            "language_detected": row[2] or row[1] or "unknown",
            "duplicate": True,
        }

    detected_language = _normalize_language_code(_detect_language(stripped_text)) or "en"
    requested_language = _normalize_language_code(metadata.language)
    language = requested_language or detected_language
    published_at = _coerce_published_at(metadata.published_at)

    doc = Document(
        title=metadata.title,
        source=metadata.source,
        language=language,
        detected_language=detected_language,
        published_at=published_at,
        region=metadata.region,
        author=metadata.author,
        topics=metadata.topics,
        document_type=metadata.document_type,
        original_filename=original_filename,
        mime_type=mime_type,
        byte_size=byte_size,
        content_hash=content_hash,
        ingestion_status="processing",
        extracted_text=text_content,
        document_metadata=_build_document_metadata(metadata, detected_language, language),
    )
    db.add(doc)
    await db.flush()

    chunks = chunk_text(text_content) or [TextChunk(text=text_content, chunk_index=0, token_count=0)]
    chunk_texts = [item.text for item in chunks]
    embeddings = await embed_passages_async(chunk_texts)

    chunk_objs: list[Chunk] = []
    for chunk_data, embedding in zip(chunks, embeddings):
        chunk = Chunk(
            document_id=doc.id,
            chunk_index=chunk_data.chunk_index,
            text=chunk_data.text,
            token_count=chunk_data.token_count,
            language=language,
            source=metadata.source,
            title=metadata.title,
            published_at=published_at,
            document_type=metadata.document_type,
            region=metadata.region,
            topics=metadata.topics,
            embedding=embedding,
        )
        db.add(chunk)
        chunk_objs.append(chunk)

    await db.flush()

    entity_count = 0
    entity_warning: str | None = None
    try:
        entity_count = await _store_entities(db, chunk_objs, chunk_texts, language)
    except Exception as exc:
        entity_warning = exc.__class__.__name__

    if entity_warning:
        document_metadata = dict(doc.document_metadata or {})
        document_metadata["entity_extraction_warning"] = entity_warning
        doc.document_metadata = document_metadata

    doc.ingestion_status = "ready"

    await db.commit()
    return {
        "document_id": str(doc.id),
        "chunks_created": len(chunk_objs),
        "entities_extracted": entity_count,
        "language": language,
        "language_detected": detected_language,
        "duplicate": False,
    }


async def _store_entities(
    db: AsyncSession,
    chunks: list[Chunk],
    texts: list[str],
    language: str,
) -> int:
    entity_count = 0
    for chunk, chunk_content in zip(chunks, texts):
        entities = extract_entities(chunk_content, language)
        entity_count += len(entities)
        for ent in entities:
            entity_id = uuid.uuid4()
            result = await db.execute(
                text("""
                    INSERT INTO entities (id, name, normalized_name, entity_type)
                    VALUES (:id, :name, :normalized_name, :entity_type)
                    ON CONFLICT (normalized_name, entity_type) DO UPDATE
                        SET name = EXCLUDED.name
                    RETURNING id
                """),
                {"id": entity_id, **ent},
            )
            persisted_entity_id = result.scalar_one()
            await db.execute(
                text("""
                    INSERT INTO chunk_entities (chunk_id, entity_id, document_id)
                    VALUES (:chunk_id, :entity_id, :document_id)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "chunk_id": chunk.id,
                    "entity_id": persisted_entity_id,
                    "document_id": chunk.document_id,
                },
            )
    return entity_count
