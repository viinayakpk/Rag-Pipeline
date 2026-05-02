from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

router = APIRouter()


@router.get("/documents/{document_id}")
async def get_document_details(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    result = await db.execute(
        text("""
            SELECT id, title, source, language, detected_language, published_at, uploaded_at,
                   region, author, topics, document_type, original_filename, mime_type,
                   byte_size, content_hash, ingestion_status, metadata
            FROM documents WHERE id = :doc_id
        """),
        {"doc_id": document_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_result = await db.execute(
        text("""
            SELECT id, chunk_index, token_count, LEFT(text, 240) AS snippet
            FROM chunks
            WHERE document_id = :doc_id
            ORDER BY chunk_index
        """),
        {"doc_id": document_id},
    )
    chunk_rows = chunk_result.fetchall()

    return {
        "id": str(row[0]),
        "title": row[1],
        "source": row[2],
        "language": row[3],
        "detected_language": row[4],
        "published_at": str(row[5]) if row[5] else None,
        "uploaded_at": str(row[6]) if row[6] else None,
        "region": row[7],
        "author": row[8],
        "topics": row[9],
        "document_type": row[10],
        "original_filename": row[11],
        "mime_type": row[12],
        "byte_size": row[13],
        "content_hash": row[14],
        "ingestion_status": row[15],
        "metadata": row[16],
        "chunks": [
            {
                "id": str(chunk_row[0]),
                "chunk_index": chunk_row[1],
                "token_count": chunk_row[2],
                "snippet": chunk_row[3],
            }
            for chunk_row in chunk_rows
        ],
    }


@router.get("/entities/{entity_name}/documents")
async def list_documents_for_entity(
    entity_name: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    result = await db.execute(
        text("""
            SELECT DISTINCT d.id, d.title, d.source, d.language, d.published_at
            FROM documents d
            JOIN chunk_entities ce ON ce.document_id = d.id
            JOIN entities e ON e.id = ce.entity_id
            WHERE e.normalized_name = :name
            ORDER BY d.published_at DESC NULLS LAST, d.title
            LIMIT 50
        """),
        {"name": entity_name.lower()},
    )
    rows = result.fetchall()
    return {
        "entity": entity_name,
        "documents": [
            {
                "id": str(r[0]),
                "title": r[1],
                "source": r[2],
                "language": r[3],
                "published_at": str(r[4]) if r[4] else None,
            }
            for r in rows
        ],
    }
