from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.ingest import IngestResponse, TextIngestRequest
from app.services.ingestion import ingest_text

router = APIRouter()


@router.post("/text", response_model=IngestResponse)
async def upload_text(
    request: TextIngestRequest,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    result = await ingest_text(db, request.text, request.metadata)
    return IngestResponse(**result)
