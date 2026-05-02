from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.ingest import DocumentMetadata, IngestResponse
from app.services.ingestion import ingest_text

router = APIRouter()


def _extract_pdf_text(content: bytes) -> str:
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="PyMuPDF is not installed. Install project dependencies before using /document.",
        ) from exc

    try:
        pdf = fitz.open(stream=content, filetype="pdf")
        pages = [page.get_text() for page in pdf]
        return "\n\n".join(pages)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc


@router.post("/document", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    text_content = _extract_pdf_text(content)
    if not text_content.strip():
        raise HTTPException(status_code=422, detail="The uploaded PDF did not contain extractable text")

    try:
        parsed_metadata = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"metadata must be valid JSON: {exc.msg}") from exc

    if not isinstance(parsed_metadata, dict):
        raise HTTPException(
            status_code=400,
            detail="metadata must be a JSON object matching the metadata schema",
        )

    try:
        doc_metadata = DocumentMetadata(**parsed_metadata)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    result = await ingest_text(
        db,
        text_content,
        doc_metadata,
        original_filename=file.filename,
        mime_type=file.content_type,
        byte_size=len(content),
    )
    return IngestResponse(**result)
