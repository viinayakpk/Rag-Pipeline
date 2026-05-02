from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from importlib.util import find_spec

from fastapi import Depends, FastAPI, Header, HTTPException, status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from .api import chat, document, documents, search, text as text_api
from .config import get_settings
from .db.session import dispose_engine, get_db, ping_database
from .errors import ServiceDependencyError

_RUNTIME_DEPENDENCY_HINTS = {
    "fitz": "PyMuPDF is required for /document",
    "langdetect": "langdetect is required for ingestion language detection",
    "langchain_text_splitters": "langchain-text-splitters is required for chunking",
    "sentence_transformers": "sentence-transformers is required for embeddings and reranking",
    "spacy": "spaCy is required for entity extraction",
    "transformers": "transformers is required for token-aware chunking",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await dispose_engine()


async def _check_api_key(x_api_key: str | None = Header(default=None)) -> None:
    s = get_settings()
    if s.api_key and x_api_key != s.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header")


app = FastAPI(
    title="Ahody RAG Backend",
    version="0.1.0",
    lifespan=lifespan,
)

_auth = [Depends(_check_api_key)]
app.include_router(text_api.router,   dependencies=_auth)
app.include_router(document.router,   dependencies=_auth)
app.include_router(search.router,     dependencies=_auth)
app.include_router(chat.router,       dependencies=_auth)
app.include_router(documents.router,  dependencies=_auth)


def _missing_runtime_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name, hint in _RUNTIME_DEPENDENCY_HINTS.items():
        if find_spec(module_name) is None:
            missing.append(f"{module_name}: {hint}")
    return missing


@app.get("/health", response_model=None)
async def health_check(
    db: AsyncSession = Depends(get_db),
) -> JSONResponse | dict[str, str]:
    try:
        await ping_database(db)
    except SQLAlchemyError as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "database": "error",
                "llm_provider": get_settings().llm_provider,
                "detail": exc.__class__.__name__,
            },
        )

    return {
        "status": "ok",
        "database": "ok",
        "llm_provider": get_settings().llm_provider,
    }


@app.exception_handler(ServiceDependencyError)
async def handle_service_dependency_error(_, exc: ServiceDependencyError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "degraded",
            "detail": str(exc),
        },
    )


@app.get("/ready", response_model=None)
async def readiness_check(
    db: AsyncSession = Depends(get_db),
) -> JSONResponse | dict[str, object]:
    try:
        await ping_database(db)
    except SQLAlchemyError as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "database": "error",
                "dependencies": "unknown",
                "llm_provider": get_settings().llm_provider,
                "detail": exc.__class__.__name__,
            },
        )

    missing_dependencies = _missing_runtime_dependencies()
    if missing_dependencies:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "database": "ok",
                "dependencies": "missing",
                "llm_provider": get_settings().llm_provider,
                "missing_dependencies": missing_dependencies,
            },
        )

    return {
        "status": "ok",
        "database": "ok",
        "dependencies": "ok",
        "llm_provider": get_settings().llm_provider,
    }
