"""FastAPI endpoint for multi-modal PDF document ingestion."""

import logging
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.exceptions import PDFParsingError, RAGEngineException
from app.models.schemas import IngestionResult
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ingestion"])

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit


def get_ingestion_service() -> IngestionService:
    """Dependency injection provider for IngestionService."""
    return IngestionService()


@router.post(
    "/ingest",
    response_model=IngestionResult,
    status_code=status.HTTP_200_OK,
    summary="Ingest a multi-modal PDF document",
    description="Parses PDF text and embedded visual figures, generates dense VLM descriptions, and stores vectors in ChromaDB.",
)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to ingest and index"),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResult:
    """Upload and process a multi-modal PDF."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid file extension: '{file.filename}'. Only .pdf files are supported.",
        )

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)",
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of 50 MB ({len(content)} bytes uploaded)",
        )

    try:
        result = await service.ingest_bytes(content, file.filename)
        return result
    except PDFParsingError as pe:
        logger.warning("PDF parsing rejection: %s", pe)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PDF parsing error: {pe.message}",
        )
    except RAGEngineException as re:
        logger.error("Domain error during ingestion: %s", re)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {re.message}",
        )
    except Exception as exc:
        logger.exception("Unexpected error during PDF ingestion: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred: {str(exc)}",
        )
