"""FastAPI endpoint for multi-modal PDF document ingestion."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.api.dependencies import get_ingestion_service
from app.api.middleware import check_rate_limit, verify_api_key
from app.core.exceptions import (
    DocumentEmptyError,
    PDFParsingError,
    RAGEngineException,
    SecurityValidationError,
)
from app.models.schemas import IngestionResult
from app.services.ingestion_service import IngestionService
from app.utils.file_utils import validate_pdf_signature

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ingestion"])


@router.post(
    "/ingest",
    response_model=IngestionResult,
    status_code=status.HTTP_200_OK,
    summary="Ingest a multi-modal PDF document",
    description="Parses PDF text and embedded visual figures, generates dense VLM descriptions, and stores vectors in ChromaDB.",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def ingest_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to ingest and index"),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResult:
    """Upload and process a multi-modal PDF."""
    req_id = getattr(request.state, "request_id", "unknown")

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

    max_allowed_bytes = service.settings.max_file_size_mb * 1024 * 1024
    chunks: list[bytes] = []
    total_bytes = 0
    chunk_buffer_size = 64 * 1024  # 64 KB read buffer

    try:
        while True:
            chunk = await file.read(chunk_buffer_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_allowed_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum allowed size of {service.settings.max_file_size_mb} MB",
                    headers={"X-Request-ID": req_id},
                )
            chunks.append(chunk)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Failed to read uploaded file [Request: %s]: %s", req_id, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file payload",
            headers={"X-Request-ID": req_id},
        )

    if total_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)",
            headers={"X-Request-ID": req_id},
        )

    content = b"".join(chunks)

    # Security check: validate PDF magic header
    if not validate_pdf_signature(content):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file header does not match valid PDF signature (%PDF-)",
            headers={"X-Request-ID": req_id},
        )

    try:
        result = await service.ingest_bytes(content, file.filename)
        return result
    except SecurityValidationError as se:
        logger.warning("Security validation failure [Request: %s]: %s", req_id, se)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Security rejection: {se.message}",
            headers={"X-Request-ID": req_id},
        )
    except DocumentEmptyError as de:
        logger.warning("Empty document rejection [Request: %s]: %s", req_id, de)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=de.message,
            headers={"X-Request-ID": req_id},
        )
    except PDFParsingError as pe:
        logger.warning("PDF parsing rejection [Request: %s]: %s", req_id, pe)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PDF parsing error: {pe.message}",
            headers={"X-Request-ID": req_id},
        )
    except RAGEngineException as re:
        logger.error("Domain error during ingestion [Request: %s]: %s", req_id, re)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {re.message}",
            headers={"X-Request-ID": req_id},
        )
    except Exception as exc:
        # Never leak raw traceback or internal server paths to API clients
        logger.exception("Unexpected error during PDF ingestion [Request: %s]: %s", req_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred while processing document. Reference: {req_id}",
            headers={"X-Request-ID": req_id},
        )
