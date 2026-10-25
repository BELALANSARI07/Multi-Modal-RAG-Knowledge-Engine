"""FastAPI endpoints for managing indexed documents in the knowledge base."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies import get_ingestion_service, get_vectorstore_service
from app.api.middleware import verify_api_key
from app.models.schemas import DocumentItem, DocumentListResponse
from app.services.ingestion_service import IngestionService
from app.services.vectorstore_service import VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Document Management"])


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all indexed documents",
    description="Returns metadata breakdown of all ingested documents and their text/image chunk counts.",
    dependencies=[Depends(verify_api_key)],
)
async def list_documents(
    vectorstore: VectorStoreService = Depends(get_vectorstore_service),
) -> DocumentListResponse:
    """List all indexed documents in the ChromaDB collection."""
    try:
        raw_docs = vectorstore.list_ingested_documents()
        items = [
            DocumentItem(
                filename=d["filename"],
                text_chunks=d["text_chunks"],
                image_chunks=d["image_chunks"],
                max_page=d["max_page"],
            )
            for d in raw_docs
        ]
        return DocumentListResponse(documents=items, total_count=len(items))
    except Exception as e:
        logger.exception("Failed to retrieve document list: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}",
        )


@router.delete(
    "/{filename}",
    status_code=status.HTTP_200_OK,
    summary="Delete an indexed document with complete filesystem cleanup",
    description="Purges all vectors from ChromaDB, deletes the uploaded source PDF, and removes all extracted image assets from disk.",
    dependencies=[Depends(verify_api_key)],
)
async def delete_document(
    filename: str,
    request: Request,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> dict:
    """Cascade deletion of document from vector index, document registry, and disk storage."""
    req_id = getattr(request.state, "request_id", "unknown")
    try:
        deleted_count = ingestion_service.delete_document_full(filename)
        if deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{filename}' not found in the vector store",
            )
        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} vectors and purged disk assets for '{filename}'",
            "deleted_vectors": deleted_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete document %s [Request: %s]: %s", filename, req_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}",
        )
