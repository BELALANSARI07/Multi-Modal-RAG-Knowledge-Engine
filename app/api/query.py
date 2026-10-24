"""FastAPI endpoint for multi-modal question answering and retrieval."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.exceptions import RAGEngineException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Retrieval & Generation"])


def get_retrieval_service() -> RetrievalService:
    """Dependency injection provider for RetrievalService."""
    return RetrievalService()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question against the multi-modal knowledge base",
    description="Retrieves relevant text passages and visual summaries from ChromaDB, augment context, and synthesizes an answer.",
)
async def query_knowledge_base(
    request: QueryRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> QueryResponse:
    """Execute multi-modal RAG search and answer synthesis."""
    try:
        response = await service.query(request)
        return response
    except RAGEngineException as re:
        logger.error("Domain error during query retrieval: %s", re)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {re.message}",
        )
    except Exception as e:
        logger.exception("Unexpected error in query endpoint: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal retrieval error: {str(e)}",
        )
