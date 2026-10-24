"""FastAPI endpoint for multi-modal question answering and retrieval."""

import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies import get_retrieval_service
from app.api.middleware import check_rate_limit, verify_api_key
from app.core.exceptions import RAGEngineException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Retrieval & Generation"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question against the multi-modal knowledge base",
    description="Retrieves relevant text passages and visual summaries from ChromaDB, augments context, and synthesizes an answer.",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def query_knowledge_base(
    request: Request,
    query_payload: QueryRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> QueryResponse:
    """Execute multi-modal RAG search and answer synthesis."""
    req_id = getattr(request.state, "request_id", "unknown")
    try:
        response = await service.query(query_payload)
        return response
    except RAGEngineException as re:
        logger.error("Domain error during query retrieval [Request: %s]: %s", req_id, re)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing error: {re.message}",
        )
    except Exception as e:
        logger.exception("Unexpected error in query endpoint [Request: %s]: %s", req_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal retrieval error occurred. Reference: {req_id}",
        )
