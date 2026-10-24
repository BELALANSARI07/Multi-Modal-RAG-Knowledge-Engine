"""Central router aggregating all v1 API sub-routers."""

from fastapi import APIRouter

from app.api import documents, ingest, query

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(ingest.router)
api_router.include_router(query.router)
api_router.include_router(documents.router)
