"""Centralized dependency injection providers reusing shared services stored in app.state."""

from fastapi import Request

from app.config import Settings, get_settings
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService
from app.services.vectorstore_service import VectorStoreService
from app.services.vlm_service import VLMService


def get_app_settings(request: Request) -> Settings:
    """Provide current application settings (from app.state if overridden, or default)."""
    return getattr(request.app.state, "settings", None) or get_settings()


def get_vectorstore_service(request: Request) -> VectorStoreService:
    """Provide shared VectorStoreService singleton from app.state."""
    service = getattr(request.app.state, "vectorstore", None)
    if service is None:
        settings = get_app_settings(request)
        service = VectorStoreService(settings)
        request.app.state.vectorstore = service
    return service


def get_embedding_service(request: Request) -> EmbeddingService:
    """Provide shared EmbeddingService singleton from app.state."""
    service = getattr(request.app.state, "embedder", None)
    if service is None:
        settings = get_app_settings(request)
        service = EmbeddingService(settings)
        request.app.state.embedder = service
    return service


def get_vlm_service(request: Request) -> VLMService:
    """Provide shared VLMService singleton from app.state."""
    service = getattr(request.app.state, "vlm", None)
    if service is None:
        settings = get_app_settings(request)
        service = VLMService(settings)
        request.app.state.vlm = service
    return service


def get_ingestion_service(request: Request) -> IngestionService:
    """Provide shared IngestionService orchestrator from app.state."""
    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        settings = get_app_settings(request)
        vectorstore = get_vectorstore_service(request)
        embedder = get_embedding_service(request)
        vlm = get_vlm_service(request)
        service = IngestionService(
            settings=settings,
            embedding_service=embedder,
            vlm_service=vlm,
            vectorstore_service=vectorstore,
        )
        request.app.state.ingestion_service = service
    return service


def get_retrieval_service(request: Request) -> RetrievalService:
    """Provide shared RetrievalService orchestrator from app.state."""
    service = getattr(request.app.state, "retrieval_service", None)
    if service is None:
        settings = get_app_settings(request)
        vectorstore = get_vectorstore_service(request)
        embedder = get_embedding_service(request)
        service = RetrievalService(
            settings=settings,
            embedding_service=embedder,
            vectorstore_service=vectorstore,
        )
        request.app.state.retrieval_service = service
    return service
