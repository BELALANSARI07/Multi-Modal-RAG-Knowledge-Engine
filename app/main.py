"""FastAPI application entrypoint with lifespan management, CORS, middleware, and health probes."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.middleware import RequestIDMiddleware, create_rate_limiter
from app.api.router import api_router
from app.config import Settings, get_settings
from app.models.schemas import HealthResponse
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService
from app.services.vectorstore_service import VectorStoreService
from app.services.vlm_service import VLMService
from app.utils.file_utils import ensure_directory
from app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown hooks."""
    app_settings = getattr(app.state, "settings", None) or get_settings()

    # 1. Initialize structured logging
    setup_logging(app_settings.log_level, log_format=app_settings.log_format)
    logger.info("Starting up %s v%s...", app_settings.app_name, app_settings.app_version)

    # 2. Ensure application directories exist
    ensure_directory(app_settings.upload_dir)
    ensure_directory(app_settings.extracted_images_dir)
    ensure_directory(app_settings.chroma_persist_dir)

    # 3. Warm up services and store singletons on app.state
    vectorstore = VectorStoreService(app_settings)
    _ = vectorstore.count()

    embedder = EmbeddingService(app_settings)
    _ = embedder.embedding_dimension

    vlm = VLMService(app_settings)

    ingestion_service = IngestionService(
        settings=app_settings,
        embedding_service=embedder,
        vlm_service=vlm,
        vectorstore_service=vectorstore,
    )

    retrieval_service = RetrievalService(
        settings=app_settings,
        embedding_service=embedder,
        vectorstore_service=vectorstore,
    )

    rate_limiter = create_rate_limiter(app_settings)

    app.state.settings = app_settings
    app.state.vectorstore = vectorstore
    app.state.embedder = embedder
    app.state.vlm = vlm
    app.state.ingestion_service = ingestion_service
    app.state.retrieval_service = retrieval_service
    app.state.rate_limiter = rate_limiter

    logger.info("Application warmup complete. Serving requests.")
    yield

    # Shutdown logic
    logger.info("Shutting down %s...", app_settings.app_name)
    rate_limiter = getattr(app.state, "rate_limiter", None)
    if rate_limiter and hasattr(rate_limiter, "close"):
        rate_limiter.close()


def create_application(custom_settings: Settings = None) -> FastAPI:
    """Application factory configuring middleware, routes, and static file mounts."""
    app_settings = custom_settings or get_settings()

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description=(
            "Production-Grade Multi-Modal Retrieval-Augmented Generation (RAG) Knowledge Engine.\n\n"
            "Ingests complex PDFs with text and visual graphics (charts, graphs, diagrams), "
            "synthesizes dense semantic summaries using Vision-Language Models, and provides "
            "unified vector retrieval and answer generation."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.state.settings = app_settings

    # 1. Correlation ID Middleware
    app.add_middleware(RequestIDMiddleware)

    # 2. CORS Middleware with restricted trusted origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # 3. Mount static folder for extracted visual images
    ensure_directory(app_settings.extracted_images_dir)
    app.mount(
        "/static/images",
        StaticFiles(directory=str(app_settings.extracted_images_dir)),
        name="images",
    )

    # 4. Include API routes
    app.include_router(api_router)

    # 5. Global Exception Handler: Shields internal server traces
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", "unknown")
        logger.exception("Unhandled server exception [Request: %s]: %s", req_id, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": f"An unexpected internal error occurred. Reference: {req_id}",
                "request_id": req_id,
            },
            headers={"X-Request-ID": req_id},
        )

    # 6. Root Endpoint
    @app.get("/", tags=["System"])
    async def root():
        return {
            "name": app_settings.app_name,
            "version": app_settings.app_version,
            "docs_url": "/docs",
            "health_url": "/health",
            "status": "operational",
        }

    # 7. Diagnostic Health Probes
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        vectorstore = getattr(app.state, "vectorstore", None) or VectorStoreService(app_settings)
        embedder = getattr(app.state, "embedder", None) or EmbeddingService(app_settings)
        vlm = getattr(app.state, "vlm", None) or VLMService(app_settings)

        try:
            total_vecs = vectorstore.count()
            chroma_ok = True
        except Exception:
            total_vecs = 0
            chroma_ok = False

        return HealthResponse(
            status="healthy" if chroma_ok else "degraded",
            app_name=app_settings.app_name,
            version=app_settings.app_version,
            chroma_connected=chroma_ok,
            total_indexed_vectors=total_vecs,
            embedding_engine=embedder.engine_type,
            vlm_provider=app_settings.vlm_provider,
            vlm_live=vlm.is_live_configured,
        )

    @app.get("/health/live", tags=["System"])
    async def liveness_probe():
        """Kubernetes/Container liveness probe."""
        return {"status": "alive"}

    @app.get("/health/ready", tags=["System"])
    async def readiness_probe():
        """Kubernetes/Container readiness probe verifying ChromaDB and embedding engine."""
        vectorstore = getattr(app.state, "vectorstore", None) or VectorStoreService(app_settings)
        try:
            _ = vectorstore.count()
            return {"status": "ready"}
        except Exception as e:
            logger.error("Readiness probe failed: %s", e)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": "Database or vector service unavailable"},
            )

    return app


app = create_application()
