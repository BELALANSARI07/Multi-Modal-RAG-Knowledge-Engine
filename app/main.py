"""FastAPI application entrypoint with lifespan management, CORS, and static mounting."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.config import get_settings
from app.models.schemas import HealthResponse
from app.services.embedding_service import EmbeddingService
from app.services.vectorstore_service import VectorStoreService
from app.services.vlm_service import VLMService
from app.utils.file_utils import ensure_directory
from app.utils.logging_config import setup_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown hooks."""
    # 1. Initialize structured logging
    setup_logging(settings.log_level)

    # 2. Ensure application directories exist
    ensure_directory(settings.upload_dir)
    ensure_directory(settings.extracted_images_dir)
    ensure_directory(settings.chroma_persist_dir)

    # 3. Warm up services
    vectorstore = VectorStoreService(settings)
    _ = vectorstore.count()

    embedder = EmbeddingService(settings)
    _ = embedder.embedding_dimension

    vlm = VLMService(settings)

    # Store shared instances on app state
    app.state.vectorstore = vectorstore
    app.state.embedder = embedder
    app.state.vlm = vlm

    yield

    # Shutdown logic
    pass


def create_application() -> FastAPI:
    """Application factory configuring middleware, routes, and static file mounts."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
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

    # 1. Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Mount static folder for extracted visual images
    ensure_directory(settings.extracted_images_dir)
    app.mount(
        "/static/images",
        StaticFiles(directory=str(settings.extracted_images_dir)),
        name="images",
    )

    # 3. Include API routes
    app.include_router(api_router)

    # 4. Root Endpoint
    @app.get("/", tags=["System"])
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs_url": "/docs",
            "health_url": "/health",
            "status": "operational",
        }

    # 5. Diagnostic Health Check Endpoint
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        vectorstore = getattr(app.state, "vectorstore", None) or VectorStoreService(settings)
        embedder = getattr(app.state, "embedder", None) or EmbeddingService(settings)
        vlm = getattr(app.state, "vlm", None) or VLMService(settings)

        try:
            total_vecs = vectorstore.count()
            chroma_ok = True
        except Exception:
            total_vecs = 0
            chroma_ok = False

        return HealthResponse(
            status="healthy" if chroma_ok else "degraded",
            app_name=settings.app_name,
            version=settings.app_version,
            chroma_connected=chroma_ok,
            total_indexed_vectors=total_vecs,
            embedding_engine=embedder.engine_type,
            vlm_provider=settings.vlm_provider,
            vlm_live=vlm.is_live_configured,
        )

    return app


app = create_application()
