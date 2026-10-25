"""Application configuration module using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Multi-Modal RAG Knowledge Engine."""

    # --- Application Metadata ---
    app_name: str = "MultiModal-RAG-Engine"
    app_version: str = "1.1.0"
    debug: bool = False  # Production default: False
    log_level: str = "INFO"
    log_format: str = "text"  # "text" or "json"

    # --- Security & Authentication ---
    api_key: str | None = None  # If set, enforces X-API-Key on mutating/query endpoints
    api_key_header_name: str = "X-API-Key"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    # --- API Rate Limiting ---
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60

    # --- VLM / LLM API ---
    vlm_provider: str = "gemini"  # "gemini" or "openai"
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    vlm_model_name: str = "gemini-1.5-flash"
    vlm_max_concurrent: int = 5
    vlm_max_retries: int = 3
    vlm_timeout_seconds: float = 30.0

    # --- Embeddings ---
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    embedding_dimension: int = 384

    # --- Vector Database ---
    chroma_persist_dir: Path = Path("./chroma_db")
    chroma_collection_name: str = "knowledge_base"

    # --- Ingestion & Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_image_dimension: int = 100  # discard images with width or height below 100px
    max_file_size_mb: int = 50
    duplicate_strategy: str = "replace"  # "replace" or "skip"

    # --- Retrieval & Guardrails ---
    default_top_k: int = 5
    min_relevance_score: float = 0.20  # filter out irrelevant chunks below 0.20
    max_context_chars: int = 8000  # context window guardrail

    # --- Storage Directories ---
    upload_dir: Path = Path("./data/uploads")
    extracted_images_dir: Path = Path("./data/extracted_images")

    @field_validator("vlm_provider")
    @classmethod
    def validate_vlm_provider(cls, v: str) -> str:
        valid = {"gemini", "openai"}
        cleaned = v.strip().lower()
        if cleaned not in valid:
            raise ValueError(f"Invalid VLM_PROVIDER '{v}'. Must be one of {valid}")
        return cleaned

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
