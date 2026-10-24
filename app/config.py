"""Application configuration module using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Multi-Modal RAG Knowledge Engine."""

    # --- Application Metadata ---
    app_name: str = "MultiModal-RAG-Engine"
    app_version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"

    # --- VLM / LLM API ---
    vlm_provider: str = "gemini"  # "gemini" or "openai"
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    vlm_model_name: str = "gemini-1.5-flash"
    vlm_max_concurrent: int = 5
    vlm_max_retries: int = 3

    # --- Embeddings ---
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    # --- Vector Database ---
    chroma_persist_dir: Path = Path("./chroma_db")
    chroma_collection_name: str = "knowledge_base"

    # --- Ingestion & Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_image_dimension: int = 100  # discard images with width or height below this threshold

    # --- Retrieval ---
    default_top_k: int = 5

    # --- Storage Directories ---
    upload_dir: Path = Path("./data/uploads")
    extracted_images_dir: Path = Path("./data/extracted_images")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
