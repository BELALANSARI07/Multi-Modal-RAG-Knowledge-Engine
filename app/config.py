"""Application configuration module using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator, model_validator
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
    api_key: SecretStr | None = None  # If set, enforces X-API-Key on mutating/query endpoints
    api_key_header_name: str = "X-API-Key"
    cors_origins: list[str] | str = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    # --- API Rate Limiting ---
    rate_limit_enabled: bool = True
    rate_limit_backend: str = "memory"  # "memory" or "redis"
    redis_url: str | None = None
    rate_limit_requests_per_minute: int = 60

    # --- VLM / LLM API ---
    vlm_provider: str = "gemini"  # "gemini" or "openai"
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
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

    @property
    def api_key_str(self) -> str | None:
        """Return the unmasked API key string if configured."""
        if self.api_key is None:
            return None
        return self.api_key.get_secret_value() if isinstance(self.api_key, SecretStr) else str(self.api_key)

    @property
    def gemini_api_key_str(self) -> str | None:
        """Return the unmasked Gemini API key string if configured."""
        if self.gemini_api_key is None:
            return None
        return (
            self.gemini_api_key.get_secret_value()
            if isinstance(self.gemini_api_key, SecretStr)
            else str(self.gemini_api_key)
        )

    @property
    def openai_api_key_str(self) -> str | None:
        """Return the unmasked OpenAI API key string if configured."""
        if self.openai_api_key is None:
            return None
        return (
            self.openai_api_key.get_secret_value()
            if isinstance(self.openai_api_key, SecretStr)
            else str(self.openai_api_key)
        )

    @field_validator("vlm_provider")
    @classmethod
    def validate_vlm_provider(cls, v: str) -> str:
        valid = {"gemini", "openai"}
        cleaned = v.strip().lower()
        if cleaned not in valid:
            raise ValueError(f"Invalid VLM_PROVIDER '{v}'. Must be one of {valid}")
        return cleaned

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        valid = {"text", "json"}
        cleaned = v.strip().lower()
        if cleaned not in valid:
            raise ValueError(f"Invalid LOG_FORMAT '{v}'. Must be one of {valid}")
        return cleaned

    @field_validator("rate_limit_backend")
    @classmethod
    def validate_rate_limit_backend(cls, v: str) -> str:
        valid = {"memory", "redis"}
        cleaned = v.strip().lower()
        if cleaned not in valid:
            raise ValueError(f"Invalid RATE_LIMIT_BACKEND '{v}'. Must be one of {valid}")
        return cleaned

    @field_validator("duplicate_strategy")
    @classmethod
    def validate_duplicate_strategy(cls, v: str) -> str:
        valid = {"replace", "skip"}
        cleaned = v.strip().lower()
        if cleaned not in valid:
            raise ValueError(f"Invalid DUPLICATE_STRATEGY '{v}'. Must be one of {valid}")
        return cleaned

    @field_validator(
        "rate_limit_requests_per_minute",
        "embedding_batch_size",
        "embedding_dimension",
        "chunk_size",
        "max_file_size_mb",
    )
    @classmethod
    def validate_positive_int(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer > 0 (got {v})")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_non_negative_overlap(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"chunk_overlap must be non-negative >= 0 (got {v})")
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors_origins(cls, v: object) -> list[str]:
        if isinstance(v, str):
            v_str = v.strip()
            if not v_str:
                raise ValueError("cors_origins must not be empty")
            if v_str.startswith("[") and v_str.endswith("]"):
                import json
                try:
                    parsed = json.loads(v_str)
                    if isinstance(parsed, list):
                        v = parsed
                except Exception:
                    pass
            if isinstance(v, str):
                v = [o.strip() for o in v_str.split(",") if o.strip()]

        if not isinstance(v, list) or not v:
            raise ValueError("cors_origins must not be empty")
        cleaned = [str(o).strip() for o in v if str(o).strip()]
        if not cleaned:
            raise ValueError("cors_origins must contain at least one non-empty origin string")
        return cleaned

    @model_validator(mode="after")
    def validate_overlap_less_than_chunk_size(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
