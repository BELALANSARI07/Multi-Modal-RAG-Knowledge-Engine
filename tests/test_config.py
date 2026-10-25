"""Unit tests for configuration validation and SecretStr credential protection."""

import pytest
from pydantic import ValidationError

from app.config import Settings


class TestConfigurationValidation:
    """Test suite verifying strict validation rules in Settings."""

    def test_default_settings_valid(self) -> None:
        """Verify default settings construct cleanly without validation error."""
        s = Settings()
        assert s.app_name == "MultiModal-RAG-Engine"
        assert s.debug is False
        assert s.chunk_overlap < s.chunk_size
        assert s.embedding_dimension == 384
        assert s.rate_limit_requests_per_minute > 0

    def test_secret_str_masks_credentials(self) -> None:
        """Verify API keys are masked in str and repr, and accessible via properties."""
        s = Settings(
            api_key="super-secret-auth-key",
            gemini_api_key="gemini-secret-12345",
            openai_api_key="openai-secret-67890",
        )
        # Representation and str must mask credentials with asterisks
        repr_str = repr(s)
        assert "super-secret-auth-key" not in repr_str
        assert "gemini-secret-12345" not in repr_str
        assert "openai-secret-67890" not in repr_str
        assert "**********" in repr_str

        # Unmasked value is accessible via property
        assert s.api_key_str == "super-secret-auth-key"
        assert s.gemini_api_key_str == "gemini-secret-12345"
        assert s.openai_api_key_str == "openai-secret-67890"

    def test_invalid_rate_limit_rejected(self) -> None:
        """Verify non-positive rate limits are rejected."""
        with pytest.raises(ValidationError, match="rate_limit_requests_per_minute"):
            Settings(rate_limit_requests_per_minute=0)

        with pytest.raises(ValidationError, match="rate_limit_requests_per_minute"):
            Settings(rate_limit_requests_per_minute=-10)

    def test_invalid_embedding_dimensions_rejected(self) -> None:
        """Verify non-positive embedding dimensions are rejected."""
        with pytest.raises(ValidationError, match="embedding_dimension"):
            Settings(embedding_dimension=0)

        with pytest.raises(ValidationError, match="embedding_batch_size"):
            Settings(embedding_batch_size=-1)

    def test_invalid_chunk_size_and_overlap_rejected(self) -> None:
        """Verify chunk size <= 0, overlap < 0, or overlap >= chunk_size are rejected."""
        with pytest.raises(ValidationError, match="chunk_size"):
            Settings(chunk_size=0)

        with pytest.raises(ValidationError, match="chunk_overlap"):
            Settings(chunk_overlap=-5)

        # Overlap equal to chunk size
        with pytest.raises(ValidationError, match="must be strictly less"):
            Settings(chunk_size=100, chunk_overlap=100)

        # Overlap greater than chunk size
        with pytest.raises(ValidationError, match="must be strictly less"):
            Settings(chunk_size=100, chunk_overlap=150)

    def test_invalid_max_file_size_rejected(self) -> None:
        """Verify non-positive maximum file size is rejected."""
        with pytest.raises(ValidationError, match="max_file_size_mb"):
            Settings(max_file_size_mb=0)

    def test_invalid_vlm_provider_rejected(self) -> None:
        """Verify unknown VLM provider names are rejected."""
        with pytest.raises(ValidationError, match="Invalid VLM_PROVIDER"):
            Settings(vlm_provider="anthropic_claude")

    def test_invalid_log_format_rejected(self) -> None:
        """Verify unknown log formats are rejected."""
        with pytest.raises(ValidationError, match="Invalid LOG_FORMAT"):
            Settings(log_format="yaml")

    def test_invalid_duplicate_strategy_rejected(self) -> None:
        """Verify invalid duplicate strategy names are rejected."""
        with pytest.raises(ValidationError, match="Invalid DUPLICATE_STRATEGY"):
            Settings(duplicate_strategy="overwrite_all")

    def test_empty_cors_origins_rejected(self) -> None:
        """Verify empty or whitespace-only CORS lists are rejected."""
        with pytest.raises(ValidationError, match="cors_origins"):
            Settings(cors_origins=[])

        with pytest.raises(ValidationError, match="cors_origins"):
            Settings(cors_origins=["   ", ""])

    def test_invalid_rate_limit_backend_rejected(self) -> None:
        """Verify unknown rate limit backend is rejected."""
        with pytest.raises(ValidationError, match="Invalid RATE_LIMIT_BACKEND"):
            Settings(rate_limit_backend="dynamodb")
