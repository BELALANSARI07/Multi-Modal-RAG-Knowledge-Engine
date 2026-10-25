"""Tests for EmbeddingService ONNX fallback, dimension verification, and lazy loading."""

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.core.exceptions import EmbeddingError
from app.services.embedding_service import EmbeddingService


class TestEmbeddingFallbackAndDimensions:
    """Test suite verifying fallback behavior from SentenceTransformers to ONNX, dimension checks, and lazy loading."""

    def test_lazy_loading_does_not_initialize_on_instantiation(self) -> None:
        """Verify model weights are not loaded into memory until first embedding operation."""
        service = EmbeddingService(settings=Settings(embedding_dimension=384))
        # Internal engine caches must remain uninitialized
        assert service._model is None
        assert service._onnx_fn is None

    def test_forced_sentence_transformers_failure_activates_onnx_fallback(self) -> None:
        """Verify that when SentenceTransformer fails (e.g. DLL block, missing torch), ONNX engine is selected."""
        settings = Settings(embedding_dimension=384)
        service = EmbeddingService(settings=settings)

        # Mock SentenceTransformer import to fail with simulated OS policy error
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Mock ChromaDB ONNX function returning 384-dimensional vector
            mock_onnx = MagicMock()
            mock_onnx.side_effect = lambda texts: [[0.05] * 384 for _ in texts]

            with patch("chromadb.utils.embedding_functions.DefaultEmbeddingFunction", return_value=mock_onnx):
                vec = service.embed_single("Test query requiring fallback")
                assert len(vec) == 384
                assert service.engine_type == "onnx"
                assert service.embedding_dimension == 384

    def test_dimension_mismatch_raises_domain_error(self) -> None:
        """Verify that if loaded model produces unexpected dimensions, EmbeddingError is raised."""
        # System configured for 384, but mock engine produces 512
        settings = Settings(embedding_dimension=384)
        service = EmbeddingService(settings=settings)

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            mock_onnx = MagicMock()
            # Returns 512 dimensions instead of 384
            mock_onnx.side_effect = lambda texts: [[0.1] * 512 for _ in texts]

            with patch("chromadb.utils.embedding_functions.DefaultEmbeddingFunction", return_value=mock_onnx):
                with pytest.raises(EmbeddingError, match="Embedding dimension mismatch"):
                    service._load_model()

    def test_both_engines_failing_raises_embedding_error(self) -> None:
        """Verify clear domain error when both primary and fallback engines fail."""
        service = EmbeddingService(settings=Settings(embedding_dimension=384))

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with patch("chromadb.utils.embedding_functions.DefaultEmbeddingFunction", side_effect=RuntimeError("ONNX DLL missing")):
                with pytest.raises(EmbeddingError, match="Both SentenceTransformer and ONNX embedding engines failed"):
                    service._load_model()
