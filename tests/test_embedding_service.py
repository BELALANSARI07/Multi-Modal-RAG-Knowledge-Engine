"""Integration tests for EmbeddingService with sentence-transformers."""

import pytest

from app.config import Settings
from app.services.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test suite for Hugging Face SentenceTransformer embeddings."""

    @pytest.fixture(scope="class")
    def embedding_service(self) -> EmbeddingService:
        settings = Settings(
            embedding_model_name="all-MiniLM-L6-v2",
            embedding_batch_size=32,
        )
        return EmbeddingService(settings=settings)

    def test_single_embedding(self, embedding_service: EmbeddingService) -> None:
        """Verify embedding dimension and normalization for a single query."""
        vec = embedding_service.embed_single("Multi-modal RAG retrieval query")
        assert isinstance(vec, list)
        assert len(vec) == 384  # MiniLM dimension
        assert all(isinstance(x, float) for x in vec)

    def test_batch_embeddings(self, embedding_service: EmbeddingService) -> None:
        """Verify batch text embedding produces vectors matching input count."""
        texts = [
            "Revenue report Q3 bar chart",
            "High performance vector database indexing with ChromaDB",
            "Dense visual descriptions extracted via Vision-Language Models",
        ]
        vectors = embedding_service.embed_texts(texts)
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 384
