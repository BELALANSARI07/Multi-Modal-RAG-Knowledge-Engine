"""Integration tests for the end-to-end IngestionService pipeline."""

import asyncio
from pathlib import Path
import pytest

from app.config import Settings
from app.core.image_extractor import ImageExtractor
from app.core.pdf_parser import PDFParser
from app.core.text_chunker import RecursiveTextChunker
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.vectorstore_service import VectorStoreService
from app.services.vlm_service import VLMService
from tests.test_pdf_parser import create_synthetic_pdf


class MockEmbeddingService(EmbeddingService):
    """Fast deterministic embedding service mock for test suites."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts):
        # Deterministic normalized vectors based on string length and ascii
        vectors = []
        for t in texts:
            val = float(len(t) % 10) / 10.0
            vec = [val] * self._dim
            vectors.append(vec)
        return vectors

    def embed_single(self, text: str):
        return self.embed_texts([text])[0]


class TestIngestionPipeline:
    """Test suite for full multi-modal document ingestion and unified vector indexing."""

    def test_full_pipeline_ingestion(self, tmp_path: Path) -> None:
        """Verify PDF -> parsing -> image extraction -> VLM summary -> embedding -> ChromaDB storage."""
        # 1. Setup isolated directories and services
        pdf_path = create_synthetic_pdf(tmp_path)
        img_dir = tmp_path / "extracted_images"
        chroma_dir = tmp_path / "chroma_test"

        settings = Settings(
            chroma_persist_dir=chroma_dir,
            chroma_collection_name="test_unified_kb",
            extracted_images_dir=img_dir,
            gemini_api_key=None,  # triggers synthetic visual summary fallback
        )

        extractor = ImageExtractor(output_dir=img_dir, min_dimension=100)
        chunker = RecursiveTextChunker(chunk_size=200, chunk_overlap=30)
        parser = PDFParser(text_chunker=chunker, image_extractor=extractor)
        vlm = VLMService(settings=settings)
        embedder = MockEmbeddingService(dim=16)
        vectorstore = VectorStoreService(settings=settings)

        ingestion_service = IngestionService(
            settings=settings,
            pdf_parser=parser,
            vlm_service=vlm,
            embedding_service=embedder,
            vectorstore_service=vectorstore,
        )

        # 2. Execute ingestion
        result = asyncio.run(ingestion_service.ingest_file(pdf_path))

        # 3. Assertions on ingestion result
        assert result.status == "success"
        assert result.filename == "sample_report.pdf"
        assert result.stats.total_pages == 2
        assert result.stats.text_chunks >= 2
        assert result.stats.images_extracted == 1
        assert result.stats.images_summarized == 1

        # 4. Verify ChromaDB index state
        total_vectors = vectorstore.count()
        assert total_vectors == result.stats.text_chunks + result.stats.images_summarized

        # 5. Verify query retrieval for visual chunk
        visual_results = vectorstore.query(
            query_embeddings=[[0.5] * 16],
            n_results=1,
            where={"chunk_type": "image_summary"},
        )
        assert len(visual_results["ids"][0]) == 1
        meta = visual_results["metadatas"][0][0]
        assert meta["chunk_type"] == "image_summary"
        assert meta["source_pdf"] == "sample_report.pdf"
        assert Path(meta["image_path"]).exists()
