"""Unit and integration tests for ChromaDB VectorStoreService."""

from pathlib import Path
import pytest

from app.config import Settings
from app.core.exceptions import VectorStoreError
from app.services.vectorstore_service import VectorStoreService


@pytest.fixture
def temp_vectorstore(tmp_path: Path) -> VectorStoreService:
    """Fixture providing an isolated ChromaDB instance in a temp directory."""
    settings = Settings(
        chroma_persist_dir=tmp_path / "chroma_test_db",
        chroma_collection_name="test_knowledge_base",
        embedding_dimension=4,
    )
    return VectorStoreService(settings=settings)


class TestVectorStoreService:
    """Test suite for ChromaDB vector operations, metadata filtering, and deletion."""

    def test_initial_count_is_zero(self, temp_vectorstore: VectorStoreService) -> None:
        """Verify new collection starts empty."""
        assert temp_vectorstore.count() == 0

    def test_upsert_and_count(self, temp_vectorstore: VectorStoreService) -> None:
        """Verify inserting documents correctly indexes records."""
        ids = ["doc1_c0", "doc1_c1", "doc1_img0"]
        docs = [
            "Introduction to deep learning and multi-modal transformers.",
            "Retrieval augmented generation architecture and indexing.",
            "Bar chart illustrating quarterly performance metrics and revenue.",
        ]
        metadatas = [
            {"source_pdf": "doc1.pdf", "page_number": 1, "chunk_type": "text", "chunk_index": 0},
            {"source_pdf": "doc1.pdf", "page_number": 1, "chunk_type": "text", "chunk_index": 1},
            {
                "source_pdf": "doc1.pdf",
                "page_number": 2,
                "chunk_type": "image_summary",
                "image_path": "/data/doc1_img0.png",
                "chunk_index": 0,
            },
        ]
        # 4-dimensional mock embeddings
        embeddings = [
            [0.1, 0.2, 0.8, 0.1],
            [0.1, 0.3, 0.7, 0.2],
            [0.9, 0.1, 0.1, 0.2],
        ]

        count = temp_vectorstore.upsert_documents(ids, docs, metadatas, embeddings)
        assert count == 3
        assert temp_vectorstore.count() == 3

    def test_query_and_similarity(self, temp_vectorstore: VectorStoreService) -> None:
        """Verify similarity query retrieves the nearest vectors."""
        ids = ["text_chunk", "visual_chunk"]
        docs = ["Text about machine learning.", "Chart showing financial growth."]
        metadatas = [
            {"source_pdf": "a.pdf", "page_number": 1, "chunk_type": "text"},
            {"source_pdf": "a.pdf", "page_number": 2, "chunk_type": "image_summary"},
        ]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        temp_vectorstore.upsert_documents(ids, docs, metadatas, embeddings)

        # Query close to visual_chunk: [0.1, 0.9, 0.0, 0.0]
        results = temp_vectorstore.query(
            query_embeddings=[[0.1, 0.9, 0.0, 0.0]],
            n_results=1,
        )

        assert len(results["ids"][0]) == 1
        assert results["ids"][0][0] == "visual_chunk"
        assert results["metadatas"][0][0]["chunk_type"] == "image_summary"

    def test_metadata_filtering(self, temp_vectorstore: VectorStoreService) -> None:
        """Verify queries with 'where' metadata filters only return matching chunk types."""
        ids = ["c1", "c2", "img1"]
        docs = ["Text 1", "Text 2", "Chart description"]
        metadatas = [
            {"source_pdf": "b.pdf", "chunk_type": "text"},
            {"source_pdf": "b.pdf", "chunk_type": "text"},
            {"source_pdf": "b.pdf", "chunk_type": "image_summary"},
        ]
        embeddings = [[0.5, 0.5, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]]
        temp_vectorstore.upsert_documents(ids, docs, metadatas, embeddings)

        # Filter strictly for image_summary chunks
        results = temp_vectorstore.query(
            query_embeddings=[[0.5, 0.5, 0.0, 0.0]],
            n_results=3,
            where={"chunk_type": "image_summary"},
        )

        retrieved_ids = results["ids"][0]
        assert len(retrieved_ids) == 1
        assert retrieved_ids[0] == "img1"

    def test_delete_by_source(self, temp_vectorstore: VectorStoreService) -> None:
        """Verify deleting by source_pdf purges only the target document's vectors."""
        ids = ["doc1_c1", "doc2_c1"]
        docs = ["Doc 1 text", "Doc 2 text"]
        metadatas = [{"source_pdf": "doc1.pdf"}, {"source_pdf": "doc2.pdf"}]
        embeddings = [[0.1, 0.2, 0.0, 0.0], [0.3, 0.4, 0.0, 0.0]]
        temp_vectorstore.upsert_documents(ids, docs, metadatas, embeddings)

        assert temp_vectorstore.count() == 2
        deleted = temp_vectorstore.delete_by_source("doc1.pdf")
        assert deleted == 1
        assert temp_vectorstore.count() == 1

        # Check remaining record
        rem = temp_vectorstore.list_ingested_documents()
        assert len(rem) == 1
        assert rem[0]["filename"] == "doc2.pdf"
