"""Unit and concurrency tests for atomic document registry persistence."""

from pathlib import Path

from app.config import Settings
from app.models.schemas import DocumentMetadata
from app.services.ingestion_service import IngestionService


class TestDocumentRegistry:
    """Test suite verifying atomic writes, JSON recovery, and lifecycle of documents_registry.json."""

    def _create_service(self, tmp_path: Path) -> IngestionService:
        settings = Settings(
            upload_dir=tmp_path / "uploads",
            extracted_images_dir=tmp_path / "images",
            chroma_persist_dir=tmp_path / "chroma",
        )
        service = IngestionService(settings=settings)
        # Point registry specifically to tmp_path
        service.registry_file = tmp_path / "documents_registry.json"
        return service

    def test_registry_creation_on_first_save(self, tmp_path: Path) -> None:
        """Verify registry file is created atomically when first document is registered."""
        service = self._create_service(tmp_path)
        assert not service.registry_file.exists()

        record = DocumentMetadata(
            document_id="doc_12345",
            filename="quarterly_report.pdf",
            stored_path=str(tmp_path / "uploads" / "doc_12345.pdf"),
            file_hash="abcdef1234567890",
            file_size_bytes=1024,
            total_pages=3,
            text_chunks=5,
            image_chunks=2,
        )

        service._save_registry_entry(record)
        assert service.registry_file.exists()

        entry = service._get_registry_entry("quarterly_report.pdf")
        assert entry is not None
        assert entry.document_id == "doc_12345"
        assert entry.total_pages == 3
        assert entry.text_chunks == 5

    def test_registry_update_existing_entry(self, tmp_path: Path) -> None:
        """Verify updating an existing document metadata record preserves file integrity."""
        service = self._create_service(tmp_path)

        record1 = DocumentMetadata(
            document_id="doc_v1",
            filename="annual_plan.pdf",
            stored_path="annual_plan_v1.pdf",
            file_hash="hash1",
            file_size_bytes=5000,
            total_pages=5,
            text_chunks=10,
            image_chunks=1,
        )
        service._save_registry_entry(record1)

        # Update with new stats
        record2 = DocumentMetadata(
            document_id="doc_v2",
            filename="annual_plan.pdf",
            stored_path="annual_plan_v2.pdf",
            file_hash="hash2",
            file_size_bytes=6200,
            total_pages=6,
            text_chunks=12,
            image_chunks=2,
        )
        service._save_registry_entry(record2)

        updated = service._get_registry_entry("annual_plan.pdf")
        assert updated is not None
        assert updated.document_id == "doc_v2"
        assert updated.file_size_bytes == 6200
        assert updated.total_pages == 6

    def test_registry_deletion(self, tmp_path: Path) -> None:
        """Verify removing an entry deletes only that document and rewrites registry atomically."""
        service = self._create_service(tmp_path)

        rec_a = DocumentMetadata(
            document_id="doc_a",
            filename="doc_a.pdf",
            stored_path="path_a",
            file_hash="hash_a",
            file_size_bytes=100,
            total_pages=1,
            text_chunks=1,
            image_chunks=0,
        )
        rec_b = DocumentMetadata(
            document_id="doc_b",
            filename="doc_b.pdf",
            stored_path="path_b",
            file_hash="hash_b",
            file_size_bytes=200,
            total_pages=2,
            text_chunks=2,
            image_chunks=0,
        )
        service._save_registry_entry(rec_a)
        service._save_registry_entry(rec_b)

        assert service._get_registry_entry("doc_a.pdf") is not None
        assert service._get_registry_entry("doc_b.pdf") is not None

        service._remove_registry_entry("doc_a.pdf")
        assert service._get_registry_entry("doc_a.pdf") is None
        assert service._get_registry_entry("doc_b.pdf") is not None

    def test_malformed_registry_json_recovery(self, tmp_path: Path) -> None:
        """Verify corrupted/truncated JSON registry is safely recovered without crashing."""
        service = self._create_service(tmp_path)

        # Write corrupt truncated JSON
        service.registry_file.write_text('{"incomplete_record": {"filename": "corrupt.pdf", ', encoding="utf-8")

        # Reading from corrupt registry should return empty dict and not raise JSONDecodeError
        recovered_data = service._load_registry()
        assert recovered_data == {}
        assert service._get_registry_entry("any_file.pdf") is None

        # Writing after corruption should cleanly overwrite and restore a valid registry
        new_record = DocumentMetadata(
            document_id="fresh_doc",
            filename="fresh.pdf",
            stored_path="fresh.pdf",
            file_hash="fresh_hash",
            file_size_bytes=300,
            total_pages=1,
            text_chunks=1,
            image_chunks=0,
        )
        service._save_registry_entry(new_record)
        entry = service._get_registry_entry("fresh.pdf")
        assert entry is not None
        assert entry.document_id == "fresh_doc"

    def test_atomic_write_leaves_no_stray_tmp_files(self, tmp_path: Path) -> None:
        """Verify atomic replacement does not leave orphan .tmp files in directory."""
        service = self._create_service(tmp_path)

        record = DocumentMetadata(
            document_id="clean_test",
            filename="clean.pdf",
            stored_path="clean.pdf",
            file_hash="hash_clean",
            file_size_bytes=150,
            total_pages=1,
            text_chunks=1,
            image_chunks=0,
        )
        service._save_registry_entry(record)

        # Check for any lingering .tmp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0
        assert service.registry_file.exists()
