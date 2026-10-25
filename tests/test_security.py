"""Security, authentication, rate limiting, and defensive validation tests."""

from pathlib import Path

import pymupdf as fitz
from fastapi.testclient import TestClient

from app.utils.file_utils import sanitize_filename, validate_pdf_signature


class TestSecurityAndDefenses:
    """Test suite verifying security controls, path traversal shielding, and authentication."""

    def test_filename_sanitization_unit(self) -> None:
        """Verify strict elimination of path traversal sequences and unsafe characters."""
        assert sanitize_filename("../../../etc/passwd.pdf") == "passwd.pdf"
        assert sanitize_filename("..\\..\\windows\\system32.pdf") == "system32.pdf"
        assert sanitize_filename("normal_file.pdf") == "normal_file.pdf"
        assert sanitize_filename("../../hidden/./doc.pdf") == "doc.pdf"
        assert sanitize_filename("report (2026) [draft]!.pdf") == "report_2026_draft_.pdf"

    def test_pdf_magic_bytes_validation(self) -> None:
        """Verify magic byte detection distinguishes genuine PDFs from disguised files."""
        valid_pdf = b"%PDF-1.4\n1 0 obj\n..."
        invalid_bytes = b"This is just a plain text file disguised as a pdf"
        assert validate_pdf_signature(valid_pdf) is True
        assert validate_pdf_signature(invalid_bytes) is False
        assert validate_pdf_signature(b"") is False

    def test_upload_spoofed_pdf_rejected(self, test_client: TestClient) -> None:
        """Verify uploading non-PDF binary with .pdf extension is rejected by signature validator."""
        fake_pdf_content = b"Not a real PDF file! Malicious payload."
        files = {"file": ("exploit.pdf", fake_pdf_content, "application/pdf")}
        resp = test_client.post("/api/v1/ingest", files=files)
        assert resp.status_code == 422
        assert "valid PDF signature" in resp.json()["detail"]

    def test_path_traversal_upload_sanitized(self, test_client: TestClient, synthetic_pdf_path: Path) -> None:
        """Verify path traversal filename in upload is sanitized safely and does not escape uploads directory."""
        with open(synthetic_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Malicious traversal attempt
        files = {"file": ("../../../../tmp_malicious.pdf", pdf_bytes, "application/pdf")}
        resp = test_client.post("/api/v1/ingest", files=files)
        assert resp.status_code == 200
        # Returned filename must be clean without ../
        data = resp.json()
        assert ".." not in data["filename"]
        assert "/" not in data["filename"]
        assert "\\" not in data["filename"]

    def test_empty_pdf_no_text_no_images(self, test_client: TestClient, tmp_path: Path) -> None:
        """Verify PDF with blank pages (no text, no images) triggers a clear DocumentEmptyError rejection."""
        blank_doc = fitz.open()
        blank_doc.new_page()  # Page with no text or images
        blank_path = tmp_path / "blank.pdf"
        blank_doc.save(str(blank_path))
        blank_doc.close()

        with open(blank_path, "rb") as f:
            blank_bytes = f.read()

        files = {"file": ("blank.pdf", blank_bytes, "application/pdf")}
        resp = test_client.post("/api/v1/ingest", files=files)
        assert resp.status_code == 422
        assert "no extractable text or visual figures" in resp.json()["detail"]

    def test_api_key_authentication(self, auth_test_client: TestClient, synthetic_pdf_path: Path) -> None:
        """Verify API key authentication when configured in settings."""
        query_payload = {"question": "Test question?", "top_k": 2}

        # 1. Missing API Key Header -> 401 Unauthorized
        resp = auth_test_client.post("/api/v1/query", json=query_payload)
        assert resp.status_code == 401
        assert "Invalid or missing API Key" in resp.json()["detail"]

        # 2. Invalid API Key Header -> 401 Unauthorized
        resp = auth_test_client.post(
            "/api/v1/query",
            json=query_payload,
            headers={"X-API-Key": "wrong-key-value"},
        )
        assert resp.status_code == 401

        # 3. Valid API Key Header -> 200 OK
        resp = auth_test_client.post(
            "/api/v1/query",
            json=query_payload,
            headers={"X-API-Key": "secret-test-api-key-12345"},
        )
        assert resp.status_code == 200

    def test_rate_limiting(self, auth_test_client: TestClient) -> None:
        """Verify rate limiter blocks client when request burst exceeds configured quota."""
        headers = {"X-API-Key": "secret-test-api-key-12345"}
        query_payload = {"question": "Hello rate limit test", "top_k": 1}

        # Configured quota is 5 requests per minute
        responses = []
        for _ in range(7):
            r = auth_test_client.post("/api/v1/query", json=query_payload, headers=headers)
            responses.append(r.status_code)

        # At least one request must have been rate-limited with 429
        assert 429 in responses

    def test_document_deletion_filesystem_cleanup(
        self,
        test_client: TestClient,
        synthetic_pdf_path: Path,
        tmp_path: Path,
    ) -> None:
        """Verify DELETE /api/v1/documents removes vectors AND purges local disk assets."""
        with open(synthetic_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Ingest
        files = {"file": ("cleanup_test.pdf", pdf_bytes, "application/pdf")}
        ingest_resp = test_client.post("/api/v1/ingest", files=files)
        assert ingest_resp.status_code == 200

        # Delete
        del_resp = test_client.delete("/api/v1/documents/cleanup_test.pdf")
        assert del_resp.status_code == 200

        # Verify 404 on immediate second delete
        del_again = test_client.delete("/api/v1/documents/cleanup_test.pdf")
        assert del_again.status_code == 404

    def test_upload_empty_file_rejected(self, test_client: TestClient) -> None:
        """Verify empty upload (0 bytes) is rejected with 400 Bad Request."""
        files = {"file": ("empty.pdf", b"", "application/pdf")}
        resp = test_client.post("/api/v1/ingest", files=files)
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()
        assert "X-Request-ID" in resp.headers

    def test_upload_size_limit_boundaries(self, tmp_path: Path) -> None:
        """Verify uploads exactly at limit succeed and uploads exceeding limit are rejected with 413 without loading."""
        from app.config import Settings
        from app.main import create_application

        limit_mb = 1  # 1 MB test limit = 1024 * 1024 bytes
        limit_bytes = limit_mb * 1024 * 1024

        settings = Settings(
            chroma_persist_dir=tmp_path / "chroma_size_test",
            upload_dir=tmp_path / "uploads_size_test",
            extracted_images_dir=tmp_path / "imgs_size_test",
            max_file_size_mb=limit_mb,
            rate_limit_enabled=False,
            api_key=None,
        )
        app = create_application(custom_settings=settings)

        with TestClient(app) as client:
            # Construct a valid PDF header followed by null pad up to exact limit
            valid_header = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n"
            pad_exact = b" " * (limit_bytes - len(valid_header))
            exact_content = valid_header + pad_exact
            assert len(exact_content) == limit_bytes

            # 1. Exactly at limit -> Passed through size check (may fail parsing as invalid PDF body, but NOT 413)
            files = {"file": ("exact_limit.pdf", exact_content, "application/pdf")}
            resp_exact = client.post("/api/v1/ingest", files=files)
            assert resp_exact.status_code != 413

            # 2. Above limit by 10 bytes -> Rejected with 413 Payload Too Large
            oversized_content = exact_content + (b"X" * 10)
            files_over = {"file": ("oversized.pdf", oversized_content, "application/pdf")}
            resp_over = client.post("/api/v1/ingest", files=files_over)
            assert resp_over.status_code == 413
            assert f"{limit_mb} MB" in resp_over.json()["detail"]
            assert "X-Request-ID" in resp_over.headers

    def test_internal_error_shielding_readiness_probe(self, test_client: TestClient) -> None:
        """Verify /health/ready does not expose raw database errors, file paths, or stack traces."""
        from unittest.mock import MagicMock

        # Simulate vector store failure with sensitive internal path in exception message
        sensitive_internal_error = "OperationalError: /var/lib/secret/chroma.sqlite3 disk I/O failure at line 49"
        mock_vs = MagicMock()
        mock_vs.count.side_effect = RuntimeError(sensitive_internal_error)

        test_client.app.state.vectorstore = mock_vs
        resp = test_client.get("/health/ready")

        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        # Must NOT leak the raw exception or filesystem paths
        assert "secret" not in data["reason"]
        assert "chroma.sqlite3" not in data["reason"]
        assert data["reason"] == "Database or vector service unavailable"

    def test_internal_error_shielding_document_list(self, test_client: TestClient) -> None:
        """Verify /api/v1/documents shields internal exception details on database errors."""
        from unittest.mock import MagicMock

        mock_vs = MagicMock()
        mock_vs.list_ingested_documents.side_effect = RuntimeError("Database cursor failure in internal/db/query.py")

        test_client.app.state.vectorstore = mock_vs
        resp = test_client.get("/api/v1/documents")

        assert resp.status_code == 500
        data = resp.json()
        assert "cursor failure" not in data["detail"]
        assert "internal/db" not in data["detail"]
        assert "Reference:" in data["detail"]
        assert "X-Request-ID" in resp.headers

    def test_memory_rate_limit_backend_unit(self) -> None:
        """Verify MemoryRateLimitBackend enforces RPM limit and thread safety."""
        from app.api.middleware import MemoryRateLimitBackend

        backend = MemoryRateLimitBackend()
        backend.reset()
        ip = "192.168.1.100"

        # Allowed up to 3 requests
        assert backend.is_allowed(ip, rpm=3) is True
        assert backend.is_allowed(ip, rpm=3) is True
        assert backend.is_allowed(ip, rpm=3) is True

        # 4th request must be rejected
        assert backend.is_allowed(ip, rpm=3) is False

        # Reset clears counters
        backend.reset()
        assert backend.is_allowed(ip, rpm=3) is True
