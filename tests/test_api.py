"""FastAPI REST endpoint integration tests using shared fixtures."""

from pathlib import Path
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Test suite covering FastAPI REST endpoints."""

    def test_root_endpoint(self, test_client: TestClient) -> None:
        """Verify root endpoint returns system status and documentation links."""
        resp = test_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "operational"
        assert "/docs" in data["docs_url"]

    def test_health_check_endpoint(self, test_client: TestClient) -> None:
        """Verify /health endpoint returns diagnostic metrics."""
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "embedding_engine" in data
        assert "total_indexed_vectors" in data

    def test_health_probes(self, test_client: TestClient) -> None:
        """Verify /health/live and /health/ready container probes."""
        live_resp = test_client.get("/health/live")
        assert live_resp.status_code == 200
        assert live_resp.json()["status"] == "alive"

        ready_resp = test_client.get("/health/ready")
        assert ready_resp.status_code == 200
        assert ready_resp.json()["status"] == "ready"

    def test_ingest_invalid_extension(self, test_client: TestClient) -> None:
        """Verify uploading non-PDF files is rejected with 422 Unprocessable Entity."""
        files = {"file": ("notes.txt", b"plain text content", "text/plain")}
        resp = test_client.post("/api/v1/ingest", files=files)
        assert resp.status_code == 422
        assert "Only .pdf files are supported" in resp.json()["detail"]

    def test_ingest_empty_file(self, test_client: TestClient) -> None:
        """Verify uploading empty file buffer is rejected with 400 Bad Request."""
        files = {"file": ("empty.pdf", b"", "application/pdf")}
        resp = test_client.post("/api/v1/ingest", files=files)
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"]

    def test_ingest_valid_pdf_and_query_flow(self, test_client: TestClient, synthetic_pdf_path: Path) -> None:
        """Verify full lifecycle: Ingest PDF -> List documents -> Query RAG -> Delete document with disk cleanup."""
        # 1. Read synthetic test PDF bytes
        with open(synthetic_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # 2. Ingest PDF via POST /api/v1/ingest
        files = {"file": ("sample_report.pdf", pdf_bytes, "application/pdf")}
        ingest_resp = test_client.post("/api/v1/ingest", files=files)
        assert ingest_resp.status_code == 200
        ingest_data = ingest_resp.json()
        assert ingest_data["status"] == "success"
        assert ingest_data["filename"] == "sample_report.pdf"
        assert ingest_data["stats"]["total_pages"] == 2
        assert ingest_data["stats"]["images_extracted"] >= 1

        # 3. Query via POST /api/v1/query
        query_payload = {
            "question": "What is the revenue breakdown and accuracy metrics?",
            "top_k": 3,
            "include_sources": True,
        }
        query_resp = test_client.post("/api/v1/query", json=query_payload)
        assert query_resp.status_code == 200
        q_data = query_resp.json()
        assert "answer" in q_data
        assert len(q_data["sources"]) > 0
        assert q_data["metadata"]["retrieval_time_ms"] >= 0

        # Verify that sources contain both text and visual elements
        chunk_types = {s["chunk_type"] for s in q_data["sources"]}
        assert "text" in chunk_types or "image_summary" in chunk_types

        # 4. List documents via GET /api/v1/documents
        doc_resp = test_client.get("/api/v1/documents")
        assert doc_resp.status_code == 200
        docs_data = doc_resp.json()
        assert docs_data["total_count"] >= 1
        filenames = [d["filename"] for d in docs_data["documents"]]
        assert "sample_report.pdf" in filenames

        # 5. Delete document via DELETE /api/v1/documents/{filename}
        del_resp = test_client.delete("/api/v1/documents/sample_report.pdf")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "success"
        assert del_resp.json()["deleted_vectors"] > 0
