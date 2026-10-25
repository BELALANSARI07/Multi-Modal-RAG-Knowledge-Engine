"""Shared pytest fixtures and test environment setup."""

import io
from pathlib import Path

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_application


@pytest.fixture
def synthetic_pdf_path(tmp_path: Path) -> Path:
    """Fixture providing a multi-page PDF containing text and an embedded test chart."""
    doc = fitz.open()

    # Page 1: Text
    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "Executive Summary: Multi-Modal RAG Architecture Overview.\n"
        "This system processes complex PDFs with text and visual graphics.",
        fontsize=12,
    )

    # Page 2: Text + Image
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Financial Charts and Revenue Breakdown Page.", fontsize=12)

    img = Image.new("RGB", (200, 200), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    p2.insert_image(fitz.Rect(50, 100, 250, 300), stream=buf.getvalue())

    pdf_file = tmp_path / "sample_report.pdf"
    doc.save(str(pdf_file))
    doc.close()
    return pdf_file


@pytest.fixture
def test_client(tmp_path: Path) -> TestClient:
    """Fixture providing an isolated FastAPI TestClient with temporary databases and no auth required."""
    test_settings = Settings(
        debug=True,
        chroma_persist_dir=tmp_path / "chroma_db",
        chroma_collection_name="test_collection",
        upload_dir=tmp_path / "uploads",
        extracted_images_dir=tmp_path / "extracted_images",
        api_key=None,  # No auth by default in basic tests
        rate_limit_enabled=False,
        gemini_api_key=None,
    )

    app = create_application(custom_settings=test_settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_test_client(tmp_path: Path) -> TestClient:
    """Fixture providing an isolated FastAPI TestClient with API Key authentication enabled."""
    test_settings = Settings(
        debug=True,
        chroma_persist_dir=tmp_path / "chroma_auth_db",
        chroma_collection_name="test_auth_collection",
        upload_dir=tmp_path / "auth_uploads",
        extracted_images_dir=tmp_path / "auth_extracted_images",
        api_key="secret-test-api-key-12345",
        rate_limit_enabled=True,
        rate_limit_requests_per_minute=5,  # low limit for testing
        gemini_api_key=None,
    )

    app = create_application(custom_settings=test_settings)
    with TestClient(app) as client:
        yield client
