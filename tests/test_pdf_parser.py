"""Integration tests for PDFParser and ImageExtractor using synthetic PDF documents."""

import asyncio
import io
from pathlib import Path

import pymupdf as fitz
import pytest
from PIL import Image

from app.core.exceptions import PDFParsingError
from app.core.image_extractor import ImageExtractor
from app.core.pdf_parser import PDFParser
from app.core.text_chunker import RecursiveTextChunker


def create_synthetic_pdf(tmp_path: Path) -> Path:
    """Helper to generate a multi-page PDF containing text and embedded test images."""
    doc = fitz.open()

    # --- Page 1: Text only ---
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "Executive Summary: Multi-Modal RAG Architecture Overview.\n"
        "This system processes complex PDFs with text and visual graphics.",
        fontsize=12,
    )

    # --- Page 2: Text + Large Image (should be extracted) + Small Icon (should be filtered) ---
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Financial Charts and Revenue Breakdown Page.", fontsize=12)

    # Create a 200x200 test RGB image (qualifies as chart/graphic)
    large_img = Image.new("RGB", (200, 200), color=(73, 109, 137))
    large_img_io = io.BytesIO()
    large_img.save(large_img_io, format="PNG")
    large_bytes = large_img_io.getvalue()

    # Create a 32x32 test icon (should be filtered by min_dimension=100)
    small_icon = Image.new("RGB", (32, 32), color=(255, 0, 0))
    small_icon_io = io.BytesIO()
    small_icon.save(small_icon_io, format="PNG")
    small_bytes = small_icon_io.getvalue()

    # Insert both images into page 2
    page2.insert_image(fitz.Rect(50, 100, 250, 300), stream=large_bytes)
    page2.insert_image(fitz.Rect(50, 350, 82, 382), stream=small_bytes)

    pdf_path = tmp_path / "sample_report.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class TestPDFParser:
    """Test suite for PDFParser parsing and visual element extraction."""

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Verify non-existent files trigger PDFParsingError."""
        parser = PDFParser()
        with pytest.raises(PDFParsingError, match="File not found"):
            parser.parse(tmp_path / "does_not_exist.pdf")

    def test_empty_bytes_raises_error(self) -> None:
        """Verify empty byte buffers trigger PDFParsingError."""
        parser = PDFParser()
        with pytest.raises(PDFParsingError, match="Cannot parse empty"):
            parser.parse_bytes(b"", "empty.pdf")

    def test_synthetic_pdf_parsing_and_image_filtering(self, tmp_path: Path) -> None:
        """Verify text chunking, large image extraction, and small image filtering."""
        pdf_path = create_synthetic_pdf(tmp_path)
        img_output_dir = tmp_path / "extracted_images"

        extractor = ImageExtractor(output_dir=img_output_dir, min_dimension=100)
        chunker = RecursiveTextChunker(chunk_size=200, chunk_overlap=30)
        parser = PDFParser(text_chunker=chunker, image_extractor=extractor)

        parsed_doc = parser.parse(pdf_path)

        # Check document structure
        assert parsed_doc.source_pdf == "sample_report.pdf"
        assert parsed_doc.total_pages == 2

        # Check text chunks
        assert len(parsed_doc.text_chunks) >= 2
        chunk_texts = " ".join(c.text for c in parsed_doc.text_chunks)
        assert "Multi-Modal RAG Architecture Overview" in chunk_texts
        assert "Financial Charts and Revenue Breakdown" in chunk_texts

        # Check image extraction: exactly 1 large image extracted (small icon discarded)
        assert len(parsed_doc.extracted_images) == 1
        extracted = parsed_doc.extracted_images[0]
        assert extracted.source_pdf == "sample_report.pdf"
        assert extracted.page_number == 2
        assert extracted.width == 200
        assert extracted.height == 200
        assert Path(extracted.local_path).exists()

    def test_async_parse_method(self, tmp_path: Path) -> None:
        """Verify asynchronous parsing method works correctly."""
        import sys
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        pdf_path = create_synthetic_pdf(tmp_path)
        extractor = ImageExtractor(output_dir=tmp_path / "async_imgs", min_dimension=100)
        parser = PDFParser(image_extractor=extractor)
        parsed_doc = asyncio.run(parser.parse_async(pdf_path))

        assert parsed_doc.total_pages == 2
        assert len(parsed_doc.text_chunks) > 0
        assert len(parsed_doc.extracted_images) == 1
