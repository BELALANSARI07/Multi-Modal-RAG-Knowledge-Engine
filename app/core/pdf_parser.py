"""PDF parsing and multi-modal extraction engine using PyMuPDF (fitz)."""

import asyncio
import logging
from pathlib import Path
from typing import NamedTuple

import pymupdf as fitz

from app.core.exceptions import DocumentEmptyError, PDFParsingError, SecurityValidationError
from app.core.image_extractor import ImageExtractor
from app.core.text_chunker import RecursiveTextChunker
from app.models.schemas import ExtractedImage, TextChunk
from app.utils.file_utils import sanitize_filename, validate_pdf_signature

logger = logging.getLogger(__name__)


class ParsedPDFDocument(NamedTuple):
    """Container holding extracted multi-modal data from a parsed PDF."""

    source_pdf: str
    total_pages: int
    text_chunks: list[TextChunk]
    extracted_images: list[ExtractedImage]


class PDFParser:
    """Extracts textual content and embedded visuals from PDF files."""

    def __init__(
        self,
        text_chunker: RecursiveTextChunker | None = None,
        image_extractor: ImageExtractor | None = None,
    ) -> None:
        self.chunker = text_chunker or RecursiveTextChunker()
        self.image_extractor = image_extractor or ImageExtractor()

    def parse(self, file_path: str | Path) -> ParsedPDFDocument:
        """Synchronously parse a PDF document from disk with signature validation."""
        path = Path(file_path)
        if not path.is_file():
            raise PDFParsingError(f"File not found: {path}")

        # Security check: validate PDF magic bytes
        try:
            with open(path, "rb") as f:
                header = f.read(1024)
            if not validate_pdf_signature(header):
                raise SecurityValidationError(f"File '{path.name}' does not contain a valid PDF signature header (%PDF-)")
        except SecurityValidationError:
            raise
        except Exception as e:
            raise PDFParsingError(f"Could not read file header for {path.name}", details=str(e))

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            raise PDFParsingError(f"Failed to open PDF at {path}", details=str(e))

        clean_filename = sanitize_filename(path.name)
        return self._process_doc(doc, source_pdf=clean_filename)

    def parse_bytes(self, pdf_bytes: bytes, filename: str) -> ParsedPDFDocument:
        """Synchronously parse a PDF document from raw in-memory bytes with signature validation."""
        if not pdf_bytes:
            raise PDFParsingError("Cannot parse empty PDF byte buffer")

        if not validate_pdf_signature(pdf_bytes):
            raise SecurityValidationError(f"Uploaded file '{filename}' does not contain a valid PDF signature header (%PDF-)")

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise PDFParsingError(f"Failed to parse PDF stream for {filename}", details=str(e))

        clean_filename = sanitize_filename(filename)
        return self._process_doc(doc, source_pdf=clean_filename)

    async def parse_async(self, file_path: str | Path) -> ParsedPDFDocument:
        """Asynchronously parse a PDF document by offloading CPU work to a thread pool."""
        return await asyncio.to_thread(self.parse, file_path)

    async def parse_bytes_async(self, pdf_bytes: bytes, filename: str) -> ParsedPDFDocument:
        """Asynchronously parse PDF bytes by offloading CPU work to a thread pool."""
        return await asyncio.to_thread(self.parse_bytes, pdf_bytes, filename)

    def _process_doc(self, doc: fitz.Document, source_pdf: str) -> ParsedPDFDocument:
        """Iterate through pages to extract text chunks and visual elements."""
        self.image_extractor.reset_deduplication()
        try:
            if doc.is_encrypted:
                raise PDFParsingError(f"PDF '{source_pdf}' is encrypted and password-protected")

            total_pages = len(doc)
            if total_pages == 0:
                raise PDFParsingError(f"PDF '{source_pdf}' contains no pages")

            all_chunks: list[TextChunk] = []
            all_images: list[ExtractedImage] = []
            global_chunk_idx = 0

            for page_index in range(total_pages):
                page_number = page_index + 1
                page = doc.load_page(page_index)

                # 1. Extract and chunk textual content
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    page_chunks = self.chunker.chunk_page_text(
                        text=page_text,
                        source_pdf=source_pdf,
                        page_number=page_number,
                        start_chunk_index=global_chunk_idx,
                    )
                    all_chunks.extend(page_chunks)
                    global_chunk_idx += len(page_chunks)

                # 2. Extract visual elements (raster images/charts)
                page_images = self.image_extractor.extract_from_page(
                    doc=doc,
                    page=page,
                    source_pdf=source_pdf,
                    page_number=page_number,
                )
                all_images.extend(page_images)

            # Check for completely empty PDF (no text and no images)
            if len(all_chunks) == 0 and len(all_images) == 0:
                raise DocumentEmptyError(
                    f"PDF '{source_pdf}' has {total_pages} page(s) but contains no extractable text or visual figures"
                )

            return ParsedPDFDocument(
                source_pdf=source_pdf,
                total_pages=total_pages,
                text_chunks=all_chunks,
                extracted_images=all_images,
            )
        finally:
            doc.close()
