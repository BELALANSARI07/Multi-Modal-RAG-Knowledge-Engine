"""Filesystem, security sanitization, and hashing utilities."""

import hashlib
from pathlib import Path, PurePath
import re
from typing import Optional, Union


PDF_MAGIC_BYTES = b"%PDF-"


def validate_pdf_signature(content: bytes) -> bool:
    """Verify that file buffer starts with the authentic PDF magic bytes."""
    if not content or len(content) < 5:
        return False
    # Check first 1024 bytes for %PDF- header (some PDF generators have leading BOM or whitespace)
    header = content[:1024]
    return PDF_MAGIC_BYTES in header


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure that a directory exists, creating parents if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_filename(name: str) -> str:
    """Strictly sanitize filename against path traversal (../, ..\\), null bytes, and unsafe characters.
    
    Guarantees that the returned string is purely a filename with no directory components.
    """
    if not name:
        return "document.pdf"

    # 1. Remove null bytes and control chars
    clean = name.replace("\x00", "").strip()

    # 2. Extract only the filename component, stripping any directory structure
    clean = PurePath(clean).name

    # 3. Replace all dangerous path traversal characters and non-alphanumeric (except dot, underscore, dash)
    clean = re.sub(r"[^\w\.\-]", "_", clean)

    # 4. Collapse consecutive dots/underscores and prevent leading dots (hidden files)
    clean = re.sub(r"\.{2,}", ".", clean)
    clean = re.sub(r"_+", "_", clean).strip("._")

    # 5. Fallback if sanitized string is empty
    if not clean or clean.lower() == ".pdf":
        clean = "document.pdf"

    # Ensure .pdf extension
    if not clean.lower().endswith(".pdf"):
        clean = f"{clean}.pdf"

    return clean


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of byte content."""
    return hashlib.sha256(file_bytes).hexdigest()


def generate_document_id(filename: str, file_hash: Optional[str] = None) -> str:
    """Generate a clean document ID combining sanitized name and unique hash."""
    base = sanitize_filename(filename).removesuffix(".pdf")
    if file_hash:
        return f"{base}_{file_hash[:8]}"
    content = f"{filename}_{hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]}"
    return content


def generate_secure_upload_path(upload_dir: Path, original_filename: str, doc_id: str) -> Path:
    """Generate a collision-resistant, sanitized storage path for uploaded files."""
    clean_name = sanitize_filename(original_filename)
    filename = f"{doc_id}_{clean_name}"
    return upload_dir / filename


def generate_chunk_id(source_pdf: str, page_number: int, chunk_index: int) -> str:
    """Generate a deterministic identifier for a text chunk."""
    base_name = sanitize_filename(source_pdf).removesuffix(".pdf")
    return f"{base_name}_p{page_number}_c{chunk_index}"


def generate_image_id(source_pdf: str, page_number: int, image_index: int) -> str:
    """Generate a deterministic identifier for an extracted image."""
    base_name = sanitize_filename(source_pdf).removesuffix(".pdf")
    return f"{base_name}_p{page_number}_img{image_index}"
