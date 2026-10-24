"""Filesystem and hashing utilities."""

import hashlib
import re
from pathlib import Path
from typing import Optional, Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure that a directory exists, creating parents if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_filename(name: str) -> str:
    """Strip special characters and whitespace from a filename for safe storage."""
    # Keep alphanumeric, dot, underscore, dash
    clean = re.sub(r"[^\w\.\-]", "_", name)
    # Remove leading dots or multiple underscores
    clean = re.sub(r"_+", "_", clean).strip("_.")
    return clean or "document"


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of byte content."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def generate_document_id(filename: str, file_hash: Optional[str] = None) -> str:
    """Generate a clean document ID combining sanitized name and unique hash."""
    base = sanitize_filename(Path(filename).stem)
    if file_hash:
        return f"{base}_{file_hash[:8]}"
    content = f"{filename}_{hashlib.md5(filename.encode('utf-8')).hexdigest()[:6]}"
    return content


def generate_chunk_id(source_pdf: str, page_number: int, chunk_index: int) -> str:
    """Generate a deterministic identifier for a text chunk."""
    base_name = sanitize_filename(Path(source_pdf).stem)
    return f"{base_name}_p{page_number}_c{chunk_index}"


def generate_image_id(source_pdf: str, page_number: int, image_index: int) -> str:
    """Generate a deterministic identifier for an extracted image."""
    base_name = sanitize_filename(Path(source_pdf).stem)
    return f"{base_name}_p{page_number}_img{image_index}"
