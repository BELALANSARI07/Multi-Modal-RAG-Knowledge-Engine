"""Image extraction and filtering from PDF documents using PyMuPDF (fitz) and Pillow."""

import hashlib
import io
import logging
from pathlib import Path
from typing import List, Set, Union

from PIL import Image
import pymupdf as fitz

from app.models.schemas import ExtractedImage
from app.utils.file_utils import ensure_directory, generate_image_id, sanitize_filename

logger = logging.getLogger(__name__)


class ImageExtractor:
    """Extracts, filters, deduplicates, and saves embedded images from PDF pages."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "./data/extracted_images",
        min_dimension: int = 100,
        max_dimension: int = 2048,
        default_format: str = "png",
    ) -> None:
        self.output_dir = ensure_directory(output_dir)
        self.min_dimension = min_dimension
        self.max_dimension = max_dimension
        self.default_format = default_format.lower()
        self._seen_image_hashes: Set[str] = set()

    def reset_deduplication(self) -> None:
        """Clear cached image hashes for a new document parsing run."""
        self._seen_image_hashes.clear()

    def extract_from_page(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        source_pdf: str,
        page_number: int,
    ) -> List[ExtractedImage]:
        """Extract embedded raster images from a single PDF page."""
        extracted: List[ExtractedImage] = []
        doc_folder_name = sanitize_filename(source_pdf).removesuffix(".pdf")
        page_save_dir = ensure_directory(self.output_dir / doc_folder_name)

        try:
            image_list = page.get_images(full=True)
        except Exception as e:
            logger.warning(
                "Failed to get image list on page %d of %s: %s",
                page_number,
                source_pdf,
                str(e),
            )
            return extracted

        valid_image_idx = 0
        for item in image_list:
            xref = item[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image.get("image") or base_image.get("bytes")
                if not image_bytes:
                    continue

                # Content-based image deduplication
                img_hash = hashlib.sha256(image_bytes).hexdigest()
                if img_hash in self._seen_image_hashes:
                    logger.debug(
                        "Skipping duplicate image (hash %s) on page %d",
                        img_hash[:8],
                        page_number,
                    )
                    continue

                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Skip small icons, buttons, decorations, or tiny logos
                if width < self.min_dimension or height < self.min_dimension:
                    logger.debug(
                        "Skipping small image on page %d (%dx%d < %dpx)",
                        page_number,
                        width,
                        height,
                        self.min_dimension,
                    )
                    continue

                # Open with Pillow to validate and normalize
                with Image.open(io.BytesIO(image_bytes)) as pil_img:
                    # Resize oversized images to preserve memory / API limits
                    if max(width, height) > self.max_dimension:
                        pil_img.thumbnail(
                            (self.max_dimension, self.max_dimension),
                            Image.Resampling.LANCZOS,
                        )
                        width, height = pil_img.size

                    # Format filename and save path
                    image_id = generate_image_id(source_pdf, page_number, valid_image_idx)
                    filename = f"{image_id}.{self.default_format}"
                    target_path = page_save_dir / filename

                    # Convert CMYK/RGBA to RGB if needed when saving as PNG/JPEG
                    if pil_img.mode in ("CMYK", "P"):
                        pil_img = pil_img.convert("RGB")

                    pil_img.save(target_path, format=self.default_format.upper())

                self._seen_image_hashes.add(img_hash)

                extracted.append(
                    ExtractedImage(
                        image_id=image_id,
                        source_pdf=source_pdf,
                        page_number=page_number,
                        image_index=valid_image_idx,
                        local_path=str(target_path.as_posix()),
                        width=width,
                        height=height,
                        image_format=self.default_format,
                    )
                )
                valid_image_idx += 1

            except Exception as exc:
                logger.warning(
                    "Error extracting image xref %d on page %d: %s",
                    xref,
                    page_number,
                    str(exc),
                )
                continue

        return extracted
