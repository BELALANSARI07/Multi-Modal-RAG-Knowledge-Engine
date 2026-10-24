"""End-to-end multi-modal ingestion pipeline orchestrator with transactional rollback."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Union

from app.config import Settings, get_settings
from app.core.exceptions import RAGEngineException
from app.core.pdf_parser import PDFParser
from app.models.enums import ChunkType
from app.models.schemas import DocumentMetadata, IngestedDocumentStats, IngestionResult
from app.services.embedding_service import EmbeddingService
from app.services.vectorstore_service import VectorStoreService
from app.services.vlm_service import VLMService
from app.utils.file_utils import (
    compute_file_hash,
    ensure_directory,
    generate_document_id,
    sanitize_filename,
)

logger = logging.getLogger(__name__)


class IngestionService:
    """Coordinates PDF parsing, visual summarization, dense embedding, and vector storage."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        pdf_parser: Optional[PDFParser] = None,
        vlm_service: Optional[VLMService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vectorstore_service: Optional[VectorStoreService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pdf_parser = pdf_parser or PDFParser()
        self.vlm_service = vlm_service or VLMService(self.settings)
        self.embedding_service = embedding_service or EmbeddingService(self.settings)
        self.vectorstore = vectorstore_service or VectorStoreService(self.settings)
        self.registry_file = ensure_directory(self.settings.upload_dir.parent) / "documents_registry.json"

    async def ingest_file(
        self,
        file_path: Union[str, Path],
        original_filename: Optional[str] = None,
    ) -> IngestionResult:
        """Execute full multimodal ingestion pipeline for a PDF on disk with transactional rollback."""
        path = Path(file_path)
        clean_filename = sanitize_filename(original_filename or path.name)
        logger.info("Starting multimodal ingestion for file: %s", clean_filename)

        with open(path, "rb") as f:
            content_bytes = f.read()

        file_hash = compute_file_hash(content_bytes)
        doc_id = generate_document_id(clean_filename, file_hash)

        # 1. Duplicate Document Handling
        existing_doc = self._get_registry_entry(clean_filename)
        if existing_doc and self.settings.duplicate_strategy == "skip":
            logger.info("Document '%s' already indexed. Duplicate strategy is 'skip'.", clean_filename)
            return IngestionResult(
                status="skipped",
                document_id=existing_doc.document_id,
                filename=clean_filename,
                stats=IngestedDocumentStats(
                    total_pages=existing_doc.total_pages,
                    text_chunks=existing_doc.text_chunks,
                    images_extracted=existing_doc.image_chunks,
                    images_summarized=existing_doc.image_chunks,
                ),
                message=f"Document '{clean_filename}' is already indexed. Skipped re-ingestion.",
            )

        if existing_doc and self.settings.duplicate_strategy == "replace":
            logger.info("Document '%s' already exists. Purging prior vectors for replacement.", clean_filename)
            self.vectorstore.delete_by_source(clean_filename)

        # Directory for extracted images of this document
        doc_folder_name = clean_filename.removesuffix(".pdf")
        doc_image_dir = self.settings.extracted_images_dir / doc_folder_name

        try:
            # 2. Parse PDF: Extract text chunks and visual figures
            parsed_doc = await self.pdf_parser.parse_async(path)

            # 3. Route extracted visuals to VLM for semantic dense summarization
            summarized_images = await self.vlm_service.summarize_extracted_images(parsed_doc.extracted_images)

            # 4. Assemble unified vectorization payload
            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[dict] = []

            # Add plain text chunks
            for chunk in parsed_doc.text_chunks:
                ids.append(chunk.chunk_id)
                documents.append(chunk.text)
                metadatas.append(
                    {
                        "source_pdf": clean_filename,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "chunk_type": ChunkType.TEXT.value,
                        "image_path": "",
                        "char_count": chunk.char_count,
                    }
                )

            # Add dense image visual summaries
            for img in summarized_images:
                if img.vlm_summary and img.vlm_summary.strip():
                    ids.append(img.image_id)
                    documents.append(img.vlm_summary)
                    metadatas.append(
                        {
                            "source_pdf": clean_filename,
                            "page_number": img.page_number,
                            "chunk_index": img.image_index,
                            "chunk_type": ChunkType.IMAGE_SUMMARY.value,
                            "image_path": img.local_path,
                            "char_count": len(img.vlm_summary),
                        }
                    )

            # 5. Generate local dense embeddings for all records
            if documents:
                embeddings = await self.embedding_service.embed_texts_async(documents)

                # 6. Upsert into unified ChromaDB collection
                await self.vectorstore.upsert_documents_async(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )

            stats = IngestedDocumentStats(
                total_pages=parsed_doc.total_pages,
                text_chunks=len(parsed_doc.text_chunks),
                images_extracted=len(parsed_doc.extracted_images),
                images_summarized=len(summarized_images),
            )

            # 7. Record Document in Persistent Registry
            metadata_record = DocumentMetadata(
                document_id=doc_id,
                filename=clean_filename,
                stored_path=str(path.as_posix()),
                file_hash=file_hash,
                file_size_bytes=len(content_bytes),
                total_pages=stats.total_pages,
                text_chunks=stats.text_chunks,
                image_chunks=stats.images_summarized,
                created_at=datetime.now(timezone.utc).isoformat(),
                status="active",
            )
            self._save_registry_entry(metadata_record)

            logger.info(
                "Ingestion finished for %s: %d pages, %d text chunks, %d visual summaries stored",
                clean_filename,
                stats.total_pages,
                stats.text_chunks,
                stats.images_summarized,
            )

            return IngestionResult(
                status="success",
                document_id=doc_id,
                filename=clean_filename,
                stats=stats,
                message=f"Successfully indexed {len(ids)} multi-modal chunks into ChromaDB",
            )

        except Exception as e:
            # Transactional Rollback: Clean up newly created images and vectors
            logger.error("Ingestion failed for '%s'. Rolling back partial artifacts: %s", clean_filename, e)
            self._rollback_artifacts(clean_filename, doc_image_dir)
            raise

    async def ingest_bytes(self, pdf_bytes: bytes, filename: str) -> IngestionResult:
        """Execute full multimodal ingestion pipeline for an uploaded PDF byte stream with path sanitization."""
        clean_filename = sanitize_filename(filename)
        doc_id = generate_document_id(clean_filename, compute_file_hash(pdf_bytes))

        # Save to upload dir with collision-free name
        ensure_directory(self.settings.upload_dir)
        save_path = self.settings.upload_dir / f"{doc_id}_{clean_filename}"

        try:
            with open(save_path, "wb") as f:
                f.write(pdf_bytes)

            result = await self.ingest_file(save_path, original_filename=clean_filename)
            return result
        except Exception:
            # Rollback saved PDF if upload or pipeline fails
            if save_path.exists():
                try:
                    save_path.unlink()
                except Exception as del_err:
                    logger.warning("Failed to delete temp uploaded file %s: %s", save_path, del_err)
            raise

    def delete_document_full(self, filename: str) -> int:
        """Comprehensively delete a document: ChromaDB vectors, source PDF file, and extracted images."""
        clean_filename = sanitize_filename(filename)

        # 1. Purge vectors from ChromaDB
        deleted_count = self.vectorstore.delete_by_source(clean_filename)

        # 2. Delete source PDF file from upload directory
        try:
            for f in self.settings.upload_dir.glob(f"*{clean_filename}"):
                if f.is_file():
                    f.unlink()
                    logger.info("Deleted source PDF file: %s", f)
        except Exception as e:
            logger.warning("Error deleting source PDF file for %s: %s", clean_filename, e)

        # 3. Delete extracted images directory
        doc_folder_name = clean_filename.removesuffix(".pdf")
        doc_image_dir = self.settings.extracted_images_dir / doc_folder_name
        if doc_image_dir.exists():
            try:
                shutil.rmtree(doc_image_dir)
                logger.info("Deleted extracted images directory: %s", doc_image_dir)
            except Exception as e:
                logger.warning("Error removing image directory %s: %s", doc_image_dir, e)

        # 4. Remove from registry
        self._remove_registry_entry(clean_filename)

        return deleted_count

    def _rollback_artifacts(self, source_pdf: str, doc_image_dir: Path) -> None:
        """Purge vectors and filesystem assets upon ingestion failure."""
        try:
            self.vectorstore.delete_by_source(source_pdf)
        except Exception as e:
            logger.warning("Rollback vector deletion warning: %s", e)

        if doc_image_dir.exists():
            try:
                shutil.rmtree(doc_image_dir)
            except Exception as e:
                logger.warning("Rollback image directory cleanup warning: %s", e)

    def _load_registry(self) -> Dict[str, dict]:
        """Load document metadata registry from JSON."""
        if not self.registry_file.exists():
            return {}
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read document registry: %s", e)
            return {}

    def _get_registry_entry(self, filename: str) -> Optional[DocumentMetadata]:
        """Fetch metadata for a document if recorded."""
        data = self._load_registry()
        raw = data.get(filename)
        return DocumentMetadata(**raw) if raw else None

    def _save_registry_entry(self, record: DocumentMetadata) -> None:
        """Save document metadata to JSON registry."""
        data = self._load_registry()
        data[record.filename] = record.model_dump()
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save document registry: %s", e)

    def _remove_registry_entry(self, filename: str) -> None:
        """Delete an entry from the document registry."""
        data = self._load_registry()
        if filename in data:
            del data[filename]
            try:
                with open(self.registry_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.warning("Failed to update registry on delete: %s", e)
