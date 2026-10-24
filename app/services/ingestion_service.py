"""End-to-end multi-modal ingestion pipeline orchestrator."""

import logging
from pathlib import Path
from typing import List, Optional, Union

from app.config import Settings, get_settings
from app.core.pdf_parser import PDFParser
from app.models.enums import ChunkType
from app.models.schemas import IngestedDocumentStats, IngestionResult
from app.services.embedding_service import EmbeddingService
from app.services.vectorstore_service import VectorStoreService
from app.services.vlm_service import VLMService
from app.utils.file_utils import generate_document_id

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

    async def ingest_file(self, file_path: Union[str, Path]) -> IngestionResult:
        """Execute full multimodal ingestion pipeline for a PDF on disk."""
        path = Path(file_path)
        logger.info("Starting multimodal ingestion for file: %s", path.name)

        # 1. Parse PDF: Extract text chunks and visual images
        parsed_doc = await self.pdf_parser.parse_async(path)

        # 2. Route extracted visuals to VLM for semantic dense summarization
        summarized_images = await self.vlm_service.summarize_extracted_images(parsed_doc.extracted_images)

        # 3. Assemble unified vectorization payload
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        # Add plain text chunks
        for chunk in parsed_doc.text_chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "source_pdf": chunk.source_pdf,
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
                        "source_pdf": img.source_pdf,
                        "page_number": img.page_number,
                        "chunk_index": img.image_index,
                        "chunk_type": ChunkType.IMAGE_SUMMARY.value,
                        "image_path": img.local_path,
                        "char_count": len(img.vlm_summary),
                    }
                )

        # 4. Generate local dense embeddings for all records
        if documents:
            embeddings = await self.embedding_service.embed_texts_async(documents)

            # 5. Upsert into unified ChromaDB collection
            await self.vectorstore.upsert_documents_async(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        doc_id = generate_document_id(path.name)
        stats = IngestedDocumentStats(
            total_pages=parsed_doc.total_pages,
            text_chunks=len(parsed_doc.text_chunks),
            images_extracted=len(parsed_doc.extracted_images),
            images_summarized=len(summarized_images),
        )

        logger.info(
            "Ingestion finished for %s: %d pages, %d text chunks, %d visual summaries stored",
            path.name,
            stats.total_pages,
            stats.text_chunks,
            stats.images_summarized,
        )

        return IngestionResult(
            status="success",
            document_id=doc_id,
            filename=path.name,
            stats=stats,
            message=f"Successfully indexed {len(ids)} multi-modal chunks into ChromaDB",
        )

    async def ingest_bytes(self, pdf_bytes: bytes, filename: str) -> IngestionResult:
        """Execute full multimodal ingestion pipeline for an uploaded PDF byte stream."""
        # Save to upload dir
        save_path = self.settings.upload_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(pdf_bytes)

        return await self.ingest_file(save_path)
