"""ChromaDB vector database service for unified multimodal storage and retrieval."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.core.exceptions import VectorStoreError
from app.utils.file_utils import ensure_directory

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages persistent vector indexing, similarity queries, and metadata filtering in ChromaDB."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.persist_dir = ensure_directory(self.settings.chroma_persist_dir)
        self.collection_name = self.settings.chroma_collection_name
        self.expected_dim = self.settings.embedding_dimension
        self._client = None
        self._collection = None

    def _get_client(self):
        """Lazy load ChromaDB persistent client."""
        if self._client is None:
            try:
                import chromadb

                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
                logger.info("Connected to ChromaDB storage at: %s", self.persist_dir)
            except Exception as e:
                logger.error("Failed to initialize ChromaDB PersistentClient: %s", e)
                raise VectorStoreError(f"Could not connect to ChromaDB at {self.persist_dir}", details=str(e))
        return self._client

    def _get_collection(self):
        """Get or initialize the target unified collection configured for cosine similarity."""
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("Initialized ChromaDB collection: '%s'", self.collection_name)
            except Exception as e:
                logger.error("Failed to acquire collection '%s': %s", self.collection_name, e)
                raise VectorStoreError(f"Failed to access collection {self.collection_name}", details=str(e))
        return self._collection

    def upsert_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """Upsert a batch of text/image chunks and their embeddings into ChromaDB with dimension validation."""
        if not ids:
            return 0

        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise VectorStoreError("Mismatched list lengths in vector store upsert")

        # Validate vector dimensionality before upserting
        for idx, vec in enumerate(embeddings):
            if len(vec) != self.expected_dim:
                raise VectorStoreError(
                    f"Vector dimension mismatch at index {idx}: expected {self.expected_dim}, got {len(vec)}"
                )

        collection = self._get_collection()
        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            logger.info("Upserted %d records into ChromaDB collection '%s'", len(ids), self.collection_name)
            return len(ids)
        except Exception as e:
            logger.error("ChromaDB upsert failed: %s", e)
            raise VectorStoreError("Failed to upsert records into ChromaDB", details=str(e))

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query the collection by embedding vectors with optional metadata filtering."""
        collection = self._get_collection()
        try:
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            return results
        except Exception as e:
            logger.error("ChromaDB vector query failed: %s", e)
            raise VectorStoreError("Failed to query ChromaDB collection", details=str(e))

    def count(self) -> int:
        """Return the total number of indexed vectors in the collection with strict error propagation."""
        collection = self._get_collection()
        try:
            return collection.count()
        except Exception as e:
            logger.error("Failed to retrieve collection count: %s", e)
            raise VectorStoreError(f"Error reading vector collection count: {str(e)}")

    def delete_by_source(self, source_pdf: str) -> int:
        """Remove all chunks associated with a specific PDF filename."""
        collection = self._get_collection()
        try:
            initial = collection.count()
            collection.delete(where={"source_pdf": source_pdf})
            final = collection.count()
            deleted = initial - final
            logger.info("Deleted %d records for source: %s", deleted, source_pdf)
            return deleted
        except Exception as e:
            logger.error("Failed to delete records for %s: %s", source_pdf, e)
            raise VectorStoreError(f"Failed to delete records for {source_pdf}", details=str(e))

    def document_exists(self, source_pdf: str) -> bool:
        """Check if any vectors for a given source PDF already exist."""
        collection = self._get_collection()
        try:
            results = collection.get(where={"source_pdf": source_pdf}, limit=1)
            return len(results.get("ids", [])) > 0
        except Exception:
            return False

    def list_ingested_documents(self) -> List[Dict[str, Any]]:
        """Scan collection metadata to compile summaries of all ingested documents."""
        collection = self._get_collection()
        try:
            total = collection.count()
            if total == 0:
                return []

            results = collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])

            docs_map: Dict[str, Dict[str, Any]] = {}
            for meta in metadatas:
                source = meta.get("source_pdf", "unknown")
                chunk_type = meta.get("chunk_type", "text")
                page_num = meta.get("page_number", 1)

                if source not in docs_map:
                    docs_map[source] = {
                        "filename": source,
                        "text_chunks": 0,
                        "image_chunks": 0,
                        "max_page": 1,
                    }

                if chunk_type == "text":
                    docs_map[source]["text_chunks"] += 1
                else:
                    docs_map[source]["image_chunks"] += 1

                docs_map[source]["max_page"] = max(docs_map[source]["max_page"], page_num)

            return list(docs_map.values())
        except Exception as e:
            logger.error("Failed to list ingested documents: %s", e)
            return []

    def reset_collection(self) -> None:
        """Clear and recreate the collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
            logger.warning("Reset ChromaDB collection: %s", self.collection_name)
        except Exception as e:
            logger.warning("Error during collection reset (may not exist): %s", e)

    async def upsert_documents_async(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """Asynchronous wrapper for document upserts."""
        return await asyncio.to_thread(self.upsert_documents, ids, documents, metadatas, embeddings)

    async def query_async(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronous wrapper for vector queries."""
        return await asyncio.to_thread(self.query, query_embeddings, n_results, where)
