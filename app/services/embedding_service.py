"""Dense embedding service powered by Hugging Face sentence-transformers with ONNX fallback."""

import asyncio
import logging

from app.config import Settings, get_settings
from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages local sentence embedding generation with batching and async thread offloading.

    Supports SentenceTransformer with transparent fallback to ONNX Runtime (all-MiniLM-L6-v2)
    to guarantee flawless operation across secure Windows environments.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.embedding_model_name
        self.batch_size = self.settings.embedding_batch_size
        self._model = None
        self._onnx_fn = None
        self._embedding_dim = 384  # Default dimension for all-MiniLM-L6-v2
        self._engine_type = "sentence_transformers"

    def _load_model(self):
        """Lazy load embedding model: tries SentenceTransformer first, then ONNX runtime fallback."""
        if self._model is not None or self._onnx_fn is not None:
            return self._model or self._onnx_fn

        # 1. Attempt standard SentenceTransformer loading
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Attempting to load SentenceTransformer: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            sample = self._model.encode("test", show_progress_bar=False)
            self._embedding_dim = len(sample)
            self._engine_type = "sentence_transformers"
            logger.info("Loaded SentenceTransformer %s (dim=%d)", self.model_name, self._embedding_dim)
            return self._model
        except Exception as e:
            logger.warning(
                "SentenceTransformer load failed (%s). Falling back to high-performance ONNX Runtime engine.",
                str(e),
            )

        # 2. Resilient Fallback: ChromaDB ONNX-based MiniLM (same weights, no torch DLL constraint)
        try:
            import chromadb.utils.embedding_functions as ef

            self._onnx_fn = ef.DefaultEmbeddingFunction()
            sample = self._onnx_fn(["test"])[0]
            self._embedding_dim = len(sample)
            self._engine_type = "onnx"
            logger.info("Loaded ONNX embedding engine (dim=%d)", self._embedding_dim)
            return self._onnx_fn
        except Exception as onnx_err:
            logger.error("Both SentenceTransformer and ONNX embedding engines failed: %s", onnx_err)
            raise EmbeddingError(
                f"Could not initialize embedding engine for {self.model_name}",
                details=str(onnx_err),
            )

    @property
    def embedding_dimension(self) -> int:
        """Return the vector dimensionality of the loaded model."""
        return self._embedding_dim

    @property
    def engine_type(self) -> str:
        """Return which engine is currently active ('sentence_transformers' or 'onnx')."""
        return self._engine_type

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Synchronously generate embeddings for a batch of text strings."""
        if not texts:
            return []

        engine = self._load_model()
        try:
            if self._engine_type == "sentence_transformers":
                embeddings = engine.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                return [vec.tolist() for vec in embeddings]
            else:
                # ONNX embedding function handles batching and returns list of lists of floats
                raw_vectors = engine(texts)
                return [list(map(float, vec)) for vec in raw_vectors]
        except Exception as e:
            raise EmbeddingError("Batch text embedding failed", details=str(e))

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding vector for a single text input."""
        results = self.embed_texts([text])
        if not results:
            raise EmbeddingError("Failed to embed single text input")
        return results[0]

    async def embed_texts_async(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously generate embeddings for a batch of texts in a worker thread."""
        return await asyncio.to_thread(self.embed_texts, texts)

    async def embed_single_async(self, text: str) -> list[float]:
        """Asynchronously generate embedding for a single text in a worker thread."""
        return await asyncio.to_thread(self.embed_single, text)
