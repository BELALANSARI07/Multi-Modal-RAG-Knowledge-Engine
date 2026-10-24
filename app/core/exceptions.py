"""Custom exception hierarchy for the Multi-Modal RAG Knowledge Engine."""

from typing import Optional


class RAGEngineException(Exception):
    """Base exception for all domain errors within the application."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class SecurityValidationError(RAGEngineException):
    """Raised when file upload violates security constraints (path traversal, invalid signature)."""
    pass


class DocumentEmptyError(RAGEngineException):
    """Raised when a PDF contains no extractable text and no valid visual figures."""
    pass


class AuthenticationError(RAGEngineException):
    """Raised when API key or authorization credentials are missing or invalid."""
    pass


class RateLimitExceededError(RAGEngineException):
    """Raised when client exceeds the allowed request rate."""
    pass


class PDFParsingError(RAGEngineException):
    """Raised when parsing a PDF fails (e.g., corrupt file, password-protected, unreadable)."""
    pass


class ImageExtractionError(RAGEngineException):
    """Raised when extracting or cropping visual objects from a PDF encounters an unrecoverable issue."""
    pass


class ChunkingError(RAGEngineException):
    """Raised when text splitting or chunking fails."""
    pass


class VLMServiceError(RAGEngineException):
    """Raised when calls to the Vision-Language Model fail after retries."""
    pass


class EmbeddingError(RAGEngineException):
    """Raised when local vector embedding generation fails."""
    pass


class VectorStoreError(RAGEngineException):
    """Raised when operations against ChromaDB fail."""
    pass
