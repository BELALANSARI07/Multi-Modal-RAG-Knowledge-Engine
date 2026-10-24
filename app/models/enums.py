"""Enumerations used throughout the Multi-Modal RAG Engine."""

from enum import Enum


class ChunkType(str, Enum):
    """Categorization of chunks ingested into the vector store."""

    TEXT = "text"
    IMAGE_SUMMARY = "image_summary"


class VLMProvider(str, Enum):
    """Supported Vision-Language Model providers."""

    GEMINI = "gemini"
    OPENAI = "openai"
