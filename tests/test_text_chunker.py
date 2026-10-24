"""Unit tests for the RecursiveTextChunker."""

import pytest

from app.core.exceptions import ChunkingError
from app.core.text_chunker import RecursiveTextChunker
from app.models.enums import ChunkType


class TestRecursiveTextChunker:
    """Test suite for RecursiveTextChunker validation and splitting behavior."""

    def test_invalid_parameters_raise_error(self) -> None:
        """Verify invalid chunk size and overlap configurations trigger ChunkingError."""
        with pytest.raises(ChunkingError, match="positive integer"):
            RecursiveTextChunker(chunk_size=0)

        with pytest.raises(ChunkingError, match="negative"):
            RecursiveTextChunker(chunk_size=100, chunk_overlap=-5)

        with pytest.raises(ChunkingError, match="strictly less"):
            RecursiveTextChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ChunkingError, match="strictly less"):
            RecursiveTextChunker(chunk_size=100, chunk_overlap=150)

    def test_empty_and_whitespace_text(self) -> None:
        """Verify empty and whitespace-only text yield empty lists without error."""
        chunker = RecursiveTextChunker(chunk_size=256, chunk_overlap=32)
        assert chunker.chunk_page_text("", "test.pdf", 1) == []
        assert chunker.chunk_page_text("   \n\n   ", "test.pdf", 1) == []

    def test_short_text_single_chunk(self) -> None:
        """Verify text smaller than chunk_size results in a single structured chunk."""
        chunker = RecursiveTextChunker(chunk_size=512, chunk_overlap=64)
        sample = "This is a short sample sentence about multi-modal retrieval."
        chunks = chunker.chunk_page_text(sample, "doc.pdf", page_number=2, start_chunk_index=3)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.text == sample
        assert chunk.chunk_id == "doc_p2_c3"
        assert chunk.source_pdf == "doc.pdf"
        assert chunk.page_number == 2
        assert chunk.chunk_index == 3
        assert chunk.chunk_type == ChunkType.TEXT
        assert chunk.char_count == len(sample)

    def test_paragraph_and_sentence_splitting(self) -> None:
        """Verify longer text is cleanly partitioned across paragraph boundaries."""
        chunker = RecursiveTextChunker(chunk_size=120, chunk_overlap=20)
        p1 = "Paragraph 1 discusses deep learning architectures in computer vision."
        p2 = "Paragraph 2 delves into vector databases and embedding metrics like cosine similarity."
        p3 = "Paragraph 3 concludes with FastAPI production deployment strategies."
        combined = f"{p1}\n\n{p2}\n\n{p3}"

        chunks = chunker.chunk_page_text(combined, "whitepaper.pdf", page_number=1)

        assert len(chunks) >= 3
        # Each paragraph should be captured without truncating abruptly
        all_text = " ".join(c.text for c in chunks)
        assert "Paragraph 1" in all_text
        assert "Paragraph 2" in all_text
        assert "Paragraph 3" in all_text

    def test_sliding_window_overlap(self) -> None:
        """Verify that consecutive chunks preserve overlapping boundary words."""
        chunker = RecursiveTextChunker(chunk_size=60, chunk_overlap=25)
        text = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda Mu"
        chunks = chunker.chunk_page_text(text, "greek.pdf", 1)

        assert len(chunks) > 1
        for i in range(len(chunks) - 1):
            curr_words = set(chunks[i].text.split())
            next_words = set(chunks[i + 1].text.split())
            # Ensure there is an intersection between consecutive chunks
            overlap = curr_words.intersection(next_words)
            assert len(overlap) > 0, f"Expected overlap between chunk {i} and {i+1}"
