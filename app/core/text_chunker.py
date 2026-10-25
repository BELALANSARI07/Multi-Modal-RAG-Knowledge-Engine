"""Recursive character-based text chunker with sliding window overlap."""


from app.core.exceptions import ChunkingError
from app.models.enums import ChunkType
from app.models.schemas import TextChunk
from app.utils.file_utils import generate_chunk_id


class RecursiveTextChunker:
    """Intelligently chunks text by recursively trying separators to preserve semantic structure."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ChunkingError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first viable separator in separators."""
        final_chunks: list[str] = []
        text = text.strip()
        if not text:
            return final_chunks

        if len(text) <= self.chunk_size:
            return [text]

        # Find the highest-priority separator present in text
        chosen_sep: str | None = None
        remaining_seps: list[str] = []
        for i, sep in enumerate(separators):
            if sep in text:
                chosen_sep = sep
                remaining_seps = separators[i + 1 :]
                break

        # If no separator found, hard split by character limit
        if chosen_sep is None:
            return self._hard_split(text)

        splits = text.split(chosen_sep)
        good_splits: list[str] = []

        for s in splits:
            s_stripped = s.strip()
            if not s_stripped:
                continue
            if len(s_stripped) <= self.chunk_size:
                good_splits.append(s_stripped)
            else:
                # Recurse on oversized segment with remaining separators
                if remaining_seps:
                    sub_splits = self._split_text(s_stripped, remaining_seps)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.extend(self._hard_split(s_stripped))

        # Recombine splits respecting chunk_size and chunk_overlap
        return self._merge_splits(good_splits, chosen_sep)

    def _hard_split(self, text: str) -> list[str]:
        """Split text rigidly by characters with overlap when no delimiters exist."""
        chunks: list[str] = []
        start = 0
        text_len = len(text)
        step = self.chunk_size - self.chunk_overlap

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            start += step
        return chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge smaller fragments together up to chunk_size with sliding overlap."""
        merged: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for segment in splits:
            seg_len = len(segment)
            sep_len = len(separator) if current_chunk else 0

            if current_len + sep_len + seg_len <= self.chunk_size:
                current_chunk.append(segment)
                current_len += sep_len + seg_len
            else:
                if current_chunk:
                    joined = separator.join(current_chunk).strip()
                    if joined:
                        merged.append(joined)
                    # Start next chunk with overlap from the tail of current_chunk
                    overlap_acc: list[str] = []
                    overlap_len = 0
                    for prev_seg in reversed(current_chunk):
                        if overlap_len + len(prev_seg) + len(separator) <= self.chunk_overlap:
                            overlap_acc.insert(0, prev_seg)
                            overlap_len += len(prev_seg) + len(separator)
                        else:
                            break
                    current_chunk = overlap_acc + [segment]
                    current_len = sum(len(s) for s in current_chunk) + (len(separator) * (len(current_chunk) - 1))
                else:
                    current_chunk = [segment]
                    current_len = seg_len

        if current_chunk:
            joined = separator.join(current_chunk).strip()
            if joined:
                merged.append(joined)

        return merged

    def chunk_page_text(
        self,
        text: str,
        source_pdf: str,
        page_number: int,
        start_chunk_index: int = 0,
    ) -> list[TextChunk]:
        """Splits raw text from a PDF page into a list of structured TextChunk instances."""
        if not text or not text.strip():
            return []

        raw_chunks = self._split_text(text, self.separators)
        results: list[TextChunk] = []

        for idx, chunk_str in enumerate(raw_chunks):
            chunk_idx = start_chunk_index + idx
            chunk_id = generate_chunk_id(
                source_pdf=source_pdf,
                page_number=page_number,
                chunk_index=chunk_idx,
            )
            results.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_str,
                    source_pdf=source_pdf,
                    page_number=page_number,
                    chunk_index=chunk_idx,
                    chunk_type=ChunkType.TEXT,
                    char_count=len(chunk_str),
                    metadata={
                        "page_number": page_number,
                        "source": source_pdf,
                    },
                )
            )

        return results
