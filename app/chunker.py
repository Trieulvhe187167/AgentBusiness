"""
Recursive character text splitter with rich metadata per chunk.
Splits by: paragraph → newline → sentence → space → character.
"""

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, chunk_size: int, chunk_overlap: int,
                separators: list[str] | None = None) -> list[str]:
    """Recursively split text using a hierarchy of separators."""
    if separators is None:
        separators = SEPARATORS.copy()

    final_chunks: list[str] = []
    separator = separators[-1]  # fallback to char-level

    # Find the best separator that exists in the text
    for sep in separators:
        if sep == "" or sep in text:
            separator = sep
            break

    # Split by the chosen separator
    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)

    # Merge small splits into chunks
    current_chunk: list[str] = []
    current_length = 0

    for piece in splits:
        piece_len = len(piece)
        join_len = len(separator) if current_chunk else 0

        if current_length + join_len + piece_len > chunk_size and current_chunk:
            # Current chunk is full — emit it
            chunk_text = separator.join(current_chunk)
            if chunk_text.strip():
                final_chunks.append(chunk_text.strip())

            # Keep overlap: trim from the front until under overlap size
            while current_chunk and current_length > chunk_overlap:
                removed = current_chunk.pop(0)
                current_length -= len(removed) + len(separator)
                current_length = max(0, current_length)

        current_chunk.append(piece)
        current_length += piece_len + (len(separator) if len(current_chunk) > 1 else 0)

    # Emit last chunk
    if current_chunk:
        chunk_text = separator.join(current_chunk)
        if chunk_text.strip():
            final_chunks.append(chunk_text.strip())

    # If any chunk is still too large, recursively split with next separator
    if len(separators) > 1:
        remaining_seps = separators[separators.index(separator) + 1:]
        if remaining_seps:
            result = []
            for chunk in final_chunks:
                if len(chunk) > chunk_size:
                    sub_chunks = _split_text(chunk, chunk_size, chunk_overlap,
                                             remaining_seps)
                    result.extend(sub_chunks)
                else:
                    result.append(chunk)
            return result

    return final_chunks


def make_chunk_id(source_id: str, offset: int, content: str) -> str:
    """Stable hash-based chunk ID."""
    raw = f"{source_id}:{offset}:{content[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_records(
    records: list[dict[str, Any]],
    source_id: str,
    filename: str,
    file_type: str,
    kb_version: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[dict[str, Any]]:
    """
    Split parsed records into chunks with rich metadata.

    Each chunk dict has:
      - chunk_id, text, source_id, filename, file_type
      - page_num / sheet_name / row_range (from parser metadata)
      - kb_version, content_preview
    """
    all_chunks = []
    offset = 0

    for record in records:
        text = record["text"]
        meta = record.get("metadata", {})

        # Split this record's text
        pieces = _split_text(text, chunk_size, chunk_overlap)

        for piece in pieces:
            chunk_id = make_chunk_id(source_id, offset, piece)
            chunk = {
                "chunk_id": chunk_id,
                "text": piece,
                "source_id": source_id,
                "filename": filename,
                "file_type": file_type,
                "kb_version": kb_version,
                "content_preview": piece[:150].replace("\n", " "),
                # Parser-specific metadata
                "page_num": meta.get("page_num"),
                "sheet_name": meta.get("sheet_name"),
                "row_num": meta.get("row_num"),
                "category": meta.get("category"),
                "keywords": meta.get("keywords"),
            }
            all_chunks.append(chunk)
            offset += 1

    logger.info(f"Chunked {filename}: {len(records)} records → {len(all_chunks)} chunks")
    return all_chunks
