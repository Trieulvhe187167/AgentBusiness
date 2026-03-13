"""
Recursive character text splitter with rich metadata per chunk.
Splits by: paragraph → newline → sentence → space → character.

chunk_strategy:
  "row"       — treat each record as one chunk (CSV/Excel/JSONL/DOCX-table rows).
                 If row text > row_max_chars, falls back to sentence-level split.
  "recursive" — recursive split (PDF / plain text / Markdown).
  "heading"   — heading-aware split (HTML / DOCX-heading sections).
"""

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
ROW_MAX_CHARS = 1500   # threshold above which "row" strategy still splits


def _split_text(text: str, chunk_size: int, chunk_overlap: int,
                separators: list[str] | None = None) -> list[str]:
    """Recursively split text using a hierarchy of separators."""
    if separators is None:
        separators = SEPARATORS.copy()

    final_chunks: list[str] = []
    separator = separators[-1]

    for sep in separators:
        if sep == "" or sep in text:
            separator = sep
            break

    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)

    current_chunk: list[str] = []
    current_length = 0

    for piece in splits:
        piece_len = len(piece)
        join_len = len(separator) if current_chunk else 0

        if current_length + join_len + piece_len > chunk_size and current_chunk:
            chunk_text = separator.join(current_chunk)
            if chunk_text.strip():
                final_chunks.append(chunk_text.strip())

            while current_chunk and current_length > chunk_overlap:
                removed = current_chunk.pop(0)
                current_length -= len(removed) + len(separator)
                current_length = max(0, current_length)

        current_chunk.append(piece)
        current_length += piece_len + (len(separator) if len(current_chunk) > 1 else 0)

    if current_chunk:
        chunk_text = separator.join(current_chunk)
        if chunk_text.strip():
            final_chunks.append(chunk_text.strip())

    if len(separators) > 1:
        remaining_seps = separators[separators.index(separator) + 1:]
        if remaining_seps:
            result = []
            for chunk in final_chunks:
                if len(chunk) > chunk_size:
                    sub_chunks = _split_text(chunk, chunk_size, chunk_overlap, remaining_seps)
                    result.extend(sub_chunks)
                else:
                    result.append(chunk)
            return result

    return final_chunks


def _split_by_sentences(text: str, max_chars: int) -> list[str]:
    """Light sentence-level split for rows that exceed ROW_MAX_CHARS."""
    return _split_text(text, max_chars, max_chars // 5, [". ", "\n", " ", ""])


def make_chunk_id(kb_id: int, file_hash: str, offset: int, ingest_signature: str) -> str:
    """Stable hash-based chunk ID scoped to KB + ingest signature."""
    raw = f"{kb_id}:{file_hash}:{offset}:{ingest_signature}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _strategy_for_file_type(file_type: str) -> str:
    """Infer chunk strategy from file type when not explicitly provided."""
    if file_type in {"csv", "excel", "jsonl", "json", "docx_table"}:
        return "row"
    if file_type in {"html", "docx"}:
        return "heading"
    return "recursive"


def chunk_records(
    records: list[dict[str, Any]],
    kb_id: int,
    source_id: str,
    filename: str,
    file_type: str,
    file_hash: str,
    kb_version: str,
    ingest_signature: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    chunk_strategy: str | None = None,
) -> list[dict[str, Any]]:
    """
    Split parsed records into chunks with rich metadata.

    Each chunk dict has:
      - chunk_id, text, source_id, filename, file_type
      - page_num / sheet_name / row_range (from parser metadata)
      - kb_version, content_preview, lang
    """
    strategy = chunk_strategy or _strategy_for_file_type(file_type)
    all_chunks = []
    offset = 0

    for record in records:
        text = record["text"]
        meta = record.get("metadata", {})

        if strategy == "row":
            # Keep row as a single chunk; only split if too long
            if len(text) <= ROW_MAX_CHARS:
                pieces = [text]
            else:
                pieces = _split_by_sentences(text, ROW_MAX_CHARS)
        else:
            # recursive / heading: standard split
            pieces = _split_text(text, chunk_size, chunk_overlap)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = make_chunk_id(kb_id, file_hash, offset, ingest_signature)
            chunk = {
                "chunk_id": chunk_id,
                "text": piece,
                "kb_id": kb_id,
                "source_id": source_id,
                "file_id": int(source_id),
                "filename": filename,
                "file_type": file_type,
                "kb_version": kb_version,
                "file_hash": file_hash,
                "ingest_signature": ingest_signature,
                "content_preview": piece[:150].replace("\n", " "),
                # Parser-specific metadata
                "page_num": meta.get("page_num"),
                "sheet_name": meta.get("sheet_name"),
                "row_num": meta.get("row_num"),
                "category": meta.get("category"),
                "keywords": meta.get("keywords"),
                "lang": meta.get("lang"),
            }
            all_chunks.append(chunk)
            offset += 1

    logger.info(
        "Chunked %s: %s records → %s chunks (strategy=%s)",
        filename, len(records), len(all_chunks), strategy,
    )
    return all_chunks
