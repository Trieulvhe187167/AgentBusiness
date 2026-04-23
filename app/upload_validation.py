"""
Upload validation helpers.

Validation strategy:
- Signature validation: pdf, xls, xlsx, docx
- Text/content heuristic: html, htm, csv, txt, md, json, jsonl
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


UPLOAD_PARSER_MAP = {
    ".xlsx": "excel",
    ".xls": "excel",
    ".csv": "csv",
    ".tsv": "csv",
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".txt": "text",
    ".md": "text",
    ".docx": "docx",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".xml": "xml",
}


SIGNATURE_VALIDATION_NOTES = {
    ".pdf": "signature",
    ".xls": "signature",
    ".xlsx": "zip_signature+member_check",
    ".docx": "zip_signature+member_check",
    ".html": "content_heuristic",
    ".htm": "content_heuristic",
    ".csv": "text_heuristic",
    ".tsv": "text_heuristic",
    ".txt": "text_heuristic",
    ".md": "text_heuristic",
    ".json": "json_heuristic",
    ".jsonl": "jsonl_heuristic",
    ".ndjson": "jsonl_heuristic",
    ".xml": "xml_heuristic",
}


class UploadValidationError(ValueError):
    def __init__(self, code: str, message: str, *, meta: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.meta = meta or {}


@dataclass(frozen=True)
class ValidatedUpload:
    original_name: str
    extension: str
    parser_type: str


def sanitize_filename(name: str) -> str:
    """Remove traversal sequences and control characters while keeping the basename."""
    base = Path(name or "").name
    base = base.replace("..", "").replace("/", "").replace("\\", "")
    base = "".join(char for char in base if ord(char) >= 32)
    return base.strip() or "unnamed"


def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validation_mode_for_extension(extension: str) -> str:
    return SIGNATURE_VALIDATION_NOTES.get(extension, "unknown")


def _decode_text_sample(content: bytes) -> str | None:
    if b"\x00" in content[:4096]:
        return None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _zip_contains(content: bytes, member_name: str) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            return member_name in archive.namelist()
    except zipfile.BadZipFile:
        return False


def _looks_like_html(content: bytes) -> bool:
    text = (_decode_text_sample(content[:8192]) or "").lower()
    return any(token in text for token in ("<!doctype", "<html", "<body", "<div", "<p"))


def _looks_like_json(content: bytes) -> bool:
    text = (_decode_text_sample(content) or "").strip()
    if not text or text[0] not in "[{":
        return False
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _looks_like_jsonl(content: bytes) -> bool:
    text = (_decode_text_sample(content) or "").strip()
    if not text:
        return False
    try:
        for line in text.splitlines():
            if line.strip():
                json.loads(line)
        return True
    except json.JSONDecodeError:
        return False


def _looks_like_xml(content: bytes) -> bool:
    text = (_decode_text_sample(content) or "").strip()
    if not text.startswith("<") and not text.startswith("<?xml"):
        return False
    try:
        ET.fromstring(text)
        return True
    except ET.ParseError:
        return False


def _looks_like_text(content: bytes) -> bool:
    text = _decode_text_sample(content[:8192])
    return bool(text and text.strip())


def validate_upload(
    *,
    filename: str | None,
    content: bytes,
    allowed_extensions: list[str],
    max_upload_bytes: int,
    max_upload_size_mb: int,
) -> ValidatedUpload:
    if not filename:
        raise UploadValidationError("missing_filename", "No filename provided")

    original_name = sanitize_filename(filename)
    extension = Path(original_name).suffix.lower()

    if extension not in allowed_extensions:
        raise UploadValidationError(
            "unsupported_extension",
            f"File type '{extension}' not allowed.",
            meta={"allowed_extensions": allowed_extensions},
        )

    if len(content) == 0:
        raise UploadValidationError("empty_file", "Empty file rejected")

    if len(content) > max_upload_bytes:
        raise UploadValidationError(
            "file_too_large",
            f"File too large ({len(content) / 1024 / 1024:.1f}MB). Max: {max_upload_size_mb}MB",
            meta={"max_upload_size_mb": max_upload_size_mb},
        )

    if extension == ".pdf" and not content.startswith(b"%PDF"):
        raise UploadValidationError("content_mismatch", "File content does not match '.pdf' format")

    if extension == ".xls" and not content.startswith(b"\xd0\xcf\x11\xe0"):
        raise UploadValidationError("content_mismatch", "File content does not match '.xls' format")

    if extension == ".xlsx":
        if not content.startswith(b"PK\x03\x04") or not _zip_contains(content, "xl/workbook.xml"):
            raise UploadValidationError("content_mismatch", "File content does not match '.xlsx' format")

    if extension == ".docx":
        if not content.startswith(b"PK\x03\x04") or not _zip_contains(content, "word/document.xml"):
            raise UploadValidationError("content_mismatch", "File content does not match '.docx' format")

    if extension in {".html", ".htm"} and not _looks_like_html(content):
        raise UploadValidationError("content_mismatch", f"File content does not match '{extension}' format")

    if extension == ".json" and not _looks_like_json(content):
        raise UploadValidationError("content_mismatch", "File content does not match '.json' format")

    if extension in {".jsonl", ".ndjson"} and not _looks_like_jsonl(content):
        raise UploadValidationError("content_mismatch", f"File content does not match '{extension}' format")

    if extension == ".xml" and not _looks_like_xml(content):
        raise UploadValidationError("content_mismatch", "File content does not match '.xml' format")

    if extension in {".csv", ".tsv", ".txt", ".md"} and not _looks_like_text(content):
        raise UploadValidationError("content_mismatch", f"File content does not match '{extension}' format")

    return ValidatedUpload(
        original_name=original_name,
        extension=extension,
        parser_type=UPLOAD_PARSER_MAP.get(extension, "unknown"),
    )
