"""
DOCX parser: extracts text section-by-section (headings) and table row-by-row.
Requires: python-docx
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", ""} else text


def parse_docx(file_path: Path) -> list[dict[str, Any]]:
    try:
        from docx import Document  # type: ignore
    except ImportError as err:
        raise ValueError(
            "DOCX parsing requires python-docx. Install with: pip install python-docx"
        ) from err

    doc = Document(str(file_path))
    records: list[dict[str, Any]] = []

    # ── Heading-based sections ────────────────────────────────────────────────
    current_heading: str = ""
    current_body: list[str] = []

    def _flush_section():
        text = "\n".join(current_body).strip()
        if text:
            records.append({
                "text": f"{current_heading}\n{text}".strip() if current_heading else text,
                "metadata": {"title": current_heading or file_path.stem},
            })

    for para in doc.paragraphs:
        style = para.style.name if para.style else ""
        text = para.text.strip()
        if not text:
            continue

        if style.startswith("Heading"):
            _flush_section()
            current_heading = text
            current_body = []
        else:
            current_body.append(text)

    _flush_section()

    # ── Tables ────────────────────────────────────────────────────────────────
    for table_idx, table in enumerate(doc.tables):
        if not table.rows:
            continue

        # First row = header
        headers = [_normalize(cell.text) for cell in table.rows[0].cells]
        if not any(headers):
            continue

        for row_idx, row in enumerate(table.rows[1:], start=2):
            values = [_normalize(cell.text) for cell in row.cells]
            parts = [
                f"{headers[i]}: {values[i]}"
                for i in range(min(len(headers), len(values)))
                if values[i]
            ]
            text = " | ".join(parts).strip()
            if text:
                records.append({
                    "text": text,
                    "metadata": {
                        "table_idx": table_idx + 1,
                        "row_num": row_idx,
                    },
                })

    logger.info(
        "Parsed DOCX: %s records (%s paragraphs, %s tables) from %s",
        len(records),
        len(doc.paragraphs),
        len(doc.tables),
        file_path.name,
    )
    return records
