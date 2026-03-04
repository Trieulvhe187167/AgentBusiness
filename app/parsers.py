"""
File parsers for CSV/Excel/PDF/HTML/TXT/JSONL/JSON.
Each parser returns list[dict]: {"text": str, "metadata": dict}
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from app.lang import detect_language

logger = logging.getLogger(__name__)

_CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

_KB_TEXT_COLS = ("title", "content")
_KB_META_COLS = ("category", "keywords", "source_url")
_FAQ_TEXT_COLS = ("question", "answer")
_FAQ_META_COLS = ("category", "tags")

# Penalty patterns for blob detection
_BLOB_RE = re.compile(r"[{}<>]|<html|<div|<span|<br", re.IGNORECASE)


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _is_kb_format(columns: list[str]) -> bool:
    lowered = [col.lower() for col in columns]
    return "title" in lowered and "content" in lowered


def _is_faq_format(columns: list[str]) -> bool:
    lowered = [col.lower() for col in columns]
    return "question" in lowered and "answer" in lowered


# ── Smart column detection ────────────────────────────────────────────────────

def _score_column(values: list[str]) -> float:
    """Score a column for likelihood of being the main text content."""
    if not values:
        return 0.0

    non_empty = [v for v in values if v]
    if not non_empty:
        return 0.0

    # % cells that are strings (non-numeric)
    numeric_re = re.compile(r"^\d+([.,]\d+)?$")
    string_ratio = sum(1 for v in non_empty if not numeric_re.match(v)) / len(non_empty)

    # Average length (capped at 500 to avoid preferring blobs)
    avg_len = min(sum(len(v) for v in non_empty) / len(non_empty), 500) / 500.0

    # Unique ratio
    unique_ratio = len(set(non_empty)) / len(non_empty)

    score = string_ratio * 0.4 + avg_len * 0.4 + unique_ratio * 0.2

    # Penalty for JSON/HTML blobs
    blob_count = sum(1 for v in non_empty if _BLOB_RE.search(v))
    blob_ratio = blob_count / len(non_empty)
    score -= blob_ratio * 0.5

    # Penalty if too many numeric-only cells
    num_count = sum(1 for v in non_empty if numeric_re.match(v))
    score -= (num_count / len(non_empty)) * 0.3

    return max(score, 0.0)


def _detect_content_column(columns: list[str], rows: list[dict[str, Any]]) -> str | None:
    """Return the column name most likely to be the main content field."""
    best_col, best_score = None, -1.0
    for col in columns:
        values = [_normalize_value(row.get(col, "")) for row in rows]
        score = _score_column(values)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col


def _build_kb_text(row: dict[str, Any], columns: list[str]) -> tuple[str, dict[str, str]]:
    lower_map = {col.lower(): col for col in columns}

    title = _normalize_value(row.get(lower_map.get("title", ""), ""))
    content = _normalize_value(row.get(lower_map.get("content", ""), ""))
    keywords = _normalize_value(row.get(lower_map.get("keywords", ""), ""))

    parts = []
    if title:
        parts.append(f"{title}:")
    if content:
        parts.append(content)
    if keywords:
        parts.append(f"Keywords: {keywords}")

    metadata: dict[str, str] = {}
    for col in _KB_META_COLS:
        source_col = lower_map.get(col)
        if not source_col:
            continue
        value = _normalize_value(row.get(source_col, ""))
        if value:
            metadata[col] = value

    return " ".join(parts).strip(), metadata


def _build_faq_text(row: dict[str, Any], columns: list[str]) -> tuple[str, dict[str, str]]:
    lower_map = {col.lower(): col for col in columns}

    question = _normalize_value(row.get(lower_map.get("question", ""), ""))
    answer = _normalize_value(row.get(lower_map.get("answer", ""), ""))
    tags = _normalize_value(row.get(lower_map.get("tags", ""), ""))
    category = _normalize_value(row.get(lower_map.get("category", ""), ""))

    parts = []
    if question:
        parts.append(f"Question: {question}")
    if answer:
        parts.append(f"Answer: {answer}")
    if tags:
        parts.append(f"Keywords: {tags}")

    metadata: dict[str, str] = {}
    if category:
        metadata["category"] = category
    if tags:
        metadata["keywords"] = tags

    return " | ".join(parts).strip(), metadata


def _build_generic_text(
    row: dict[str, Any],
    columns: list[str],
    content_col: str | None,
) -> tuple[str, dict[str, str]]:
    """
    Generic mode: use smart-detected content column as main text,
    remaining columns as metadata.
    """
    metadata: dict[str, str] = {}
    main_text = ""

    for col in columns:
        val = _normalize_value(row.get(col, ""))
        if not val:
            continue
        if col == content_col:
            main_text = val
        else:
            metadata[col] = val

    # Fallback: join all if we couldn't isolate content
    if not main_text:
        parts = [f"{col}: {_normalize_value(row.get(col, ''))}" for col in columns if _normalize_value(row.get(col, ""))]
        main_text = " | ".join(parts)

    return main_text.strip(), metadata


def _rows_to_records(
    rows: list[dict[str, Any]],
    columns: list[str],
    base_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert parsed rows to records list using KB/FAQ/generic logic."""
    records: list[dict[str, Any]] = []

    kb = _is_kb_format(columns)
    faq = _is_faq_format(columns)
    content_col = None if (kb or faq) else _detect_content_column(columns, rows)

    for row_index, row in enumerate(rows, start=2):
        if kb:
            text, extra = _build_kb_text(row, columns)
        elif faq:
            text, extra = _build_faq_text(row, columns)
        else:
            text, extra = _build_generic_text(row, columns, content_col)

        text = text.strip()
        if not text:
            continue

        # Attach language detection
        lang = detect_language(text)

        records.append({
            "text": text,
            "metadata": {
                **base_meta,
                "row_num": row_index,
                "columns": columns,
                "lang": lang,
                **extra,
            },
        })

    return records


# ── CSV ───────────────────────────────────────────────────────────────────────

def parse_csv(file_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] | None = None
    columns: list[str] = []
    last_err: Exception | None = None

    for encoding in _CSV_ENCODINGS:
        try:
            with file_path.open("r", encoding=encoding, newline="") as fobj:
                reader = csv.DictReader(fobj)
                columns = list(reader.fieldnames or [])
                rows = [{(k or ""): _normalize_value(v) for k, v in row.items()} for row in reader]
            logger.info("CSV decoded with encoding=%s", encoding)
            break
        except Exception as err:
            last_err = err
            continue

    if rows is None:
        raise ValueError(f"Cannot decode CSV: {last_err}")

    records = _rows_to_records(rows, columns, {})
    logger.info("Parsed CSV: %s records from %s", len(records), file_path.name)
    return records


# ── Excel ─────────────────────────────────────────────────────────────────────

def _parse_excel_with_openpyxl(file_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    workbook = load_workbook(filename=file_path, read_only=True, data_only=True)

    for sheet in workbook.worksheets:
        rows_iter = sheet.iter_rows(values_only=True)
        try:
            header_row = next(rows_iter)
        except StopIteration:
            continue

        raw_cols = [_normalize_value(cell) for cell in header_row]
        if not any(raw_cols):
            continue

        normalized_columns = [col if col else f"col_{idx+1}" for idx, col in enumerate(raw_cols)]

        raw_rows = []
        for values in rows_iter:
            row_map = {
                normalized_columns[idx]: _normalize_value(values[idx] if idx < len(values) else "")
                for idx in range(len(normalized_columns))
            }
            raw_rows.append(row_map)

        sheet_records = _rows_to_records(
            raw_rows, normalized_columns, {"sheet_name": sheet.title}
        )
        # Fix row_num offset (already set inside _rows_to_records starting at 2)
        records.extend(sheet_records)

    workbook.close()
    return records


def _parse_excel_with_pandas_fallback(file_path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception as err:
        raise ValueError(
            "Legacy .xls requires optional dependency pandas/xlrd. "
            "Please convert to .xlsx or install pandas + xlrd."
        ) from err

    records: list[dict[str, Any]] = []
    excel_file = pd.ExcelFile(file_path)
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name, dtype=str, keep_default_na=False)
        columns = list(df.columns)
        raw_rows = [{col: _normalize_value(row[col]) for col in columns} for _, row in df.iterrows()]
        records.extend(_rows_to_records(raw_rows, columns, {"sheet_name": sheet_name}))

    return records


def parse_excel(file_path: Path) -> list[dict[str, Any]]:
    try:
        records = _parse_excel_with_openpyxl(file_path)
        logger.info("Parsed Excel via openpyxl: %s records from %s", len(records), file_path.name)
        return records
    except InvalidFileException:
        records = _parse_excel_with_pandas_fallback(file_path)
        logger.info("Parsed Excel via pandas fallback: %s records from %s", len(records), file_path.name)
        return records


# ── PDF ───────────────────────────────────────────────────────────────────────

def parse_pdf(file_path: Path) -> list[dict[str, Any]]:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    records: list[dict[str, Any]] = []
    for page_index, layout in enumerate(extract_pages(str(file_path)), start=1):
        texts: list[str] = []
        for element in layout:
            if isinstance(element, LTTextContainer):
                txt = element.get_text().strip()
                if txt:
                    texts.append(txt)

        page_text = "\n".join(texts).strip()
        if not page_text:
            continue

        lang = detect_language(page_text)
        records.append({"text": page_text, "metadata": {"page_num": page_index, "lang": lang}})

    logger.info("Parsed PDF: %s pages from %s", len(records), file_path.name)
    return records


# ── HTML ──────────────────────────────────────────────────────────────────────

def parse_html(file_path: Path) -> list[dict[str, Any]]:
    from bs4 import BeautifulSoup

    content = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(content, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    body = soup.body if soup.body else soup
    text = body.get_text(separator="\n", strip=True).strip()

    if not text:
        return []

    lang = detect_language(text)
    return [{"text": text, "metadata": {"title": title, "lang": lang}}]


# ── TXT / Markdown ────────────────────────────────────────────────────────────

def parse_text(file_path: Path) -> list[dict[str, Any]]:
    content = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return []
    lang = detect_language(content)
    return [{"text": content, "metadata": {"title": file_path.stem, "lang": lang}}]


# ── JSONL / JSON ──────────────────────────────────────────────────────────────

def _json_object_to_row(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return {str(k): _normalize_value(v) for k, v in obj.items()}
    return {"value": _normalize_value(obj)}


def parse_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Parse .jsonl (one JSON object per line) or .json (list of objects)."""
    raw_objs: list[Any] = []
    last_err: Exception | None = None

    for encoding in _CSV_ENCODINGS:
        try:
            content = file_path.read_text(encoding=encoding)
            stripped = content.strip()
            if stripped.startswith("["):
                # JSON array
                raw_objs = json.loads(stripped)
            else:
                # JSONL
                raw_objs = [json.loads(line) for line in stripped.splitlines() if line.strip()]
            break
        except Exception as err:
            last_err = err
            continue

    if not raw_objs:
        logger.warning("JSONL/JSON parse returned 0 objects (last_err=%s)", last_err)
        return []

    rows = [_json_object_to_row(obj) for obj in raw_objs]
    columns = list(rows[0].keys()) if rows else []

    records = _rows_to_records(rows, columns, {})
    logger.info("Parsed JSONL/JSON: %s records from %s", len(records), file_path.name)
    return records


# ── Registry ──────────────────────────────────────────────────────────────────

PARSERS = {
    "csv": parse_csv,
    "excel": parse_excel,
    "pdf": parse_pdf,
    "html": parse_html,
    "text": parse_text,
    "jsonl": parse_jsonl,
    "json": parse_jsonl,
}


def parse_file(file_path: Path, parser_type: str) -> list[dict[str, Any]]:
    if parser_type == "docx":
        from app.parsers_docx import parse_docx
        return parse_docx(file_path)
    parser = PARSERS.get(parser_type)
    if not parser:
        raise ValueError(f"Unknown parser type: {parser_type}")
    return parser(file_path)
