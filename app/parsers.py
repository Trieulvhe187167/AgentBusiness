"""
File parsers for CSV/Excel/PDF/HTML/TXT.
Each parser returns list[dict]: {"text": str, "metadata": dict}
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

logger = logging.getLogger(__name__)

_CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

_KB_TEXT_COLS = ("title", "content")
_KB_META_COLS = ("category", "keywords", "source_url")
_FAQ_TEXT_COLS = ("question", "answer")
_FAQ_META_COLS = ("category", "tags")


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


def parse_csv(file_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] | None = None
    columns: list[str] = []
    last_err: Exception | None = None

    for encoding in _CSV_ENCODINGS:
        try:
            with file_path.open("r", encoding=encoding, newline="") as file_obj:
                reader = csv.DictReader(file_obj)
                columns = list(reader.fieldnames or [])
                rows = [{(key or ""): _normalize_value(val) for key, val in row.items()} for row in reader]
            logger.info("CSV decoded with encoding=%s", encoding)
            break
        except Exception as err:
            last_err = err
            continue

    if rows is None:
        raise ValueError(f"Cannot decode CSV: {last_err}")

    kb = _is_kb_format(columns)
    faq = _is_faq_format(columns)
    for row_index, row in enumerate(rows, start=2):
        if kb:
            text, extra = _build_kb_text(row, columns)
        elif faq:
            text, extra = _build_faq_text(row, columns)
        else:
            parts = [f"{col}: {row.get(col, '')}" for col in columns if _normalize_value(row.get(col, ""))]
            text = " | ".join(parts)
            extra = {}

        text = text.strip()
        if not text:
            continue

        records.append(
            {
                "text": text,
                "metadata": {
                    "row_num": row_index,
                    "columns": columns,
                    **extra,
                },
            }
        )

    logger.info("Parsed CSV: %s records from %s (kb=%s, faq=%s)", len(records), file_path.name, kb, faq)
    return records


def _parse_excel_with_openpyxl(file_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    workbook = load_workbook(filename=file_path, read_only=True, data_only=True)
    for sheet in workbook.worksheets:
        rows_iter = sheet.iter_rows(values_only=True)
        try:
            header_row = next(rows_iter)
        except StopIteration:
            continue

        columns = [_normalize_value(cell) for cell in header_row]
        if not any(columns):
            continue

        normalized_columns = [col if col else f"col_{idx+1}" for idx, col in enumerate(columns)]
        kb = _is_kb_format(normalized_columns)
        faq = _is_faq_format(normalized_columns)

        for row_idx, values in enumerate(rows_iter, start=2):
            row_map = {
                normalized_columns[idx]: _normalize_value(values[idx] if idx < len(values) else "")
                for idx in range(len(normalized_columns))
            }

            if kb:
                text, extra = _build_kb_text(row_map, normalized_columns)
            elif faq:
                text, extra = _build_faq_text(row_map, normalized_columns)
            else:
                parts = [
                    f"{col}: {row_map.get(col, '')}"
                    for col in normalized_columns
                    if _normalize_value(row_map.get(col, ""))
                ]
                text = " | ".join(parts)
                extra = {}

            text = text.strip()
            if not text:
                continue

            records.append(
                {
                    "text": text,
                    "metadata": {
                        "sheet_name": sheet.title,
                        "row_num": row_idx,
                        "columns": normalized_columns,
                        **extra,
                    },
                }
            )

    workbook.close()
    return records


def _parse_excel_with_pandas_fallback(file_path: Path) -> list[dict[str, Any]]:
    """
    Optional fallback for legacy .xls files.
    Requires pandas (+ xlrd for old xls).
    """
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
        kb = _is_kb_format(columns)
        faq = _is_faq_format(columns)

        for row_idx, (_, row) in enumerate(df.iterrows(), start=2):
            row_map = {col: _normalize_value(row[col]) for col in columns}
            if kb:
                text, extra = _build_kb_text(row_map, columns)
            elif faq:
                text, extra = _build_faq_text(row_map, columns)
            else:
                parts = [f"{col}: {row_map.get(col, '')}" for col in columns if _normalize_value(row_map.get(col, ""))]
                text = " | ".join(parts)
                extra = {}

            text = text.strip()
            if not text:
                continue

            records.append(
                {
                    "text": text,
                    "metadata": {
                        "sheet_name": sheet_name,
                        "row_num": row_idx,
                        "columns": columns,
                        **extra,
                    },
                }
            )

    return records


def parse_excel(file_path: Path) -> list[dict[str, Any]]:
    try:
        records = _parse_excel_with_openpyxl(file_path)
        logger.info("Parsed Excel via openpyxl: %s records from %s", len(records), file_path.name)
        return records
    except InvalidFileException:
        # openpyxl cannot read old .xls
        records = _parse_excel_with_pandas_fallback(file_path)
        logger.info("Parsed Excel via pandas fallback: %s records from %s", len(records), file_path.name)
        return records


def parse_pdf(file_path: Path) -> list[dict[str, Any]]:
    """Parse PDF page-by-page using pdfminer.six (pure Python path)."""
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

        records.append(
            {
                "text": page_text,
                "metadata": {"page_num": page_index},
            }
        )

    logger.info("Parsed PDF: %s pages from %s", len(records), file_path.name)
    return records


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

    return [{"text": text, "metadata": {"title": title}}]


def parse_text(file_path: Path) -> list[dict[str, Any]]:
    content = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return []
    return [{"text": content, "metadata": {"title": file_path.stem}}]


PARSERS = {
    "csv": parse_csv,
    "excel": parse_excel,
    "pdf": parse_pdf,
    "html": parse_html,
    "text": parse_text,
}


def parse_file(file_path: Path, parser_type: str) -> list[dict[str, Any]]:
    parser = PARSERS.get(parser_type)
    if not parser:
        raise ValueError(f"Unknown parser type: {parser_type}")
    return parser(file_path)
