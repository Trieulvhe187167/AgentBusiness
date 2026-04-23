"""
File parsers for CSV/TSV/Excel/PDF/HTML/TXT/XML/JSONL/NDJSON/JSON.

Each parser returns list[dict]: {"text": str, "metadata": dict}
"""

from __future__ import annotations

import csv
import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from app.lang import detect_language

logger = logging.getLogger(__name__)

_TEXT_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

_KB_TEXT_COLS = ("title", "content")
_KB_META_COLS = ("category", "keywords", "source_url")
_FAQ_TEXT_COLS = ("question", "answer")
_FAQ_META_COLS = ("category", "tags")
_STRUCTURED_RECORD_CONTAINER_KEYS = ("data", "items", "records", "results", "rows")

_BLOB_RE = re.compile(r"[{}<>]|<html|<div|<span|<br", re.IGNORECASE)


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _read_text_with_fallbacks(file_path: Path, encodings: list[str] | None = None) -> tuple[str, str]:
    last_err: Exception | None = None
    for encoding in encodings or _TEXT_ENCODINGS:
        try:
            return file_path.read_text(encoding=encoding), encoding
        except Exception as err:
            last_err = err
    raise ValueError(f"Cannot decode text file {file_path.name}: {last_err}")


def _collect_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_str = str(key)
            if key_str in seen:
                continue
            seen.add(key_str)
            columns.append(key_str)
    return columns


def _is_kb_format(columns: list[str]) -> bool:
    lowered = [col.lower() for col in columns]
    return "title" in lowered and "content" in lowered


def _is_faq_format(columns: list[str]) -> bool:
    lowered = [col.lower() for col in columns]
    return "question" in lowered and "answer" in lowered


def _score_column(values: list[str]) -> float:
    if not values:
        return 0.0

    non_empty = [value for value in values if value]
    if not non_empty:
        return 0.0

    numeric_re = re.compile(r"^\d+([.,]\d+)?$")
    string_ratio = sum(1 for value in non_empty if not numeric_re.match(value)) / len(non_empty)
    avg_len = min(sum(len(value) for value in non_empty) / len(non_empty), 500) / 500.0
    unique_ratio = len(set(non_empty)) / len(non_empty)

    score = string_ratio * 0.4 + avg_len * 0.4 + unique_ratio * 0.2

    blob_count = sum(1 for value in non_empty if _BLOB_RE.search(value))
    score -= (blob_count / len(non_empty)) * 0.5

    num_count = sum(1 for value in non_empty if numeric_re.match(value))
    score -= (num_count / len(non_empty)) * 0.3

    return max(score, 0.0)


def _detect_content_column(columns: list[str], rows: list[dict[str, Any]]) -> str | None:
    best_col: str | None = None
    best_score = -1.0
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

    parts: list[str] = []
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

    for col in columns:
        if col.lower() in _KB_TEXT_COLS or col.lower() in _KB_META_COLS:
            continue
        value = _normalize_value(row.get(col, ""))
        if value:
            metadata[col] = value

    return " ".join(parts).strip(), metadata


def _build_faq_text(row: dict[str, Any], columns: list[str]) -> tuple[str, dict[str, str]]:
    lower_map = {col.lower(): col for col in columns}

    question = _normalize_value(row.get(lower_map.get("question", ""), ""))
    answer = _normalize_value(row.get(lower_map.get("answer", ""), ""))
    tags = _normalize_value(row.get(lower_map.get("tags", ""), ""))
    category = _normalize_value(row.get(lower_map.get("category", ""), ""))

    parts: list[str] = []
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

    for col in columns:
        if col.lower() in _FAQ_TEXT_COLS or col.lower() in _FAQ_META_COLS:
            continue
        value = _normalize_value(row.get(col, ""))
        if value:
            metadata[col] = value

    return " | ".join(parts).strip(), metadata


def _build_generic_text(
    row: dict[str, Any],
    columns: list[str],
    content_col: str | None,
) -> tuple[str, dict[str, str]]:
    metadata: dict[str, str] = {}
    main_text = ""

    for col in columns:
        value = _normalize_value(row.get(col, ""))
        if not value:
            continue
        if col == content_col:
            main_text = value
        else:
            metadata[col] = value

    if not main_text:
        parts = [f"{col}: {_normalize_value(row.get(col, ''))}" for col in columns if _normalize_value(row.get(col, ""))]
        main_text = " | ".join(parts)

    return main_text.strip(), metadata


def _rows_to_records(
    rows: list[dict[str, Any]],
    columns: list[str],
    base_meta: dict[str, Any],
) -> list[dict[str, Any]]:
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

        records.append(
            {
                "text": text,
                "metadata": {
                    **base_meta,
                    "row_num": row_index,
                    "columns": columns,
                    "lang": detect_language(text),
                    **extra,
                },
            }
        )

    return records


def _sniff_csv_dialect(sample: str) -> csv.Dialect | None:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return None


def _read_delimited_rows(file_path: Path) -> tuple[list[dict[str, Any]], list[str], str, str]:
    last_err: Exception | None = None

    for encoding in _TEXT_ENCODINGS:
        try:
            with file_path.open("r", encoding=encoding, newline="") as fobj:
                sample = fobj.read(4096)
                fobj.seek(0)
                dialect = _sniff_csv_dialect(sample)
                reader = csv.DictReader(fobj, dialect=dialect) if dialect else csv.DictReader(fobj)
                columns = list(reader.fieldnames or [])
                rows = [{(key or ""): _normalize_value(value) for key, value in row.items()} for row in reader]
                delimiter = getattr(dialect, "delimiter", ",")
                return rows, columns, encoding, delimiter
        except Exception as err:
            last_err = err

    raise ValueError(f"Cannot decode delimited file: {last_err}")


def parse_csv(file_path: Path) -> list[dict[str, Any]]:
    rows, columns, encoding, delimiter = _read_delimited_rows(file_path)
    records = _rows_to_records(rows, columns, {"encoding": encoding, "delimiter": delimiter})
    logger.info(
        "Parsed delimited file: %s records from %s (encoding=%s delimiter=%r)",
        len(records),
        file_path.name,
        encoding,
        delimiter,
    )
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

        raw_cols = [_normalize_value(cell) for cell in header_row]
        if not any(raw_cols):
            continue

        normalized_columns = [col if col else f"col_{index + 1}" for index, col in enumerate(raw_cols)]
        raw_rows: list[dict[str, Any]] = []
        for values in rows_iter:
            row_map = {
                normalized_columns[index]: _normalize_value(values[index] if index < len(values) else "")
                for index in range(len(normalized_columns))
            }
            raw_rows.append(row_map)

        records.extend(_rows_to_records(raw_rows, normalized_columns, {"sheet_name": sheet.title}))

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


def parse_pdf(file_path: Path) -> list[dict[str, Any]]:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    records: list[dict[str, Any]] = []
    for page_index, layout in enumerate(extract_pages(str(file_path)), start=1):
        texts: list[str] = []
        for element in layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                if text:
                    texts.append(text)

        page_text = "\n".join(texts).strip()
        if not page_text:
            continue

        records.append({"text": page_text, "metadata": {"page_num": page_index, "lang": detect_language(page_text)}})

    logger.info("Parsed PDF: %s pages from %s", len(records), file_path.name)
    return records


def parse_html(file_path: Path) -> list[dict[str, Any]]:
    from bs4 import BeautifulSoup

    content, encoding = _read_text_with_fallbacks(file_path)
    soup = BeautifulSoup(content, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    body = soup.body if soup.body else soup
    text = body.get_text(separator="\n", strip=True).strip()
    if not text:
        return []

    return [{"text": text, "metadata": {"title": title, "lang": detect_language(text), "encoding": encoding}}]


def parse_text(file_path: Path) -> list[dict[str, Any]]:
    content, encoding = _read_text_with_fallbacks(file_path)
    content = content.strip()
    if not content:
        return []
    return [{"text": content, "metadata": {"title": file_path.stem, "lang": detect_language(content), "encoding": encoding}}]


def _flatten_scalar(value: Any) -> str:
    if isinstance(value, str):
        return _normalize_value(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return _normalize_value(value)
    return _normalize_value(json.dumps(value, ensure_ascii=False))


def _flatten_mapping(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict):
        flattened: dict[str, Any] = {}
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_mapping(value, next_prefix))
        return flattened

    if isinstance(obj, list):
        flattened: dict[str, Any] = {}
        if not obj:
            flattened[prefix or "value"] = ""
            return flattened
        for index, value in enumerate(obj):
            next_prefix = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(_flatten_mapping(value, next_prefix))
        return flattened

    return {prefix or "value": _flatten_scalar(obj)}


def _extract_structured_rows(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        return [_flatten_mapping(item) for item in payload], {}

    if not isinstance(payload, dict):
        return [_flatten_mapping(payload)], {}

    envelope = {
        key: _flatten_scalar(value)
        for key, value in payload.items()
        if not isinstance(value, (dict, list))
    }

    for key in _STRUCTURED_RECORD_CONTAINER_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            rows = []
            for item in value:
                flattened = _flatten_mapping(item)
                rows.append(
                    {
                        **{f"envelope.{item_key}": item_value for item_key, item_value in envelope.items()},
                        **flattened,
                    }
                )
            return rows, {"record_container": key}

    list_keys = [key for key, value in payload.items() if isinstance(value, list)]
    if len(list_keys) == 1:
        key = list_keys[0]
        rows = []
        for item in payload[key]:
            flattened = _flatten_mapping(item)
            rows.append(
                {
                    **{f"envelope.{item_key}": item_value for item_key, item_value in envelope.items()},
                    **flattened,
                }
            )
        return rows, {"record_container": key}

    return [_flatten_mapping(payload)], {}


def _xml_to_data(element: ET.Element) -> Any:
    children = list(element)
    text = (element.text or "").strip()

    if not children:
        if element.attrib:
            data = {f"@{key}": _normalize_value(value) for key, value in element.attrib.items()}
            if text:
                data["#text"] = _normalize_value(text)
            return data
        return _normalize_value(text)

    grouped: dict[str, list[Any]] = {}
    for child in children:
        grouped.setdefault(child.tag, []).append(_xml_to_data(child))

    data: dict[str, Any] = {f"@{key}": _normalize_value(value) for key, value in element.attrib.items()}
    for tag, values in grouped.items():
        data[tag] = values[0] if len(values) == 1 else values
    if text:
        data["#text"] = _normalize_value(text)
    return data


def parse_jsonl(file_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra_meta: dict[str, Any] = {}
    detected_mode = "jsonl"
    last_err: Exception | None = None

    for encoding in _TEXT_ENCODINGS:
        try:
            content = file_path.read_text(encoding=encoding)
            stripped = content.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                payload = json.loads(stripped)
                rows, extra_meta = _extract_structured_rows(payload)
                detected_mode = "json"
            else:
                rows = [_flatten_mapping(json.loads(line)) for line in stripped.splitlines() if line.strip()]
            records = _rows_to_records(
                rows,
                _collect_columns(rows),
                {"structured_format": detected_mode, "encoding": encoding, **extra_meta},
            )
            logger.info("Parsed JSON/JSONL: %s records from %s", len(records), file_path.name)
            return records
        except Exception as err:
            last_err = err

    raise ValueError(f"Cannot decode JSON/JSONL file {file_path.name}: {last_err}")


def parse_xml(file_path: Path) -> list[dict[str, Any]]:
    content, encoding = _read_text_with_fallbacks(file_path)
    try:
        root = ET.fromstring(content)
    except ET.ParseError as err:
        raise ValueError(f"Cannot parse XML: {err}") from err

    payload = _xml_to_data(root)
    rows, extra_meta = _extract_structured_rows(payload)
    records = _rows_to_records(
        rows,
        _collect_columns(rows),
        {"structured_format": "xml", "xml_root": root.tag, "encoding": encoding, **extra_meta},
    )
    logger.info("Parsed XML: %s records from %s", len(records), file_path.name)
    return records


PARSERS = {
    "csv": parse_csv,
    "tsv": parse_csv,
    "excel": parse_excel,
    "pdf": parse_pdf,
    "html": parse_html,
    "text": parse_text,
    "jsonl": parse_jsonl,
    "ndjson": parse_jsonl,
    "json": parse_jsonl,
    "xml": parse_xml,
}


def parse_file(file_path: Path, parser_type: str) -> list[dict[str, Any]]:
    if parser_type == "docx":
        from app.parsers_docx import parse_docx

        return parse_docx(file_path)

    parser = PARSERS.get(parser_type)
    if not parser:
        raise ValueError(f"Unknown parser type: {parser_type}")
    return parser(file_path)
