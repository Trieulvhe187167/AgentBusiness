from __future__ import annotations

from app.chunker import chunk_records
from app import parsers
from app.config import settings


def test_pdf_table_rows_are_serialized_with_headers():
    records = parsers._pdf_table_rows_to_records(
        [
            ["Product", "Quantity", "Unit price"],
            ["Air purifier", "50", "2,500,000"],
            ["Vacuum", "30", "1,800,000"],
        ],
        page_num=3,
        table_idx=2,
    )

    assert records == [
        {
            "text": "Product: Air purifier | Quantity: 50 | Unit price: 2,500,000",
            "metadata": {
                "page_num": 3,
                "table_idx": 2,
                "row_num": 2,
                "lang": "en",
                "pdf_extraction": "pdfplumber_table",
            },
        },
        {
            "text": "Product: Vacuum | Quantity: 30 | Unit price: 1,800,000",
            "metadata": {
                "page_num": 3,
                "table_idx": 2,
                "row_num": 3,
                "lang": "en",
                "pdf_extraction": "pdfplumber_table",
            },
        },
    ]


def test_parse_pdf_includes_pdfplumber_table_records_when_text_is_available(tmp_path, monkeypatch):
    pdf_path = tmp_path / "text-table.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    called = {"ocr": False}

    monkeypatch.setattr(settings, "pdf_ocr_enabled", True)
    monkeypatch.setattr(settings, "pdf_ocr_min_text_chars", 20)
    monkeypatch.setattr(parsers, "_pdf_text_pages", lambda _: {1: "This page already has enough text."})
    monkeypatch.setattr(
        parsers,
        "_pdf_table_records",
        lambda _: [
            {
                "text": "Product: Vacuum | Quantity: 30",
                "metadata": {
                    "page_num": 1,
                    "table_idx": 1,
                    "row_num": 2,
                    "lang": "en",
                    "pdf_extraction": "pdfplumber_table",
                },
            }
        ],
    )

    def _unexpected_ocr(*_args, **_kwargs):
        called["ocr"] = True
        return {}

    monkeypatch.setattr(parsers, "_ocr_pdf_pages", _unexpected_ocr)

    records = parsers.parse_pdf(pdf_path)

    assert called["ocr"] is False
    assert [record["metadata"]["pdf_extraction"] for record in records] == ["pdfminer", "pdfplumber_table"]
    assert records[1]["text"] == "Product: Vacuum | Quantity: 30"
    assert records[1]["metadata"]["table_idx"] == 1


def test_pdf_table_metadata_survives_chunking():
    chunks = chunk_records(
        [
            {
                "text": "Product: Vacuum | Quantity: 30",
                "metadata": {"page_num": 1, "table_idx": 2, "row_num": 3, "lang": "en"},
            }
        ],
        kb_id=1,
        source_id="10",
        filename="prices.pdf",
        file_type="pdf",
        file_hash="hash",
        kb_version="v1",
        ingest_signature="sig",
    )

    assert chunks[0]["page_num"] == 1
    assert chunks[0]["table_idx"] == 2
    assert chunks[0]["row_num"] == 3


def test_parse_pdf_skips_ocr_when_text_is_available(tmp_path, monkeypatch):
    pdf_path = tmp_path / "text.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    called = {"ocr": False}

    monkeypatch.setattr(settings, "pdf_ocr_enabled", True)
    monkeypatch.setattr(settings, "pdf_ocr_min_text_chars", 20)
    monkeypatch.setattr(parsers, "_pdf_text_pages", lambda _: {1: "This page already has enough text."})

    def _unexpected_ocr(*_args, **_kwargs):
        called["ocr"] = True
        return {}

    monkeypatch.setattr(parsers, "_ocr_pdf_pages", _unexpected_ocr)

    records = parsers.parse_pdf(pdf_path)

    assert called["ocr"] is False
    assert records[0]["text"] == "This page already has enough text."
    assert records[0]["metadata"]["pdf_extraction"] == "pdfminer"


def test_parse_pdf_uses_ocr_for_scanned_pages(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(settings, "pdf_ocr_enabled", True)
    monkeypatch.setattr(settings, "pdf_ocr_min_text_chars", 20)
    monkeypatch.setattr(parsers, "_pdf_text_pages", lambda _: {1: "", 2: "short"})
    monkeypatch.setattr(
        parsers,
        "_ocr_pdf_pages",
        lambda _path, page_numbers: {page: f"OCR text page {page}" for page in page_numbers},
    )

    records = parsers.parse_pdf(pdf_path)

    assert [record["metadata"]["page_num"] for record in records] == [1, 2]
    assert records[0]["text"] == "OCR text page 1"
    assert records[1]["text"] == "OCR text page 2"
    assert all(record["metadata"]["pdf_extraction"] == "pdfminer_ocr" for record in records)


def test_parse_pdf_reports_missing_ocr_dependencies_for_scan(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(settings, "pdf_ocr_enabled", True)
    monkeypatch.setattr(settings, "pdf_ocr_min_text_chars", 20)
    monkeypatch.setattr(parsers, "_pdf_text_pages", lambda _: {1: ""})

    def _raise_missing_deps(_path, _page_numbers):
        raise ValueError("PDF OCR requires optional dependencies pdf2image and pytesseract.")

    monkeypatch.setattr(parsers, "_ocr_pdf_pages", _raise_missing_deps)

    try:
        parsers.parse_pdf(pdf_path)
    except ValueError as err:
        assert "PDF OCR requires optional dependencies" in str(err)
    else:
        raise AssertionError("Expected OCR dependency error")
