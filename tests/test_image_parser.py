from __future__ import annotations

from app.chunker import chunk_records
from app.parsers import parse_file, parse_image


def test_parse_image_uses_ocr_and_returns_image_metadata(tmp_path, monkeypatch):
    from PIL import Image

    image_path = tmp_path / "policy.png"
    Image.new("RGB", (80, 40), color="white").save(image_path)
    monkeypatch.setattr("app.parsers._ocr_image_text", lambda _image: "Warranty is 12 months.")

    records = parse_image(image_path)

    assert records == [
        {
            "text": "Warranty is 12 months.",
            "metadata": {
                "title": "policy",
                "lang": "en",
                "image_format": "PNG",
                "image_width": 80,
                "image_height": 40,
                "ocr_extraction": "image_ocr",
            },
        }
    ]


def test_parse_file_dispatches_image_parser(tmp_path, monkeypatch):
    from PIL import Image

    image_path = tmp_path / "scan.jpg"
    Image.new("RGB", (60, 30), color="white").save(image_path)
    monkeypatch.setattr("app.parsers._ocr_image_text", lambda _image: "Invoice total: 1000000")

    records = parse_file(image_path, "image")

    assert records[0]["text"] == "Invoice total: 1000000"
    assert records[0]["metadata"]["image_format"] == "JPEG"


def test_image_metadata_survives_chunking():
    chunks = chunk_records(
        [
            {
                "text": "Invoice total: 1000000",
                "metadata": {
                    "title": "invoice",
                    "lang": "en",
                    "image_format": "PNG",
                    "image_width": 100,
                    "image_height": 50,
                    "ocr_extraction": "image_ocr",
                },
            }
        ],
        kb_id=1,
        source_id="10",
        filename="invoice.png",
        file_type="image",
        file_hash="hash",
        kb_version="v1",
        ingest_signature="sig",
    )

    assert chunks[0]["title"] == "invoice"
    assert chunks[0]["image_format"] == "PNG"
    assert chunks[0]["image_width"] == 100
    assert chunks[0]["image_height"] == 50
    assert chunks[0]["ocr_extraction"] == "image_ocr"
