from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
from tests.conftest import admin_headers


def test_upload_rejects_fake_pdf(isolated_client: TestClient):
    response = isolated_client.post(
        "/api/upload",
        files={"file": ("fake.pdf", b"not really a pdf", "application/pdf")},
        headers=admin_headers(),
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": {
            "code": "content_mismatch",
            "message": "File content does not match '.pdf' format",
        }
    }


def test_upload_rejects_empty_file(isolated_client: TestClient):
    response = isolated_client.post(
        "/api/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
        headers=admin_headers(),
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": {
            "code": "empty_file",
            "message": "Empty file rejected",
        }
    }


def test_upload_rejects_file_over_limit(isolated_client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_size_mb", 0)

    response = isolated_client.post(
        "/api/upload",
        files={"file": ("big.csv", b"id,value\n1,alpha\n", "text/csv")},
        headers=admin_headers(),
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": {
            "code": "file_too_large",
            "message": "File too large (0.0MB). Max: 0MB",
            "meta": {"max_upload_size_mb": 0},
        }
    }


def test_upload_sanitizes_path_traversal_filename(isolated_client: TestClient):
    response = isolated_client.post(
        "/api/upload",
        files={"file": ("..\\..\\secret.csv", b"name,answer\nshipping,free\n", "text/csv")},
        headers=admin_headers(),
    )

    assert response.status_code == 200, response.text
    payload = response.json()["file"]
    assert payload["original_name"] == "secret.csv"
    assert ".." not in payload["filename"]
    assert "\\" not in payload["filename"]
    assert "/" not in payload["filename"]
