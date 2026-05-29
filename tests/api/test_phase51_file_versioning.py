from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.database import fetch_all_sync, fetch_one_sync
from app.upload_service import import_content_to_uploaded_file
from tests.conftest import admin_headers, poll_jobs, run


def test_upload_creates_initial_file_version_with_snapshot(isolated_client: TestClient):
    response = isolated_client.post(
        "/api/upload",
        files={"file": ("policy.csv", b"topic,answer\nshipping,free\n", "text/csv")},
        headers=admin_headers(user_id="admin-version"),
    )

    assert response.status_code == 200, response.text
    file_id = response.json()["file"]["id"]

    versions = isolated_client.get(f"/api/files/{file_id}/versions", headers=admin_headers())

    assert versions.status_code == 200, versions.text
    payload = versions.json()
    assert payload["current_version"] == 1
    assert payload["versions"][0]["version_number"] == 1
    assert payload["versions"][0]["is_current"] is True
    assert payload["versions"][0]["has_snapshot"] is True


def test_existing_file_update_snapshots_previous_and_creates_next_version(isolated_client: TestClient):
    first = isolated_client.post(
        "/api/upload",
        files={"file": ("policy.csv", b"topic,answer\nshipping,free\n", "text/csv")},
        headers=admin_headers(user_id="admin-version"),
    )
    assert first.status_code == 200, first.text
    file_id = int(first.json()["file"]["id"])

    run(
        import_content_to_uploaded_file(
            filename="policy.csv",
            content=b"topic,answer\nshipping,paid\n",
            existing_file_id=file_id,
            owner_user_id="admin-version",
        )
    )

    rows = fetch_all_sync(
        """
        SELECT version_number, file_hash, snapshot_path
        FROM file_versions
        WHERE file_id = ?
        ORDER BY version_number ASC
        """,
        (file_id,),
    )

    assert [row["version_number"] for row in rows] == [1, 2]
    assert rows[0]["file_hash"] != rows[1]["file_hash"]
    assert rows[0]["snapshot_path"]
    assert rows[1]["snapshot_path"]

    versions = isolated_client.get(f"/api/files/{file_id}/versions", headers=admin_headers())
    assert versions.status_code == 200, versions.text
    payload = versions.json()
    assert payload["current_version"] == 2
    assert [item["version_number"] for item in payload["versions"]] == [2, 1]


def test_admin_replace_file_content_endpoint_creates_next_version(isolated_client: TestClient):
    admin = admin_headers(user_id="admin-version")
    first = isolated_client.post(
        "/api/upload",
        files={"file": ("policy.csv", b"topic,answer\nshipping,free\n", "text/csv")},
        headers=admin,
    )
    assert first.status_code == 200, first.text
    file_id = int(first.json()["file"]["id"])

    replaced = isolated_client.post(
        f"/api/files/{file_id}/content",
        files={"file": ("policy.csv", b"topic,answer\nshipping,paid\n", "text/csv")},
        headers=admin,
    )
    assert replaced.status_code == 200, replaced.text
    assert replaced.json()["file"]["id"] == file_id

    versions = isolated_client.get(f"/api/files/{file_id}/versions", headers=admin)
    assert versions.status_code == 200, versions.text
    payload = versions.json()
    assert payload["current_version"] == 2
    assert [item["version_number"] for item in payload["versions"]] == [2, 1]


def test_ingest_marks_current_file_version_active(isolated_client: TestClient):
    admin = admin_headers()
    upload = isolated_client.post(
        "/api/upload",
        files={"file": ("policy.csv", b"topic,answer\nshipping,free\n", "text/csv")},
        headers=admin,
    )
    assert upload.status_code == 200, upload.text
    file_id = int(upload.json()["file"]["id"])

    kb = isolated_client.get("/api/kbs/default", headers=admin).json()
    ingest = isolated_client.post(f"/api/kbs/{kb['id']}/ingest", headers=admin)
    assert ingest.status_code == 200, ingest.text
    poll_jobs(isolated_client, ingest.json()["jobs"])

    version = fetch_one_sync(
        "SELECT id, chunk_count, ingest_signature FROM file_versions WHERE file_id = ?",
        (file_id,),
    )
    assert version is not None
    assert int(version["chunk_count"]) > 0
    assert version["ingest_signature"]

    active = fetch_one_sync(
        """
        SELECT is_active, chunk_count
        FROM file_version_ingests
        WHERE file_version_id = ?
        """,
        (version["id"],),
    )
    assert active == {"is_active": 1, "chunk_count": version["chunk_count"]}


def test_rollback_restores_previous_binary_as_new_current_version(isolated_client: TestClient):
    original_content = b"topic,answer\nshipping,free\n"
    updated_content = b"topic,answer\nshipping,paid\n"
    upload = isolated_client.post(
        "/api/upload",
        files={"file": ("policy.csv", original_content, "text/csv")},
        headers=admin_headers(user_id="admin-version"),
    )
    assert upload.status_code == 200, upload.text
    file_id = int(upload.json()["file"]["id"])

    run(
        import_content_to_uploaded_file(
            filename="policy.csv",
            content=updated_content,
            existing_file_id=file_id,
            owner_user_id="admin-version",
        )
    )

    before_rows = fetch_all_sync(
        """
        SELECT version_number, file_hash
        FROM file_versions
        WHERE file_id = ?
        ORDER BY version_number ASC
        """,
        (file_id,),
    )
    assert [row["version_number"] for row in before_rows] == [1, 2]

    response = isolated_client.post(
        f"/api/files/{file_id}/versions/1/rollback",
        json={"reingest": False, "reason": "bad update"},
        headers=admin_headers(user_id="admin-version"),
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["changed"] is True
    assert payload["jobs"] == []
    assert payload["restored_from"]["version_number"] == 1
    assert payload["restored_as"]["version_number"] == 3
    assert payload["restored_as"]["file_hash"] == before_rows[0]["file_hash"]

    file_row = fetch_one_sync("SELECT * FROM uploaded_files WHERE id = ?", (file_id,))
    assert file_row is not None
    assert file_row["file_hash"] == before_rows[0]["file_hash"]
    assert (settings.raw_upload_dir / file_row["filename"]).read_bytes() == original_content

    versions = isolated_client.get(f"/api/files/{file_id}/versions", headers=admin_headers())
    assert versions.status_code == 200, versions.text
    version_payload = versions.json()
    assert version_payload["current_version"] == 3
    assert [item["version_number"] for item in version_payload["versions"]] == [3, 2, 1]
    assert [item["is_current"] for item in version_payload["versions"]] == [True, False, False]


def test_rollback_can_reingest_restored_version(isolated_client: TestClient):
    admin = admin_headers(user_id="admin-version")
    upload = isolated_client.post(
        "/api/upload",
        files={"file": ("policy.csv", b"topic,answer\nshipping,free\n", "text/csv")},
        headers=admin,
    )
    assert upload.status_code == 200, upload.text
    file_id = int(upload.json()["file"]["id"])

    run(
        import_content_to_uploaded_file(
            filename="policy.csv",
            content=b"topic,answer\nshipping,paid\n",
            existing_file_id=file_id,
            owner_user_id="admin-version",
        )
    )

    kb = isolated_client.get("/api/kbs/default", headers=admin).json()
    ingest = isolated_client.post(f"/api/kbs/{kb['id']}/ingest", headers=admin)
    assert ingest.status_code == 200, ingest.text
    poll_jobs(isolated_client, ingest.json()["jobs"])

    response = isolated_client.post(
        f"/api/files/{file_id}/versions/1/rollback",
        json={"reingest": True, "kb_id": kb["id"]},
        headers=admin,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["changed"] is True
    assert len(payload["jobs"]) == 1
    poll_jobs(isolated_client, payload["jobs"])

    restored = fetch_one_sync(
        """
        SELECT id, version_number, chunk_count, ingest_signature
        FROM file_versions
        WHERE file_id = ?
        ORDER BY version_number DESC
        LIMIT 1
        """,
        (file_id,),
    )
    assert restored is not None
    assert restored["version_number"] == 3
    assert int(restored["chunk_count"]) > 0
    assert restored["ingest_signature"]

    active = fetch_one_sync(
        """
        SELECT file_version_id, is_active
        FROM file_version_ingests
        WHERE kb_id = ? AND file_id = ? AND is_active = 1
        """,
        (kb["id"], file_id),
    )
    assert active == {"file_version_id": restored["id"], "is_active": 1}


def test_version_diff_returns_unified_diff_for_snapshots(isolated_client: TestClient):
    upload = isolated_client.post(
        "/api/upload",
        files={"file": ("policy.csv", b"topic,answer\nshipping,free\nrefund,7 days\n", "text/csv")},
        headers=admin_headers(user_id="admin-version"),
    )
    assert upload.status_code == 200, upload.text
    file_id = int(upload.json()["file"]["id"])

    run(
        import_content_to_uploaded_file(
            filename="policy.csv",
            content=b"topic,answer\nshipping,paid\nrefund,14 days\n",
            existing_file_id=file_id,
            owner_user_id="admin-version",
        )
    )

    response = isolated_client.get(
        f"/api/files/{file_id}/versions/1/diff/2",
        headers=admin_headers(),
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["changed"] is True
    assert payload["from_version"]["version_number"] == 1
    assert payload["to_version"]["version_number"] == 2
    assert payload["additions"] >= 1
    assert payload["deletions"] >= 1
    assert payload["truncated"] is False
    diff_text = "\n".join(payload["diff_lines"])
    assert "-shipping,free" in diff_text
    assert "+shipping,paid" in diff_text
    assert "-refund,7 days" in diff_text
    assert "+refund,14 days" in diff_text


def test_version_diff_returns_conflict_when_snapshot_is_not_retained(isolated_client: TestClient):
    upload = isolated_client.post(
        "/api/upload",
        files={"file": ("policy.csv", b"topic,answer\nshipping,free\n", "text/csv")},
        headers=admin_headers(user_id="admin-version"),
    )
    assert upload.status_code == 200, upload.text
    file_id = int(upload.json()["file"]["id"])

    run(
        import_content_to_uploaded_file(
            filename="policy.csv",
            content=b"topic,answer\nshipping,paid\n",
            existing_file_id=file_id,
            owner_user_id="admin-version",
        )
    )
    snapshot = fetch_one_sync(
        "SELECT snapshot_path FROM file_versions WHERE file_id = ? AND version_number = 1",
        (file_id,),
    )
    assert snapshot and snapshot["snapshot_path"]
    Path(snapshot["snapshot_path"]).unlink()

    response = isolated_client.get(
        f"/api/files/{file_id}/versions/1/diff/2",
        headers=admin_headers(),
    )

    assert response.status_code == 409, response.text
    assert "snapshot" in response.json()["detail"].lower()
