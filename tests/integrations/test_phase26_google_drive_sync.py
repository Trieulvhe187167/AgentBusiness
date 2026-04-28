from __future__ import annotations

import app.database as database
from fastapi.testclient import TestClient

from app.integrations.google_drive import normalize_google_drive_id
from tests.conftest import admin_headers, poll_background_job, poll_jobs


class _FakeDriveClient:
    files: list[dict] = []
    contents: dict[str, bytes] = {}

    async def list_files(self, folder_id: str, *, shared_drive_id: str | None = None, recursive: bool = True):
        return list(self.files)

    async def download_file(self, item: dict):
        file_id = item["id"]
        return self.contents[file_id], item["name"], "csv"


def test_normalize_google_drive_id_accepts_folder_url():
    assert (
        normalize_google_drive_id("https://drive.google.com/drive/folders/1AbCdEfGhIJ?usp=sharing")
        == "1AbCdEfGhIJ"
    )


def _patch_sync_dependencies(monkeypatch, *, content: bytes, version: str = "1"):
    _FakeDriveClient.files = [
        {
            "id": "drive-file-1",
            "name": "pricing.csv",
            "mimeType": "text/csv",
            "md5Checksum": f"md5-{version}",
            "modifiedTime": f"2026-04-23T10:00:0{version}Z",
            "parents": ["folder-1"],
            "version": version,
            "size": str(len(content)),
        }
    ]
    _FakeDriveClient.contents = {"drive-file-1": content}
    monkeypatch.setattr("app.drive_sync.GoogleDriveClient", _FakeDriveClient)
    monkeypatch.setattr("app.ingest.embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])


def _sync_drive_source(client: TestClient, source_id: int, admin: dict[str, str]) -> dict:
    sync = client.post(
        f"/api/admin/google-drive/sources/{source_id}/sync",
        headers=admin,
    )
    assert sync.status_code == 200, sync.text
    job_payload = sync.json()
    assert job_payload["job_type"] == "google_drive_sync"
    sync_job = poll_background_job(client, job_payload["job_id"])
    sync_payload = sync_job["result"]
    assert sync_payload["status"] == "success"
    poll_jobs(client, [{"job_id": job_id} for job_id in sync_payload.get("queued_job_ids", [])])
    return sync_payload


def test_google_drive_source_can_sync_into_default_kb(isolated_client: TestClient, monkeypatch):
    _patch_sync_dependencies(
        monkeypatch,
        content=b"question,answer\nshipping,30k\n",
        version="1",
    )
    admin = admin_headers()

    create = isolated_client.post(
        "/api/admin/google-drive/sources",
        headers=admin,
        json={
            "kb_key": "default",
            "name": "Ops Drive",
            "folder_id": "folder-1",
        },
    )
    assert create.status_code == 200, create.text
    source_id = create.json()["id"]

    sync_payload = _sync_drive_source(isolated_client, source_id, admin)
    assert sync_payload["status"] == "success"
    assert sync_payload["imported_count"] == 1
    assert len(sync_payload["queued_job_ids"]) == 1

    status = isolated_client.get(
        f"/api/admin/google-drive/sources/{source_id}/status",
        headers=admin,
    )
    assert status.status_code == 200, status.text
    status_payload = status.json()
    assert status_payload["last_run"]["status"] == "success"
    assert status_payload["last_run"]["imported_count"] == 1

    kb_sources = isolated_client.get(
        "/api/kb/sources",
        params={"kb_key": "default"},
        headers=admin,
    )
    assert kb_sources.status_code == 200, kb_sources.text
    assert any(item["filename"] == "pricing.csv" for item in kb_sources.json())

    drive_file_row = database.fetch_one_sync(
        """
        SELECT sync_status, uploaded_file_id
        FROM google_drive_files
        WHERE source_id = ?
        """,
        (source_id,),
    )
    assert drive_file_row is not None
    assert drive_file_row["sync_status"] == "synced"
    assert int(drive_file_row["uploaded_file_id"]) > 0


def test_google_drive_source_creation_normalizes_folder_url(isolated_client: TestClient):
    admin = admin_headers()
    create = isolated_client.post(
        "/api/admin/google-drive/sources",
        headers=admin,
        json={
            "kb_key": "default",
            "name": "Ops Drive",
            "folder_id": "https://drive.google.com/drive/folders/folder-1?usp=sharing",
        },
    )
    assert create.status_code == 200, create.text
    assert create.json()["folder_id"] == "folder-1"


def test_force_full_google_drive_sync_creates_pending_action(isolated_client: TestClient):
    admin = admin_headers()
    create = isolated_client.post(
        "/api/admin/google-drive/sources",
        headers=admin,
        json={
            "kb_key": "default",
            "name": "Ops Drive",
            "folder_id": "folder-1",
        },
    )
    assert create.status_code == 200, create.text
    source_id = create.json()["id"]

    sync = isolated_client.post(
        f"/api/admin/google-drive/sources/{source_id}/sync?force_full=true",
        headers=admin,
    )
    assert sync.status_code == 200, sync.text
    payload = sync.json()
    assert payload["status"] == "draft"
    assert payload["action_type"] == "sync_google_drive_source"
    assert payload["payload"] == {"force_full": True, "source_id": source_id}


def test_google_drive_resync_updates_same_uploaded_file_record(isolated_client: TestClient, monkeypatch):
    admin = admin_headers()
    _patch_sync_dependencies(
        monkeypatch,
        content=b"question,answer\nshipping,30k\n",
        version="1",
    )

    create = isolated_client.post(
        "/api/admin/google-drive/sources",
        headers=admin,
        json={
            "kb_key": "default",
            "name": "Ops Drive",
            "folder_id": "folder-1",
        },
    )
    assert create.status_code == 200, create.text
    source_id = create.json()["id"]

    _sync_drive_source(isolated_client, source_id, admin)

    first_row = database.fetch_one_sync(
        """
        SELECT uploaded_file_id
        FROM google_drive_files
        WHERE source_id = ? AND drive_file_id = 'drive-file-1'
        """,
        (source_id,),
    )
    assert first_row is not None
    first_file_id = int(first_row["uploaded_file_id"])

    _patch_sync_dependencies(
        monkeypatch,
        content=b"question,answer\nshipping,35k\n",
        version="2",
    )

    second_payload = _sync_drive_source(isolated_client, source_id, admin)
    assert second_payload["changed_count"] == 1

    second_row = database.fetch_one_sync(
        """
        SELECT uploaded_file_id, revision_id, md5_checksum
        FROM google_drive_files
        WHERE source_id = ? AND drive_file_id = 'drive-file-1'
        """,
        (source_id,),
    )
    assert second_row is not None
    assert int(second_row["uploaded_file_id"]) == first_file_id
    assert second_row["revision_id"] == "2"
    assert second_row["md5_checksum"] == "md5-2"


def test_delete_google_drive_source_unlink_keeps_imported_file(isolated_client: TestClient, monkeypatch):
    admin = admin_headers()
    _patch_sync_dependencies(
        monkeypatch,
        content=b"question,answer\nshipping,30k\n",
        version="1",
    )

    create = isolated_client.post(
        "/api/admin/google-drive/sources",
        headers=admin,
        json={
            "kb_key": "default",
            "name": "Ops Drive",
            "folder_id": "folder-1",
        },
    )
    assert create.status_code == 200, create.text
    source_id = create.json()["id"]

    _sync_drive_source(isolated_client, source_id, admin)

    drive_row = database.fetch_one_sync(
        "SELECT uploaded_file_id FROM google_drive_files WHERE source_id = ?",
        (source_id,),
    )
    assert drive_row is not None
    uploaded_file_id = int(drive_row["uploaded_file_id"])

    delete_resp = isolated_client.delete(
        f"/api/admin/google-drive/sources/{source_id}?mode=unlink",
        headers=admin,
    )
    assert delete_resp.status_code == 200, delete_resp.text
    action = delete_resp.json()
    assert action["status"] == "draft"
    assert action["action_type"] == "delete_google_drive_source"
    assert database.fetch_one_sync("SELECT id FROM google_drive_sources WHERE id = ?", (source_id,)) == {"id": source_id}

    approve_resp = isolated_client.post(f"/api/admin/pending-actions/{action['id']}/approve", headers=admin)
    assert approve_resp.status_code == 200, approve_resp.text
    execute_resp = isolated_client.post(f"/api/admin/pending-actions/{action['id']}/execute", headers=admin)
    assert execute_resp.status_code == 200, execute_resp.text
    payload = poll_background_job(isolated_client, execute_resp.json()["job_id"])["result"]["result"]
    assert payload["mode"] == "unlink"
    assert payload["deleted_file_count"] == 0

    source_row = database.fetch_one_sync("SELECT id FROM google_drive_sources WHERE id = ?", (source_id,))
    assert source_row is None
    file_row = database.fetch_one_sync("SELECT id FROM uploaded_files WHERE id = ?", (uploaded_file_id,))
    assert file_row == {"id": uploaded_file_id}


def test_delete_google_drive_source_purge_removes_imported_file_when_exclusive(isolated_client: TestClient, monkeypatch):
    admin = admin_headers()
    _patch_sync_dependencies(
        monkeypatch,
        content=b"question,answer\nshipping,30k\n",
        version="1",
    )

    create = isolated_client.post(
        "/api/admin/google-drive/sources",
        headers=admin,
        json={
            "kb_key": "default",
            "name": "Ops Drive",
            "folder_id": "folder-1",
        },
    )
    assert create.status_code == 200, create.text
    source_id = create.json()["id"]

    _sync_drive_source(isolated_client, source_id, admin)

    drive_row = database.fetch_one_sync(
        "SELECT uploaded_file_id FROM google_drive_files WHERE source_id = ?",
        (source_id,),
    )
    assert drive_row is not None
    uploaded_file_id = int(drive_row["uploaded_file_id"])

    delete_resp = isolated_client.delete(
        f"/api/admin/google-drive/sources/{source_id}?mode=purge",
        headers=admin,
    )
    assert delete_resp.status_code == 200, delete_resp.text
    action = delete_resp.json()
    assert action["status"] == "draft"
    assert action["action_type"] == "delete_google_drive_source"

    approve_resp = isolated_client.post(f"/api/admin/pending-actions/{action['id']}/approve", headers=admin)
    assert approve_resp.status_code == 200, approve_resp.text
    execute_resp = isolated_client.post(f"/api/admin/pending-actions/{action['id']}/execute", headers=admin)
    assert execute_resp.status_code == 200, execute_resp.text
    payload = poll_background_job(isolated_client, execute_resp.json()["job_id"])["result"]["result"]
    assert payload["mode"] == "purge"
    assert payload["detached_file_count"] == 1
    assert payload["deleted_file_count"] == 1

    source_row = database.fetch_one_sync("SELECT id FROM google_drive_sources WHERE id = ?", (source_id,))
    assert source_row is None
    file_row = database.fetch_one_sync("SELECT id FROM uploaded_files WHERE id = ?", (uploaded_file_id,))
    assert file_row is None
    mapping_row = database.fetch_one_sync("SELECT kb_id FROM kb_files WHERE file_id = ?", (uploaded_file_id,))
    assert mapping_row is None
