from __future__ import annotations

import app.database as database
from app.scheduled_sync import (
    UpsertSyncScheduleInput,
    list_sync_schedules,
    run_due_sync_schedules_once,
    upsert_sync_schedule,
)
from tests.conftest import admin_headers, configure_test_env
from app.models import AuthContext


def _admin_auth() -> AuthContext:
    return AuthContext(user_id="admin-1", roles=["admin"], channel="admin")


def _create_drive_source() -> int:
    return int(
        database.execute_sync(
            """
            INSERT INTO google_drive_sources (
                kb_id, name, folder_id, recursive, include_patterns_json,
                exclude_patterns_json, supported_mime_types_json, delete_policy,
                status, created_by_user_id, created_at, updated_at
            ) VALUES (1, 'Ops Drive', 'folder-1', 1, '[]', '[]', '[]', 'detach', 'active', 'admin-1', datetime('now'), datetime('now'))
            """
        )
        or 0
    )


def test_due_drive_schedule_enqueues_once_and_skips_active_duplicate(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    source_id = _create_drive_source()
    schedule = upsert_sync_schedule(
        UpsertSyncScheduleInput(
            schedule_type="google_drive_sync",
            target_id=source_id,
            interval_seconds=60,
        ),
        auth=_admin_auth(),
    )
    database.execute_sync(
        "UPDATE sync_schedules SET next_run_at = '2000-01-01T00:00:00+00:00' WHERE id = ?",
        (schedule["id"],),
    )

    assert run_due_sync_schedules_once() == 1
    first_job = database.fetch_one_sync(
        "SELECT job_id, job_type, payload_json, status FROM background_jobs WHERE job_type = 'google_drive_sync'"
    )
    assert first_job is not None
    assert first_job["status"] == "queued"
    assert f'"source_id": {source_id}' in first_job["payload_json"]

    database.execute_sync(
        "UPDATE sync_schedules SET next_run_at = '2000-01-01T00:00:00+00:00' WHERE id = ?",
        (schedule["id"],),
    )
    assert run_due_sync_schedules_once() == 0
    total = database.fetch_one_sync("SELECT COUNT(*) AS total FROM background_jobs WHERE job_type = 'google_drive_sync'")
    assert total == {"total": 1}


def test_support_email_schedule_routes(isolated_client):
    admin = admin_headers()
    create = isolated_client.post(
        "/api/admin/sync-schedules",
        headers=admin,
        json={
            "schedule_type": "support_email_sync",
            "name": "Support Email",
            "enabled": True,
            "interval_seconds": 120,
            "payload": {"limit": 30, "unread_only": False},
        },
    )
    assert create.status_code == 200, create.text
    schedule_id = create.json()["id"]

    listed = isolated_client.get("/api/admin/sync-schedules", headers=admin)
    assert listed.status_code == 200, listed.text
    assert listed.json()["total"] >= 1

    update = isolated_client.patch(
        f"/api/admin/sync-schedules/{schedule_id}",
        headers=admin,
        json={"enabled": False, "interval_seconds": 180},
    )
    assert update.status_code == 200, update.text
    assert update.json()["enabled"] is False
    assert update.json()["interval_seconds"] == 180

    delete = isolated_client.delete(f"/api/admin/sync-schedules/{schedule_id}", headers=admin)
    assert delete.status_code == 200, delete.text
    assert delete.json()["id"] == schedule_id


def test_list_sync_schedules_service(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    upsert_sync_schedule(
        UpsertSyncScheduleInput(
            schedule_type="support_email_sync",
            interval_seconds=120,
            payload={"limit": 10, "unread_only": True},
        ),
        auth=_admin_auth(),
    )

    schedules = list_sync_schedules()
    assert schedules["total"] == 1
    assert schedules["items"][0]["schedule_type"] == "support_email_sync"
    assert schedules["items"][0]["payload"] == {"limit": 10, "unread_only": True}
