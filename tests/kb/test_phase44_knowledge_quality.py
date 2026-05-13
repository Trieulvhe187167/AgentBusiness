from __future__ import annotations

import json

from fastapi.testclient import TestClient

import app.main as main
from app.database import execute_sync, fetch_one_sync, utcnow_iso
from tests.conftest import admin_headers, attach_file, configure_test_env, fetch_default_kb, insert_file, mark_ingested


def _insert_feedback_chat_log(*, kb_id: int, filename: str, mode: str = "fallback", rating: str = "down") -> int:
    now = utcnow_iso()
    chat_log_id = int(
        execute_sync(
            """
            INSERT INTO chat_logs (
                session_id, request_id, user_id, roles_json, channel,
                tenant_id, org_id, kb_id, kb_key, user_message, merged_query,
                mode, top_score, answer_text, citations_json, latency_ms,
                llm_provider, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session-quality-1",
                "req-quality-1",
                "employee-1",
                json.dumps(["employee"]),
                "portal",
                None,
                None,
                kb_id,
                "default",
                "What is tuition?",
                "What is tuition?",
                mode,
                0.4,
                "No answer returned.",
                json.dumps([{"filename": filename, "chunk_id": "chunk-1"}]),
                80,
                "none",
                now,
            ),
        )
        or 0
    )
    execute_sync(
        """
        INSERT INTO chat_feedback (
            chat_log_id, request_id, rating, reason_code, comment,
            created_by_user_id, roles_json, channel, tenant_id, org_id,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chat_log_id,
            "req-quality-1",
            rating,
            "not_helpful",
            "Needs fresher content.",
            "employee-1",
            json.dumps(["employee"]),
            "portal",
            None,
            None,
            now,
            now,
        ),
    )
    return chat_log_id


def test_quality_report_scores_lifecycle_and_feedback_signals(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    kb = fetch_default_kb()
    file_id = insert_file("tuition.csv")
    attach_file(kb.id, file_id)
    mark_ingested(kb.id, file_id)
    file_row = fetch_one_sync("SELECT filename FROM uploaded_files WHERE id = ?", (file_id,))
    assert file_row
    _insert_feedback_chat_log(kb_id=kb.id, filename=file_row["filename"])

    with TestClient(main.app) as client:
        report = client.get(f"/api/kbs/{kb.id}/quality", headers=admin_headers())
        updated = client.patch(
            f"/api/kbs/{kb.id}/files/{file_id}/lifecycle",
            headers=admin_headers(user_id="reviewer-1"),
            json={"lifecycle_status": "published"},
        )
        refreshed = client.get(f"/api/kbs/{kb.id}/quality", headers=admin_headers())

    assert report.status_code == 200, report.text
    payload = report.json()
    assert payload["total_files"] == 1
    assert payload["draft_count"] == 1
    assert payload["stale_count"] == 1
    item = payload["items"][0]
    assert item["feedback_down"] == 1
    assert item["fallback_count"] == 1
    assert item["citation_count"] == 1
    assert item["quality_score"] < 70
    assert item["stale_reason"] == "Document is still in draft lifecycle."
    assert item["stale_detected_at"]

    assert updated.status_code == 200, updated.text
    assert updated.json()["lifecycle_status"] == "published"
    assert updated.json()["reviewed_by_user_id"] == "reviewer-1"
    assert updated.json()["stale"] is False

    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["published_count"] == 1
    assert refreshed.json()["stale_count"] == 0


def test_kb_file_diff_detects_changed_drive_source(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    kb = fetch_default_kb()
    file_id = insert_file("drive-policy.csv")
    attach_file(kb.id, file_id)
    mark_ingested(kb.id, file_id)
    now = utcnow_iso()
    source_id = int(
        execute_sync(
            """
            INSERT INTO google_drive_sources (
                kb_id, name, folder_id, shared_drive_id, recursive,
                include_patterns_json, exclude_patterns_json, supported_mime_types_json,
                delete_policy, status, tenant_id, org_id, created_by_user_id,
                last_sync_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kb.id,
                "Ops Drive",
                "folder-1",
                None,
                1,
                json.dumps(["*.csv"]),
                json.dumps([]),
                json.dumps([]),
                "detach",
                "active",
                None,
                None,
                "admin-1",
                now,
                now,
                now,
            ),
        )
        or 0
    )
    execute_sync(
        """
        INSERT INTO google_drive_files (
            source_id, drive_file_id, drive_parent_id, name, mime_type, export_ext,
            revision_id, md5_checksum, etag, size_bytes, modified_time,
            uploaded_file_id, sync_status, last_seen_at, last_synced_at, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_id,
            "drive-file-1",
            "folder-1",
            "drive-policy.csv",
            "text/csv",
            None,
            "rev-2",
            "md5-2",
            "etag-2",
            1024,
            "2099-01-01T00:00:00+00:00",
            file_id,
            "synced",
            now,
            now,
            "{}",
        ),
    )

    with TestClient(main.app) as client:
        lifecycle = client.patch(
            f"/api/kbs/{kb.id}/files/{file_id}/lifecycle",
            headers=admin_headers(),
            json={"lifecycle_status": "published"},
        )
        diff = client.get(f"/api/kbs/{kb.id}/files/{file_id}/diff", headers=admin_headers())

    assert lifecycle.status_code == 200, lifecycle.text
    assert diff.status_code == 200, diff.text
    payload = diff.json()
    assert payload["has_drive_source"] is True
    assert payload["changed"] is True
    assert payload["reason"] == "Drive file changed after last ingest."


def test_review_queue_prioritizes_drive_changed_and_low_quality_docs(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    kb = fetch_default_kb()
    draft_file_id = insert_file("draft.csv")
    low_quality_file_id = insert_file("low-quality.csv")
    drive_file_id = insert_file("drive-changed.csv")
    for file_id in [draft_file_id, low_quality_file_id, drive_file_id]:
        attach_file(kb.id, file_id)
        mark_ingested(kb.id, file_id)
    low_quality_row = fetch_one_sync("SELECT filename FROM uploaded_files WHERE id = ?", (low_quality_file_id,))
    assert low_quality_row
    _insert_feedback_chat_log(kb_id=kb.id, filename=low_quality_row["filename"], rating="down", mode="fallback")

    now = utcnow_iso()
    source_id = int(
        execute_sync(
            """
            INSERT INTO google_drive_sources (
                kb_id, name, folder_id, shared_drive_id, recursive,
                include_patterns_json, exclude_patterns_json, supported_mime_types_json,
                delete_policy, status, tenant_id, org_id, created_by_user_id,
                last_sync_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kb.id,
                "Ops Drive",
                "folder-2",
                None,
                1,
                json.dumps(["*.csv"]),
                json.dumps([]),
                json.dumps([]),
                "detach",
                "active",
                None,
                None,
                "admin-1",
                now,
                now,
                now,
            ),
        )
        or 0
    )
    execute_sync(
        """
        INSERT INTO google_drive_files (
            source_id, drive_file_id, drive_parent_id, name, mime_type, export_ext,
            revision_id, md5_checksum, etag, size_bytes, modified_time,
            uploaded_file_id, sync_status, last_seen_at, last_synced_at, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_id,
            "drive-file-2",
            "folder-2",
            "drive-changed.csv",
            "text/csv",
            None,
            "rev-2",
            "md5-2",
            "etag-2",
            1024,
            "2099-01-01T00:00:00+00:00",
            drive_file_id,
            "synced",
            now,
            now,
            "{}",
        ),
    )

    with TestClient(main.app) as client:
        client.patch(f"/api/kbs/{kb.id}/files/{drive_file_id}/lifecycle", headers=admin_headers(), json={"lifecycle_status": "published"})
        queue = client.get(f"/api/kbs/{kb.id}/review-queue", headers=admin_headers())
        drive_only = client.get(f"/api/kbs/{kb.id}/review-queue?issue_type=drive_changed", headers=admin_headers())

    assert queue.status_code == 200, queue.text
    payload = queue.json()
    assert payload["total"] >= 3
    assert payload["drive_changed_count"] == 1
    assert payload["low_quality_count"] >= 1
    assert payload["draft_count"] >= 1
    assert payload["items"][0]["issue_type"] == "drive_changed"
    assert payload["items"][0]["priority"] == "P1"

    assert drive_only.status_code == 200, drive_only.text
    assert drive_only.json()["total"] == 1
    assert drive_only.json()["items"][0]["file_id"] == drive_file_id
