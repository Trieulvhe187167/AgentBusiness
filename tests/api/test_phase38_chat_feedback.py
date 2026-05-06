from __future__ import annotations

import json

from fastapi.testclient import TestClient

import app.main as main
from app.database import execute_sync, fetch_one_sync, utcnow_iso
from tests.conftest import admin_headers, auth_headers, configure_test_env


def _insert_chat_log(
    *,
    request_id: str,
    user_id: str | None = "user-1",
    kb_id: int | None = 1,
    kb_key: str | None = "default",
) -> int:
    now = utcnow_iso()
    return int(
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
                f"session-{request_id}",
                request_id,
                user_id,
                json.dumps(["customer"]),
                "web",
                "tenant-a",
                "org-a",
                kb_id,
                kb_key,
                "What is shipping?",
                "What is shipping?",
                "answer",
                0.9,
                "Shipping takes 3 days.",
                "[]",
                120,
                "none",
                now,
            ),
        )
        or 0
    )


def test_user_submits_feedback_by_request_id_and_can_update(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    chat_log_id = _insert_chat_log(request_id="req-feedback-1", user_id="user-1")

    with TestClient(main.app) as client:
        created = client.post(
            "/api/feedback/chat",
            headers=auth_headers(user_id="user-1", roles=["customer"], channel="web"),
            json={"request_id": "req-feedback-1", "rating": "up", "reason_code": "good_answer"},
        )
        updated = client.post(
            "/api/feedback/chat",
            headers=auth_headers(user_id="user-1", roles=["customer"], channel="web"),
            json={"chat_log_id": chat_log_id, "rating": "down", "reason_code": "not_helpful", "comment": "Too generic"},
        )

    assert created.status_code == 200, created.text
    assert created.json()["rating"] == "up"
    assert updated.status_code == 200, updated.text
    assert updated.json()["rating"] == "down"
    assert updated.json()["comment"] == "Too generic"

    count = fetch_one_sync("SELECT COUNT(*) AS total FROM chat_feedback WHERE chat_log_id = ?", (chat_log_id,))
    assert count == {"total": 1}


def test_user_cannot_feedback_another_users_chat_log(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _insert_chat_log(request_id="req-feedback-2", user_id="user-1")

    with TestClient(main.app) as client:
        response = client.post(
            "/api/feedback/chat",
            headers=auth_headers(user_id="user-2", roles=["customer"], channel="web"),
            json={"request_id": "req-feedback-2", "rating": "up"},
        )

    assert response.status_code == 403, response.text
    assert "another user's chat log" in response.json()["detail"]


def test_admin_can_feedback_any_chat_log_and_anonymous_log(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    user_log_id = _insert_chat_log(request_id="req-feedback-admin", user_id="user-1")
    anonymous_log_id = _insert_chat_log(request_id="req-feedback-anon", user_id=None, kb_id=None, kb_key=None)

    with TestClient(main.app) as client:
        user_feedback = client.post(
            "/api/feedback/chat",
            headers=admin_headers(),
            json={"chat_log_id": user_log_id, "rating": "down", "reason_code": "wrong_answer"},
        )
        anonymous_feedback = client.post(
            "/api/feedback/chat",
            headers=admin_headers(),
            json={"chat_log_id": anonymous_log_id, "rating": "up", "reason_code": "good_answer"},
        )

    assert user_feedback.status_code == 200, user_feedback.text
    assert anonymous_feedback.status_code == 200, anonymous_feedback.text
    assert anonymous_feedback.json()["created_by_user_id"] == "admin-1"


def test_non_admin_cannot_feedback_anonymous_chat_log(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _insert_chat_log(request_id="req-feedback-anon-user", user_id=None)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/feedback/chat",
            headers=auth_headers(user_id="user-1", roles=["customer"], channel="web"),
            json={"request_id": "req-feedback-anon-user", "rating": "up"},
        )

    assert response.status_code == 403, response.text
    assert "anonymous chat logs" in response.json()["detail"]


def test_feedback_validation_and_missing_target(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        missing_selector = client.post(
            "/api/feedback/chat",
            headers=auth_headers(user_id="user-1", roles=["customer"], channel="web"),
            json={"rating": "up"},
        )
        missing_log = client.post(
            "/api/feedback/chat",
            headers=auth_headers(user_id="user-1", roles=["customer"], channel="web"),
            json={"request_id": "missing-request", "rating": "up"},
        )

    assert missing_selector.status_code == 422, missing_selector.text
    assert missing_log.status_code == 404, missing_log.text


def test_admin_lists_feedback_summary_and_chat_log_aggregates(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    log_one = _insert_chat_log(request_id="req-feedback-list-1", user_id="user-1", kb_id=1, kb_key="default")
    log_two = _insert_chat_log(request_id="req-feedback-list-2", user_id="user-2", kb_id=2, kb_key="sales")

    with TestClient(main.app) as client:
        client.post(
            "/api/feedback/chat",
            headers=auth_headers(user_id="user-1", roles=["customer"], channel="web"),
            json={"chat_log_id": log_one, "rating": "up"},
        ).raise_for_status()
        client.post(
            "/api/feedback/chat",
            headers=auth_headers(user_id="user-2", roles=["customer"], channel="web"),
            json={"chat_log_id": log_two, "rating": "down", "reason_code": "missing_context"},
        ).raise_for_status()

        listed = client.get("/api/admin/feedback?limit=10", headers=admin_headers())
        down_only = client.get("/api/admin/feedback?rating=down&limit=10", headers=admin_headers())
        summary = client.get("/api/admin/feedback/summary", headers=admin_headers())
        logs = client.get("/api/admin/chat-logs?limit=10", headers=admin_headers())

    assert listed.status_code == 200, listed.text
    assert listed.json()["total"] == 2
    assert down_only.status_code == 200, down_only.text
    assert down_only.json()["total"] == 1
    assert down_only.json()["items"][0]["rating"] == "down"
    assert summary.status_code == 200, summary.text
    summary_payload = summary.json()
    assert summary_payload["total"] == 2
    assert summary_payload["up"] == 1
    assert summary_payload["down"] == 1
    assert summary_payload["positive_rate"] == 0.5
    assert len(summary_payload["by_kb"]) == 2

    assert logs.status_code == 200, logs.text
    by_request = {item["request_id"]: item for item in logs.json()}
    assert by_request["req-feedback-list-1"]["feedback_up"] == 1
    assert by_request["req-feedback-list-1"]["feedback_down"] == 0
    assert by_request["req-feedback-list-2"]["feedback_up"] == 0
    assert by_request["req-feedback-list-2"]["feedback_down"] == 1
