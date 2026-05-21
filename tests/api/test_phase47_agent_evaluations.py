from __future__ import annotations

import json

from fastapi.testclient import TestClient

import app.main as main
from app.database import execute_sync, utcnow_iso
from tests.conftest import admin_headers, auth_headers, configure_test_env


def _insert_eval_chat(
    request_id: str,
    *,
    mode: str = "answer",
    top_score: float | None = 0.82,
    answer_text: str = "This is a grounded answer with enough detail.",
    citations_json: str = '[{"source":"kb"}]',
    rating: str | None = None,
) -> int:
    now = utcnow_iso()
    chat_id = int(
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
                "user-1",
                json.dumps(["employee"]),
                "web",
                "tenant-a",
                "org-a",
                1,
                "default",
                f"Question {request_id}",
                f"Question {request_id}",
                mode,
                top_score,
                answer_text,
                citations_json,
                100,
                "none",
                now,
            ),
        )
        or 0
    )
    if rating:
        execute_sync(
            """
            INSERT INTO chat_feedback (
                chat_log_id, request_id, rating, reason_code, comment,
                created_by_user_id, roles_json, channel, tenant_id, org_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                request_id,
                rating,
                "not_helpful" if rating == "down" else "good_answer",
                None,
                "user-1",
                json.dumps(["employee"]),
                "web",
                "tenant-a",
                "org-a",
                now,
                now,
            ),
        )
    return chat_id


def test_admin_can_create_list_and_get_agent_eval_run(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _insert_eval_chat("eval-pass", rating="up")
    _insert_eval_chat(
        "eval-fail",
        mode="fallback",
        top_score=0.1,
        answer_text="",
        citations_json="[]",
        rating="down",
    )

    with TestClient(main.app) as client:
        created = client.post(
            "/api/admin/evaluations/runs",
            headers=admin_headers(),
            json={"name": "Smoke eval", "days": 7, "limit": 10, "kb_id": 1},
        )
        listed = client.get("/api/admin/evaluations/runs?limit=10", headers=admin_headers())
        detail = client.get(f"/api/admin/evaluations/runs/{created.json()['id']}", headers=admin_headers())

    assert created.status_code == 200, created.text
    payload = created.json()
    assert payload["name"] == "Smoke eval"
    assert payload["sample_size"] == 2
    assert payload["pass_count"] == 1
    assert payload["fail_count"] == 1
    assert payload["results"][0]["verdict"] == "fail"
    assert listed.status_code == 200, listed.text
    assert listed.json()["total"] == 1
    assert detail.status_code == 200, detail.text
    assert detail.json()["config"]["scorer"] == "rule_based_v1"


def test_agent_eval_requires_analytics_role(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/admin/evaluations/runs",
            headers=auth_headers(user_id="user-1", roles=["employee"]),
            json={"days": 7},
        )

    assert response.status_code == 403, response.text
