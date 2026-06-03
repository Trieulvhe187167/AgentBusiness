from __future__ import annotations

import json

from fastapi.testclient import TestClient

import app.main as main
from app.database import execute_sync, utcnow_iso
from tests.conftest import admin_headers, auth_headers, configure_test_env


def _insert_chat(
    request_id: str,
    *,
    kb_id: int = 1,
    kb_key: str = "default",
    mode: str = "answer",
    llm_input_tokens: int = 0,
    llm_output_tokens: int = 0,
    llm_cached_tokens: int = 0,
) -> int:
    now = utcnow_iso()
    return int(
        execute_sync(
            """
            INSERT INTO chat_logs (
                session_id, request_id, user_id, roles_json, channel,
                tenant_id, org_id, kb_id, kb_key, user_message, merged_query,
                mode, top_score, answer_text, citations_json, latency_ms,
                llm_provider, llm_input_tokens, llm_output_tokens, llm_total_tokens,
                llm_cached_tokens, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"session-{request_id}",
                request_id,
                "user-1",
                json.dumps(["customer"]),
                "web",
                "tenant-a",
                "org-a",
                kb_id,
                kb_key,
                "Question",
                "Question",
                mode,
                0.82,
                "Answer",
                "[]",
                150,
                "none",
                llm_input_tokens,
                llm_output_tokens,
                llm_input_tokens + llm_output_tokens,
                llm_cached_tokens,
                now,
            ),
        )
        or 0
    )


def _seed_analytics_data():
    now = utcnow_iso()
    chat_id = _insert_chat("analytics-req-1", llm_input_tokens=1400, llm_output_tokens=12, llm_cached_tokens=1024)
    _insert_chat("analytics-req-2", mode="fallback")
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
            "analytics-req-1",
            "up",
            "good_answer",
            None,
            "user-1",
            json.dumps(["customer"]),
            "web",
            "tenant-a",
            "org-a",
            now,
            now,
        ),
    )
    execute_sync(
        """
        INSERT INTO tool_audit_logs (
            tool_call_id, request_id, session_id, user_id, roles_json, channel,
            tenant_id, org_id, kb_id, kb_key, tool_name, args_json, result_summary,
            tool_status, latency_ms, error_message, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "tool-analytics-1",
            "analytics-req-1",
            "session-analytics",
            "user-1",
            json.dumps(["customer"]),
            "web",
            "tenant-a",
            "org-a",
            1,
            "default",
            "search_kb",
            "{}",
            "ok",
            "success",
            40,
            None,
            now,
        ),
    )
    execute_sync(
        """
        INSERT INTO background_jobs (
            job_id, job_type, status, payload_json, result_json, error_message,
            progress, created_by_user_id, tenant_id, org_id, kb_id, kb_key,
            attempts, max_attempts, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "BGJ-ANALYTICS-1",
            "drive_sync",
            "failed",
            "{}",
            None,
            "boom",
            0.2,
            "admin-1",
            "tenant-a",
            "org-a",
            1,
            "default",
            1,
            1,
            now,
            now,
        ),
    )
    execute_sync(
        """
        INSERT INTO pending_actions (
            action_type, risk_level, status, title, summary, payload_json,
            created_by_user_id, tenant_id, org_id, kb_id, kb_key, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "send_email_reply",
            "high",
            "draft",
            "Reply",
            "Needs approval",
            "{}",
            "agent",
            "tenant-a",
            "org-a",
            1,
            "default",
            now,
            now,
        ),
    )
    execute_sync(
        """
        INSERT INTO support_tickets (
            ticket_code, issue_type, message, contact, status, created_by_user_id,
            channel, tenant_id, org_id, kb_id, kb_key, intent, priority,
            workflow_status, sla_breached_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "TCK-ANALYTICS",
            "other",
            "Need help",
            "user@example.com",
            "open",
            "user-1",
            "web",
            "tenant-a",
            "org-a",
            1,
            "default",
            "order_status",
            "P1",
            "escalated",
            now,
            now,
            now,
        ),
    )


def test_admin_analytics_dashboard_returns_operational_metrics(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _seed_analytics_data()

    with TestClient(main.app) as client:
        response = client.get("/api/admin/analytics?days=7&kb_id=1", headers=admin_headers())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["period_days"] == 7
    assert payload["kb_id"] == 1
    assert payload["summary"]["chat_count"] == 2
    assert payload["summary"]["fallback_count"] == 1
    assert payload["summary"]["llm_input_tokens"] == 1400
    assert payload["summary"]["llm_output_tokens"] == 12
    assert payload["summary"]["llm_total_tokens"] == 1412
    assert payload["summary"]["llm_cached_tokens"] == 1024
    assert payload["summary"]["llm_cached_input_rate"] == 0.7314
    assert payload["summary"]["feedback_up"] == 1
    assert payload["summary"]["tool_calls"] == 1
    assert payload["summary"]["background_jobs_failed"] == 1
    assert payload["summary"]["pending_actions_open"] == 1
    assert payload["summary"]["support_tickets_escalated"] == 1
    assert len(payload["timeseries"]) == 7
    assert payload["top_tools"][0]["key"] == "search_kb"


def test_analytics_dashboard_requires_admin(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        response = client.get("/api/admin/analytics?days=7", headers=auth_headers(user_id="user-1", roles=["customer"]))

    assert response.status_code == 403, response.text
