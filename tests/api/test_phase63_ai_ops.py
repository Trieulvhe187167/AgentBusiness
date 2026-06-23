from __future__ import annotations

import json

from fastapi.testclient import TestClient

import app.main as main
import app.rag as rag
from app.database import execute_sync, utcnow_iso
from tests.conftest import admin_headers, auth_headers, configure_test_env


def _insert_ai_ops_chat(
    request_id: str,
    *,
    user_message: str = "Where is my order?",
    answer_text: str = "Your order is on the way.",
    latency_ms: int = 100,
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
                "customer@example.com",
                json.dumps(["customer"]),
                "web",
                "tenant-secret",
                "org-secret",
                1,
                "default",
                user_message,
                user_message,
                "answer",
                0.82,
                answer_text,
                "[]",
                latency_ms,
                "openai",
                llm_input_tokens,
                llm_output_tokens,
                llm_input_tokens + llm_output_tokens,
                llm_cached_tokens,
                now,
            ),
        )
        or 0
    )


def _seed_ai_ops_summary_data() -> int:
    now = utcnow_iso()
    chat_id = _insert_ai_ops_chat(
        "aiops-req-1",
        latency_ms=4200,
        llm_input_tokens=2000,
        llm_output_tokens=120,
        llm_cached_tokens=100,
    )
    _insert_ai_ops_chat("aiops-req-2", latency_ms=100)
    for index, status in enumerate(["success", "error"]):
        execute_sync(
            """
            INSERT INTO tool_audit_logs (
                tool_call_id, request_id, session_id, user_id, roles_json, channel,
                tenant_id, org_id, kb_id, kb_key, tool_name, args_json, result_summary,
                tool_status, latency_ms, error_message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"aiops-tool-{index}",
                "aiops-req-1",
                "session-aiops",
                "customer@example.com",
                json.dumps(["customer"]),
                "web",
                "tenant-secret",
                "org-secret",
                1,
                "default",
                "search_kb",
                "{}",
                "ok" if status == "success" else "boom",
                status,
                50,
                None if status == "success" else "boom",
                now,
            ),
        )
    execute_sync(
        """
        INSERT INTO pending_actions (
            action_type, risk_level, status, title, summary, payload_json,
            created_by_user_id, tenant_id, org_id, kb_id, kb_key,
            created_at, updated_at
        ) VALUES (
            'send_email_reply', 'high', 'draft', 'Reply', 'Needs approval', '{}',
            'agent', 'tenant-secret', 'org-secret', 1, 'default',
            datetime('now', '-3 hours'), datetime('now', '-3 hours')
        )
        """
    )
    execute_sync(
        """
        INSERT INTO agent_eval_runs (
            name, status, source, kb_id, kb_key, period_days, sample_size,
            pass_count, warn_count, fail_count, avg_score, config_json,
            metrics_json, comparison_json, gate_status, created_by_user_id,
            created_at, completed_at
        ) VALUES (?, 'completed', 'golden_dataset', 1, 'default', 7, 10, 5, 2, 3, 68.5, '{}', ?, ?, 'failed', 'admin-1', ?, ?)
        """,
        (
            "AI Ops gate",
            json.dumps({"recall_at_k": 0.72}),
            json.dumps({"regressions": [{"metric": "recall_at_k"}]}),
            now,
            now,
        ),
    )
    return chat_id


def test_admin_ai_ops_summary_returns_alerts_and_eval_trend(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _seed_ai_ops_summary_data()

    with TestClient(main.app) as client:
        response = client.get("/api/admin/ai-ops/summary?days=7&kb_id=1", headers=admin_headers())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "critical"
    assert payload["cost"]["input_tokens"] == 2000
    assert payload["cost"]["cached_input_tokens"] == 100
    assert payload["cost"]["cache_reuse_rate"] == 0.05
    assert payload["cost"]["billable_input_token_estimate"] == 1900
    assert payload["latency"]["p95_ms"] == 4200
    assert payload["tooling"]["calls"] == 2
    assert payload["tooling"]["failures"] == 1
    assert payload["approvals"]["open_count"] == 1
    assert payload["eval_trend"][0]["gate_status"] == "failed"
    assert {"latency_p95_high", "tool_error_budget", "eval_gate_failed", "approval_backlog", "cache_reuse_low"} <= {
        item["code"] for item in payload["alerts"]
    }


def test_admin_ai_ops_replay_chat_log_is_retrieval_only_and_redacted(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    chat_id = _insert_ai_ops_chat(
        "aiops-replay",
        user_message="Can you email john@example.com about order 12345678?",
        answer_text="Sure, john@example.com is the contact.",
    )
    captured = {}

    def fake_retrieve(query, top_k=None, *, kb_id=None, kb_key=None, auth_context=None, runtime_context=None):
        captured.update(
            {
                "query": query,
                "top_k": top_k,
                "kb_id": kb_id,
                "kb_key": kb_key,
                "auth_context": auth_context,
                "runtime_context": runtime_context,
            }
        )
        return [
            {
                "similarity": 0.91,
                "retrieval_score": 0.91,
                "final_score": 0.91,
                "filename": "orders.csv",
                "text": "Contact john@example.com for order updates.",
            }
        ]

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)

    with TestClient(main.app) as client:
        response = client.post(
            f"/api/admin/ai-ops/replay/chat-logs/{chat_id}",
            headers=admin_headers(),
            json={"mode": "retrieval_only", "top_k": 3},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["mode"] == "retrieval_only"
    assert payload["predicted_mode"] == "answer"
    assert payload["chat_log"]["user_id"] == "[redacted]"
    assert payload["chat_log"]["tenant_id"] == "[redacted]"
    assert "[redacted-email]" in payload["query"]
    assert "[redacted-email]" in payload["results"][0]["snippet"]
    assert captured["query"] == "Can you email john@example.com about order 12345678?"
    assert captured["top_k"] == 3
    assert captured["kb_id"] == 1
    assert captured["runtime_context"]["disable_corrective_rag"] is True


def test_ai_ops_endpoints_require_analytics_role(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        summary = client.get(
            "/api/admin/ai-ops/summary?days=7",
            headers=auth_headers(user_id="user-1", roles=["customer"]),
        )
        replay = client.post(
            "/api/admin/ai-ops/replay/chat-logs/1",
            headers=auth_headers(user_id="user-1", roles=["customer"]),
            json={"mode": "retrieval_only"},
        )

    assert summary.status_code == 403, summary.text
    assert replay.status_code == 403, replay.text
