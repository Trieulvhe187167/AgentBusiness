from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main
from app.agent_runs import (
    create_agent_run,
    get_agent_run,
    pause_agent_run_for_pending_action,
    record_agent_route,
    run_agent_step_once,
)
from app.database import fetch_all_sync
from app.models import AuthContext, RequestContext
from app.pending_actions import approve_pending_action, create_pending_action, execute_pending_action, reject_pending_action
from tests.conftest import admin_headers, auth_headers, configure_test_env, run


def _admin_context(request_id: str = "agent-run-test") -> RequestContext:
    return RequestContext(
        request_id=request_id,
        session_id="agent-run-session",
        auth=AuthContext(user_id="admin-1", roles=["admin"], channel="admin"),
    )


def _paused_run_with_action() -> tuple[int, int, RequestContext]:
    context = _admin_context()
    agent_run_id = create_agent_run(query="review risky action", context=context)
    record_agent_route(agent_run_id, route="tool", tool_name="send_email_reply", reason="test")
    action = create_pending_action(
        action_type="support_case_review",
        risk_level="high",
        title="Review test action",
        summary="Approval lifecycle test",
        payload={},
        context=context,
    )
    action_id = int(action["id"])
    pause_agent_run_for_pending_action(
        agent_run_id,
        pending_action_id=action_id,
        tool_name="send_email_reply",
        tool_call_id="tool-call-test",
    )
    return agent_run_id, action_id, context


def test_agent_run_auto_resumes_after_pending_action_execution(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    agent_run_id, action_id, context = _paused_run_with_action()

    approve_pending_action(action_id, auth=context.auth)
    executed = run(execute_pending_action(action_id, context=context))
    detail = get_agent_run(agent_run_id)

    assert executed["result"]["auto_resumed_agent_run_ids"] == [agent_run_id]
    assert detail["status"] == "completed"
    assert detail["pending_action_id"] == action_id
    assert {step["step_key"] for step in detail["steps"]} == {
        "route",
        "waiting_approval",
        "approval_granted",
        "approval_terminal",
    }
    terminal = next(step for step in detail["steps"] if step["step_key"] == "approval_terminal")
    assert terminal["side_effect"] is True
    assert terminal["attempt_count"] == 1
    assert terminal["idempotency_key"].endswith(f"pending-action:{action_id}:terminal:executed")


def test_pending_action_events_show_approval_and_agent_resume(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    agent_run_id, action_id, context = _paused_run_with_action()

    approve_pending_action(action_id, auth=context.auth)
    run(execute_pending_action(action_id, context=context))

    with TestClient(main.app) as client:
        response = client.get(
            f"/api/admin/pending-actions/{action_id}/events",
            headers=admin_headers(),
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["action"]["id"] == action_id
    event_types = [event["event_type"] for event in payload["events"]]
    assert "pending_action.created" in event_types
    assert "pending_action.approved" in event_types
    assert "pending_action.executed" in event_types
    assert "agent_run.approval_terminal" in event_types
    terminal = next(event for event in payload["events"] if event["event_type"] == "agent_run.approval_terminal")
    assert terminal["entity_id"] == agent_run_id
    assert terminal["details"]["side_effect"] is True
    assert terminal["details"]["attempt_count"] == 1
    assert terminal["details"]["idempotency_key"].endswith(f"pending-action:{action_id}:terminal:executed")


def test_agent_run_step_once_reuses_completed_side_effect(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    agent_run_id = create_agent_run(query="send once", context=_admin_context("step-once"))
    calls = {"count": 0}

    def send_email_once():
        calls["count"] += 1
        return {"message_id": "email-1"}

    first = run_agent_step_once(
        agent_run_id,
        step_key="send_email",
        step_type="tool",
        fn=send_email_once,
        idempotency_key=f"agent-run:{agent_run_id}:send-email",
        side_effect=True,
    )
    second = run_agent_step_once(
        agent_run_id,
        step_key="send_email",
        step_type="tool",
        fn=send_email_once,
        idempotency_key=f"agent-run:{agent_run_id}:send-email",
        side_effect=True,
    )
    steps = fetch_all_sync("SELECT * FROM agent_run_steps WHERE agent_run_id = ?", (agent_run_id,))

    assert first == {"message_id": "email-1"}
    assert second == {"message_id": "email-1"}
    assert calls["count"] == 1
    assert len(steps) == 1
    assert steps[0]["attempt_count"] == 1


def test_agent_run_step_once_retries_failed_non_side_effect(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    agent_run_id = create_agent_run(query="retry step", context=_admin_context("step-retry"))
    calls = {"count": 0}

    def flaky_parse():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary parse error")
        return {"parsed": True}

    try:
        run_agent_step_once(agent_run_id, step_key="parse", step_type="compute", fn=flaky_parse)
    except RuntimeError:
        pass
    result = run_agent_step_once(agent_run_id, step_key="parse", step_type="compute", fn=flaky_parse)
    steps = fetch_all_sync("SELECT status, attempt_count, output_json FROM agent_run_steps WHERE agent_run_id = ?", (agent_run_id,))

    assert result == {"parsed": True}
    assert calls["count"] == 2
    assert len(steps) == 1
    assert steps[0]["status"] == "done"
    assert steps[0]["attempt_count"] == 2


def test_agent_run_is_cancelled_when_pending_action_is_rejected(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    agent_run_id, action_id, context = _paused_run_with_action()

    rejected = reject_pending_action(action_id, auth=context.auth, note="Operator rejected")
    detail = get_agent_run(agent_run_id)

    assert rejected["status"] == "rejected"
    assert detail["status"] == "cancelled"
    assert detail["result"]["trigger_status"] == "rejected"


def test_chat_tool_route_is_visible_in_agent_run_admin_api(isolated_client: TestClient):
    chat = isolated_client.post(
        "/api/chat",
        headers=admin_headers(),
        json={"session_id": "phase57-agent-run", "message": "list kbs", "lang": "en"},
    )

    assert chat.status_code == 200, chat.text
    assert '"agent_run_id":' in chat.text
    listed = isolated_client.get(
        "/api/admin/agent-runs",
        params={"session_id": "phase57-agent-run"},
        headers=admin_headers(),
    )
    assert listed.status_code == 200, listed.text
    items = listed.json()["items"]
    assert len(items) == 1
    assert items[0]["route"] == "tool"
    assert items[0]["tool_name"] == "list_kbs"
    assert items[0]["status"] == "completed"

    detail = isolated_client.get(f"/api/admin/agent-runs/{items[0]['id']}", headers=admin_headers())
    assert detail.status_code == 200, detail.text
    assert [step["step_key"] for step in detail.json()["steps"]] == ["route", "tool_call", "tool_result"]


def test_agent_run_admin_api_requires_operations_role(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        denied = client.get(
            "/api/admin/agent-runs",
            headers=auth_headers(user_id="employee-1", roles=["employee"], channel="portal"),
        )

    assert denied.status_code == 403, denied.text
