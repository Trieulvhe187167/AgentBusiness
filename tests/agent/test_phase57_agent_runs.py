from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main
from app.agent_runs import create_agent_run, get_agent_run, pause_agent_run_for_pending_action, record_agent_route
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
