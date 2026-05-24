from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main
from app.database import fetch_one_sync
from app.models import RequestContext
from app.support_ticket_service import create_support_ticket
from tests.conftest import admin_headers, auth_headers, configure_test_env, run


def _create_ticket(message: str, *, issue_type: str = "other") -> int:
    ticket = create_support_ticket(
        issue_type=issue_type,
        message=message,
        contact="workflow-customer@example.com",
        context=RequestContext(
            request_id="durable-workflow-test",
            auth={"user_id": "customer-1", "roles": ["user"], "channel": "web"},
        ),
    )
    row = fetch_one_sync("SELECT id FROM support_tickets WHERE ticket_code = ?", (ticket["ticket_code"],))
    assert row
    return int(row["id"])


def test_support_case_workflow_persists_run_and_steps(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = _create_ticket("Tôi muốn hỏi về chính sách học phí của FPT University.", issue_type="question")

    with TestClient(main.app) as client:
        handled = client.post(f"/api/admin/support-workflows/tickets/{ticket_id}/handle", headers=admin_headers())
        listed = client.get(
            f"/api/admin/workflows/runs?entity_type=support_ticket&entity_id={ticket_id}",
            headers=admin_headers(),
        )

    assert handled.status_code == 200, handled.text
    assert listed.status_code == 200, listed.text
    runs = listed.json()["items"]
    assert len(runs) == 1
    assert runs[0]["workflow_type"] == "support_ticket_case"
    assert runs[0]["status"] in {"completed", "paused"}

    with TestClient(main.app) as client:
        detail = client.get(f"/api/admin/workflows/runs/{runs[0]['id']}", headers=admin_headers())

    assert detail.status_code == 200, detail.text
    step_keys = {step["step_key"] for step in detail.json()["steps"]}
    assert {"load_ticket", "classify_case", "assign_priority", "enrich_context", "build_action_plan", "update_ticket_workflow"} <= step_keys


def test_paused_workflow_can_be_cancelled_and_retried(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = _create_ticket("Tôi muốn hoàn tiền cho đơn DH99999 vì giao hàng thất bại.", issue_type="refund")

    with TestClient(main.app) as client:
        handled = client.post(f"/api/admin/support-workflows/tickets/{ticket_id}/handle", headers=admin_headers())
        listed = client.get(
            f"/api/admin/workflows/runs?status=paused&entity_type=support_ticket&entity_id={ticket_id}",
            headers=admin_headers(),
        )

    assert handled.status_code == 200, handled.text
    assert listed.status_code == 200, listed.text
    run_id = listed.json()["items"][0]["id"]

    with TestClient(main.app) as client:
        blocked_resume = client.post(f"/api/admin/workflows/runs/{run_id}/resume", headers=admin_headers())
        cancelled = client.post(
            f"/api/admin/workflows/runs/{run_id}/cancel",
            json={"reason": "operator test cancel"},
            headers=admin_headers(),
        )
        retried = client.post(f"/api/admin/workflows/runs/{run_id}/retry", headers=admin_headers())

    assert blocked_resume.status_code == 400, blocked_resume.text
    assert "pending approval" in blocked_resume.json()["detail"]
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    assert retried.status_code == 200, retried.text
    assert retried.json()["status"] == "paused"
    assert retried.json()["step_count"] > cancelled.json()["step_count"]


def test_support_case_workflow_auto_resumes_after_pending_action_execution(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = _create_ticket("Tôi muốn hoàn tiền cho đơn DH99999 vì giao hàng thất bại.", issue_type="refund")

    with TestClient(main.app) as client:
        handled = client.post(f"/api/admin/support-workflows/tickets/{ticket_id}/handle", headers=admin_headers())
        assert handled.status_code == 200, handled.text
        pending = fetch_one_sync(
            """
            SELECT id
            FROM pending_actions
            WHERE action_type = 'support_case_review'
              AND json_extract(payload_json, '$.ticket_id') = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (ticket_id,),
        )
        assert pending
        action_id = int(pending["id"])
        approved = client.post(f"/api/admin/pending-actions/{action_id}/approve", headers=admin_headers())
        queued = client.post(f"/api/admin/pending-actions/{action_id}/execute", headers=admin_headers())

        assert approved.status_code == 200, approved.text
        assert queued.status_code == 200, queued.text
        from app.background_jobs import run_due_background_jobs_once

        assert run(run_due_background_jobs_once()) is True
        listed = client.get(
            f"/api/admin/workflows/runs?entity_type=support_ticket&entity_id={ticket_id}",
            headers=admin_headers(),
        )
        timeline = client.get(f"/api/admin/support-tickets/{ticket_id}/timeline", headers=admin_headers())

    assert listed.status_code == 200, listed.text
    run_item = listed.json()["items"][0]
    assert run_item["status"] == "completed"
    assert run_item["result"]["resumed"] is True
    assert run_item["result"]["trigger_status"] == "executed"

    assert timeline.status_code == 200, timeline.text
    events = timeline.json()["events"]
    assert any(event["stage"] == "workflow_step" and "resume_after_approval" in event["title"] for event in events)


def test_workflow_run_endpoints_require_operations_role(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        response = client.get(
            "/api/admin/workflows/runs",
            headers=auth_headers(user_id="employee-1", roles=["employee"], channel="portal"),
        )

    assert response.status_code == 403, response.text
