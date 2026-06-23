from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main
from app.database import execute_sync, fetch_one_sync, utcnow_iso
from app.models import RequestContext
from app.support_ticket_service import create_support_ticket
from tests.conftest import admin_headers, auth_headers, configure_test_env, poll_background_job, run


def _create_ticket(message: str, *, issue_type: str = "other") -> int:
    ticket = create_support_ticket(
        issue_type=issue_type,
        message=message,
        contact="customer@example.com",
        context=RequestContext(request_id="wf-test", auth={"user_id": "customer-1", "roles": ["user"], "channel": "web"}),
    )
    row = fetch_one_sync("SELECT id FROM support_tickets WHERE ticket_code = ?", (ticket["ticket_code"],))
    assert row
    return int(row["id"])


def test_user_ticket_created_from_chat_answer_records_source_context(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/support-tickets",
            headers=auth_headers(user_id="employee-1", roles=["employee"], channel="web"),
            json={
                "issue_type": "question",
                "message": "The assistant answer was not enough.",
                "kb_id": 1,
                "kb_key": "default",
                "source_chat_request_id": "chat-req-123",
                "source_session_id": "session-abc",
                "source_question": "What is the return policy?",
                "source_answer": "Returns are available for standard orders.",
                "source_citations": [
                    {
                        "filename": "policy.csv",
                        "chunk_id": "chunk-1",
                        "content_preview": "Returns are available for standard orders.",
                    }
                ],
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ticket_code"].startswith("TCK-")
    assert payload["note_count"] == 1
    note = fetch_one_sync(
        "SELECT note_type, visibility, body, metadata_json FROM support_ticket_notes WHERE ticket_id = ?",
        (payload["id"],),
    )
    assert note
    assert note["note_type"] == "source_chat"
    assert note["visibility"] == "internal"
    assert "What is the return policy?" in note["body"]
    assert "policy.csv" in note["body"]
    assert '"request_id": "chat-req-123"' in note["metadata_json"]
    assert '"citation_count": 1' in note["metadata_json"]


def test_support_workflow_resolves_low_risk_order_case(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    now = utcnow_iso()
    execute_sync(
        """
        INSERT INTO order_status_cache
            (order_code, user_id, status, last_update, tracking_code, carrier, source, cached_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'snapshot', ?, ?)
        """,
        ("DH12345", "admin-1", "in_transit", now, "TRK-1", "Carrier", now, now),
    )
    ticket_id = _create_ticket("Đơn hàng DH12345 của tôi đang ở đâu?", issue_type="shipping")

    with TestClient(main.app) as client:
        response = client.post(f"/api/admin/support-workflows/tickets/{ticket_id}/handle", headers=admin_headers())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["classification"]["intent"] == "order_status"
    assert payload["lifecycle_status"] == "resolved"
    assert payload["context"]["order_status"]["status"] == "in_transit"
    assert payload["resolution_summary"]

    row = fetch_one_sync("SELECT status, intent, priority, workflow_status, action_plan_json FROM support_tickets WHERE id = ?", (ticket_id,))
    assert row
    assert row["status"] == "resolved"
    assert row["intent"] == "order_status"
    assert row["priority"] == "P2"
    assert row["workflow_status"] == "resolved"
    assert "get_order_status" in row["action_plan_json"]


def test_support_workflow_creates_pending_review_for_refund(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = _create_ticket("Tôi muốn hoàn tiền cho đơn DH99999 vì giao hàng thất bại.", issue_type="refund")

    with TestClient(main.app) as client:
        response = client.post(f"/api/admin/support-workflows/tickets/{ticket_id}/handle", headers=admin_headers())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["classification"]["intent"] == "refund_request"
    assert payload["lifecycle_status"] == "waiting_approval"
    assert payload["pending_actions"]
    assert payload["escalation"]["priority"] == "P1"

    pending = fetch_one_sync("SELECT action_type, risk_level, status, payload_json FROM pending_actions ORDER BY id DESC LIMIT 1")
    assert pending
    assert pending["action_type"] == "support_case_review"
    assert pending["risk_level"] == "high"
    assert pending["status"] == "draft"
    assert "DH99999" in pending["payload_json"]


def test_support_workflow_context_and_manual_escalation(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = _create_ticket("Tôi cần gặp nhân viên hỗ trợ.", issue_type="other")

    with TestClient(main.app) as client:
        escalate = client.post(
            f"/api/admin/support-tickets/{ticket_id}/escalate",
            json={"note": "Customer explicitly asked for a human."},
            headers=admin_headers(),
        )
        context = client.get(f"/api/admin/support-tickets/{ticket_id}/context", headers=admin_headers())

    assert escalate.status_code == 200, escalate.text
    assert context.status_code == 200, context.text
    payload = context.json()
    assert payload["ticket"]["workflow_status"] == "escalated"
    assert payload["escalation"]["suggested_next_action"] == "Customer explicitly asked for a human."


def test_support_operations_list_assign_note_and_status(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = _create_ticket("Please check invoice for order DH77777.", issue_type="payment")

    with TestClient(main.app) as client:
        list_response = client.get("/api/admin/support-tickets?limit=10", headers=admin_headers())
        assign_response = client.post(
            f"/api/admin/support-tickets/{ticket_id}/assign",
            json={"assigned_team": "billing", "assigned_user_id": "agent-1", "note": "Assign to billing queue."},
            headers=admin_headers(),
        )
        note_response = client.post(
            f"/api/admin/support-tickets/{ticket_id}/notes",
            json={"body": "Customer needs invoice follow-up.", "note_type": "internal", "visibility": "internal"},
            headers=admin_headers(),
        )
        status_response = client.post(
            f"/api/admin/support-tickets/{ticket_id}/status",
            json={"status": "waiting_customer", "note": "Asked for missing tax details."},
            headers=admin_headers(),
        )
        notes_response = client.get(f"/api/admin/support-tickets/{ticket_id}/notes", headers=admin_headers())

    assert list_response.status_code == 200, list_response.text
    assert list_response.json()["total"] == 1
    assert assign_response.status_code == 200, assign_response.text
    assert assign_response.json()["assigned_team"] == "billing"
    assert assign_response.json()["assigned_user_id"] == "agent-1"
    assert note_response.status_code == 200, note_response.text
    assert note_response.json()["note_type"] == "internal"
    assert status_response.status_code == 200, status_response.text
    assert status_response.json()["workflow_status"] == "waiting_customer"
    assert notes_response.status_code == 200, notes_response.text
    assert notes_response.json()["total"] == 3


def test_support_case_timeline_combines_workflow_notes_actions_and_jobs(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = _create_ticket("I want a refund for order DH99999 because delivery failed.", issue_type="refund")

    with TestClient(main.app) as client:
        handle = client.post(f"/api/admin/support-workflows/tickets/{ticket_id}/handle", headers=admin_headers())
        note = client.post(
            f"/api/admin/support-tickets/{ticket_id}/notes",
            json={"body": "Review refund policy before replying.", "note_type": "internal", "visibility": "internal"},
            headers=admin_headers(),
        )
        enqueue = client.post(f"/api/admin/support-workflows/tickets/{ticket_id}/enqueue", headers=admin_headers())
        timeline = client.get(f"/api/admin/support-tickets/{ticket_id}/timeline", headers=admin_headers())

    assert handle.status_code == 200, handle.text
    assert note.status_code == 200, note.text
    assert enqueue.status_code == 200, enqueue.text
    assert timeline.status_code == 200, timeline.text
    payload = timeline.json()
    stages = {item["stage"] for item in payload["events"]}
    assert payload["ticket_id"] == ticket_id
    assert "user_request" in stages
    assert "classification" in stages
    assert "action_plan" in stages
    assert "pending_action" in stages
    assert "support_note" in stages
    assert "background_job" in stages


def test_support_case_ai_draft_reply_fills_reviewable_draft(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = _create_ticket(
        "Tôi muốn hỏi về học phí 3 học kỳ đầu của chương trình FPT University.",
        issue_type="question",
    )

    with TestClient(main.app) as client:
        response = client.post(
            f"/api/admin/support-tickets/{ticket_id}/draft-reply",
            json={"tone": "professional"},
            headers=admin_headers(),
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ticket_id"] == ticket_id
    assert payload["ticket_code"].startswith("TCK-")
    assert payload["draft_reply"]
    assert payload["used_llm"] is False
    assert "retrieval_query" in payload
    assert "citations" in payload
    assert payload["customer_reply"] == payload["draft_reply"]
    assert "review_packet" in payload
    assert "internal_risk" in payload["review_packet"]
    assert "evidence_used" in payload["review_packet"]


def test_support_ticket_next_action_and_canned_more_info(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = _create_ticket("I need help accessing my account.", issue_type="account")

    with TestClient(main.app) as client:
        before = client.get(f"/api/admin/support-tickets/{ticket_id}", headers=admin_headers())
        canned = client.post(
            f"/api/admin/support-tickets/{ticket_id}/canned-action",
            json={
                "action": "ask_more_info",
                "reply_body": "Please share the email address and the error message you see.",
                "note": "Need account identifiers before troubleshooting.",
            },
            headers=admin_headers(),
        )
        notes = client.get(f"/api/admin/support-tickets/{ticket_id}/notes", headers=admin_headers())

    assert before.status_code == 200, before.text
    assert before.json()["next_action"]["key"] == "assign_owner"
    assert canned.status_code == 200, canned.text
    payload = canned.json()
    assert payload["action"] == "ask_more_info"
    assert payload["ticket"]["workflow_status"] == "waiting_customer"
    assert payload["note"]["visibility"] == "public"
    assert payload["pending_action"] is None
    assert any(item["note_type"] == "public_more_info_request" for item in notes.json()["items"])


def test_high_risk_canned_refund_creates_pending_approval(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = _create_ticket("Please refund order DH99999.", issue_type="refund")

    with TestClient(main.app) as client:
        response = client.post(
            f"/api/admin/support-tickets/{ticket_id}/canned-action",
            json={
                "action": "refund_requires_approval",
                "reply_body": "We are reviewing your refund request.",
                "note": "Refund requested; approval required.",
            },
            headers=admin_headers(),
        )
        actions = client.get("/api/admin/pending-actions?limit=10", headers=admin_headers())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ticket"]["workflow_status"] == "waiting_approval"
    assert payload["pending_action"]["action_type"] == "support_case_review"
    assert payload["pending_action"]["risk_level"] == "high"
    assert payload["pending_action"]["payload"]["canned_action"] == "refund_requires_approval"
    assert payload["next_action"]["key"] == "approval_review"
    assert actions.status_code == 200, actions.text
    assert any(item["payload"].get("ticket_id") == ticket_id for item in actions.json()["items"])


def test_support_sla_monitor_escalates_overdue_ticket(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = _create_ticket("This complaint needs attention.", issue_type="other")
    execute_sync(
        """
        UPDATE support_tickets
        SET priority = 'P1',
            workflow_status = 'planned',
            status = 'planned',
            sla_due_at = '2000-01-01T00:00:00+00:00'
        WHERE id = ?
        """,
        (ticket_id,),
    )

    with TestClient(main.app) as client:
        response = client.post("/api/admin/support-workflows/sla/monitor?limit=10", headers=admin_headers())
        context = client.get(f"/api/admin/support-tickets/{ticket_id}/context", headers=admin_headers())

    assert response.status_code == 200, response.text
    assert response.json()["breached"] == 1
    row = fetch_one_sync("SELECT workflow_status, sla_breached_at FROM support_tickets WHERE id = ?", (ticket_id,))
    assert row
    assert row["workflow_status"] == "escalated"
    assert row["sla_breached_at"]
    assert context.status_code == 200, context.text
    assert context.json()["escalation"]["suggested_next_action"].startswith("SLA is overdue")


def test_support_workflow_can_run_as_background_job(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = _create_ticket("Where is order DH54321?", issue_type="shipping")

    with TestClient(main.app) as client:
        enqueue = client.post(f"/api/admin/support-workflows/tickets/{ticket_id}/enqueue", headers=admin_headers())
        assert enqueue.status_code == 200, enqueue.text
        from app.background_jobs import run_due_background_jobs_once

        assert run(run_due_background_jobs_once()) is True
        job = poll_background_job(client, enqueue.json()["job_id"])

    assert job["status"] == "done"
    assert job["result"]["ticket_id"] == ticket_id
    row = fetch_one_sync("SELECT workflow_status, action_plan_json FROM support_tickets WHERE id = ?", (ticket_id,))
    assert row
    assert row["workflow_status"] in {"resolved", "escalated"}
    assert row["action_plan_json"]
