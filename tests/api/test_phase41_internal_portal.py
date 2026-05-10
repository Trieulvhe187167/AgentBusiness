from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main
from app.database import execute_sync
from tests.conftest import auth_headers, configure_test_env


def test_internal_portal_page_is_served(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        response = client.get("/portal")

    assert response.status_code == 200, response.text
    assert "Internal User Portal" in response.text
    assert "/api/support-tickets" in response.text


def test_internal_user_can_create_and_list_own_ticket(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        created = client.post(
            "/api/support-tickets",
            headers=auth_headers(user_id="employee-1", roles=["employee"], channel="portal"),
            json={
                "issue_type": "technical_issue",
                "message": "Laptop VPN cannot connect.",
                "contact": "employee-1@example.com",
            },
        )
        listed = client.get(
            "/api/support-tickets?limit=10",
            headers=auth_headers(user_id="employee-1", roles=["employee"], channel="portal"),
        )

    assert created.status_code == 200, created.text
    payload = created.json()
    assert payload["ticket_code"].startswith("TCK-")
    assert payload["created_by_user_id"] == "employee-1"
    assert payload["channel"] == "portal"
    assert payload["issue_type"] == "technical_issue"

    assert listed.status_code == 200, listed.text
    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["ticket_code"] == payload["ticket_code"]


def test_internal_user_cannot_read_another_users_ticket(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = execute_sync(
        """
        INSERT INTO support_tickets (
            ticket_code, issue_type, message, status, created_by_user_id,
            channel, created_at, updated_at
        ) VALUES ('TCK-OTHERUSER', 'question', 'Private request', 'open', 'employee-1',
                  'portal', datetime('now'), datetime('now'))
        """
    )

    with TestClient(main.app) as client:
        response = client.get(
            f"/api/support-tickets/{ticket_id}",
            headers=auth_headers(user_id="employee-2", roles=["employee"], channel="portal"),
        )

    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Support ticket access denied"


def test_internal_ticket_api_requires_authenticated_user(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        response = client.get("/api/support-tickets")

    assert response.status_code == 401, response.text
    assert response.json()["detail"] == "Authentication required"


def test_internal_user_can_only_read_public_ticket_notes(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = execute_sync(
        """
        INSERT INTO support_tickets (
            ticket_code, issue_type, message, status, workflow_status, created_by_user_id,
            channel, created_at, updated_at
        ) VALUES ('TCK-PUBLICREPLY', 'question', 'Need tuition info', 'open', 'open', 'employee-1',
                  'portal', datetime('now'), datetime('now'))
        """
    )
    execute_sync(
        """
        INSERT INTO support_ticket_notes (
            ticket_id, note_type, visibility, body, metadata_json,
            created_by_user_id, roles_json, created_at
        ) VALUES (?, 'public_reply', 'public', 'Public answer', '{}', 'support-1', '["support_agent"]', datetime('now'))
        """,
        (ticket_id,),
    )
    execute_sync(
        """
        INSERT INTO support_ticket_notes (
            ticket_id, note_type, visibility, body, metadata_json,
            created_by_user_id, roles_json, created_at
        ) VALUES (?, 'internal', 'internal', 'Private support note', '{}', 'support-1', '["support_agent"]', datetime('now'))
        """,
        (ticket_id,),
    )

    with TestClient(main.app) as client:
        response = client.get(
            f"/api/support-tickets/{ticket_id}/notes",
            headers=auth_headers(user_id="employee-1", roles=["employee"], channel="portal"),
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["body"] == "Public answer"
    assert payload["items"][0]["visibility"] == "public"


def test_internal_user_can_reply_to_own_ticket_and_reopen(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = execute_sync(
        """
        INSERT INTO support_tickets (
            ticket_code, issue_type, message, status, workflow_status, created_by_user_id,
            channel, created_at, updated_at
        ) VALUES ('TCK-EMPLOYEE-REPLY', 'question', 'Need tuition info', 'waiting_customer', 'waiting_customer',
                  'employee-1', 'portal', datetime('now'), datetime('now'))
        """
    )

    with TestClient(main.app) as client:
        created = client.post(
            f"/api/support-tickets/{ticket_id}/notes",
            headers=auth_headers(user_id="employee-1", roles=["employee"], channel="portal"),
            json={"body": "Thanks. Can you confirm whether this includes lab fees?"},
        )
        ticket = client.get(
            f"/api/support-tickets/{ticket_id}",
            headers=auth_headers(user_id="employee-1", roles=["employee"], channel="portal"),
        )

    assert created.status_code == 200, created.text
    payload = created.json()
    assert payload["note_type"] == "customer_reply"
    assert payload["visibility"] == "public"
    assert payload["created_by_user_id"] == "employee-1"
    assert ticket.status_code == 200, ticket.text
    assert ticket.json()["workflow_status"] == "open"


def test_internal_user_cannot_reply_to_another_users_ticket(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = execute_sync(
        """
        INSERT INTO support_tickets (
            ticket_code, issue_type, message, status, workflow_status, created_by_user_id,
            channel, created_at, updated_at
        ) VALUES ('TCK-NOT-YOURS', 'question', 'Private request', 'waiting_customer', 'waiting_customer',
                  'employee-1', 'portal', datetime('now'), datetime('now'))
        """
    )

    with TestClient(main.app) as client:
        response = client.post(
            f"/api/support-tickets/{ticket_id}/notes",
            headers=auth_headers(user_id="employee-2", roles=["employee"], channel="portal"),
            json={"body": "Trying to reply"},
        )

    assert response.status_code == 403, response.text


def test_internal_user_cannot_reply_to_closed_ticket(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    ticket_id = execute_sync(
        """
        INSERT INTO support_tickets (
            ticket_code, issue_type, message, status, workflow_status, created_by_user_id,
            channel, created_at, updated_at
        ) VALUES ('TCK-CLOSED-PORTAL', 'question', 'Closed request', 'closed', 'closed',
                  'employee-1', 'portal', datetime('now'), datetime('now'))
        """
    )

    with TestClient(main.app) as client:
        response = client.post(
            f"/api/support-tickets/{ticket_id}/notes",
            headers=auth_headers(user_id="employee-1", roles=["employee"], channel="portal"),
            json={"body": "I still need help"},
        )

    assert response.status_code == 409, response.text
