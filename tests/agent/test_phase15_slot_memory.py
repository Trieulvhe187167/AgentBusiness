from __future__ import annotations

import json

import app.database as database
from fastapi.testclient import TestClient

from tests.conftest import (
    add_vector,
    attach_file,
    auth_headers,
    fetch_default_kb,
    insert_file,
    isolated_client,
    mark_ingested,
)


def _load_slots(session_id: str) -> dict:
    row = database.fetch_one_sync("SELECT slots_json FROM chat_sessions WHERE session_id = ?", (session_id,))
    assert row is not None
    return json.loads(row["slots_json"] or "{}")


def _seed_default_kb_with_vectors() -> int:
    kb = fetch_default_kb()
    file_id = insert_file("phase15-default.csv")
    attach_file(kb.id, file_id)
    mark_ingested(kb.id, file_id)
    add_vector(
        kb.id,
        file_id,
        "Shipping fee policy for phase 15.",
        filename="phase15-default.csv",
        kb_version=kb.kb_version,
        chunk_id="chunk-phase15-default",
    )
    return kb.id


def test_slot_memory_resumes_pending_support_ticket(isolated_client: TestClient):
    session_id = "phase15-ticket-resume"

    first = isolated_client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "Tạo ticket giao hàng giúp tôi",
            "lang": "vi",
        },
    )
    assert first.status_code == 200, first.text
    assert '"route": "clarify"' in first.text

    slots_after_clarify = _load_slots(session_id)
    assert slots_after_clarify["pending_tool_name"] == "create_support_ticket"
    assert slots_after_clarify["pending_tool_arguments"]["issue_type"] == "shipping"

    second = isolated_client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "user@example.com",
            "lang": "vi",
        },
    )
    assert second.status_code == 200, second.text
    assert '"route": "tool"' in second.text
    assert '"tool_name": "create_support_ticket"' in second.text
    assert '"status": "success"' in second.text

    ticket_row = database.fetch_one_sync(
        "SELECT ticket_code, issue_type, contact FROM support_tickets ORDER BY id DESC LIMIT 1"
    )
    assert ticket_row == {
        "ticket_code": ticket_row["ticket_code"],
        "issue_type": "shipping",
        "contact": "user@example.com",
    }

    slots_after_success = _load_slots(session_id)
    assert slots_after_success["last_tool"] == "create_support_ticket"
    assert slots_after_success["last_ticket_code"] == ticket_row["ticket_code"]
    assert "pending_tool_name" not in slots_after_success


def test_slot_memory_answers_recent_ticket_reference(isolated_client: TestClient):
    session_id = "phase15-ticket-memory"

    create = isolated_client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "Tạo ticket giao hàng, liên hệ user@example.com",
            "lang": "vi",
        },
    )
    assert create.status_code == 200, create.text
    ticket_row = database.fetch_one_sync(
        "SELECT ticket_code FROM support_tickets ORDER BY id DESC LIMIT 1"
    )
    assert ticket_row is not None

    recall = isolated_client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "Mã ticket vừa tạo là gì?",
            "lang": "vi",
        },
    )
    assert recall.status_code == 200, recall.text
    assert '"route": "memory"' in recall.text
    assert ticket_row["ticket_code"] in recall.text


def test_slot_memory_reuses_last_kb_for_follow_up_stats(isolated_client: TestClient):
    session_id = "phase15-kb-memory"
    kb_id = _seed_default_kb_with_vectors()
    admin = auth_headers(user_id="admin-1", roles=["admin"])

    first = isolated_client.post(
        "/api/chat",
        headers=admin,
        json={
            "session_id": session_id,
            "message": "kb stats",
            "lang": "en",
            "kb_id": kb_id,
        },
    )
    assert first.status_code == 200, first.text
    assert '"tool_name": "get_kb_stats"' in first.text

    second = isolated_client.post(
        "/api/chat",
        headers=admin,
        json={
            "session_id": session_id,
            "message": "how many vectors?",
            "lang": "en",
        },
    )
    assert second.status_code == 200, second.text
    assert '"route": "tool"' in second.text
    assert '"tool_name": "get_kb_stats"' in second.text
    assert '"status": "success"' in second.text

    audit_row = database.fetch_one_sync(
        """
        SELECT args_json
        FROM tool_audit_logs
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (session_id,),
    )
    assert audit_row is not None
    args_payload = json.loads(audit_row["args_json"] or "{}")
    assert args_payload["kb_id"] == kb_id

    slots = _load_slots(session_id)
    assert slots["last_kb_id"] == kb_id
    assert slots["subject_type"] == "kb"
