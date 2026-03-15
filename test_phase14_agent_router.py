from __future__ import annotations

import app.database as database
from fastapi.testclient import TestClient

from tests.conftest import add_vector, attach_file, fetch_default_kb, insert_file, isolated_client, mark_ingested


def _patch_retrieval(monkeypatch):
    monkeypatch.setattr("app.rag.expand_query", lambda query: [query])
    monkeypatch.setattr("app.rag.embed_query", lambda query: [1.0, 0.0])
    monkeypatch.setattr("app.rag.rerank", lambda query, items: items)


def _seed_default_kb_answer() -> int:
    kb = fetch_default_kb()
    file_id = insert_file("agent-default.csv")
    attach_file(kb.id, file_id)
    mark_ingested(kb.id, file_id)
    add_vector(
        kb.id,
        file_id,
        "Phí giao hàng tiêu chuẩn là 30.000 VND.",
        filename="agent-default.csv",
        kb_version=kb.kb_version,
        chunk_id="chunk-agent-default",
    )
    return kb.id


def test_chat_agent_routes_kb_questions_to_rag(isolated_client: TestClient, monkeypatch):
    _patch_retrieval(monkeypatch)
    kb_id = _seed_default_kb_answer()

    chat = isolated_client.post(
        "/api/chat",
        json={
            "session_id": "phase14-rag",
            "message": "Phí giao hàng là bao nhiêu?",
            "lang": "vi",
            "kb_id": kb_id,
        },
    )

    assert chat.status_code == 200, chat.text
    assert "event: route" in chat.text
    assert '"route": "rag"' in chat.text
    assert "event: start" in chat.text
    assert '"kb_key": "default"' in chat.text
    assert "event: done" in chat.text

    chat_logs = isolated_client.get("/api/admin/chat-logs", params={"limit": 10})
    assert chat_logs.status_code == 200, chat_logs.text
    assert any(item["session_id"] == f"phase14-rag::kb:{kb_id}" for item in chat_logs.json())


def test_chat_agent_creates_support_ticket_via_tool(isolated_client: TestClient):
    chat = isolated_client.post(
        "/api/chat",
        json={
            "session_id": "phase14-ticket",
            "message": "Tao ticket giao hang, lien he user@example.com",
            "lang": "vi",
        },
    )

    assert chat.status_code == 200, chat.text
    assert "event: route" in chat.text
    assert '"route": "tool"' in chat.text
    assert "event: tool_call" in chat.text
    assert '"tool_name": "create_support_ticket"' in chat.text
    assert '"status": "success"' in chat.text
    assert "event: done" in chat.text

    ticket_row = database.fetch_one_sync(
        """
        SELECT issue_type, contact, channel
        FROM support_tickets
        ORDER BY id DESC
        LIMIT 1
        """
    )
    assert ticket_row == {
        "issue_type": "shipping",
        "contact": "user@example.com",
        "channel": "web",
    }

    audit_row = database.fetch_one_sync(
        """
        SELECT tool_name, tool_status, session_id
        FROM tool_audit_logs
        ORDER BY id DESC
        LIMIT 1
        """
    )
    assert audit_row == {
        "tool_name": "create_support_ticket",
        "tool_status": "success",
        "session_id": "phase14-ticket",
    }


def test_chat_agent_clarifies_when_ticket_contact_is_missing(isolated_client: TestClient):
    chat = isolated_client.post(
        "/api/chat",
        json={
            "session_id": "phase14-ticket-clarify",
            "message": "Tao ticket giao hang giup toi",
            "lang": "vi",
        },
    )

    assert chat.status_code == 200, chat.text
    assert "event: route" in chat.text
    assert '"route": "clarify"' in chat.text
    assert "event: tool_call" not in chat.text
    assert "event: done" in chat.text

    count_row = database.fetch_one_sync("SELECT COUNT(*) AS total FROM support_tickets")
    assert count_row == {"total": 0}


def test_chat_agent_rejects_admin_tool_without_role(isolated_client: TestClient):
    chat = isolated_client.post(
        "/api/chat",
        json={
            "session_id": "phase14-admin-denied",
            "message": "list kbs",
            "lang": "en",
        },
    )

    assert chat.status_code == 200, chat.text
    assert "event: route" in chat.text
    assert '"route": "tool"' in chat.text
    assert "event: tool_call" in chat.text
    assert '"tool_name": "list_kbs"' in chat.text
    assert '"status": "failed"' in chat.text
    assert "permission_denied" in chat.text

    audit_row = database.fetch_one_sync(
        """
        SELECT tool_name, tool_status, session_id
        FROM tool_audit_logs
        ORDER BY id DESC
        LIMIT 1
        """
    )
    assert audit_row == {
        "tool_name": "list_kbs",
        "tool_status": "permission_denied",
        "session_id": "phase14-admin-denied",
    }


def test_chat_agent_runs_admin_tool_with_role(isolated_client: TestClient):
    chat = isolated_client.post(
        "/api/chat",
        json={
            "session_id": "phase14-admin-ok",
            "message": "list kbs",
            "lang": "en",
            "user_id": "admin-1",
            "roles": ["admin"],
            "channel": "admin",
        },
    )

    assert chat.status_code == 200, chat.text
    assert "event: tool_call" in chat.text
    assert '"tool_name": "list_kbs"' in chat.text
    assert '"status": "success"' in chat.text
    assert "event: done" in chat.text
