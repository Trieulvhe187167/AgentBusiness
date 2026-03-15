from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main
import app.rag as rag
from app.models import AuthContext, ChatRequest, RequestContext
from app.tool_audit import log_tool_call
from tests.conftest import configure_test_env, poll_jobs


def _prepare_ingested_default_kb(client: TestClient) -> int:
    sample_path = Path("kb_sample.csv")
    with sample_path.open("rb") as handle:
        upload = client.post(
            "/api/upload",
            files={"file": (sample_path.name, handle, "text/csv")},
        )
    upload.raise_for_status()

    kb = client.get("/api/kbs/default")
    kb.raise_for_status()
    kb_id = kb.json()["id"]

    ingest = client.post(f"/api/kbs/{kb_id}/ingest")
    ingest.raise_for_status()
    jobs = ingest.json().get("jobs") or []
    if jobs:
        poll_jobs(client, jobs)
    return kb_id


def test_chat_logs_capture_request_and_auth_context(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        kb_id = _prepare_ingested_default_kb(client)

        chat_req = ChatRequest(
            session_id="phase12-context",
            request_id="req-phase12-chat",
            message="Phí giao hàng là bao nhiêu?",
            lang="vi",
            kb_id=kb_id,
            user_id="User-001",
            roles=["Member", "admin", "Member"],
            channel="WEB",
            tenant_id="Tenant-A",
            org_id="Org-1",
        )
        context = chat_req.build_request_context(chat_req.request_id or "req-phase12-chat")
        events = list(
            rag.rag_stream(
                query=chat_req.message,
                session_id=chat_req.resolved_session_id,
                lang=chat_req.lang,
                kb_id=chat_req.kb_id,
                kb_key=chat_req.kb_key,
                request_context=context,
            )
        )
        start_event = next(event for event in events if event["event"] == "start")
        assert start_event["data"]["request_id"] == "req-phase12-chat"

        logs = client.get("/api/admin/chat-logs", params={"limit": 20})
        logs.raise_for_status()
        payload = logs.json()
        row = next(item for item in payload if item["request_id"] == "req-phase12-chat")

        assert row["session_id"] == f"phase12-context::kb:{kb_id}"
        assert row["user_id"] == "User-001"
        assert row["roles"] == ["member", "admin"]
        assert row["channel"] == "web"
        assert row["tenant_id"] == "Tenant-A"
        assert row["org_id"] == "Org-1"
        assert row["kb_id"] == kb_id
        assert row["kb_key"] == "default"


def test_tool_audit_endpoint_returns_logged_entries(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    request_context = RequestContext(
        request_id="req-tool-1",
        session_id="session-tool-1",
        kb_id=7,
        kb_key="support",
        auth=AuthContext(
            user_id="agent-9",
            roles=["support", "admin"],
            channel="api",
            tenant_id="tenant-z",
            org_id="org-z",
        ),
    )
    tool_call_id = log_tool_call(
        "create_support_ticket",
        request_context=request_context,
        args={"issue_type": "payment", "message": "Card declined"},
        tool_status="success",
        result_summary="Ticket TCK-100 created",
        latency_ms=82,
    )

    with TestClient(main.app) as client:
        logs = client.get("/api/admin/tool-audit-logs", params={"limit": 10})
        logs.raise_for_status()
        payload = logs.json()
        row = next(item for item in payload if item["tool_call_id"] == tool_call_id)

        assert row["request_id"] == "req-tool-1"
        assert row["session_id"] == "session-tool-1"
        assert row["user_id"] == "agent-9"
        assert row["roles"] == ["support", "admin"]
        assert row["channel"] == "api"
        assert row["tenant_id"] == "tenant-z"
        assert row["org_id"] == "org-z"
        assert row["kb_id"] == 7
        assert row["kb_key"] == "support"
        assert row["tool_name"] == "create_support_ticket"
        assert row["tool_status"] == "success"
        assert row["result_summary"] == "Ticket TCK-100 created"
