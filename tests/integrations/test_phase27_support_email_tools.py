from __future__ import annotations

import app.database as database
from app.integrations import support_email
from app.integrations.support_email import ParsedEmail, upsert_support_email_message
from app.models import AuthContext, RequestContext
from app.tools import build_default_tool_registry
from tests.conftest import configure_test_env, run


def _seed_email() -> int:
    return upsert_support_email_message(
        ParsedEmail(
            provider="imap_smtp",
            mailbox="INBOX",
            provider_message_id="uid-1",
            thread_id="thread-1",
            message_id_header="<m1@example.com>",
            in_reply_to=None,
            references_header=None,
            from_address="customer@example.com",
            from_name="Customer",
            to_addresses=["support@example.com"],
            cc_addresses=[],
            subject="Shipping issue ORDER-12345",
            body_text="My order ORDER-12345 has not arrived.",
            snippet="My order ORDER-12345 has not arrived.",
            received_at="2026-04-24T00:00:00+00:00",
        )
    )


def _admin_context() -> RequestContext:
    return RequestContext(
        request_id="req-email-tool",
        auth=AuthContext(user_id="admin-1", roles=["admin"], channel="admin"),
    )


def test_support_email_tools_list_read_ticket_and_reply(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    email_id = _seed_email()
    registry = build_default_tool_registry()

    listed = run(
        registry.execute(
            "list_support_emails",
            {"limit": 10, "sync_first": False},
            request_context=_admin_context(),
        )
    )
    assert listed.output["total"] == 1
    assert listed.output["items"][0]["id"] == email_id

    thread = run(
        registry.execute(
            "read_email_thread",
            {"email_id": email_id},
            request_context=_admin_context(),
        )
    )
    assert thread.output["thread_id"] == "thread-1"
    assert thread.output["messages"][0]["subject"] == "Shipping issue ORDER-12345"

    ticket = run(
        registry.execute(
            "create_ticket_from_email",
            {"email_id": email_id, "issue_type": "shipping"},
            request_context=_admin_context(),
        )
    )
    assert ticket.output["ticket_code"].startswith("TCK-")
    assert ticket.output["order_code"] == "ORDER-12345"

    ticket_row = database.fetch_one_sync(
        "SELECT ticket_code, contact FROM support_tickets WHERE ticket_code = ?",
        (ticket.output["ticket_code"],),
    )
    assert ticket_row == {
        "ticket_code": ticket.output["ticket_code"],
        "contact": "customer@example.com",
    }

    monkeypatch.setattr(
        support_email,
        "_send_smtp_reply",
        lambda **kwargs: "<reply@example.com>",
    )
    reply = run(
        registry.execute(
            "send_email_reply",
            {"email_id": email_id, "body": "We received your request."},
            request_context=_admin_context(),
        )
    )
    assert reply.output["status"] == "sent"
    assert reply.output["to_address"] == "customer@example.com"

    outbound = database.fetch_one_sync(
        "SELECT direction, status, thread_id FROM support_email_messages WHERE id = ?",
        (reply.output["reply_message_id"],),
    )
    assert outbound == {"direction": "outbound", "status": "sent", "thread_id": "thread-1"}
