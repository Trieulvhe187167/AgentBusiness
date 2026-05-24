from __future__ import annotations

from email.message import EmailMessage

import app.database as database
from app.integrations import support_email
from app.integrations.support_email import (
    ParsedEmail,
    _html_to_display_text,
    _message_text_body,
    read_support_email_thread,
    upsert_support_email_message,
)
from app.models import AuthContext, RequestContext
from app.pending_actions import approve_pending_action, execute_pending_action
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
    assert reply.output["status"] == "draft"
    assert reply.output["action_type"] == "send_email_reply"
    assert reply.output["payload"]["email_id"] == email_id

    approve_pending_action(reply.output["id"], auth=_admin_context().auth)
    executed = run(execute_pending_action(reply.output["id"], context=_admin_context()))
    assert executed["status"] == "executed"
    assert executed["result"]["status"] == "sent"
    assert executed["result"]["to_address"] == "customer@example.com"

    outbound = database.fetch_one_sync(
        "SELECT direction, status, thread_id FROM support_email_messages WHERE id = ?",
        (executed["result"]["reply_message_id"],),
    )
    assert outbound == {"direction": "outbound", "status": "sent", "thread_id": "thread-1"}


def test_support_email_html_body_is_rendered_as_readable_text(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    raw_html = """
    <!DOCTYPE html>
    <html>
      <head><style>.hidden{display:none}</style><script>track()</script></head>
      <body>
        <div class="preheader">Hidden preview text</div>
        <img src="https://tracker.example/open.aspx" width="1" height="1" alt="">
        <h1>Upskill in AI, business, or marketing.</h1>
        <p>No matter which field you are in, digital transformation is about adapting.</p>
        <a href="https://click.sfmc.edx.org/?qs=tracking">Find your course</a>
      </body>
    </html>
    """
    email_id = upsert_support_email_message(
        ParsedEmail(
            provider="imap_smtp",
            mailbox="INBOX",
            provider_message_id="uid-html",
            thread_id="thread-html",
            message_id_header="<html@example.com>",
            in_reply_to=None,
            references_header=None,
            from_address="marketing@example.com",
            from_name="Marketing",
            to_addresses=["support@example.com"],
            cc_addresses=[],
            subject="HTML newsletter",
            body_text=raw_html,
            snippet=raw_html[:260],
            received_at="2026-05-22T00:00:00+00:00",
        )
    )

    thread = read_support_email_thread(email_id=email_id)
    body_text = thread["messages"][0]["body_text"]
    snippet = thread["messages"][0]["snippet"]

    assert "Upskill in AI, business, or marketing." in body_text
    assert "No matter which field" in body_text
    assert "Find your course" in body_text
    assert "Hidden preview text" not in body_text
    assert "<html" not in body_text.lower()
    assert "<style" not in body_text.lower()
    assert "click.sfmc" not in snippet


def test_support_email_parser_cleans_text_html_parts():
    message = EmailMessage()
    message["Subject"] = "Only HTML"
    message.set_content(
        "<html><body><p>Hello support team</p><p>Order ORDER-12345 needs help.</p></body></html>",
        subtype="html",
    )

    body_text = _message_text_body(message)

    assert "Hello support team" in body_text
    assert "ORDER-12345" in body_text
    assert "<p>" not in body_text


def test_support_email_html_cleaner_tolerates_tags_without_attrs(monkeypatch):
    class BrokenAttrsTag:
        attrs = None

        def decompose(self):
            return None

    class FakeSoup:
        def __call__(self, _names):
            return []

        def find_all(self, name=True):
            if name is True:
                return [BrokenAttrsTag()]
            return []

        def get_text(self, *_args, **_kwargs):
            return "Readable fallback text"

    monkeypatch.setattr(support_email, "BeautifulSoup", lambda *_args, **_kwargs: FakeSoup())

    assert _html_to_display_text("<html><body>Readable fallback text</body></html>") == "Readable fallback text"
