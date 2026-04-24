"""
Support email adapter and SQLite snapshot helpers.

The tool surface is provider-agnostic. This first adapter uses IMAP/SMTP,
which works with Gmail and Outlook mailboxes when the account is configured
with app passwords or equivalent mailbox credentials.
"""

from __future__ import annotations

import asyncio
import email
import imaplib
import json
import re
import smtplib
import ssl
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.header import decode_header, make_header
from email.message import EmailMessage, Message
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from typing import Any

from app.config import settings
from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.models import RequestContext
from app.support_ticket_service import create_support_ticket
from app.tools.registry import ToolExecutionError, ToolValidationError

_ORDER_CODE_RE = re.compile(r"\b(?:DH|ORD|ORDER)[A-Z0-9-]{3,}\b", re.IGNORECASE)


@dataclass(slots=True)
class ParsedEmail:
    provider: str
    mailbox: str
    provider_message_id: str
    thread_id: str
    message_id_header: str | None
    in_reply_to: str | None
    references_header: str | None
    from_address: str | None
    from_name: str | None
    to_addresses: list[str]
    cc_addresses: list[str]
    subject: str
    body_text: str
    snippet: str
    received_at: str | None
    direction: str = "inbound"
    status: str = "new"
    raw: dict[str, Any] | None = None


def _provider_name() -> str:
    return (settings.email_provider or "imap_smtp").strip().lower()


def _mailbox_name() -> str:
    return (settings.email_imap_mailbox or "INBOX").strip()


def _decode_mime_header(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return value.strip()


def _message_text_body(msg: Message) -> str:
    if msg.is_multipart():
        html_fallback = ""
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition") or "").lower()
            if "attachment" in disposition:
                continue
            try:
                payload = part.get_payload(decode=True)
            except Exception:
                payload = None
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                text = payload.decode(charset, errors="replace")
            except LookupError:
                text = payload.decode("utf-8", errors="replace")
            if content_type == "text/plain" and text.strip():
                return text.strip()
            if content_type == "text/html" and text.strip() and not html_fallback:
                html_fallback = re.sub(r"<[^>]+>", " ", text)
        return " ".join(html_fallback.split())

    payload = msg.get_payload(decode=True)
    if payload is None:
        raw = msg.get_payload()
        return str(raw or "").strip()
    charset = msg.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace").strip()
    except LookupError:
        return payload.decode("utf-8", errors="replace").strip()


def _parsed_received_at(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _addresses(raw: str | None) -> list[str]:
    return [addr for _, addr in getaddresses([raw or ""]) if addr]


def _thread_id(message_id: str | None, in_reply_to: str | None, references_header: str | None, fallback: str) -> str:
    refs = [part for part in re.split(r"\s+", references_header or "") if part]
    if refs:
        return refs[0].strip("<>")
    if in_reply_to:
        return in_reply_to.strip("<>")
    if message_id:
        return message_id.strip("<>")
    return fallback


def _parse_message(uid: str, raw_bytes: bytes) -> ParsedEmail:
    msg = email.message_from_bytes(raw_bytes)
    subject = _decode_mime_header(msg.get("Subject"))
    from_name, from_address = parseaddr(_decode_mime_header(msg.get("From")))
    message_id = (msg.get("Message-ID") or "").strip() or None
    in_reply_to = (msg.get("In-Reply-To") or "").strip() or None
    references_header = (msg.get("References") or "").strip() or None
    body_text = _message_text_body(msg)
    snippet = " ".join(body_text.split())[:260]
    thread_id = _thread_id(message_id, in_reply_to, references_header, uid)
    return ParsedEmail(
        provider=_provider_name(),
        mailbox=_mailbox_name(),
        provider_message_id=uid,
        thread_id=thread_id,
        message_id_header=message_id,
        in_reply_to=in_reply_to,
        references_header=references_header,
        from_address=from_address or None,
        from_name=from_name or None,
        to_addresses=_addresses(msg.get("To")),
        cc_addresses=_addresses(msg.get("Cc")),
        subject=subject,
        body_text=body_text,
        snippet=snippet,
        received_at=_parsed_received_at(msg.get("Date")) or utcnow_iso(),
        raw={
            "uid": uid,
            "headers": {
                "message_id": message_id,
                "in_reply_to": in_reply_to,
                "references": references_header,
            },
        },
    )


def _imap_fetch_recent(limit: int, unread_only: bool) -> list[ParsedEmail]:
    if not settings.email_integration_enabled:
        raise ToolExecutionError("Support email integration is disabled")
    if not settings.email_imap_host.strip() or not settings.email_imap_username.strip():
        raise ToolExecutionError("IMAP host and username are required")

    if settings.email_imap_use_ssl:
        client = imaplib.IMAP4_SSL(settings.email_imap_host, settings.email_imap_port)
    else:
        client = imaplib.IMAP4(settings.email_imap_host, settings.email_imap_port)
    try:
        client.login(settings.email_imap_username, settings.email_imap_password)
        status, _ = client.select(_mailbox_name())
        if status != "OK":
            raise ToolExecutionError(f"Unable to select mailbox {_mailbox_name()}")

        since = (datetime.now(timezone.utc) - timedelta(days=max(1, settings.email_lookback_days))).strftime("%d-%b-%Y")
        criteria = f'(SINCE "{since}")'
        if unread_only:
            criteria = f'(UNSEEN SINCE "{since}")'
        status, data = client.uid("SEARCH", None, criteria)
        if status != "OK":
            raise ToolExecutionError("IMAP search failed")

        uids = (data[0] or b"").split()
        latest = list(reversed(uids[-max(1, limit):]))
        messages: list[ParsedEmail] = []
        for raw_uid in latest:
            uid = raw_uid.decode("ascii", errors="ignore")
            status, fetched = client.uid("FETCH", raw_uid, "(RFC822)")
            if status != "OK" or not fetched:
                continue
            for item in fetched:
                if isinstance(item, tuple) and item[1]:
                    messages.append(_parse_message(uid, item[1]))
                    break
        return messages
    finally:
        try:
            client.logout()
        except Exception:
            pass


def _send_smtp_reply(*, to_address: str, subject: str, body: str, original: dict[str, Any] | None = None) -> str:
    if not settings.email_integration_enabled:
        raise ToolExecutionError("Support email integration is disabled")
    if not settings.email_smtp_host.strip():
        raise ToolExecutionError("SMTP host is required to send email replies")

    from_address = (settings.email_from_address or settings.email_smtp_username or settings.email_support_address).strip()
    if not from_address:
        raise ToolExecutionError("Email from address is required")

    msg = EmailMessage()
    msg["From"] = from_address
    msg["To"] = to_address
    msg["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    if original:
        message_id = original.get("message_id_header")
        references = original.get("references_header") or message_id
        if message_id:
            msg["In-Reply-To"] = message_id
        if references:
            msg["References"] = references
    msg.set_content(body)

    if settings.email_smtp_use_ssl:
        smtp = smtplib.SMTP_SSL(settings.email_smtp_host, settings.email_smtp_port, context=ssl.create_default_context())
    else:
        smtp = smtplib.SMTP(settings.email_smtp_host, settings.email_smtp_port)
    try:
        if settings.email_smtp_use_tls and not settings.email_smtp_use_ssl:
            smtp.starttls(context=ssl.create_default_context())
        if settings.email_smtp_username:
            smtp.login(settings.email_smtp_username, settings.email_smtp_password)
        smtp.send_message(msg)
        return msg.get("Message-ID") or f"outbound-{uuid.uuid4().hex}"
    finally:
        try:
            smtp.quit()
        except Exception:
            pass


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "provider": row.get("provider") or "",
        "mailbox": row.get("mailbox") or "",
        "provider_message_id": row.get("provider_message_id") or "",
        "thread_id": row.get("thread_id") or "",
        "from_address": row.get("from_address"),
        "from_name": row.get("from_name"),
        "to_addresses": json.loads(row.get("to_addresses_json") or "[]"),
        "cc_addresses": json.loads(row.get("cc_addresses_json") or "[]"),
        "subject": row.get("subject") or "",
        "snippet": row.get("snippet") or "",
        "body_text": row.get("body_text") or "",
        "received_at": row.get("received_at"),
        "direction": row.get("direction") or "inbound",
        "status": row.get("status") or "new",
        "ticket_code": row.get("ticket_code"),
    }


def upsert_support_email_message(item: ParsedEmail) -> int:
    now = utcnow_iso()
    execute_sync(
        """
        INSERT INTO support_email_messages (
            provider, mailbox, provider_message_id, thread_id, message_id_header,
            in_reply_to, references_header, from_address, from_name,
            to_addresses_json, cc_addresses_json, subject, body_text, snippet,
            received_at, direction, status, raw_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, mailbox, provider_message_id) DO UPDATE SET
            thread_id=excluded.thread_id,
            message_id_header=excluded.message_id_header,
            in_reply_to=excluded.in_reply_to,
            references_header=excluded.references_header,
            from_address=excluded.from_address,
            from_name=excluded.from_name,
            to_addresses_json=excluded.to_addresses_json,
            cc_addresses_json=excluded.cc_addresses_json,
            subject=excluded.subject,
            body_text=excluded.body_text,
            snippet=excluded.snippet,
            received_at=excluded.received_at,
            direction=excluded.direction,
            raw_json=excluded.raw_json,
            updated_at=excluded.updated_at
        """,
        (
            item.provider,
            item.mailbox,
            item.provider_message_id,
            item.thread_id,
            item.message_id_header,
            item.in_reply_to,
            item.references_header,
            item.from_address,
            item.from_name,
            json.dumps(item.to_addresses, ensure_ascii=False),
            json.dumps(item.cc_addresses, ensure_ascii=False),
            item.subject,
            item.body_text,
            item.snippet,
            item.received_at,
            item.direction,
            item.status,
            json.dumps(item.raw or {}, ensure_ascii=False),
            now,
            now,
        ),
    )
    row = fetch_one_sync(
        """
        SELECT id FROM support_email_messages
        WHERE provider = ? AND mailbox = ? AND provider_message_id = ?
        """,
        (item.provider, item.mailbox, item.provider_message_id),
    )
    if not row:
        raise ToolExecutionError("Support email message was not persisted")
    return int(row["id"])


async def sync_support_emails(*, limit: int | None = None, unread_only: bool = False) -> dict[str, Any]:
    started = utcnow_iso()
    run_id = execute_sync(
        """
        INSERT INTO support_email_sync_runs (provider, mailbox, status, started_at)
        VALUES (?, ?, 'running', ?)
        """,
        (_provider_name(), _mailbox_name(), started),
    )
    try:
        fetch_limit = limit or settings.email_fetch_limit
        messages = await asyncio.to_thread(_imap_fetch_recent, max(1, fetch_limit), unread_only)
        imported = 0
        for item in messages:
            upsert_support_email_message(item)
            imported += 1
        execute_sync(
            """
            UPDATE support_email_sync_runs
            SET status = 'success', scanned_count = ?, imported_count = ?, finished_at = ?
            WHERE id = ?
            """,
            (len(messages), imported, utcnow_iso(), run_id),
        )
        return {
            "run_id": int(run_id or 0),
            "status": "success",
            "scanned_count": len(messages),
            "imported_count": imported,
        }
    except Exception as err:
        execute_sync(
            """
            UPDATE support_email_sync_runs
            SET status = 'failed', finished_at = ?, error_message = ?
            WHERE id = ?
            """,
            (utcnow_iso(), str(err), run_id),
        )
        raise


async def list_support_emails(*, limit: int = 20, unread_only: bool = False, sync_first: bool = True) -> dict[str, Any]:
    sync_result = None
    if sync_first and settings.email_integration_enabled:
        sync_result = await sync_support_emails(limit=limit, unread_only=unread_only)

    rows = fetch_all_sync(
        """
        SELECT *
        FROM support_email_messages
        WHERE direction = 'inbound'
        ORDER BY COALESCE(received_at, created_at) DESC, id DESC
        LIMIT ?
        """,
        (max(1, min(limit, 100)),),
    )
    items = [_serialize_row(dict(row)) for row in rows]
    return {
        "total": len(items),
        "items": items,
        "sync": sync_result,
        "source": "imap" if sync_result else "snapshot",
    }


def read_support_email_thread(*, email_id: int | None = None, thread_id: str | None = None) -> dict[str, Any]:
    if email_id is not None and thread_id is None:
        row = fetch_one_sync("SELECT thread_id FROM support_email_messages WHERE id = ?", (email_id,))
        if not row:
            raise ToolValidationError("Email message not found")
        thread_id = str(row["thread_id"])
    if not thread_id:
        raise ToolValidationError("email_id or thread_id is required")

    rows = fetch_all_sync(
        """
        SELECT *
        FROM support_email_messages
        WHERE thread_id = ?
        ORDER BY COALESCE(received_at, created_at) ASC, id ASC
        """,
        (thread_id,),
    )
    if not rows:
        raise ToolValidationError("Email thread not found")
    messages = [_serialize_row(dict(row)) for row in rows]
    return {
        "thread_id": thread_id,
        "total": len(messages),
        "messages": messages,
    }


def _extract_order_code(text: str) -> str | None:
    match = _ORDER_CODE_RE.search(text.upper())
    return match.group(0).upper() if match else None


def create_ticket_from_email(
    *,
    email_id: int,
    issue_type: str,
    context: RequestContext,
    message_override: str | None = None,
) -> dict[str, Any]:
    row = fetch_one_sync("SELECT * FROM support_email_messages WHERE id = ?", (email_id,))
    if not row:
        raise ToolValidationError("Email message not found")
    if row.get("ticket_code"):
        return {
            "email_id": email_id,
            "thread_id": row["thread_id"],
            "ticket_code": row["ticket_code"],
            "issue_type": issue_type,
            "status": "open",
            "order_code": _extract_order_code(f"{row.get('subject') or ''}\n{row.get('body_text') or ''}"),
            "created_at": row.get("updated_at") or utcnow_iso(),
        }

    body = row.get("body_text") or row.get("snippet") or ""
    message = message_override or f"Email subject: {row.get('subject') or '(no subject)'}\nFrom: {row.get('from_address') or '-'}\n\n{body}"
    ticket = create_support_ticket(
        issue_type=issue_type,
        message=message[:2000],
        contact=row.get("from_address"),
        context=context,
    )
    execute_sync(
        """
        UPDATE support_email_messages
        SET ticket_code = ?, status = 'ticket_created', updated_at = ?
        WHERE id = ?
        """,
        (ticket["ticket_code"], utcnow_iso(), email_id),
    )
    return {
        "email_id": email_id,
        "thread_id": row["thread_id"],
        "ticket_code": ticket["ticket_code"],
        "issue_type": ticket["issue_type"],
        "status": ticket["status"],
        "order_code": _extract_order_code(f"{row.get('subject') or ''}\n{body}"),
        "created_at": ticket["created_at"],
    }


async def send_email_reply(*, email_id: int, body: str, context: RequestContext, to_address: str | None = None) -> dict[str, Any]:
    row = fetch_one_sync("SELECT * FROM support_email_messages WHERE id = ?", (email_id,))
    if not row:
        raise ToolValidationError("Email message not found")

    recipient = (to_address or row.get("from_address") or "").strip()
    if not recipient:
        raise ToolValidationError("Reply recipient is required")

    message_id = await asyncio.to_thread(
        _send_smtp_reply,
        to_address=recipient,
        subject=row.get("subject") or "(no subject)",
        body=body.strip(),
        original=row,
    )
    outbound = ParsedEmail(
        provider=_provider_name(),
        mailbox=_mailbox_name(),
        provider_message_id=f"outbound:{uuid.uuid4().hex}",
        thread_id=row["thread_id"],
        message_id_header=message_id,
        in_reply_to=row.get("message_id_header"),
        references_header=row.get("references_header") or row.get("message_id_header"),
        from_address=(settings.email_from_address or settings.email_smtp_username or settings.email_support_address or None),
        from_name=None,
        to_addresses=[recipient],
        cc_addresses=[],
        subject=row.get("subject") or "",
        body_text=body.strip(),
        snippet=" ".join(body.strip().split())[:260],
        received_at=utcnow_iso(),
        direction="outbound",
        status="sent",
        raw={"sent_by_user_id": context.auth.user_id},
    )
    outbound_id = upsert_support_email_message(outbound)
    execute_sync(
        "UPDATE support_email_messages SET status = 'replied', updated_at = ? WHERE id = ?",
        (utcnow_iso(), email_id),
    )
    return {
        "email_id": email_id,
        "thread_id": row["thread_id"],
        "reply_message_id": outbound_id,
        "to_address": recipient,
        "status": "sent",
    }
