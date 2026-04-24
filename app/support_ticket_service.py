"""
Shared support ticket persistence helpers.
"""

from __future__ import annotations

import uuid

from app.database import execute_sync, fetch_one_sync, utcnow_iso
from app.models import RequestContext
from app.tools.registry import ToolValidationError


def new_ticket_code() -> str:
    return f"TCK-{uuid.uuid4().hex[:10].upper()}"


def create_support_ticket(
    *,
    issue_type: str,
    message: str,
    contact: str | None,
    context: RequestContext,
) -> dict:
    auth = context.auth
    normalized_contact = contact.strip() if contact else None
    if not auth.user_id and not normalized_contact:
        raise ToolValidationError("Anonymous ticket creation requires contact information")

    ticket_code = new_ticket_code()
    now = utcnow_iso()
    execute_sync(
        """
        INSERT INTO support_tickets (
            ticket_code,
            issue_type,
            message,
            contact,
            status,
            created_by_user_id,
            channel,
            tenant_id,
            org_id,
            kb_id,
            kb_key,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticket_code,
            issue_type,
            message.strip(),
            normalized_contact,
            auth.user_id,
            auth.channel,
            auth.tenant_id,
            auth.org_id,
            context.kb_id,
            context.kb_key,
            now,
            now,
        ),
    )
    row = fetch_one_sync(
        """
        SELECT ticket_code, issue_type, status, contact, created_by_user_id, created_at
        FROM support_tickets
        WHERE ticket_code = ?
        """,
        (ticket_code,),
    )
    if not row:
        raise ToolValidationError("Support ticket was not persisted")
    return dict(row)
