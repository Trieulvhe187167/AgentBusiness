"""
Support-oriented tools.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from app.database import execute_sync, fetch_one_sync, utcnow_iso
from app.models import RequestContext
from app.tools.registry import ToolAuthPolicy, ToolSpec, ToolValidationError


class CreateSupportTicketInput(BaseModel):
    issue_type: Literal["payment", "shipping", "refund", "account", "technical", "other"]
    message: str = Field(..., min_length=5, max_length=2000)
    contact: str | None = Field(default=None, max_length=200)


class CreateSupportTicketOutput(BaseModel):
    ticket_code: str
    issue_type: str
    status: str
    contact: str | None = None
    created_by_user_id: str | None = None
    created_at: str


def _new_ticket_code() -> str:
    return f"TCK-{uuid.uuid4().hex[:10].upper()}"


async def _create_support_ticket_tool(payload: CreateSupportTicketInput, context: RequestContext) -> dict:
    auth = context.auth
    contact = payload.contact.strip() if payload.contact else None
    if not auth.user_id and not contact:
        raise ToolValidationError("Anonymous ticket creation requires contact information")

    ticket_code = _new_ticket_code()
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
            payload.issue_type,
            payload.message.strip(),
            contact,
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


def build_create_support_ticket_tool() -> ToolSpec:
    return ToolSpec(
        name="create_support_ticket",
        description="Create a support ticket for a customer issue and return the ticket code.",
        input_model=CreateSupportTicketInput,
        output_model=CreateSupportTicketOutput,
        auth_policy=ToolAuthPolicy(allow_anonymous=True, scope="support"),
        timeout_seconds=10,
        idempotent=False,
        handler=_create_support_ticket_tool,
        summarize_result=lambda payload: f"create_support_ticket created {payload.get('ticket_code', '')}",
    )
