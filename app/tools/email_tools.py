"""
Support email action tools.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.integrations.support_email import (
    create_ticket_from_email,
    list_support_emails,
    read_support_email_thread,
    send_email_reply,
)
from app.models import RequestContext
from app.tools.registry import ToolAuthPolicy, ToolSpec


class SupportEmailItem(BaseModel):
    id: int
    provider: str
    mailbox: str
    provider_message_id: str
    thread_id: str
    from_address: str | None = None
    from_name: str | None = None
    to_addresses: list[str] = Field(default_factory=list)
    cc_addresses: list[str] = Field(default_factory=list)
    subject: str
    snippet: str
    body_text: str = ""
    received_at: str | None = None
    direction: str
    status: str
    ticket_code: str | None = None


class SupportEmailSyncSummary(BaseModel):
    run_id: int
    status: str
    scanned_count: int
    imported_count: int


class ListSupportEmailsInput(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)
    unread_only: bool = False
    sync_first: bool = True


class ListSupportEmailsOutput(BaseModel):
    total: int
    items: list[SupportEmailItem]
    sync: SupportEmailSyncSummary | None = None
    source: str


class ReadEmailThreadInput(BaseModel):
    email_id: int | None = Field(default=None, ge=1)
    thread_id: str | None = Field(default=None, min_length=1, max_length=300)


class ReadEmailThreadOutput(BaseModel):
    thread_id: str
    total: int
    messages: list[SupportEmailItem]


class CreateTicketFromEmailInput(BaseModel):
    email_id: int = Field(..., ge=1)
    issue_type: Literal["payment", "shipping", "refund", "account", "technical", "other"] = "other"
    message_override: str | None = Field(default=None, max_length=2000)


class CreateTicketFromEmailRequest(BaseModel):
    issue_type: Literal["payment", "shipping", "refund", "account", "technical", "other"] = "other"
    message_override: str | None = Field(default=None, max_length=2000)


class CreateTicketFromEmailOutput(BaseModel):
    email_id: int
    thread_id: str
    ticket_code: str
    issue_type: str
    status: str
    order_code: str | None = None
    created_at: str


class SendEmailReplyInput(BaseModel):
    email_id: int = Field(..., ge=1)
    body: str = Field(..., min_length=2, max_length=4000)
    to_address: str | None = Field(default=None, max_length=300)


class SendEmailReplyRequest(BaseModel):
    body: str = Field(..., min_length=2, max_length=4000)
    to_address: str | None = Field(default=None, max_length=300)


class SendEmailReplyOutput(BaseModel):
    email_id: int
    thread_id: str
    reply_message_id: int
    to_address: str
    status: str


async def _list_support_emails_tool(payload: ListSupportEmailsInput, _: RequestContext) -> dict[str, Any]:
    return await list_support_emails(
        limit=payload.limit,
        unread_only=payload.unread_only,
        sync_first=payload.sync_first,
    )


async def _read_email_thread_tool(payload: ReadEmailThreadInput, _: RequestContext) -> dict[str, Any]:
    return read_support_email_thread(email_id=payload.email_id, thread_id=payload.thread_id)


async def _create_ticket_from_email_tool(payload: CreateTicketFromEmailInput, context: RequestContext) -> dict[str, Any]:
    return create_ticket_from_email(
        email_id=payload.email_id,
        issue_type=payload.issue_type,
        message_override=payload.message_override,
        context=context,
    )


async def _send_email_reply_tool(payload: SendEmailReplyInput, context: RequestContext) -> dict[str, Any]:
    return await send_email_reply(
        email_id=payload.email_id,
        body=payload.body,
        to_address=payload.to_address,
        context=context,
    )


def _support_email_policy(*, risk_level: str, scope: str = "support_email") -> ToolAuthPolicy:
    return ToolAuthPolicy(
        required_roles=["admin"],
        allowed_channels=["admin"],
        risk_level=risk_level,
        scope=scope,
    )


def build_list_support_emails_tool() -> ToolSpec:
    return ToolSpec(
        name="list_support_emails",
        description="List recent inbound support emails, optionally syncing the mailbox first.",
        input_model=ListSupportEmailsInput,
        output_model=ListSupportEmailsOutput,
        auth_policy=_support_email_policy(risk_level="high"),
        timeout_seconds=45,
        idempotent=True,
        handler=_list_support_emails_tool,
        summarize_result=lambda payload: f"listed {payload.get('total', 0)} support email(s)",
    )


def build_read_email_thread_tool() -> ToolSpec:
    return ToolSpec(
        name="read_email_thread",
        description="Read a support email thread from the local email snapshot.",
        input_model=ReadEmailThreadInput,
        output_model=ReadEmailThreadOutput,
        auth_policy=_support_email_policy(risk_level="high"),
        timeout_seconds=10,
        idempotent=True,
        handler=_read_email_thread_tool,
        summarize_result=lambda payload: f"read email thread {payload.get('thread_id')}",
    )


def build_create_ticket_from_email_tool() -> ToolSpec:
    return ToolSpec(
        name="create_ticket_from_email",
        description="Create a support ticket from an inbound support email and link the ticket code back to the email.",
        input_model=CreateTicketFromEmailInput,
        output_model=CreateTicketFromEmailOutput,
        auth_policy=_support_email_policy(risk_level="high", scope="support"),
        timeout_seconds=10,
        idempotent=False,
        handler=_create_ticket_from_email_tool,
        summarize_result=lambda payload: f"created ticket {payload.get('ticket_code')} from email {payload.get('email_id')}",
    )


def build_send_email_reply_tool() -> ToolSpec:
    return ToolSpec(
        name="send_email_reply",
        description="Send a plain-text reply to a support email thread through the configured SMTP backend.",
        input_model=SendEmailReplyInput,
        output_model=SendEmailReplyOutput,
        auth_policy=_support_email_policy(risk_level="critical"),
        timeout_seconds=30,
        idempotent=False,
        handler=_send_email_reply_tool,
        summarize_result=lambda payload: f"sent reply for email {payload.get('email_id')}",
    )
