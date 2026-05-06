"""
Support-oriented tools.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models import RequestContext
from app.support_ticket_service import create_support_ticket
from app.support_workflows import add_ticket_note, assign_ticket, list_support_tickets, update_ticket_status
from app.support_workflows.schemas import (
    AddTicketNoteInput,
    AssignTicketInput,
    ListSupportTicketsOutput,
    SupportTicketItem,
    SupportTicketNoteItem,
    UpdateTicketStatusInput,
)
from app.tools.registry import ToolAuthPolicy, ToolSpec


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


class ListCustomerTicketsInput(BaseModel):
    contact: str | None = Field(default=None, max_length=200)
    user_id: str | None = Field(default=None, max_length=120)
    limit: int = Field(default=10, ge=1, le=50)


class UpdateTicketStatusToolInput(UpdateTicketStatusInput):
    ticket_id: int = Field(..., ge=1)


class AssignTicketToolInput(AssignTicketInput):
    ticket_id: int = Field(..., ge=1)


class AddTicketInternalNoteToolInput(BaseModel):
    ticket_id: int = Field(..., ge=1)
    body: str = Field(..., min_length=1, max_length=4000)


async def _create_support_ticket_tool(payload: CreateSupportTicketInput, context: RequestContext) -> dict:
    return create_support_ticket(
        issue_type=payload.issue_type,
        message=payload.message,
        contact=payload.contact,
        context=context,
    )


async def _list_customer_tickets_tool(payload: ListCustomerTicketsInput, context: RequestContext) -> dict:
    user_id = payload.user_id or context.auth.user_id
    if not payload.contact and not user_id:
        return {"total": 0, "items": []}
    return list_support_tickets(
        contact=payload.contact,
        created_by_user_id=user_id,
        limit=payload.limit,
    )


async def _update_ticket_status_tool(payload: UpdateTicketStatusToolInput, context: RequestContext) -> dict:
    return update_ticket_status(
        payload.ticket_id,
        UpdateTicketStatusInput(
            status=payload.status,
            resolution_summary=payload.resolution_summary,
            note=payload.note,
        ),
        context=context,
    )


async def _assign_ticket_tool(payload: AssignTicketToolInput, context: RequestContext) -> dict:
    return assign_ticket(
        payload.ticket_id,
        AssignTicketInput(
            assigned_team=payload.assigned_team,
            assigned_user_id=payload.assigned_user_id,
            note=payload.note,
        ),
        context=context,
    )


async def _add_ticket_internal_note_tool(payload: AddTicketInternalNoteToolInput, context: RequestContext) -> dict:
    return add_ticket_note(
        payload.ticket_id,
        AddTicketNoteInput(body=payload.body, note_type="internal", visibility="internal"),
        context=context,
    )


def build_create_support_ticket_tool() -> ToolSpec:
    return ToolSpec(
        name="create_support_ticket",
        description="Create a support ticket for a customer issue and return the ticket code.",
        input_model=CreateSupportTicketInput,
        output_model=CreateSupportTicketOutput,
        auth_policy=ToolAuthPolicy(
            allow_anonymous=True,
            allowed_channels=["web", "chat", "admin"],
            risk_level="medium",
            scope="support",
        ),
        timeout_seconds=10,
        idempotent=False,
        handler=_create_support_ticket_tool,
        summarize_result=lambda payload: f"create_support_ticket created {payload.get('ticket_code', '')}",
    )


def _admin_support_policy(*, risk_level: str = "medium") -> ToolAuthPolicy:
    return ToolAuthPolicy(
        required_roles=["admin"],
        allowed_channels=["admin"],
        risk_level=risk_level,
        scope="support",
    )


def build_list_customer_tickets_tool() -> ToolSpec:
    return ToolSpec(
        name="list_customer_tickets",
        description="List recent support tickets for a customer by contact email or authenticated user id.",
        input_model=ListCustomerTicketsInput,
        output_model=ListSupportTicketsOutput,
        auth_policy=_admin_support_policy(risk_level="medium"),
        timeout_seconds=10,
        idempotent=True,
        handler=_list_customer_tickets_tool,
        summarize_result=lambda payload: f"listed {payload.get('total', 0)} customer ticket(s)",
    )


def build_update_ticket_status_tool() -> ToolSpec:
    return ToolSpec(
        name="update_ticket_status",
        description="Update a support ticket lifecycle status and optionally store a resolution summary or internal note.",
        input_model=UpdateTicketStatusToolInput,
        output_model=SupportTicketItem,
        auth_policy=_admin_support_policy(risk_level="medium"),
        timeout_seconds=10,
        idempotent=False,
        handler=_update_ticket_status_tool,
        summarize_result=lambda payload: f"updated ticket {payload.get('ticket_code')} to {payload.get('status')}",
    )


def build_assign_ticket_tool() -> ToolSpec:
    return ToolSpec(
        name="assign_ticket",
        description="Assign a support ticket to a team or human owner.",
        input_model=AssignTicketToolInput,
        output_model=SupportTicketItem,
        auth_policy=_admin_support_policy(risk_level="medium"),
        timeout_seconds=10,
        idempotent=False,
        handler=_assign_ticket_tool,
        summarize_result=lambda payload: f"assigned ticket {payload.get('ticket_code')}",
    )


def build_add_ticket_internal_note_tool() -> ToolSpec:
    return ToolSpec(
        name="add_ticket_internal_note",
        description="Add an internal note to a support ticket for handoff, audit, or case history.",
        input_model=AddTicketInternalNoteToolInput,
        output_model=SupportTicketNoteItem,
        auth_policy=_admin_support_policy(risk_level="low"),
        timeout_seconds=10,
        idempotent=False,
        handler=_add_ticket_internal_note_tool,
        summarize_result=lambda payload: f"added note {payload.get('id')} to ticket {payload.get('ticket_id')}",
    )
