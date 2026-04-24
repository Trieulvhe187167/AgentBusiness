"""
Support-oriented tools.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models import RequestContext
from app.support_ticket_service import create_support_ticket
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


async def _create_support_ticket_tool(payload: CreateSupportTicketInput, context: RequestContext) -> dict:
    return create_support_ticket(
        issue_type=payload.issue_type,
        message=payload.message,
        contact=payload.contact,
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
