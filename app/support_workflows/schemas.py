from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SupportIntent = Literal[
    "order_status",
    "refund_request",
    "cancel_order",
    "technical_issue",
    "account_access",
    "billing_invoice",
    "product_question",
    "complaint",
    "bug_report",
    "human_request",
    "unknown",
]

TicketLifecycleStatus = Literal[
    "new",
    "classified",
    "enriched",
    "planned",
    "waiting_customer",
    "waiting_approval",
    "escalated",
    "resolved",
    "closed",
]


class CaseClassification(BaseModel):
    intent: SupportIntent
    confidence: float = Field(ge=0.0, le=1.0)
    entities: dict[str, Any] = Field(default_factory=dict)
    customer_sentiment: str = "neutral"
    requires_auth: bool = False
    risk_level: str = "low"
    reason: str = ""


class PriorityAssessment(BaseModel):
    priority: Literal["P0", "P1", "P2", "P3"]
    sla_due_at: str
    assigned_team: str
    reason: str


class CaseContext(BaseModel):
    ticket: dict[str, Any]
    email_thread: dict[str, Any] | None = None
    previous_tickets: list[dict[str, Any]] = Field(default_factory=list)
    order_status: dict[str, Any] | None = None
    kb_suggestions: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)


class ActionPlanStep(BaseModel):
    type: str
    description: str
    tool: str | None = None
    risk: str = "low"
    status: str = "ready"
    requires_approval: bool = False
    result: dict[str, Any] | None = None


class ActionPlan(BaseModel):
    case_id: str
    goal: str
    requires_approval: bool = False
    should_escalate: bool = False
    steps: list[ActionPlanStep] = Field(default_factory=list)


class EscalationPackage(BaseModel):
    summary: str
    intent: str
    priority: str
    customer_sentiment: str
    entities: dict[str, Any] = Field(default_factory=dict)
    tools_used: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    suggested_next_action: str
    draft_reply: str | None = None
    conversation_transcript: str | None = None


class WorkflowResult(BaseModel):
    ticket_id: int
    ticket_code: str
    lifecycle_status: TicketLifecycleStatus
    classification: CaseClassification
    priority: PriorityAssessment
    context: CaseContext
    action_plan: ActionPlan
    pending_actions: list[dict[str, Any]] = Field(default_factory=list)
    escalation: EscalationPackage | None = None
    resolution_summary: str | None = None


class SupportTicketItem(BaseModel):
    id: int
    ticket_code: str
    issue_type: str
    message: str
    contact: str | None = None
    status: str
    workflow_status: str | None = None
    intent: str | None = None
    intent_confidence: float | None = None
    priority: str | None = None
    sla_due_at: str | None = None
    sla_breached_at: str | None = None
    assigned_team: str | None = None
    assigned_user_id: str | None = None
    risk_level: str | None = None
    sentiment: str | None = None
    resolution_summary: str | None = None
    created_by_user_id: str | None = None
    channel: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    kb_id: int | None = None
    kb_key: str | None = None
    created_at: str
    updated_at: str
    workflow_updated_at: str | None = None
    note_count: int = 0
    pending_action_count: int = 0


class ListSupportTicketsOutput(BaseModel):
    total: int
    items: list[SupportTicketItem]


class SupportTicketNoteItem(BaseModel):
    id: int
    ticket_id: int
    note_type: str
    visibility: str
    body: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_by_user_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    created_at: str


class ListSupportTicketNotesOutput(BaseModel):
    total: int
    items: list[SupportTicketNoteItem]


class AddTicketNoteInput(BaseModel):
    body: str = Field(..., min_length=1, max_length=4000)
    note_type: str = Field(default="internal", max_length=60)
    visibility: str = Field(default="internal", max_length=60)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssignTicketInput(BaseModel):
    assigned_team: str | None = Field(default=None, max_length=120)
    assigned_user_id: str | None = Field(default=None, max_length=120)
    note: str | None = Field(default=None, max_length=1000)


class UpdateTicketStatusInput(BaseModel):
    status: TicketLifecycleStatus
    resolution_summary: str | None = Field(default=None, max_length=4000)
    note: str | None = Field(default=None, max_length=1000)


class SlaMonitorResult(BaseModel):
    scanned: int
    breached: int
    escalated_ticket_ids: list[int] = Field(default_factory=list)
