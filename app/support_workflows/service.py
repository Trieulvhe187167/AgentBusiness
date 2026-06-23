from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.integrations.live_data import get_order_status
from app.integrations.support_email import create_ticket_from_email, read_support_email_thread
from app.models import RequestContext
from app.pending_actions import create_pending_action
from app.support_workflows.classifier import classify_text
from app.support_workflows.priority import assign_priority
from app.support_workflows.schemas import (
    ActionPlan,
    ActionPlanStep,
    AddTicketNoteInput,
    AssignTicketInput,
    CaseClassification,
    CaseContext,
    EscalationPackage,
    ListSupportTicketNotesOutput,
    ListSupportTicketsOutput,
    PriorityAssessment,
    SlaMonitorResult,
    SupportCannedActionInput,
    SupportCannedActionOutput,
    SupportTicketItem,
    SupportTicketNoteItem,
    UpdateTicketStatusInput,
    WorkflowResult,
)
from app.workflow_engine import WorkflowRecorder


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _ticket_by_id(ticket_id: int) -> dict[str, Any]:
    row = fetch_one_sync("SELECT * FROM support_tickets WHERE id = ?", (ticket_id,))
    if not row:
        raise ValueError("Support ticket not found")
    return row


def _ticket_by_code(ticket_code: str) -> dict[str, Any]:
    row = fetch_one_sync("SELECT * FROM support_tickets WHERE ticket_code = ?", (ticket_code,))
    if not row:
        raise ValueError("Support ticket not found")
    return row


def _ticket_text(ticket: dict[str, Any]) -> str:
    return f"{ticket.get('issue_type') or ''}\n{ticket.get('message') or ''}\n{ticket.get('contact') or ''}"


def _parse_json_field(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _roles_json(context: RequestContext) -> str:
    return _json_dumps(context.auth.roles)


def _serialize_note(row: dict[str, Any]) -> dict[str, Any]:
    return SupportTicketNoteItem(
        id=int(row["id"]),
        ticket_id=int(row["ticket_id"]),
        note_type=row["note_type"],
        visibility=row["visibility"],
        body=row["body"],
        metadata=_parse_json_field(row.get("metadata_json"), {}),
        created_by_user_id=row.get("created_by_user_id"),
        roles=_parse_json_field(row.get("roles_json"), []),
        created_at=row["created_at"],
    ).model_dump()


def _next_action_for_ticket(row: dict[str, Any]) -> dict[str, Any]:
    status = row.get("workflow_status") or row.get("status") or "new"
    priority = row.get("priority") or "unclassified"
    risk = row.get("risk_level") or "unknown"
    pending_count = int(row.get("pending_action_count") or 0)
    breached = bool(row.get("sla_breached_at"))
    assigned = bool(row.get("assigned_team") or row.get("assigned_user_id"))
    if breached:
        return {
            "key": "sla_breached",
            "label": "Escalate SLA breach",
            "description": "SLA is overdue. Assign an owner and send a customer-facing update.",
            "severity": "critical",
            "requires_approval": False,
        }
    if status == "waiting_approval" or pending_count:
        return {
            "key": "approval_review",
            "label": "Review approval",
            "description": "A support_case_review action is waiting in the approval queue.",
            "severity": "warning",
            "requires_approval": True,
        }
    if not assigned and status not in {"resolved", "closed"}:
        return {
            "key": "assign_owner",
            "label": "Assign owner",
            "description": "Assign a support owner before continuing the case.",
            "severity": "warning" if priority in {"P0", "P1"} else "info",
            "requires_approval": False,
        }
    if status in {"new", "open", "classified", "planned"}:
        return {
            "key": "draft_reply",
            "label": "Generate draft reply",
            "description": "Generate or review a customer-facing reply with evidence before sending.",
            "severity": "warning" if risk in {"high", "critical"} else "info",
            "requires_approval": risk in {"high", "critical"},
        }
    if status == "waiting_customer":
        return {
            "key": "wait_customer",
            "label": "Wait for customer",
            "description": "The last public reply is sent. Reopen when the employee responds.",
            "severity": "info",
            "requires_approval": False,
        }
    if status == "escalated":
        return {
            "key": "human_followup",
            "label": "Human follow-up",
            "description": "Escalation package is ready. Support owner should reply or resolve.",
            "severity": "warning",
            "requires_approval": risk in {"high", "critical"},
        }
    return {
        "key": "none",
        "label": "No next action",
        "description": "This case is in a terminal or low-touch state.",
        "severity": "info",
        "requires_approval": False,
    }


def _serialize_ticket(row: dict[str, Any]) -> dict[str, Any]:
    return SupportTicketItem(
        id=int(row["id"]),
        ticket_code=row["ticket_code"],
        issue_type=row["issue_type"],
        message=row["message"],
        contact=row.get("contact"),
        status=row["status"],
        workflow_status=row.get("workflow_status"),
        intent=row.get("intent"),
        intent_confidence=row.get("intent_confidence"),
        priority=row.get("priority"),
        sla_due_at=row.get("sla_due_at"),
        sla_breached_at=row.get("sla_breached_at"),
        assigned_team=row.get("assigned_team"),
        assigned_user_id=row.get("assigned_user_id"),
        risk_level=row.get("risk_level"),
        sentiment=row.get("sentiment"),
        resolution_summary=row.get("resolution_summary"),
        created_by_user_id=row.get("created_by_user_id"),
        channel=row.get("channel"),
        tenant_id=row.get("tenant_id"),
        org_id=row.get("org_id"),
        kb_id=row.get("kb_id"),
        kb_key=row.get("kb_key"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        workflow_updated_at=row.get("workflow_updated_at"),
        note_count=int(row.get("note_count") or 0),
        pending_action_count=int(row.get("pending_action_count") or 0),
        next_action=_next_action_for_ticket(row),
    ).model_dump()


def list_support_tickets(
    *,
    status: str | None = None,
    workflow_status: str | None = None,
    priority: str | None = None,
    assigned_user_id: str | None = None,
    contact: str | None = None,
    created_by_user_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("t.status = ?")
        params.append(status)
    if workflow_status:
        clauses.append("t.workflow_status = ?")
        params.append(workflow_status)
    if priority:
        clauses.append("t.priority = ?")
        params.append(priority)
    if assigned_user_id:
        clauses.append("t.assigned_user_id = ?")
        params.append(assigned_user_id)
    if contact:
        clauses.append("t.contact = ?")
        params.append(contact)
    if created_by_user_id:
        clauses.append("t.created_by_user_id = ?")
        params.append(created_by_user_id)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, min(limit, 200)))
    rows = fetch_all_sync(
        f"""
        SELECT t.*,
               (SELECT COUNT(*) FROM support_ticket_notes n WHERE n.ticket_id = t.id) AS note_count,
               (SELECT COUNT(*) FROM pending_actions p
                WHERE p.action_type = 'support_case_review'
                  AND json_extract(p.payload_json, '$.ticket_id') = t.id
                  AND p.status IN ('draft', 'approved')) AS pending_action_count
        FROM support_tickets t
        {where_sql}
        ORDER BY
            CASE t.priority WHEN 'P0' THEN 0 WHEN 'P1' THEN 1 WHEN 'P2' THEN 2 WHEN 'P3' THEN 3 ELSE 4 END,
            COALESCE(t.sla_due_at, t.updated_at) ASC,
            t.id DESC
        LIMIT ?
        """,
        tuple(params),
    )
    items = [_serialize_ticket(row) for row in rows]
    return ListSupportTicketsOutput(total=len(items), items=items).model_dump()


def get_support_ticket(ticket_id: int) -> dict[str, Any]:
    row = fetch_one_sync(
        """
        SELECT t.*,
               (SELECT COUNT(*) FROM support_ticket_notes n WHERE n.ticket_id = t.id) AS note_count,
               (SELECT COUNT(*) FROM pending_actions p
                WHERE p.action_type = 'support_case_review'
                  AND json_extract(p.payload_json, '$.ticket_id') = t.id
                  AND p.status IN ('draft', 'approved')) AS pending_action_count
        FROM support_tickets t
        WHERE t.id = ?
        """,
        (ticket_id,),
    )
    if not row:
        raise ValueError("Support ticket not found")
    return _serialize_ticket(row)


def get_support_ticket_by_code(ticket_code: str) -> dict[str, Any]:
    row = fetch_one_sync(
        """
        SELECT t.*,
               (SELECT COUNT(*) FROM support_ticket_notes n WHERE n.ticket_id = t.id) AS note_count,
               (SELECT COUNT(*) FROM pending_actions p
                WHERE p.action_type = 'support_case_review'
                  AND json_extract(p.payload_json, '$.ticket_id') = t.id
                  AND p.status IN ('draft', 'approved')) AS pending_action_count
        FROM support_tickets t
        WHERE t.ticket_code = ?
        """,
        (ticket_code,),
    )
    if not row:
        raise ValueError("Support ticket not found")
    return _serialize_ticket(row)


def list_ticket_notes(ticket_id: int, *, visibility: str | None = None) -> dict[str, Any]:
    _ticket_by_id(ticket_id)
    params: list[Any] = [ticket_id]
    visibility_sql = ""
    if visibility:
        visibility_sql = "AND visibility = ?"
        params.append(visibility)
    rows = fetch_all_sync(
        f"""
        SELECT *
        FROM support_ticket_notes
        WHERE ticket_id = ?
        {visibility_sql}
        ORDER BY created_at DESC, id DESC
        """,
        tuple(params),
    )
    items = [_serialize_note(row) for row in rows]
    return ListSupportTicketNotesOutput(total=len(items), items=items).model_dump()


def add_ticket_note(ticket_id: int, payload: AddTicketNoteInput, *, context: RequestContext) -> dict[str, Any]:
    _ticket_by_id(ticket_id)
    now = utcnow_iso()
    note_id = execute_sync(
        """
        INSERT INTO support_ticket_notes (
            ticket_id, note_type, visibility, body, metadata_json,
            created_by_user_id, roles_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticket_id,
            payload.note_type.strip() or "internal",
            payload.visibility.strip() or "internal",
            payload.body.strip(),
            _json_dumps(payload.metadata),
            context.auth.user_id,
            _roles_json(context),
            now,
        ),
    )
    execute_sync("UPDATE support_tickets SET updated_at = ? WHERE id = ?", (now, ticket_id))
    row = fetch_one_sync("SELECT * FROM support_ticket_notes WHERE id = ?", (int(note_id or 0),))
    if not row:
        raise ValueError("Support ticket note was not persisted")
    return _serialize_note(row)


def assign_ticket(ticket_id: int, payload: AssignTicketInput, *, context: RequestContext) -> dict[str, Any]:
    _ticket_by_id(ticket_id)
    now = utcnow_iso()
    execute_sync(
        """
        UPDATE support_tickets
        SET assigned_team = ?,
            assigned_user_id = ?,
            updated_at = ?,
            workflow_updated_at = ?
        WHERE id = ?
        """,
        (
            payload.assigned_team.strip() if payload.assigned_team else None,
            payload.assigned_user_id.strip() if payload.assigned_user_id else None,
            now,
            now,
            ticket_id,
        ),
    )
    if payload.note:
        add_ticket_note(
            ticket_id,
            AddTicketNoteInput(body=payload.note, note_type="assignment", visibility="internal"),
            context=context,
        )
    return get_support_ticket(ticket_id)


def update_ticket_status(ticket_id: int, payload: UpdateTicketStatusInput, *, context: RequestContext) -> dict[str, Any]:
    _ticket_by_id(ticket_id)
    now = utcnow_iso()
    execute_sync(
        """
        UPDATE support_tickets
        SET status = ?,
            workflow_status = ?,
            resolution_summary = COALESCE(?, resolution_summary),
            workflow_updated_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (payload.status, payload.status, payload.resolution_summary, now, now, ticket_id),
    )
    if payload.note:
        add_ticket_note(
            ticket_id,
            AddTicketNoteInput(body=payload.note, note_type="status_change", visibility="internal"),
            context=context,
        )
    return get_support_ticket(ticket_id)


def _default_public_reply(ticket: dict[str, Any], action: str) -> str:
    if action == "ask_more_info":
        return (
            "Thank you for contacting support. Could you please share the missing details "
            "so we can verify your request and respond accurately?"
        )
    if action == "resolve_with_kb":
        return (
            "Thank you for contacting support. Based on the available knowledge base information, "
            "we have provided the relevant answer and will mark this case resolved. Reply here if you need more help."
        )
    return "Support has reviewed this case and will follow up with the next step."


def _canned_approval_payload(ticket: dict[str, Any], payload: SupportCannedActionInput) -> dict[str, Any]:
    action_label = "refund" if payload.action == "refund_requires_approval" else "cancellation"
    return {
        "ticket_id": int(ticket["id"]),
        "ticket_code": ticket["ticket_code"],
        "canned_action": payload.action,
        "customer_impact": action_label,
        "requested_reply": payload.reply_body,
        "operator_note": payload.note,
        "source": "support_workspace_canned_action",
        "approval_reason": f"{action_label.title()} action requires human approval before execution.",
    }


def apply_canned_support_action(
    ticket_id: int,
    payload: SupportCannedActionInput,
    *,
    context: RequestContext,
) -> dict[str, Any]:
    ticket = _ticket_by_id(ticket_id)
    action = payload.action
    note: dict[str, Any] | None = None
    pending_action: dict[str, Any] | None = None

    if action == "ask_more_info":
        body = payload.reply_body or _default_public_reply(ticket, action)
        note = add_ticket_note(
            ticket_id,
            AddTicketNoteInput(
                body=body,
                note_type="public_more_info_request",
                visibility="public",
                metadata={"source": "canned_action", "action": action},
            ),
            context=context,
        )
        updated = update_ticket_status(
            ticket_id,
            UpdateTicketStatusInput(
                status="waiting_customer",
                note=payload.note or "Canned action: asked customer for more information.",
            ),
            context=context,
        )
        message = "Asked the employee for more information."
    elif action == "resolve_with_kb":
        body = payload.reply_body or _default_public_reply(ticket, action)
        note = add_ticket_note(
            ticket_id,
            AddTicketNoteInput(
                body=body,
                note_type="public_resolution",
                visibility="public",
                metadata={"source": "canned_action", "action": action},
            ),
            context=context,
        )
        updated = update_ticket_status(
            ticket_id,
            UpdateTicketStatusInput(
                status="resolved",
                resolution_summary=body,
                note=payload.note or "Canned action: resolved with KB-backed reply.",
            ),
            context=context,
        )
        message = "Sent a KB-backed resolution and marked the case resolved."
    elif action == "escalate_to_team":
        team = payload.assigned_team or ticket.get("assigned_team") or "support"
        assigned_user_id = payload.assigned_user_id or ticket.get("assigned_user_id")
        assign_ticket(
            ticket_id,
            AssignTicketInput(
                assigned_team=team,
                assigned_user_id=assigned_user_id,
                note=payload.note or f"Canned action: escalated to {team}.",
            ),
            context=context,
        )
        updated_context = escalate_ticket(
            ticket_id,
            reason=payload.note or f"Canned action: escalate to {team}.",
            context=context,
        )
        updated = updated_context["ticket"]
        message = f"Escalated case to {team}."
    elif action in {"refund_requires_approval", "cancel_requires_approval"}:
        high_risk = "refund" if action == "refund_requires_approval" else "cancel"
        pending_action = create_pending_action(
            action_type="support_case_review",
            risk_level="high",
            title=f"Approve {high_risk} action for {ticket['ticket_code']}",
            summary=f"{high_risk.title()} action requested from Support Workspace. Approval required before customer-impacting execution.",
            payload=_canned_approval_payload(ticket, payload),
            context=context,
        )
        add_ticket_note(
            ticket_id,
            AddTicketNoteInput(
                body=payload.note or f"Canned action drafted: {high_risk} requires approval.",
                note_type="approval_request",
                visibility="internal",
                metadata={"source": "canned_action", "action": action, "pending_action_id": pending_action.get("id")},
            ),
            context=context,
        )
        updated = update_ticket_status(
            ticket_id,
            UpdateTicketStatusInput(status="waiting_approval", note=f"{high_risk.title()} action is waiting for approval."),
            context=context,
        )
        message = f"{high_risk.title()} approval action created."
    else:
        raise ValueError(f"Unsupported canned action: {action}")

    return SupportCannedActionOutput(
        ticket=SupportTicketItem(**updated),
        action=action,
        message=message,
        note=SupportTicketNoteItem(**note) if note else None,
        pending_action=pending_action,
        next_action=updated.get("next_action") or {},
    ).model_dump()


def classify_ticket(ticket_id: int) -> CaseClassification:
    ticket = _ticket_by_id(ticket_id)
    classification = classify_text(_ticket_text(ticket), issue_type=ticket.get("issue_type"))
    _update_ticket_workflow(
        ticket_id,
        lifecycle_status="classified",
        classification=classification,
    )
    return classification


async def _build_context(ticket: dict[str, Any], classification: CaseClassification, context: RequestContext) -> CaseContext:
    findings: list[str] = []
    email_thread = None
    email_row = fetch_one_sync(
        "SELECT id, thread_id FROM support_email_messages WHERE ticket_code = ? ORDER BY id DESC LIMIT 1",
        (ticket["ticket_code"],),
    )
    if email_row:
        email_thread = read_support_email_thread(thread_id=email_row["thread_id"])
        findings.append(f"Loaded email thread {email_row['thread_id']} with {email_thread.get('total', 0)} message(s).")

    previous_tickets = fetch_all_sync(
        """
        SELECT id, ticket_code, issue_type, status, intent, priority, created_at
        FROM support_tickets
        WHERE id != ?
          AND (
            (? IS NOT NULL AND contact = ?)
            OR (? IS NOT NULL AND created_by_user_id = ?)
          )
        ORDER BY created_at DESC
        LIMIT 5
        """,
        (
            ticket["id"],
            ticket.get("contact"),
            ticket.get("contact"),
            ticket.get("created_by_user_id"),
            ticket.get("created_by_user_id"),
        ),
    )
    if previous_tickets:
        findings.append(f"Found {len(previous_tickets)} previous support ticket(s).")

    order_status = None
    order_code = classification.entities.get("order_code")
    if order_code:
        try:
            order_status = await get_order_status(str(order_code), user_id=context.auth.user_id or ticket.get("created_by_user_id"))
            findings.append(f"Order {order_code} status is {order_status.get('status')}.")
        except Exception as err:
            findings.append(f"Order lookup failed for {order_code}: {err}")

    return CaseContext(
        ticket=ticket,
        email_thread=email_thread,
        previous_tickets=previous_tickets,
        order_status=order_status,
        kb_suggestions=[],
        findings=findings,
    )


def _build_action_plan(ticket: dict[str, Any], classification: CaseClassification, priority: PriorityAssessment, case_context: CaseContext) -> ActionPlan:
    steps: list[ActionPlanStep] = []
    order_code = classification.entities.get("order_code")
    if order_code:
        steps.append(
            ActionPlanStep(
                type="tool_call",
                tool="get_order_status",
                description=f"Check current status for order {order_code}.",
                risk="low",
                status="done" if case_context.order_status else "failed",
                result=case_context.order_status,
            )
        )

    steps.append(
        ActionPlanStep(
            type="context_review",
            description="Review ticket, prior tickets, email thread, and operational findings.",
            risk="low",
            status="done",
            result={"findings": case_context.findings},
        )
    )

    requires_approval = classification.risk_level in {"high", "critical"} or classification.intent in {"refund_request", "cancel_order"}
    should_escalate = (
        requires_approval
        or classification.intent in {"human_request", "unknown"}
        or classification.confidence < 0.7
        or priority.priority in {"P0", "P1"}
    )

    if requires_approval:
        steps.append(
            ActionPlanStep(
                type="human_approval",
                description=f"Create review task for {classification.intent}.",
                risk=classification.risk_level,
                status="requires_review",
                requires_approval=True,
            )
        )
    elif should_escalate:
        steps.append(
            ActionPlanStep(
                type="escalation",
                description="Escalate case to human support with context package.",
                risk="medium",
                status="ready",
            )
        )
    else:
        steps.append(
            ActionPlanStep(
                type="auto_resolution",
                description="Resolve low-risk case with gathered operational context.",
                risk="low",
                status="done",
            )
        )

    return ActionPlan(
        case_id=ticket["ticket_code"],
        goal=f"Handle {classification.intent} support case with priority {priority.priority}.",
        requires_approval=requires_approval,
        should_escalate=should_escalate,
        steps=steps,
    )


def _build_escalation(
    ticket: dict[str, Any],
    classification: CaseClassification,
    priority: PriorityAssessment,
    case_context: CaseContext,
    *,
    suggested_next_action: str,
) -> EscalationPackage:
    messages = []
    if case_context.email_thread:
        for msg in case_context.email_thread.get("messages", []):
            messages.append(f"[{msg.get('received_at') or '-'}] {msg.get('from_address') or '-'}: {msg.get('snippet') or ''}")
    return EscalationPackage(
        summary=f"Ticket {ticket['ticket_code']} classified as {classification.intent}.",
        intent=classification.intent,
        priority=priority.priority,
        customer_sentiment=classification.customer_sentiment,
        entities=classification.entities,
        tools_used=[step.tool for step in _build_action_plan(ticket, classification, priority, case_context).steps if step.tool],
        findings=case_context.findings,
        suggested_next_action=suggested_next_action,
        draft_reply=_draft_reply(ticket, classification, case_context),
        conversation_transcript="\n".join(messages) if messages else ticket.get("message"),
    )


def _draft_reply(ticket: dict[str, Any], classification: CaseClassification, case_context: CaseContext) -> str:
    order = case_context.order_status
    if classification.intent == "order_status" and order:
        return (
            f"Chúng tôi đã kiểm tra đơn {order.get('order_code')}. "
            f"Trạng thái hiện tại: {order.get('status')}. "
            "Nếu bạn cần thêm hỗ trợ, vui lòng phản hồi lại ticket này."
        )
    return (
        "Chúng tôi đã tiếp nhận yêu cầu và đang chuyển thông tin cần thiết cho đội hỗ trợ phụ trách. "
        "Bạn sẽ nhận được phản hồi trong thời gian SLA."
    )


def _create_review_pending_action(
    ticket: dict[str, Any],
    classification: CaseClassification,
    priority: PriorityAssessment,
    case_context: CaseContext,
    context: RequestContext,
) -> dict[str, Any]:
    return create_pending_action(
        action_type="support_case_review",
        risk_level=classification.risk_level,
        title=f"Review support case {ticket['ticket_code']}: {classification.intent}",
        summary=f"{priority.priority} {classification.intent}. {priority.reason}",
        payload={
            "ticket_id": ticket["id"],
            "ticket_code": ticket["ticket_code"],
            "classification": classification.model_dump(),
            "priority": priority.model_dump(),
            "context_findings": case_context.findings,
            "draft_reply": _draft_reply(ticket, classification, case_context),
        },
        context=context,
    )


def _update_ticket_workflow(
    ticket_id: int,
    *,
    lifecycle_status: str,
    classification: CaseClassification | None = None,
    priority: PriorityAssessment | None = None,
    case_context: CaseContext | None = None,
    action_plan: ActionPlan | None = None,
    escalation: EscalationPackage | None = None,
    resolution_summary: str | None = None,
) -> None:
    now = utcnow_iso()
    execute_sync(
        """
        UPDATE support_tickets
        SET status = ?,
            workflow_status = ?,
            intent = COALESCE(?, intent),
            intent_confidence = COALESCE(?, intent_confidence),
            priority = COALESCE(?, priority),
            sla_due_at = COALESCE(?, sla_due_at),
            assigned_team = COALESCE(?, assigned_team),
            risk_level = COALESCE(?, risk_level),
            sentiment = COALESCE(?, sentiment),
            resolution_summary = COALESCE(?, resolution_summary),
            classification_json = COALESCE(?, classification_json),
            context_summary_json = COALESCE(?, context_summary_json),
            action_plan_json = COALESCE(?, action_plan_json),
            escalation_package_json = COALESCE(?, escalation_package_json),
            workflow_updated_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            lifecycle_status,
            lifecycle_status,
            classification.intent if classification else None,
            classification.confidence if classification else None,
            priority.priority if priority else None,
            priority.sla_due_at if priority else None,
            priority.assigned_team if priority else None,
            classification.risk_level if classification else None,
            classification.customer_sentiment if classification else None,
            resolution_summary,
            _json_dumps(classification.model_dump()) if classification else None,
            _json_dumps(case_context.model_dump()) if case_context else None,
            _json_dumps(action_plan.model_dump()) if action_plan else None,
            _json_dumps(escalation.model_dump()) if escalation else None,
            now,
            now,
            ticket_id,
        ),
    )


async def handle_ticket_case(
    ticket_id: int,
    context: RequestContext,
    *,
    workflow_run_id: int | None = None,
) -> WorkflowResult:
    recorder = WorkflowRecorder.start(
        workflow_type="support_ticket_case",
        entity_type="support_ticket",
        entity_id=ticket_id,
        input_payload={"ticket_id": ticket_id},
        context=context,
        run_id=workflow_run_id,
    )
    try:
        ticket = recorder.step(
            "load_ticket",
            "read",
            lambda: _ticket_by_id(ticket_id),
            input_payload={"ticket_id": ticket_id},
        )
        classification = recorder.step(
            "classify_case",
            "classifier",
            lambda: classify_text(_ticket_text(ticket), issue_type=ticket.get("issue_type")),
            input_payload={"issue_type": ticket.get("issue_type")},
        )
        priority = recorder.step(
            "assign_priority",
            "rule",
            lambda: assign_priority(classification),
            input_payload={"intent": classification.intent, "risk_level": classification.risk_level},
        )
        case_context = await recorder.async_step(
            "enrich_context",
            "context_builder",
            lambda: _build_context(ticket, classification, context),
            input_payload={"ticket_id": ticket_id, "intent": classification.intent},
        )
        plan = recorder.step(
            "build_action_plan",
            "planner",
            lambda: _build_action_plan(ticket, classification, priority, case_context),
            input_payload={"ticket_id": ticket_id, "priority": priority.priority},
        )

        pending_actions: list[dict[str, Any]] = []
        escalation = None
        resolution_summary = None
        lifecycle = "planned"

        if plan.requires_approval:
            pending_actions = recorder.step(
                "create_pending_approval",
                "approval",
                lambda: [_create_review_pending_action(ticket, classification, priority, case_context, context)],
                input_payload={"ticket_id": ticket_id, "risk_level": classification.risk_level},
            )
            lifecycle = "waiting_approval"
            escalation = recorder.step(
                "build_escalation_package",
                "handoff",
                lambda: _build_escalation(
                    ticket,
                    classification,
                    priority,
                    case_context,
                    suggested_next_action="Review approval package and decide the customer-impacting action.",
                ),
                input_payload={"reason": "approval_required"},
            )
        elif plan.should_escalate:
            lifecycle = "escalated"
            escalation = recorder.step(
                "build_escalation_package",
                "handoff",
                lambda: _build_escalation(
                    ticket,
                    classification,
                    priority,
                    case_context,
                    suggested_next_action="Human support should review the case context and respond.",
                ),
                input_payload={"reason": "plan_should_escalate"},
            )
        else:
            lifecycle = "resolved"
            resolution_summary = recorder.step(
                "draft_resolution",
                "response",
                lambda: _draft_reply(ticket, classification, case_context),
                input_payload={"ticket_id": ticket_id, "intent": classification.intent},
            )

        recorder.step(
            "update_ticket_workflow",
            "write",
            lambda: _update_ticket_workflow(
                ticket_id,
                lifecycle_status=lifecycle,
                classification=classification,
                priority=priority,
                case_context=case_context,
                action_plan=plan,
                escalation=escalation,
                resolution_summary=resolution_summary,
            )
            or {"workflow_status": lifecycle},
            input_payload={"ticket_id": ticket_id, "workflow_status": lifecycle},
        )
        updated_ticket = _ticket_by_id(ticket_id)
        case_context.ticket = updated_ticket
        result = WorkflowResult(
            ticket_id=ticket_id,
            ticket_code=ticket["ticket_code"],
            lifecycle_status=lifecycle,  # type: ignore[arg-type]
            classification=classification,
            priority=priority,
            context=case_context,
            action_plan=plan,
            pending_actions=pending_actions,
            escalation=escalation,
            resolution_summary=resolution_summary,
        )
        if lifecycle == "waiting_approval":
            recorder.pause(
                current_step="waiting_approval",
                reason="Waiting for support_case_review pending action approval/execution.",
                state={"ticket_id": ticket_id, "pending_action_ids": [item.get("id") for item in pending_actions]},
            )
        else:
            recorder.complete(result=result.model_dump(), state={"ticket_id": ticket_id, "workflow_status": lifecycle})
        return result
    except Exception as err:
        recorder.fail(err, state={"ticket_id": ticket_id})
        raise


async def handle_email_case(email_id: int, context: RequestContext) -> WorkflowResult:
    row = fetch_one_sync("SELECT * FROM support_email_messages WHERE id = ?", (email_id,))
    if not row:
        raise ValueError("Email message not found")
    issue_type = classify_text(f"{row.get('subject') or ''}\n{row.get('body_text') or row.get('snippet') or ''}").intent
    if issue_type not in {"refund_request", "order_status", "technical_issue", "account_access"}:
        mapped_issue_type = "other"
    elif issue_type == "refund_request":
        mapped_issue_type = "refund"
    elif issue_type == "technical_issue":
        mapped_issue_type = "technical"
    else:
        mapped_issue_type = "shipping"
    ticket = create_ticket_from_email(email_id=email_id, issue_type=mapped_issue_type, context=context)
    ticket_row = _ticket_by_code(ticket["ticket_code"])
    return await handle_ticket_case(int(ticket_row["id"]), context)


def get_ticket_context(ticket_id: int) -> dict[str, Any]:
    ticket = _ticket_by_id(ticket_id)
    return {
        "ticket": ticket,
        "classification": _parse_json_field(ticket.get("classification_json"), {}),
        "context": _parse_json_field(ticket.get("context_summary_json"), {}),
        "action_plan": _parse_json_field(ticket.get("action_plan_json"), {}),
        "escalation": _parse_json_field(ticket.get("escalation_package_json"), {}),
    }


def escalate_ticket(ticket_id: int, *, reason: str, context: RequestContext) -> dict[str, Any]:
    ticket = _ticket_by_id(ticket_id)
    classification = classify_text(_ticket_text(ticket), issue_type=ticket.get("issue_type"))
    priority = assign_priority(classification)
    case_context = CaseContext(ticket=ticket, findings=[reason or "Manual escalation requested."])
    escalation = _build_escalation(
        ticket,
        classification,
        priority,
        case_context,
        suggested_next_action=reason or "Manual escalation requested by admin.",
    )
    _update_ticket_workflow(
        ticket_id,
        lifecycle_status="escalated",
        classification=classification,
        priority=priority,
        case_context=case_context,
        escalation=escalation,
    )
    add_ticket_note(
        ticket_id,
        AddTicketNoteInput(
            body=reason or "Manual escalation requested.",
            note_type="escalation",
            visibility="internal",
            metadata={"source": "manual"},
        ),
        context=context,
    )
    return get_ticket_context(ticket_id)


def process_sla_breaches(*, context: RequestContext, limit: int = 50) -> dict[str, Any]:
    now = utcnow_iso()
    rows = fetch_all_sync(
        """
        SELECT *
        FROM support_tickets
        WHERE sla_due_at IS NOT NULL
          AND sla_due_at <= ?
          AND sla_breached_at IS NULL
          AND COALESCE(workflow_status, status) NOT IN ('resolved', 'closed')
        ORDER BY sla_due_at ASC, id ASC
        LIMIT ?
        """,
        (now, max(1, min(limit, 200))),
    )
    escalated_ids: list[int] = []
    for row in rows:
        ticket_id = int(row["id"])
        classification = classify_text(_ticket_text(row), issue_type=row.get("issue_type"))
        priority = assign_priority(classification)
        reason = f"SLA breached at {now}; due at {row.get('sla_due_at')}."
        case_context = CaseContext(ticket=row, findings=[reason])
        escalation = _build_escalation(
            row,
            classification,
            priority,
            case_context,
            suggested_next_action="SLA is overdue. Assign a human owner and respond to the customer.",
        )
        execute_sync(
            """
            UPDATE support_tickets
            SET status = 'escalated',
                workflow_status = 'escalated',
                sla_breached_at = ?,
                escalation_package_json = ?,
                workflow_updated_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (now, _json_dumps(escalation.model_dump()), now, now, ticket_id),
        )
        add_ticket_note(
            ticket_id,
            AddTicketNoteInput(
                body=reason,
                note_type="sla_breach",
                visibility="internal",
                metadata={"sla_due_at": row.get("sla_due_at")},
            ),
            context=context,
        )
        try:
            from app.notifications import create_notification

            create_notification(
                event_type="support.sla_breached",
                severity="critical",
                title=f"SLA breached: {row.get('ticket_code') or ticket_id}",
                message=reason,
                entity_type="support_ticket",
                entity_id=ticket_id,
                payload={"ticket_code": row.get("ticket_code"), "sla_due_at": row.get("sla_due_at")},
                context=context,
            )
        except Exception:
            pass
        escalated_ids.append(ticket_id)

    return SlaMonitorResult(scanned=len(rows), breached=len(escalated_ids), escalated_ticket_ids=escalated_ids).model_dump()


def workflow_summary() -> dict[str, Any]:
    rows = fetch_all_sync(
        """
        SELECT COALESCE(workflow_status, status, 'unknown') AS status, COUNT(*) AS count
        FROM support_tickets
        GROUP BY COALESCE(workflow_status, status, 'unknown')
        ORDER BY count DESC
        """
    )
    priority_rows = fetch_all_sync(
        """
        SELECT COALESCE(priority, 'unclassified') AS priority, COUNT(*) AS count
        FROM support_tickets
        GROUP BY COALESCE(priority, 'unclassified')
        ORDER BY priority ASC
        """
    )
    overdue = fetch_one_sync(
        """
        SELECT COUNT(*) AS count
        FROM support_tickets
        WHERE sla_due_at IS NOT NULL
          AND sla_due_at <= ?
          AND COALESCE(workflow_status, status) NOT IN ('resolved', 'closed')
        """,
        (utcnow_iso(),),
    )
    return {
        "by_status": rows,
        "by_priority": priority_rows,
        "overdue_sla_count": int((overdue or {}).get("count") or 0),
    }
