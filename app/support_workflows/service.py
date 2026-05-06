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
    SupportTicketItem,
    SupportTicketNoteItem,
    UpdateTicketStatusInput,
    WorkflowResult,
)


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


def list_ticket_notes(ticket_id: int) -> dict[str, Any]:
    _ticket_by_id(ticket_id)
    rows = fetch_all_sync(
        """
        SELECT *
        FROM support_ticket_notes
        WHERE ticket_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (ticket_id,),
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


async def handle_ticket_case(ticket_id: int, context: RequestContext) -> WorkflowResult:
    ticket = _ticket_by_id(ticket_id)
    classification = classify_text(_ticket_text(ticket), issue_type=ticket.get("issue_type"))
    priority = assign_priority(classification)
    case_context = await _build_context(ticket, classification, context)
    plan = _build_action_plan(ticket, classification, priority, case_context)

    pending_actions: list[dict[str, Any]] = []
    escalation = None
    resolution_summary = None
    lifecycle = "planned"

    if plan.requires_approval:
        pending_actions.append(_create_review_pending_action(ticket, classification, priority, case_context, context))
        lifecycle = "waiting_approval"
        escalation = _build_escalation(
            ticket,
            classification,
            priority,
            case_context,
            suggested_next_action="Review approval package and decide the customer-impacting action.",
        )
    elif plan.should_escalate:
        lifecycle = "escalated"
        escalation = _build_escalation(
            ticket,
            classification,
            priority,
            case_context,
            suggested_next_action="Human support should review the case context and respond.",
        )
    else:
        lifecycle = "resolved"
        resolution_summary = _draft_reply(ticket, classification, case_context)

    _update_ticket_workflow(
        ticket_id,
        lifecycle_status=lifecycle,
        classification=classification,
        priority=priority,
        case_context=case_context,
        action_plan=plan,
        escalation=escalation,
        resolution_summary=resolution_summary,
    )
    updated_ticket = _ticket_by_id(ticket_id)
    case_context.ticket = updated_ticket
    return WorkflowResult(
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
