"""Support case trace timeline assembled from operational tables."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.database import fetch_all_sync, fetch_one_sync


class CaseTimelineEvent(BaseModel):
    id: str
    timestamp: str
    stage: str
    title: str
    summary: str = ""
    status: str | None = None
    severity: str = "info"
    actor: str | None = None
    source_table: str
    source_id: str | int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class CaseTimelineOutput(BaseModel):
    ticket_id: int
    ticket_code: str
    total: int
    events: list[CaseTimelineEvent]


def _parse_json(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _compact(value: str | None, *, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _append(events: list[CaseTimelineEvent], **kwargs: Any) -> None:
    if not kwargs.get("timestamp"):
        return
    events.append(CaseTimelineEvent(**kwargs))


def _ticket(ticket_id: int) -> dict[str, Any]:
    row = fetch_one_sync("SELECT * FROM support_tickets WHERE id = ?", (ticket_id,))
    if not row:
        raise ValueError("Support ticket not found")
    return row


def _workflow_events(ticket: dict[str, Any]) -> list[CaseTimelineEvent]:
    events: list[CaseTimelineEvent] = []
    workflow_time = ticket.get("workflow_updated_at") or ticket.get("updated_at") or ticket.get("created_at")
    classification = _parse_json(ticket.get("classification_json"), {})
    context_summary = _parse_json(ticket.get("context_summary_json"), {})
    action_plan = _parse_json(ticket.get("action_plan_json"), {})
    escalation = _parse_json(ticket.get("escalation_package_json"), {})

    if classification:
        _append(
            events,
            id=f"classification:{ticket['id']}",
            timestamp=workflow_time,
            stage="classification",
            title=f"Classified as {classification.get('intent') or ticket.get('intent') or 'unknown'}",
            summary=(
                f"confidence={classification.get('confidence', '-')}, "
                f"risk={classification.get('risk_level') or ticket.get('risk_level') or '-'}, "
                f"sentiment={classification.get('customer_sentiment') or ticket.get('sentiment') or '-'}"
            ),
            status=ticket.get("workflow_status") or ticket.get("status"),
            severity="info" if (classification.get("risk_level") or ticket.get("risk_level")) == "low" else "warning",
            actor="agent",
            source_table="support_tickets",
            source_id=ticket["id"],
            details=classification,
        )

    if context_summary:
        findings = context_summary.get("findings") or []
        _append(
            events,
            id=f"enrichment:{ticket['id']}",
            timestamp=workflow_time,
            stage="enrichment",
            title="Context enriched",
            summary=_compact("; ".join(str(item) for item in findings) or "Ticket context package was built."),
            status="done",
            severity="info",
            actor="agent",
            source_table="support_tickets",
            source_id=ticket["id"],
            details=context_summary,
        )

    if action_plan:
        steps = action_plan.get("steps") or []
        requires_approval = bool(action_plan.get("requires_approval"))
        _append(
            events,
            id=f"action_plan:{ticket['id']}",
            timestamp=workflow_time,
            stage="action_plan",
            title="Action plan created",
            summary=_compact(action_plan.get("goal") or f"{len(steps)} planned step(s)."),
            status="requires_approval" if requires_approval else "ready",
            severity="warning" if requires_approval else "info",
            actor="agent",
            source_table="support_tickets",
            source_id=ticket["id"],
            details=action_plan,
        )
        for index, step in enumerate(steps, start=1):
            stage = "tool_call" if step.get("type") == "tool_call" else step.get("type") or "plan_step"
            _append(
                events,
                id=f"action_step:{ticket['id']}:{index}",
                timestamp=workflow_time,
                stage=stage,
                title=step.get("tool") or step.get("description") or f"Plan step {index}",
                summary=_compact(step.get("description") or ""),
                status=step.get("status"),
                severity="warning" if step.get("requires_approval") else "info",
                actor="agent",
                source_table="support_tickets.action_plan_json",
                source_id=ticket["id"],
                details=step,
            )

    if escalation:
        _append(
            events,
            id=f"escalation_package:{ticket['id']}",
            timestamp=workflow_time,
            stage="escalation",
            title="Escalation package created",
            summary=_compact(escalation.get("summary") or escalation.get("suggested_next_action") or ""),
            status=ticket.get("workflow_status") or ticket.get("status"),
            severity="critical" if ticket.get("priority") in {"P0", "P1"} else "warning",
            actor="agent",
            source_table="support_tickets",
            source_id=ticket["id"],
            details=escalation,
        )

    return events


def _note_events(ticket_id: int) -> list[CaseTimelineEvent]:
    rows = fetch_all_sync(
        """
        SELECT *
        FROM support_ticket_notes
        WHERE ticket_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (ticket_id,),
    )
    events: list[CaseTimelineEvent] = []
    for row in rows:
        note_type = row.get("note_type") or "note"
        visibility = row.get("visibility") or "internal"
        stage = "support_reply" if visibility == "public" else "support_note"
        if note_type in {"status_change", "assignment", "escalation", "sla_breach"}:
            stage = note_type
        _append(
            events,
            id=f"note:{row['id']}",
            timestamp=row["created_at"],
            stage=stage,
            title=f"{note_type.replace('_', ' ').title()} ({visibility})",
            summary=_compact(row.get("body")),
            status=visibility,
            severity="critical" if note_type in {"sla_breach", "escalation"} else "info",
            actor=row.get("created_by_user_id"),
            source_table="support_ticket_notes",
            source_id=row["id"],
            details={
                "note_type": note_type,
                "visibility": visibility,
                "metadata": _parse_json(row.get("metadata_json"), {}),
                "roles": _parse_json(row.get("roles_json"), []),
            },
        )
    return events


def _pending_action_events(ticket_id: int) -> list[CaseTimelineEvent]:
    rows = fetch_all_sync(
        """
        SELECT *
        FROM pending_actions
        WHERE json_extract(payload_json, '$.ticket_id') = ?
        ORDER BY created_at ASC, id ASC
        """,
        (ticket_id,),
    )
    events: list[CaseTimelineEvent] = []
    for row in rows:
        payload = _parse_json(row.get("payload_json"), {})
        result = _parse_json(row.get("result_json"), None)
        severity = "critical" if row.get("risk_level") == "critical" else "warning"
        _append(
            events,
            id=f"pending_action:{row['id']}:created",
            timestamp=row["created_at"],
            stage="pending_action",
            title=row["title"],
            summary=_compact(row.get("summary")),
            status=row.get("status"),
            severity=severity,
            actor=row.get("created_by_user_id"),
            source_table="pending_actions",
            source_id=row["id"],
            details={
                "action_type": row.get("action_type"),
                "risk_level": row.get("risk_level"),
                "payload": payload,
            },
        )
        if row.get("approved_at"):
            _append(
                events,
                id=f"pending_action:{row['id']}:approved",
                timestamp=row["approved_at"],
                stage="approval",
                title=f"Pending action approved: {row['title']}",
                summary=_compact(row.get("summary")),
                status=row.get("status"),
                severity="warning",
                actor=row.get("approved_by_user_id"),
                source_table="pending_actions",
                source_id=row["id"],
                details={"action_type": row.get("action_type")},
            )
        if row.get("executed_at"):
            _append(
                events,
                id=f"pending_action:{row['id']}:executed",
                timestamp=row["executed_at"],
                stage="execution",
                title=f"Pending action executed: {row['title']}",
                summary=_compact(row.get("error_message") or row.get("summary")),
                status=row.get("status"),
                severity="critical" if row.get("status") == "failed" else "info",
                actor=row.get("executed_by_user_id"),
                source_table="pending_actions",
                source_id=row["id"],
                details={"action_type": row.get("action_type"), "result": result},
            )
    return events


def _background_job_events(ticket_id: int) -> tuple[list[CaseTimelineEvent], list[str]]:
    rows = fetch_all_sync(
        """
        SELECT *
        FROM background_jobs
        WHERE json_extract(payload_json, '$.ticket_id') = ?
        ORDER BY created_at ASC, id ASC
        """,
        (ticket_id,),
    )
    events: list[CaseTimelineEvent] = []
    job_ids: list[str] = []
    for row in rows:
        job_ids.append(row["job_id"])
        payload = _parse_json(row.get("payload_json"), {})
        result = _parse_json(row.get("result_json"), None)
        _append(
            events,
            id=f"background_job:{row['job_id']}",
            timestamp=row["created_at"],
            stage="background_job",
            title=f"Background job queued: {row['job_type']}",
            summary=f"{row['job_id']} progress={row.get('progress')}",
            status=row.get("status"),
            severity="critical" if row.get("status") == "failed" else "info",
            actor=row.get("created_by_user_id"),
            source_table="background_jobs",
            source_id=row["job_id"],
            details={"payload": payload, "result": result, "error_message": row.get("error_message")},
        )
        if row.get("started_at"):
            _append(
                events,
                id=f"background_job:{row['job_id']}:started",
                timestamp=row["started_at"],
                stage="background_job",
                title=f"Background job started: {row['job_type']}",
                summary=row["job_id"],
                status="running",
                severity="info",
                actor=row.get("worker_id"),
                source_table="background_jobs",
                source_id=row["job_id"],
                details={"worker_id": row.get("worker_id")},
            )
        if row.get("finished_at"):
            _append(
                events,
                id=f"background_job:{row['job_id']}:finished",
                timestamp=row["finished_at"],
                stage="background_job",
                title=f"Background job finished: {row['job_type']}",
                summary=_compact(row.get("error_message") or "Completed background processing."),
                status=row.get("status"),
                severity="critical" if row.get("status") == "failed" else "info",
                actor=row.get("worker_id"),
                source_table="background_jobs",
                source_id=row["job_id"],
                details={"result": result, "error_message": row.get("error_message")},
            )
    return events, job_ids


def _tool_audit_events(job_ids: list[str]) -> list[CaseTimelineEvent]:
    if not job_ids:
        return []
    placeholders = ",".join("?" for _ in job_ids)
    rows = fetch_all_sync(
        f"""
        SELECT *
        FROM tool_audit_logs
        WHERE request_id IN ({placeholders})
        ORDER BY created_at ASC, id ASC
        """,
        tuple(job_ids),
    )
    events: list[CaseTimelineEvent] = []
    for row in rows:
        _append(
            events,
            id=f"tool_audit:{row['id']}",
            timestamp=row["created_at"],
            stage="tool_call",
            title=f"Tool call: {row['tool_name']}",
            summary=_compact(row.get("result_summary") or row.get("error_message") or ""),
            status=row.get("tool_status"),
            severity="critical" if row.get("tool_status") == "error" else "info",
            actor=row.get("user_id"),
            source_table="tool_audit_logs",
            source_id=row["id"],
            details={
                "tool_call_id": row.get("tool_call_id"),
                "request_id": row.get("request_id"),
                "latency_ms": row.get("latency_ms"),
                "args": _parse_json(row.get("args_json"), {}),
                "error_message": row.get("error_message"),
            },
        )
    return events


def _notification_events(ticket_id: int) -> list[CaseTimelineEvent]:
    rows = fetch_all_sync(
        """
        SELECT *
        FROM notifications
        WHERE entity_type = 'support_ticket'
          AND entity_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (str(ticket_id),),
    )
    events: list[CaseTimelineEvent] = []
    for row in rows:
        _append(
            events,
            id=f"notification:{row['id']}",
            timestamp=row["created_at"],
            stage="notification",
            title=row["title"],
            summary=_compact(row.get("message")),
            status=row.get("status"),
            severity=row.get("severity") or "info",
            actor=row.get("created_by_user_id"),
            source_table="notifications",
            source_id=row["id"],
            details={"event_type": row.get("event_type"), "payload": _parse_json(row.get("payload_json"), {})},
        )
    return events


def build_case_timeline(ticket_id: int) -> dict[str, Any]:
    ticket = _ticket(ticket_id)
    events: list[CaseTimelineEvent] = []
    _append(
        events,
        id=f"ticket:{ticket['id']}:created",
        timestamp=ticket["created_at"],
        stage="user_request",
        title=f"Ticket created: {ticket['ticket_code']}",
        summary=_compact(ticket.get("message")),
        status=ticket.get("status"),
        severity="info",
        actor=ticket.get("created_by_user_id") or ticket.get("contact"),
        source_table="support_tickets",
        source_id=ticket["id"],
        details={
            "ticket_code": ticket.get("ticket_code"),
            "issue_type": ticket.get("issue_type"),
            "contact": ticket.get("contact"),
            "channel": ticket.get("channel"),
            "tenant_id": ticket.get("tenant_id"),
            "org_id": ticket.get("org_id"),
            "kb_id": ticket.get("kb_id"),
            "kb_key": ticket.get("kb_key"),
        },
    )
    events.extend(_workflow_events(ticket))
    events.extend(_note_events(ticket_id))
    events.extend(_pending_action_events(ticket_id))
    job_events, job_ids = _background_job_events(ticket_id)
    events.extend(job_events)
    events.extend(_tool_audit_events(job_ids))
    events.extend(_notification_events(ticket_id))

    events.sort(key=lambda event: (event.timestamp, event.id))
    return CaseTimelineOutput(
        ticket_id=int(ticket["id"]),
        ticket_code=ticket["ticket_code"],
        total=len(events),
        events=events,
    ).model_dump()
