"""Approval event timeline assembled from existing audit state."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.database import fetch_all_sync
from app.pending_actions import PendingActionItem, get_pending_action


class ApprovalEventItem(BaseModel):
    event_type: str
    label: str
    status: str
    actor_user_id: str | None = None
    created_at: str
    entity_type: str | None = None
    entity_id: str | int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ApprovalEventsOutput(BaseModel):
    action: PendingActionItem
    events: list[ApprovalEventItem] = Field(default_factory=list)


def _parse_json(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _compact(value: Any, *, limit: int = 1400) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact(item, limit=limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact(item, limit=limit) for item in value[:25]]
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...[truncated]"
    return value


def _event(
    *,
    event_type: str,
    label: str,
    status: str,
    created_at: str | None,
    actor_user_id: str | None = None,
    entity_type: str | None = None,
    entity_id: str | int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "label": label,
        "status": status,
        "actor_user_id": actor_user_id,
        "created_at": created_at or "",
        "entity_type": entity_type,
        "entity_id": entity_id,
        "details": _compact(details or {}),
    }


def _pending_action_events(action: PendingActionItem) -> list[dict[str, Any]]:
    events = [
        _event(
            event_type="pending_action.created",
            label="Draft created",
            status="done",
            actor_user_id=action.created_by_user_id,
            created_at=action.created_at,
            entity_type="pending_action",
            entity_id=action.id,
            details={
                "action_type": action.action_type,
                "risk_level": action.risk_level,
                "title": action.title,
                "summary": action.summary,
                "agent_run_id": action.agent_run_id,
                "kb_id": action.kb_id,
                "kb_key": action.kb_key,
            },
        )
    ]
    if action.approved_at:
        events.append(
            _event(
                event_type="pending_action.approved",
                label="Approved",
                status="done",
                actor_user_id=action.approved_by_user_id,
                created_at=action.approved_at,
                entity_type="pending_action",
                entity_id=action.id,
            )
        )
    if action.status == "rejected":
        events.append(
            _event(
                event_type="pending_action.rejected",
                label="Rejected",
                status="failed",
                actor_user_id=action.approved_by_user_id,
                created_at=action.updated_at,
                entity_type="pending_action",
                entity_id=action.id,
                details={"reason": action.error_message},
            )
        )
    if action.executed_at and action.status == "executed":
        events.append(
            _event(
                event_type="pending_action.executed",
                label="Executed",
                status="done",
                actor_user_id=action.executed_by_user_id,
                created_at=action.executed_at,
                entity_type="pending_action",
                entity_id=action.id,
                details={"result": action.result or {}},
            )
        )
    if action.status == "failed":
        events.append(
            _event(
                event_type="pending_action.failed",
                label="Execution failed",
                status="failed",
                actor_user_id=action.executed_by_user_id,
                created_at=action.executed_at or action.updated_at,
                entity_type="pending_action",
                entity_id=action.id,
                details={"error_message": action.error_message},
            )
        )
    return events


def _notification_events(action_id: int) -> list[dict[str, Any]]:
    rows = fetch_all_sync(
        """
        SELECT *
        FROM notifications
        WHERE entity_type = 'pending_action' AND entity_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (str(action_id),),
    )
    return [
        _event(
            event_type=f"notification.{row['event_type']}",
            label=str(row.get("title") or row["event_type"]),
            status=str(row.get("severity") or "info"),
            actor_user_id=row.get("created_by_user_id"),
            created_at=row.get("created_at"),
            entity_type="notification",
            entity_id=int(row["id"]),
            details={
                "event_type": row.get("event_type"),
                "message": row.get("message"),
                "payload": _parse_json(row.get("payload_json"), {}),
            },
        )
        for row in rows
    ]


def _agent_run_label(step_key: str, status: str, output: dict[str, Any]) -> str:
    if step_key == "waiting_approval":
        return "Agent run paused for approval"
    if step_key == "approval_granted":
        return "Agent approval checkpoint"
    if step_key == "approval_terminal":
        trigger = str(output.get("trigger_status") or status)
        if trigger == "executed":
            return "Agent run resumed"
        if trigger == "rejected":
            return "Agent run cancelled"
        return "Agent run failed"
    return f"Agent step: {step_key}"


def _agent_run_events(agent_run_id: int | None) -> list[dict[str, Any]]:
    if not agent_run_id:
        return []
    rows = fetch_all_sync(
        """
        SELECT *
        FROM agent_run_steps
        WHERE agent_run_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (int(agent_run_id),),
    )
    events: list[dict[str, Any]] = []
    for row in rows:
        output = _parse_json(row.get("output_json"), {})
        status = str(row.get("status") or "unknown")
        events.append(
            _event(
                event_type=f"agent_run.{row['step_key']}",
                label=_agent_run_label(str(row["step_key"]), status, output if isinstance(output, dict) else {}),
                status="done" if status == "done" else status,
                created_at=row.get("completed_at") or row.get("created_at"),
                entity_type="agent_run",
                entity_id=int(agent_run_id),
                details={
                    "step_key": row.get("step_key"),
                    "step_type": row.get("step_type"),
                    "side_effect": bool(row.get("side_effect")),
                    "idempotency_key": row.get("idempotency_key"),
                    "attempt_count": int(row.get("attempt_count") or 1),
                    "last_attempt_at": row.get("last_attempt_at"),
                    "output": output,
                    "error_message": row.get("error_message"),
                },
            )
        )
    return events


def _workflow_events(action: PendingActionItem) -> list[dict[str, Any]]:
    ticket_id = action.payload.get("ticket_id")
    if not ticket_id:
        return []
    rows = fetch_all_sync(
        """
        SELECT wr.id AS run_id, ws.*
        FROM workflow_runs wr
        JOIN workflow_steps ws ON ws.run_id = wr.id
        WHERE wr.entity_type = 'support_ticket'
          AND wr.entity_id = ?
          AND ws.step_key IN ('resume_after_approval', 'create_pending_actions')
        ORDER BY ws.completed_at ASC, ws.id ASC
        """,
        (str(ticket_id),),
    )
    events: list[dict[str, Any]] = []
    for row in rows:
        output = _parse_json(row.get("output_json"), {})
        events.append(
            _event(
                event_type=f"workflow.{row['step_key']}",
                label="Workflow resumed" if row["step_key"] == "resume_after_approval" else "Workflow created approval",
                status=str(row.get("status") or "unknown"),
                created_at=row.get("completed_at") or row.get("started_at"),
                entity_type="workflow_run",
                entity_id=int(row["run_id"]),
                details={
                    "step_key": row.get("step_key"),
                    "step_type": row.get("step_type"),
                    "attempts": int(row.get("attempts") or 1),
                    "output": output,
                    "error_message": row.get("error_message"),
                },
            )
        )
    return events


def list_approval_events(action_id: int) -> dict[str, Any]:
    action = get_pending_action(int(action_id))
    events = [
        *_pending_action_events(action),
        *_notification_events(action.id),
        *_agent_run_events(action.agent_run_id),
        *_workflow_events(action),
    ]
    events = sorted(enumerate(events), key=lambda item: (str(item[1].get("created_at") or ""), item[0]))
    return ApprovalEventsOutput(
        action=action,
        events=[ApprovalEventItem(**event) for _, event in events],
    ).model_dump()
