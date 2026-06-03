"""
Durable agent orchestration run ledger.

This complements workflow_engine: agent runs checkpoint one chat orchestration
turn, while workflow runs continue to own longer-lived business workflows.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.models import AuthContext, RequestContext

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class AgentRunDecisionInput(BaseModel):
    reason: str | None = Field(default=None, max_length=1000)


class AgentRunStepItem(BaseModel):
    id: int
    agent_run_id: int
    step_key: str
    step_type: str
    status: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    error_message: str | None = None
    created_at: str
    completed_at: str | None = None


class AgentRunItem(BaseModel):
    id: int
    request_id: str
    session_id: str | None = None
    route: str | None = None
    tool_name: str | None = None
    status: str
    input: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error_message: str | None = None
    blocked_reason: str | None = None
    pending_action_id: int | None = None
    created_by_user_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    channel: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    kb_id: int | None = None
    kb_key: str | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None
    step_count: int = 0


class AgentRunDetail(AgentRunItem):
    steps: list[AgentRunStepItem] = Field(default_factory=list)


class ListAgentRunsOutput(BaseModel):
    total: int
    items: list[AgentRunItem]


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _parse_json(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _run_query() -> str:
    return """
        SELECT ar.*,
               (SELECT COUNT(*) FROM agent_run_steps ars WHERE ars.agent_run_id = ar.id) AS step_count
        FROM agent_runs ar
    """


def _serialize_run(row: dict[str, Any]) -> AgentRunItem:
    roles = _parse_json(row.get("roles_json"), [])
    return AgentRunItem(
        id=int(row["id"]),
        request_id=row["request_id"],
        session_id=row.get("session_id"),
        route=row.get("route"),
        tool_name=row.get("tool_name"),
        status=row["status"],
        input=_parse_json(row.get("input_json"), {}),
        state=_parse_json(row.get("state_json"), {}),
        result=_parse_json(row.get("result_json"), None),
        error_message=row.get("error_message"),
        blocked_reason=row.get("blocked_reason"),
        pending_action_id=row.get("pending_action_id"),
        created_by_user_id=row.get("created_by_user_id"),
        roles=[str(role) for role in roles] if isinstance(roles, list) else [],
        channel=row.get("channel"),
        tenant_id=row.get("tenant_id"),
        org_id=row.get("org_id"),
        kb_id=row.get("kb_id"),
        kb_key=row.get("kb_key"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row.get("completed_at"),
        step_count=int(row.get("step_count") or 0),
    )


def _serialize_step(row: dict[str, Any]) -> AgentRunStepItem:
    return AgentRunStepItem(
        id=int(row["id"]),
        agent_run_id=int(row["agent_run_id"]),
        step_key=row["step_key"],
        step_type=row["step_type"],
        status=row["status"],
        input=_parse_json(row.get("input_json"), {}),
        output=_parse_json(row.get("output_json"), None),
        error_message=row.get("error_message"),
        created_at=row["created_at"],
        completed_at=row.get("completed_at"),
    )


def create_agent_run(*, query: str, context: RequestContext) -> int:
    now = utcnow_iso()
    return int(
        execute_sync(
            """
            INSERT INTO agent_runs (
                request_id, session_id, status, input_json, state_json,
                created_by_user_id, roles_json, channel, tenant_id, org_id,
                kb_id, kb_key, created_at, updated_at
            ) VALUES (?, ?, 'running', ?, '{}', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                context.request_id,
                context.session_id,
                _json_dumps({"query": query}),
                context.auth.user_id,
                _json_dumps(context.auth.roles),
                context.auth.channel,
                context.auth.tenant_id,
                context.auth.org_id,
                context.kb_id,
                context.kb_key,
                now,
                now,
            ),
        )
        or 0
    )


def record_agent_run_step(
    agent_run_id: int,
    *,
    step_key: str,
    step_type: str,
    status: str = "done",
    input_payload: dict[str, Any] | None = None,
    output: Any = None,
    error_message: str | None = None,
) -> int:
    now = utcnow_iso()
    return int(
        execute_sync(
            """
            INSERT INTO agent_run_steps (
                agent_run_id, step_key, step_type, status, input_json,
                output_json, error_message, created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(agent_run_id),
                step_key,
                step_type,
                status,
                _json_dumps(input_payload or {}),
                _json_dumps(output) if output is not None else None,
                error_message,
                now,
                now,
            ),
        )
        or 0
    )


def record_agent_route(agent_run_id: int, *, route: str, tool_name: str | None, reason: str) -> None:
    execute_sync(
        """
        UPDATE agent_runs
        SET route = ?, tool_name = ?, updated_at = ?
        WHERE id = ?
        """,
        (route, tool_name, utcnow_iso(), int(agent_run_id)),
    )
    record_agent_run_step(
        agent_run_id,
        step_key="route",
        step_type="route",
        output={"route": route, "tool_name": tool_name, "reason": reason},
    )


def _set_agent_run_status(
    agent_run_id: int,
    *,
    status: str,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
    blocked_reason: str | None = None,
    completed: bool = False,
) -> None:
    now = utcnow_iso()
    execute_sync(
        """
        UPDATE agent_runs
        SET status = ?,
            result_json = COALESCE(?, result_json),
            error_message = ?,
            blocked_reason = ?,
            updated_at = ?,
            completed_at = CASE WHEN ? THEN ? ELSE completed_at END
        WHERE id = ?
        """,
        (
            status,
            _json_dumps(result) if result is not None else None,
            error_message,
            blocked_reason,
            now,
            1 if completed else 0,
            now,
            int(agent_run_id),
        ),
    )


def complete_agent_run(agent_run_id: int, *, result: dict[str, Any] | None = None) -> None:
    _set_agent_run_status(agent_run_id, status="completed", result=result, completed=True)


def fail_agent_run(agent_run_id: int, *, error_message: str, result: dict[str, Any] | None = None) -> None:
    _set_agent_run_status(
        agent_run_id,
        status="failed",
        result=result,
        error_message=error_message,
        completed=True,
    )


def pause_agent_run_for_pending_action(
    agent_run_id: int,
    *,
    pending_action_id: int,
    tool_name: str,
    tool_call_id: str,
) -> None:
    now = utcnow_iso()
    reason = f"Waiting for pending action #{pending_action_id} approval and execution."
    execute_sync(
        """
        UPDATE agent_runs
        SET status = 'paused',
            pending_action_id = ?,
            blocked_reason = ?,
            state_json = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            int(pending_action_id),
            reason,
            _json_dumps({"pending_action_id": int(pending_action_id), "tool_call_id": tool_call_id}),
            now,
            int(agent_run_id),
        ),
    )
    execute_sync(
        "UPDATE pending_actions SET agent_run_id = ?, updated_at = ? WHERE id = ?",
        (int(agent_run_id), now, int(pending_action_id)),
    )
    record_agent_run_step(
        agent_run_id,
        step_key="waiting_approval",
        step_type="approval",
        status="paused",
        input_payload={"tool_name": tool_name, "tool_call_id": tool_call_id},
        output={"pending_action_id": int(pending_action_id)},
    )


def record_agent_run_approval(action_id: int, *, auth: AuthContext) -> dict[str, Any] | None:
    row = fetch_one_sync("SELECT agent_run_id FROM pending_actions WHERE id = ?", (int(action_id),))
    if not row or not row.get("agent_run_id"):
        return None
    agent_run_id = int(row["agent_run_id"])
    record_agent_run_step(
        agent_run_id,
        step_key="approval_granted",
        step_type="approval",
        input_payload={"pending_action_id": int(action_id)},
        output={"approved_by_user_id": auth.user_id},
    )
    return get_agent_run(agent_run_id)


def auto_resume_agent_run_for_pending_action(
    action_id: int,
    *,
    trigger_status: str,
    context: RequestContext,
) -> dict[str, Any] | None:
    row = fetch_one_sync("SELECT agent_run_id FROM pending_actions WHERE id = ?", (int(action_id),))
    if not row or not row.get("agent_run_id"):
        return None
    agent_run_id = int(row["agent_run_id"])
    item = _get_agent_run_item(agent_run_id)
    if item.status != "paused":
        return get_agent_run(agent_run_id)

    output = {
        "pending_action_id": int(action_id),
        "trigger_status": trigger_status,
        "request_id": context.request_id,
    }
    record_agent_run_step(
        agent_run_id,
        step_key="approval_terminal",
        step_type="resume",
        status="done" if trigger_status == "executed" else trigger_status,
        output=output,
    )
    if trigger_status == "executed":
        complete_agent_run(agent_run_id, result=output)
    elif trigger_status == "rejected":
        _set_agent_run_status(
            agent_run_id,
            status="cancelled",
            result=output,
            error_message="Pending action was rejected.",
            completed=True,
        )
    else:
        fail_agent_run(agent_run_id, error_message=f"Pending action ended with status '{trigger_status}'.", result=output)
    return get_agent_run(agent_run_id)


def list_agent_runs(
    *,
    status: str | None = None,
    session_id: str | None = None,
    pending_action_id: int | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("ar.status = ?")
        params.append(status)
    if session_id:
        clauses.append("ar.session_id = ?")
        params.append(session_id)
    if pending_action_id is not None:
        clauses.append("ar.pending_action_id = ?")
        params.append(int(pending_action_id))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, min(int(limit), 200)))
    rows = fetch_all_sync(
        f"""
        {_run_query()}
        {where}
        ORDER BY ar.updated_at DESC, ar.id DESC
        LIMIT ?
        """,
        tuple(params),
    )
    items = [_serialize_run(row).model_dump() for row in rows]
    return ListAgentRunsOutput(total=len(items), items=items).model_dump()


def get_agent_run(agent_run_id: int) -> dict[str, Any]:
    row = fetch_one_sync(f"{_run_query()} WHERE ar.id = ?", (int(agent_run_id),))
    if not row:
        raise ValueError("Agent run not found")
    steps = fetch_all_sync(
        "SELECT * FROM agent_run_steps WHERE agent_run_id = ? ORDER BY id ASC",
        (int(agent_run_id),),
    )
    run = _serialize_run(row)
    return AgentRunDetail(
        **run.model_dump(),
        steps=[_serialize_step(step).model_dump() for step in steps],
    ).model_dump()


def _get_agent_run_item(agent_run_id: int) -> AgentRunItem:
    return AgentRunItem.model_validate(get_agent_run(agent_run_id))


def cancel_agent_run(agent_run_id: int, *, reason: str | None, auth: AuthContext) -> dict[str, Any]:
    item = _get_agent_run_item(agent_run_id)
    if item.status in TERMINAL_STATUSES:
        raise ValueError(f"Cannot cancel agent run in status '{item.status}'")
    reason_text = reason or f"Cancelled by {auth.user_id or 'operator'}"
    record_agent_run_step(
        agent_run_id,
        step_key="cancelled",
        step_type="operator",
        output={"reason": reason_text},
    )
    _set_agent_run_status(
        agent_run_id,
        status="cancelled",
        error_message=reason_text,
        blocked_reason=None,
        completed=True,
    )
    return get_agent_run(agent_run_id)


def resume_agent_run(agent_run_id: int, *, context: RequestContext) -> dict[str, Any]:
    item = _get_agent_run_item(agent_run_id)
    if item.status != "paused":
        raise ValueError(f"Only paused agent runs can be resumed, got '{item.status}'")
    if not item.pending_action_id:
        raise ValueError("Paused agent run has no pending action")
    action = fetch_one_sync("SELECT status FROM pending_actions WHERE id = ?", (int(item.pending_action_id),))
    if not action:
        raise ValueError("Pending action not found")
    action_status = str(action["status"])
    if action_status in {"draft", "approved"}:
        raise ValueError("Agent run is waiting for pending approval to finish")
    return auto_resume_agent_run_for_pending_action(
        item.pending_action_id,
        trigger_status=action_status,
        context=context,
    ) or get_agent_run(agent_run_id)
