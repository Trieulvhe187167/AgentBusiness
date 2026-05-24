"""
Durable workflow engine primitives.

V1 is intentionally lightweight: workflows run in-process but every run and
step is checkpointed to SQLite so operators can inspect, cancel, retry, and
resume safe workflow states without losing context across restarts.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.models import AuthContext, RequestContext
from app.observability import trace_span, workflow_trace_attrs


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class WorkflowDecisionInput(BaseModel):
    reason: str | None = Field(default=None, max_length=1000)


class WorkflowStepItem(BaseModel):
    id: int
    run_id: int
    step_key: str
    step_type: str
    status: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    error_message: str | None = None
    attempts: int
    started_at: str | None = None
    completed_at: str | None = None


class WorkflowRunItem(BaseModel):
    id: int
    workflow_type: str
    entity_type: str
    entity_id: str
    status: str
    current_step: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error_message: str | None = None
    blocked_reason: str | None = None
    created_by_user_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    tenant_id: str | None = None
    org_id: str | None = None
    kb_id: int | None = None
    kb_key: str | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None
    step_count: int = 0


class WorkflowRunDetail(WorkflowRunItem):
    steps: list[WorkflowStepItem] = Field(default_factory=list)


class ListWorkflowRunsOutput(BaseModel):
    total: int
    items: list[WorkflowRunItem]


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


def _parse_roles(raw: str | None) -> list[str]:
    parsed = _parse_json(raw, [])
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _serialize_run(row: dict[str, Any]) -> WorkflowRunItem:
    return WorkflowRunItem(
        id=int(row["id"]),
        workflow_type=row["workflow_type"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        status=row["status"],
        current_step=row.get("current_step"),
        input=_parse_json(row.get("input_json"), {}),
        state=_parse_json(row.get("state_json"), {}),
        result=_parse_json(row.get("result_json"), None),
        error_message=row.get("error_message"),
        blocked_reason=row.get("blocked_reason"),
        created_by_user_id=row.get("created_by_user_id"),
        roles=_parse_roles(row.get("roles_json")),
        tenant_id=row.get("tenant_id"),
        org_id=row.get("org_id"),
        kb_id=row.get("kb_id"),
        kb_key=row.get("kb_key"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row.get("completed_at"),
        step_count=int(row.get("step_count") or 0),
    )


def _serialize_step(row: dict[str, Any]) -> WorkflowStepItem:
    return WorkflowStepItem(
        id=int(row["id"]),
        run_id=int(row["run_id"]),
        step_key=row["step_key"],
        step_type=row["step_type"],
        status=row["status"],
        input=_parse_json(row.get("input_json"), {}),
        output=_parse_json(row.get("output_json"), None),
        error_message=row.get("error_message"),
        attempts=int(row.get("attempts") or 1),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
    )


def _run_query() -> str:
    return """
        SELECT wr.*,
               (SELECT COUNT(*) FROM workflow_steps ws WHERE ws.run_id = wr.id) AS step_count
        FROM workflow_runs wr
    """


def list_workflow_runs(
    *,
    status: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("wr.status = ?")
        params.append(status)
    if entity_type:
        clauses.append("wr.entity_type = ?")
        params.append(entity_type)
    if entity_id:
        clauses.append("wr.entity_id = ?")
        params.append(str(entity_id))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, min(int(limit), 200)))
    rows = fetch_all_sync(
        f"""
        {_run_query()}
        {where}
        ORDER BY wr.updated_at DESC, wr.id DESC
        LIMIT ?
        """,
        tuple(params),
    )
    items = [_serialize_run(row).model_dump() for row in rows]
    return ListWorkflowRunsOutput(total=len(items), items=items).model_dump()


def get_workflow_run(run_id: int) -> dict[str, Any]:
    row = fetch_one_sync(f"{_run_query()} WHERE wr.id = ?", (int(run_id),))
    if not row:
        raise ValueError("Workflow run not found")
    steps = fetch_all_sync(
        """
        SELECT *
        FROM workflow_steps
        WHERE run_id = ?
        ORDER BY id ASC
        """,
        (int(run_id),),
    )
    run = _serialize_run(row)
    return WorkflowRunDetail(
        **run.model_dump(),
        steps=[_serialize_step(step).model_dump() for step in steps],
    ).model_dump()


def _get_workflow_run_item(run_id: int) -> WorkflowRunItem:
    return WorkflowRunItem.model_validate(get_workflow_run(run_id))


def create_workflow_run(
    *,
    workflow_type: str,
    entity_type: str,
    entity_id: str | int,
    input_payload: dict[str, Any],
    context: RequestContext,
) -> int:
    now = utcnow_iso()
    return int(
        execute_sync(
            """
            INSERT INTO workflow_runs (
                workflow_type, entity_type, entity_id, status, current_step,
                input_json, state_json, created_by_user_id, roles_json,
                tenant_id, org_id, kb_id, kb_key, created_at, updated_at
            ) VALUES (?, ?, ?, 'running', NULL, ?, '{}', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workflow_type,
                entity_type,
                str(entity_id),
                _json_dumps(input_payload),
                context.auth.user_id,
                _json_dumps(context.auth.roles),
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


def _set_run_status(
    run_id: int,
    *,
    status: str,
    current_step: str | None = None,
    state: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
    blocked_reason: str | None = None,
    completed: bool = False,
) -> None:
    now = utcnow_iso()
    execute_sync(
        """
        UPDATE workflow_runs
        SET status = ?,
            current_step = ?,
            state_json = COALESCE(?, state_json),
            result_json = ?,
            error_message = ?,
            blocked_reason = ?,
            updated_at = ?,
            completed_at = CASE WHEN ? THEN ? ELSE completed_at END
        WHERE id = ?
        """,
        (
            status,
            current_step,
            _json_dumps(state) if state is not None else None,
            _json_dumps(result) if result is not None else None,
            error_message,
            blocked_reason,
            now,
            1 if completed else 0,
            now,
            int(run_id),
        ),
    )


def _insert_system_step(
    run_id: int,
    *,
    step_key: str,
    step_type: str,
    status: str = "done",
    input_payload: dict[str, Any] | None = None,
    output: Any = None,
) -> None:
    now = utcnow_iso()
    attempts_row = fetch_one_sync(
        "SELECT COALESCE(MAX(attempts), 0) AS attempts FROM workflow_steps WHERE run_id = ? AND step_key = ?",
        (int(run_id), step_key),
    )
    execute_sync(
        """
        INSERT INTO workflow_steps (
            run_id, step_key, step_type, status, input_json,
            output_json, attempts, started_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(run_id),
            step_key,
            step_type,
            status,
            _json_dumps(input_payload or {}),
            _json_dumps(output) if output is not None else None,
            int((attempts_row or {}).get("attempts") or 0) + 1,
            now,
            now,
        ),
    )


def cancel_workflow_run(run_id: int, *, reason: str | None, auth: AuthContext) -> dict[str, Any]:
    item = _get_workflow_run_item(run_id)
    if item.status in TERMINAL_STATUSES:
        raise ValueError(f"Cannot cancel workflow in status '{item.status}'")
    reason_text = reason or f"Cancelled by {auth.user_id or 'admin'}"
    _set_run_status(
        run_id,
        status="cancelled",
        error_message=reason_text,
        blocked_reason=None,
        completed=True,
    )
    return get_workflow_run(run_id)


def _pending_approval_count_for_ticket(ticket_id: str) -> int:
    row = fetch_one_sync(
        """
        SELECT COUNT(*) AS total
        FROM pending_actions
        WHERE action_type = 'support_case_review'
          AND status IN ('draft', 'approved')
          AND json_extract(payload_json, '$.ticket_id') = ?
        """,
        (int(ticket_id),),
    )
    return int((row or {}).get("total") or 0)


async def resume_workflow_run(run_id: int, *, context: RequestContext) -> dict[str, Any]:
    item = _get_workflow_run_item(run_id)
    if item.status != "paused":
        raise ValueError(f"Only paused workflows can be resumed, got '{item.status}'")
    if item.workflow_type == "support_ticket_case" and item.entity_type == "support_ticket":
        if _pending_approval_count_for_ticket(item.entity_id) > 0:
            raise ValueError("Workflow is waiting for pending approval to be executed")
        _insert_system_step(
            run_id,
            step_key="resume_after_approval",
            step_type="resume",
            input_payload={"trigger": "manual", "request_id": context.request_id},
            output={"resumed": True, "reason": "No open support_case_review approval remains."},
        )
        _set_run_status(
            run_id,
            status="completed",
            current_step="resume_after_approval",
            result={"resumed": True, "reason": "No open support_case_review approval remains."},
            completed=True,
        )
        return get_workflow_run(run_id)
    raise ValueError(f"Resume is not implemented for workflow type '{item.workflow_type}'")


def auto_resume_paused_workflows_for_ticket(
    ticket_id: int,
    *,
    context: RequestContext,
    trigger_action_id: int | None = None,
    trigger_status: str = "executed",
) -> list[dict[str, Any]]:
    rows = fetch_all_sync(
        """
        SELECT id
        FROM workflow_runs
        WHERE workflow_type = 'support_ticket_case'
          AND entity_type = 'support_ticket'
          AND entity_id = ?
          AND status = 'paused'
        ORDER BY id ASC
        """,
        (str(ticket_id),),
    )
    if not rows or _pending_approval_count_for_ticket(str(ticket_id)) > 0:
        return []

    resumed: list[dict[str, Any]] = []
    for row in rows:
        run_id = int(row["id"])
        output = {
            "resumed": True,
            "trigger": "pending_action_terminal",
            "trigger_action_id": trigger_action_id,
            "trigger_status": trigger_status,
        }
        with trace_span(
            "workflow.auto_resume",
            workflow_trace_attrs(
                workflow_type="support_ticket_case",
                run_id=run_id,
                step="resume_after_approval",
                step_type="resume",
                status="completed",
                entity_type="support_ticket",
                entity_id=ticket_id,
            ),
        ):
            _insert_system_step(
                run_id,
                step_key="resume_after_approval",
                step_type="resume",
                input_payload={
                    "trigger": "pending_action_terminal",
                    "trigger_action_id": trigger_action_id,
                    "trigger_status": trigger_status,
                    "request_id": context.request_id,
                },
                output=output,
            )
            _set_run_status(
                run_id,
                status="completed",
                current_step="resume_after_approval",
                result=output,
                state={"ticket_id": ticket_id, "auto_resumed": True},
                blocked_reason=None,
                completed=True,
            )
            resumed.append(get_workflow_run(run_id))
    return resumed


async def retry_workflow_run(run_id: int, *, context: RequestContext) -> dict[str, Any]:
    item = _get_workflow_run_item(run_id)
    if item.status not in {"failed", "cancelled"}:
        raise ValueError(f"Only failed or cancelled workflows can be retried, got '{item.status}'")
    if item.workflow_type == "support_ticket_case" and item.entity_type == "support_ticket":
        from app.support_workflows import handle_ticket_case

        result = await handle_ticket_case(int(item.entity_id), context, workflow_run_id=run_id)
        return get_workflow_run(run_id) | {"retry_result": result.model_dump()}
    raise ValueError(f"Retry is not implemented for workflow type '{item.workflow_type}'")


class WorkflowRecorder:
    def __init__(self, run_id: int, *, workflow_type: str = "unknown", entity_type: str | None = None, entity_id: str | int | None = None):
        self.run_id = int(run_id)
        self.workflow_type = workflow_type
        self.entity_type = entity_type
        self.entity_id = entity_id

    @classmethod
    def start(
        cls,
        *,
        workflow_type: str,
        entity_type: str,
        entity_id: str | int,
        input_payload: dict[str, Any],
        context: RequestContext,
        run_id: int | None = None,
    ) -> "WorkflowRecorder":
        if run_id is None:
            run_id = create_workflow_run(
                workflow_type=workflow_type,
                entity_type=entity_type,
                entity_id=entity_id,
                input_payload=input_payload,
                context=context,
            )
        else:
            _set_run_status(run_id, status="running", current_step=None, error_message=None, blocked_reason=None)
        return cls(run_id, workflow_type=workflow_type, entity_type=entity_type, entity_id=entity_id)

    def _insert_step(self, step_key: str, step_type: str, input_payload: dict[str, Any] | None) -> int:
        attempts_row = fetch_one_sync(
            "SELECT COALESCE(MAX(attempts), 0) AS attempts FROM workflow_steps WHERE run_id = ? AND step_key = ?",
            (self.run_id, step_key),
        )
        attempts = int((attempts_row or {}).get("attempts") or 0) + 1
        now = utcnow_iso()
        _set_run_status(self.run_id, status="running", current_step=step_key)
        return int(
            execute_sync(
                """
                INSERT INTO workflow_steps (
                    run_id, step_key, step_type, status, input_json,
                    attempts, started_at
                ) VALUES (?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    self.run_id,
                    step_key,
                    step_type,
                    _json_dumps(input_payload or {}),
                    attempts,
                    now,
                ),
            )
            or 0
        )

    def _complete_step(self, step_id: int, *, output: Any = None, status: str = "done", error: str | None = None) -> None:
        execute_sync(
            """
            UPDATE workflow_steps
            SET status = ?,
                output_json = ?,
                error_message = ?,
                completed_at = ?
            WHERE id = ?
            """,
            (
                status,
                _json_dumps(output) if output is not None else None,
                error,
                utcnow_iso(),
                int(step_id),
            ),
        )

    def step(
        self,
        step_key: str,
        step_type: str,
        fn: Callable[[], Any],
        *,
        input_payload: dict[str, Any] | None = None,
    ) -> Any:
        step_id = self._insert_step(step_key, step_type, input_payload)
        with trace_span(
            "workflow.step",
            workflow_trace_attrs(
                workflow_type=self.workflow_type,
                run_id=self.run_id,
                step=step_key,
                step_type=step_type,
                status="running",
                entity_type=self.entity_type,
                entity_id=self.entity_id,
            ),
        ):
            try:
                output = fn()
            except Exception as err:
                self._complete_step(step_id, status="failed", error=str(err))
                raise
        self._complete_step(step_id, output=output)
        return output

    async def async_step(
        self,
        step_key: str,
        step_type: str,
        fn: Callable[[], Awaitable[Any] | Any],
        *,
        input_payload: dict[str, Any] | None = None,
    ) -> Any:
        step_id = self._insert_step(step_key, step_type, input_payload)
        with trace_span(
            "workflow.step",
            workflow_trace_attrs(
                workflow_type=self.workflow_type,
                run_id=self.run_id,
                step=step_key,
                step_type=step_type,
                status="running",
                entity_type=self.entity_type,
                entity_id=self.entity_id,
            ),
        ):
            try:
                output = fn()
                if inspect.isawaitable(output):
                    output = await output
            except Exception as err:
                self._complete_step(step_id, status="failed", error=str(err))
                raise
        self._complete_step(step_id, output=output)
        return output

    def pause(self, *, current_step: str, reason: str, state: dict[str, Any] | None = None) -> None:
        _set_run_status(
            self.run_id,
            status="paused",
            current_step=current_step,
            state=state,
            blocked_reason=reason,
        )

    def complete(self, *, result: dict[str, Any], state: dict[str, Any] | None = None) -> None:
        _set_run_status(
            self.run_id,
            status="completed",
            current_step=None,
            state=state,
            result=result,
            blocked_reason=None,
            completed=True,
        )

    def fail(self, err: Exception, *, current_step: str | None = None, state: dict[str, Any] | None = None) -> None:
        _set_run_status(
            self.run_id,
            status="failed",
            current_step=current_step,
            state=state,
            error_message=str(err),
            blocked_reason=None,
            completed=True,
        )
