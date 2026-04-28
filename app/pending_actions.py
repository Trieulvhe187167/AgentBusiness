"""
Pending action approval workflow.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.models import AuthContext, RequestContext

PendingActionStatus = Literal["draft", "approved", "executed", "rejected", "failed"]
PendingActionType = Literal["send_email_reply", "delete_google_drive_source", "sync_google_drive_source"]


class PendingActionItem(BaseModel):
    id: int
    action_type: str
    risk_level: str
    status: str
    title: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error_message: str | None = None
    created_by_user_id: str | None = None
    approved_by_user_id: str | None = None
    executed_by_user_id: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    kb_id: int | None = None
    kb_key: str | None = None
    created_at: str
    updated_at: str
    approved_at: str | None = None
    executed_at: str | None = None
    expires_at: str | None = None


class ListPendingActionsOutput(BaseModel):
    total: int
    items: list[PendingActionItem]


class CreatePendingActionInput(BaseModel):
    action_type: PendingActionType
    risk_level: str = Field(default="high", min_length=1, max_length=40)
    title: str = Field(..., min_length=1, max_length=240)
    summary: str = Field(default="", max_length=1000)
    payload: dict[str, Any] = Field(default_factory=dict)


class PendingActionDecisionInput(BaseModel):
    note: str | None = Field(default=None, max_length=1000)


def _parse_json(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else None


def _serialize_row(row: dict[str, Any]) -> PendingActionItem:
    return PendingActionItem(
        id=int(row["id"]),
        action_type=row["action_type"],
        risk_level=row["risk_level"],
        status=row["status"],
        title=row["title"],
        summary=row.get("summary") or "",
        payload=_parse_json(row.get("payload_json")) or {},
        result=_parse_json(row.get("result_json")),
        error_message=row.get("error_message"),
        created_by_user_id=row.get("created_by_user_id"),
        approved_by_user_id=row.get("approved_by_user_id"),
        executed_by_user_id=row.get("executed_by_user_id"),
        tenant_id=row.get("tenant_id"),
        org_id=row.get("org_id"),
        kb_id=row.get("kb_id"),
        kb_key=row.get("kb_key"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        approved_at=row.get("approved_at"),
        executed_at=row.get("executed_at"),
        expires_at=row.get("expires_at"),
    )


def get_pending_action(action_id: int) -> PendingActionItem:
    row = fetch_one_sync("SELECT * FROM pending_actions WHERE id = ?", (action_id,))
    if not row:
        raise ValueError("Pending action not found")
    return _serialize_row(row)


def list_pending_actions(*, status: str | None = None, limit: int = 50) -> dict[str, Any]:
    if status:
        rows = fetch_all_sync(
            """
            SELECT * FROM pending_actions
            WHERE status = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (status, max(1, min(limit, 200))),
        )
    else:
        rows = fetch_all_sync(
            """
            SELECT * FROM pending_actions
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        )
    items = [_serialize_row(row).model_dump() for row in rows]
    return {"total": len(items), "items": items}


def create_pending_action(
    *,
    action_type: str,
    title: str,
    summary: str,
    payload: dict[str, Any],
    risk_level: str,
    context: RequestContext,
) -> dict[str, Any]:
    now = utcnow_iso()
    action_id = execute_sync(
        """
        INSERT INTO pending_actions (
            action_type, risk_level, status, title, summary, payload_json,
            created_by_user_id, tenant_id, org_id, kb_id, kb_key,
            created_at, updated_at
        ) VALUES (?, ?, 'draft', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            action_type,
            risk_level,
            title,
            summary,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            context.auth.user_id,
            context.auth.tenant_id,
            context.auth.org_id,
            context.kb_id,
            context.kb_key,
            now,
            now,
        ),
    )
    return get_pending_action(int(action_id or 0)).model_dump()


def approve_pending_action(action_id: int, *, auth: AuthContext) -> dict[str, Any]:
    item = get_pending_action(action_id)
    if item.status != "draft":
        raise ValueError("Only draft pending actions can be approved")
    now = utcnow_iso()
    execute_sync(
        """
        UPDATE pending_actions
        SET status = 'approved',
            approved_by_user_id = ?,
            approved_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (auth.user_id, now, now, action_id),
    )
    return get_pending_action(action_id).model_dump()


def reject_pending_action(action_id: int, *, auth: AuthContext, note: str | None = None) -> dict[str, Any]:
    item = get_pending_action(action_id)
    if item.status not in {"draft", "approved"}:
        raise ValueError("Only draft or approved pending actions can be rejected")
    now = utcnow_iso()
    execute_sync(
        """
        UPDATE pending_actions
        SET status = 'rejected',
            approved_by_user_id = ?,
            error_message = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (auth.user_id, note, now, action_id),
    )
    return get_pending_action(action_id).model_dump()


async def execute_pending_action(action_id: int, *, context: RequestContext) -> dict[str, Any]:
    item = get_pending_action(action_id)
    if item.status != "approved":
        raise ValueError("Pending action must be approved before execution")

    try:
        result = await _dispatch_pending_action(item, context=context)
    except Exception as err:
        now = utcnow_iso()
        execute_sync(
            """
            UPDATE pending_actions
            SET status = 'failed',
                error_message = ?,
                executed_by_user_id = ?,
                executed_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (str(err), context.auth.user_id, now, now, action_id),
        )
        raise

    now = utcnow_iso()
    execute_sync(
        """
        UPDATE pending_actions
        SET status = 'executed',
            result_json = ?,
            executed_by_user_id = ?,
            executed_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (json.dumps(result, ensure_ascii=False, sort_keys=True), context.auth.user_id, now, now, action_id),
    )
    return get_pending_action(action_id).model_dump()


async def _dispatch_pending_action(item: PendingActionItem, *, context: RequestContext) -> dict[str, Any]:
    payload = item.payload
    if item.action_type == "send_email_reply":
        from app.integrations.support_email import send_email_reply

        return await send_email_reply(
            email_id=int(payload["email_id"]),
            body=str(payload["body"]),
            to_address=payload.get("to_address"),
            context=context,
        )

    if item.action_type == "delete_google_drive_source":
        from app.drive_sync import delete_google_drive_source

        return delete_google_drive_source(int(payload["source_id"]), mode=str(payload.get("mode") or "unlink"))

    if item.action_type == "sync_google_drive_source":
        from app.drive_sync import sync_google_drive_source

        return await sync_google_drive_source(
            int(payload["source_id"]),
            triggered_by_user_id=context.auth.user_id,
            trigger_mode="approved_action",
            force_full=bool(payload.get("force_full")),
        )

    raise RuntimeError(f"Unsupported pending action type: {item.action_type}")


def draft_email_reply_action(
    *,
    email_id: int,
    body: str,
    to_address: str | None,
    context: RequestContext,
) -> dict[str, Any]:
    payload = {"email_id": email_id, "body": body, "to_address": to_address}
    return create_pending_action(
        action_type="send_email_reply",
        risk_level="critical",
        title=f"Send email reply for message {email_id}",
        summary=f"Reply body preview: {body.strip()[:180]}",
        payload=payload,
        context=context,
    )


def draft_drive_delete_action(*, source_id: int, mode: str, context: RequestContext) -> dict[str, Any]:
    return create_pending_action(
        action_type="delete_google_drive_source",
        risk_level="critical" if mode == "purge" else "high",
        title=f"Delete Google Drive source {source_id}",
        summary=f"Mode: {mode}",
        payload={"source_id": source_id, "mode": mode},
        context=context,
    )


def draft_drive_full_sync_action(*, source_id: int, force_full: bool, context: RequestContext) -> dict[str, Any]:
    return create_pending_action(
        action_type="sync_google_drive_source",
        risk_level="high",
        title=f"Run full Google Drive sync for source {source_id}",
        summary="Force full sync can re-import many files and queue ingest jobs.",
        payload={"source_id": source_id, "force_full": force_full},
        context=context,
    )
