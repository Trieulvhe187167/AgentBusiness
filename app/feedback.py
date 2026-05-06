"""
Chat answer feedback persistence and authorization.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException

from app.authorization import can_manage_kb
from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.models import (
    AuthContext,
    ChatFeedbackItem,
    FeedbackSummaryOutput,
    FeedbackSummaryGroup,
    ListChatFeedbackOutput,
    RequestContext,
    SubmitChatFeedbackInput,
)


def _parse_roles(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _roles_json(auth: AuthContext) -> str:
    return json.dumps(auth.roles, ensure_ascii=False, sort_keys=True)


def _positive_rate(up: int, down: int) -> float | None:
    total = up + down
    if total <= 0:
        return None
    return round(up / total, 4)


def _resolve_chat_log(payload: SubmitChatFeedbackInput) -> dict[str, Any]:
    if payload.chat_log_id is not None:
        row = fetch_one_sync("SELECT * FROM chat_logs WHERE id = ?", (payload.chat_log_id,))
    else:
        row = fetch_one_sync(
            """
            SELECT *
            FROM chat_logs
            WHERE request_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (payload.request_id,),
        )
    if not row:
        raise HTTPException(status_code=404, detail="Chat log not found")
    return row


def _ensure_feedback_allowed(chat_log: dict[str, Any], context: RequestContext) -> None:
    auth = context.auth
    if can_manage_kb(auth):
        return
    if not auth.user_id:
        raise HTTPException(status_code=403, detail="Authentication required to submit chat feedback")
    owner_user_id = chat_log.get("user_id")
    if not owner_user_id:
        raise HTTPException(status_code=403, detail="Only admins can submit feedback for anonymous chat logs")
    if owner_user_id != auth.user_id:
        raise HTTPException(status_code=403, detail="Cannot submit feedback for another user's chat log")


def _serialize_feedback(row: dict[str, Any]) -> dict[str, Any]:
    return ChatFeedbackItem(
        id=int(row["id"]),
        chat_log_id=int(row["chat_log_id"]),
        request_id=row.get("request_id"),
        rating=row["rating"],
        reason_code=row.get("reason_code"),
        comment=row.get("comment"),
        created_by_user_id=row["created_by_user_id"],
        roles=_parse_roles(row.get("roles_json")),
        channel=row.get("channel"),
        tenant_id=row.get("tenant_id"),
        org_id=row.get("org_id"),
        chat_session_id=row.get("chat_session_id"),
        kb_id=row.get("kb_id"),
        kb_key=row.get("kb_key"),
        chat_user_message=row.get("chat_user_message"),
        chat_answer_text=row.get("chat_answer_text"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    ).model_dump()


def submit_chat_feedback(payload: SubmitChatFeedbackInput, *, context: RequestContext) -> dict[str, Any]:
    chat_log = _resolve_chat_log(payload)
    _ensure_feedback_allowed(chat_log, context)
    if not context.auth.user_id:
        raise HTTPException(status_code=403, detail="Authentication required to submit chat feedback")

    now = utcnow_iso()
    execute_sync(
        """
        INSERT INTO chat_feedback (
            chat_log_id, request_id, rating, reason_code, comment,
            created_by_user_id, roles_json, channel, tenant_id, org_id,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chat_log_id, created_by_user_id) DO UPDATE SET
            request_id = excluded.request_id,
            rating = excluded.rating,
            reason_code = excluded.reason_code,
            comment = excluded.comment,
            roles_json = excluded.roles_json,
            channel = excluded.channel,
            tenant_id = excluded.tenant_id,
            org_id = excluded.org_id,
            updated_at = excluded.updated_at
        """,
        (
            int(chat_log["id"]),
            chat_log.get("request_id"),
            payload.rating,
            payload.reason_code,
            payload.comment,
            context.auth.user_id,
            _roles_json(context.auth),
            context.auth.channel,
            context.auth.tenant_id,
            context.auth.org_id,
            now,
            now,
        ),
    )
    row = fetch_one_sync(
        """
        SELECT cf.*, cl.session_id AS chat_session_id, cl.kb_id, cl.kb_key,
               cl.user_message AS chat_user_message, cl.answer_text AS chat_answer_text
        FROM chat_feedback cf
        JOIN chat_logs cl ON cl.id = cf.chat_log_id
        WHERE cf.chat_log_id = ? AND cf.created_by_user_id = ?
        """,
        (int(chat_log["id"]), context.auth.user_id),
    )
    if not row:
        raise HTTPException(status_code=500, detail="Chat feedback was not persisted")
    return _serialize_feedback(row)


def list_chat_feedback(*, rating: str | None = None, kb_id: int | None = None, limit: int = 50) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if rating:
        normalized = rating.strip().lower()
        if normalized not in {"up", "down"}:
            raise HTTPException(status_code=422, detail="rating must be 'up' or 'down'")
        clauses.append("cf.rating = ?")
        params.append(normalized)
    if kb_id is not None:
        clauses.append("cl.kb_id = ?")
        params.append(int(kb_id))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, min(limit, 500)))
    rows = fetch_all_sync(
        f"""
        SELECT cf.*, cl.session_id AS chat_session_id, cl.kb_id, cl.kb_key,
               cl.user_message AS chat_user_message, cl.answer_text AS chat_answer_text
        FROM chat_feedback cf
        JOIN chat_logs cl ON cl.id = cf.chat_log_id
        {where}
        ORDER BY cf.updated_at DESC, cf.id DESC
        LIMIT ?
        """,
        tuple(params),
    )
    items = [_serialize_feedback(row) for row in rows]
    return ListChatFeedbackOutput(total=len(items), items=items).model_dump()


def feedback_summary() -> dict[str, Any]:
    totals = fetch_one_sync(
        """
        SELECT
            COUNT(*) AS total,
            COALESCE(SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END), 0) AS up,
            COALESCE(SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END), 0) AS down
        FROM chat_feedback
        """
    ) or {"total": 0, "up": 0, "down": 0}
    up = int(totals.get("up") or 0)
    down = int(totals.get("down") or 0)
    rows = fetch_all_sync(
        """
        SELECT
            cl.kb_id,
            cl.kb_key,
            COUNT(*) AS total,
            COALESCE(SUM(CASE WHEN cf.rating = 'up' THEN 1 ELSE 0 END), 0) AS up,
            COALESCE(SUM(CASE WHEN cf.rating = 'down' THEN 1 ELSE 0 END), 0) AS down
        FROM chat_feedback cf
        JOIN chat_logs cl ON cl.id = cf.chat_log_id
        GROUP BY cl.kb_id, cl.kb_key
        ORDER BY total DESC, cl.kb_key ASC
        """
    )
    groups = [
        FeedbackSummaryGroup(
            kb_id=row.get("kb_id"),
            kb_key=row.get("kb_key"),
            total=int(row.get("total") or 0),
            up=int(row.get("up") or 0),
            down=int(row.get("down") or 0),
            positive_rate=_positive_rate(int(row.get("up") or 0), int(row.get("down") or 0)),
        )
        for row in rows
    ]
    return FeedbackSummaryOutput(
        total=int(totals.get("total") or 0),
        up=up,
        down=down,
        positive_rate=_positive_rate(up, down),
        by_kb=groups,
    ).model_dump()
