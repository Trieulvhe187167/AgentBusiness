"""
Helpers for tool-call audit logging.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from app.database import execute_sync, utcnow_iso
from app.models import RequestContext


def _coerce_request_context(request_context: RequestContext | dict[str, Any] | None) -> dict[str, Any]:
    if request_context is None:
        return {}
    if isinstance(request_context, RequestContext):
        return request_context.model_dump()
    return dict(request_context)


def log_tool_call(
    tool_name: str,
    *,
    tool_call_id: str | None = None,
    request_context: RequestContext | dict[str, Any] | None = None,
    args: dict[str, Any] | None = None,
    tool_status: str,
    result_summary: str | None = None,
    latency_ms: int | None = None,
    error_message: str | None = None,
) -> str:
    context = _coerce_request_context(request_context)
    auth = context.get("auth") or {}
    call_id = tool_call_id or uuid.uuid4().hex[:12]

    execute_sync(
        """
        INSERT INTO tool_audit_logs (
            tool_call_id,
            request_id,
            session_id,
            user_id,
            roles_json,
            channel,
            tenant_id,
            org_id,
            kb_id,
            kb_key,
            tool_name,
            args_json,
            result_summary,
            tool_status,
            latency_ms,
            error_message,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            call_id,
            context.get("request_id"),
            context.get("session_id"),
            auth.get("user_id"),
            json.dumps(auth.get("roles") or [], ensure_ascii=False),
            auth.get("channel"),
            auth.get("tenant_id"),
            auth.get("org_id"),
            context.get("kb_id"),
            context.get("kb_key"),
            tool_name,
            json.dumps(args or {}, ensure_ascii=False),
            result_summary,
            tool_status,
            latency_ms,
            error_message,
            utcnow_iso(),
        ),
    )
    return call_id
