"""
Helpers for authorization audit logging.
"""

from __future__ import annotations

import json
from typing import Any

from app.database import execute_sync, utcnow_iso
from app.models import AuthContext, RequestContext


def _coerce_auth_context(auth_context: AuthContext | dict[str, Any] | None) -> dict[str, Any]:
    if auth_context is None:
        return {}
    if isinstance(auth_context, AuthContext):
        return auth_context.model_dump()
    return dict(auth_context)


def _coerce_request_context(request_context: RequestContext | dict[str, Any] | None) -> dict[str, Any]:
    if request_context is None:
        return {}
    if isinstance(request_context, RequestContext):
        return request_context.model_dump()
    return dict(request_context)


def log_auth_decision(
    *,
    resource_type: str,
    action: str,
    decision: str,
    resource_id: str | None = None,
    reason: str | None = None,
    auth_context: AuthContext | dict[str, Any] | None = None,
    request_context: RequestContext | dict[str, Any] | None = None,
) -> None:
    context = _coerce_request_context(request_context)
    auth = _coerce_auth_context(auth_context or context.get("auth"))

    execute_sync(
        """
        INSERT INTO auth_audit_logs (
            request_id,
            user_id,
            roles_json,
            channel,
            tenant_id,
            org_id,
            resource_type,
            resource_id,
            action,
            decision,
            reason,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            context.get("request_id"),
            auth.get("user_id"),
            json.dumps(auth.get("roles") or [], ensure_ascii=False),
            auth.get("channel"),
            auth.get("tenant_id"),
            auth.get("org_id"),
            resource_type,
            resource_id,
            action,
            decision,
            reason,
            utcnow_iso(),
        ),
    )
