"""
Shared authorization policy helpers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException

from app.auth_audit import log_auth_decision
from app.models import AuthContext

INTERNAL_ACCESS_ROLES = {"employee", "staff", "internal", "support", "admin"}
ADMIN_ACCESS_ROLES = {"admin"}


class AuthorizationDeniedError(PermissionError):
    pass


def coerce_auth_context(auth_context: AuthContext | Mapping[str, Any] | None) -> AuthContext:
    if isinstance(auth_context, AuthContext):
        return auth_context
    if isinstance(auth_context, Mapping):
        return AuthContext.model_validate(dict(auth_context))
    return AuthContext()


def _read_attr(source: object, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def has_any_role(auth_context: AuthContext | Mapping[str, Any] | None, roles: set[str] | list[str] | tuple[str, ...]) -> bool:
    auth = coerce_auth_context(auth_context)
    return bool(set(auth.roles).intersection({str(item).strip().lower() for item in roles if str(item).strip()}))


def can_manage_kb(auth_context: AuthContext | Mapping[str, Any] | None) -> bool:
    return has_any_role(auth_context, ADMIN_ACCESS_ROLES)


def can_view_logs(auth_context: AuthContext | Mapping[str, Any] | None) -> bool:
    return has_any_role(auth_context, ADMIN_ACCESS_ROLES)


def can_access_kb(
    kb: Mapping[str, Any] | object,
    auth_context: AuthContext | Mapping[str, Any] | None,
) -> bool:
    auth = coerce_auth_context(auth_context)
    access_level = str(_read_attr(kb, "access_level", "public") or "public").lower()
    roles = set(auth.roles)
    is_admin = bool(roles.intersection(ADMIN_ACCESS_ROLES))
    tenant_id = _read_attr(kb, "tenant_id")
    org_id = _read_attr(kb, "org_id")

    if access_level == "public":
        allowed = True
    if access_level == "internal":
        allowed = bool(auth.user_id and roles.intersection(INTERNAL_ACCESS_ROLES))
    if access_level == "admin":
        allowed = is_admin
    if access_level not in {"public", "internal", "admin"}:
        return False

    if not allowed:
        return False
    if is_admin:
        return True
    if tenant_id and str(tenant_id).strip() != str(auth.tenant_id or "").strip():
        return False
    if org_id and str(org_id).strip() != str(auth.org_id or "").strip():
        return False
    return True


def ensure_can_access_kb(
    kb: Mapping[str, Any] | object,
    auth_context: AuthContext | Mapping[str, Any] | None,
    *,
    request_context: Mapping[str, Any] | object | None = None,
) -> None:
    auth = coerce_auth_context(auth_context)
    access_level = str(_read_attr(kb, "access_level", "public") or "public").lower()
    tenant_id = _read_attr(kb, "tenant_id")
    org_id = _read_attr(kb, "org_id")
    resource_id = str(_read_attr(kb, "id", None) or _read_attr(kb, "key", None) or "")

    if can_access_kb(kb, auth):
        log_auth_decision(
            resource_type="knowledge_base",
            resource_id=resource_id or None,
            action="read",
            decision="allow",
            reason=f"access_level={access_level}",
            auth_context=auth,
            request_context=request_context,
        )
        return
    if access_level == "internal":
        log_auth_decision(
            resource_type="knowledge_base",
            resource_id=resource_id or None,
            action="read",
            decision="deny",
            reason="internal_access_required",
            auth_context=auth,
            request_context=request_context,
        )
        raise HTTPException(status_code=403, detail="Internal Knowledge Base access required")
    if access_level == "admin":
        log_auth_decision(
            resource_type="knowledge_base",
            resource_id=resource_id or None,
            action="read",
            decision="deny",
            reason="admin_access_required",
            auth_context=auth,
            request_context=request_context,
        )
        raise HTTPException(status_code=403, detail="Admin Knowledge Base access required")
    if tenant_id and str(tenant_id).strip() != str(auth.tenant_id or "").strip():
        log_auth_decision(
            resource_type="knowledge_base",
            resource_id=resource_id or None,
            action="read",
            decision="deny",
            reason="tenant_scope_required",
            auth_context=auth,
            request_context=request_context,
        )
        raise HTTPException(status_code=403, detail="Tenant-scoped Knowledge Base access required")
    if org_id and str(org_id).strip() != str(auth.org_id or "").strip():
        log_auth_decision(
            resource_type="knowledge_base",
            resource_id=resource_id or None,
            action="read",
            decision="deny",
            reason="org_scope_required",
            auth_context=auth,
            request_context=request_context,
        )
        raise HTTPException(status_code=403, detail="Org-scoped Knowledge Base access required")
    log_auth_decision(
        resource_type="knowledge_base",
        resource_id=resource_id or None,
        action="read",
        decision="deny",
        reason="knowledge_base_access_denied",
        auth_context=auth,
        request_context=request_context,
    )
    raise HTTPException(status_code=403, detail="Knowledge Base access denied")


def authorize_tool_access(
    tool_name: str,
    auth_policy: Mapping[str, Any] | object,
    *,
    context: Mapping[str, Any] | object | None,
    arguments: Mapping[str, Any] | None = None,
) -> None:
    auth = coerce_auth_context(_read_attr(context, "auth"))
    resource_id = tool_name or None

    require_user_id = bool(_read_attr(auth_policy, "require_user_id", False))
    allow_anonymous = bool(_read_attr(auth_policy, "allow_anonymous", False))
    required_roles = [
        str(item).strip().lower()
        for item in (_read_attr(auth_policy, "required_roles", []) or [])
        if str(item).strip()
    ]
    allowed_channels = [
        str(item).strip().lower()
        for item in (_read_attr(auth_policy, "allowed_channels", []) or [])
        if str(item).strip()
    ]
    requires_tenant_match = bool(_read_attr(auth_policy, "requires_tenant_match", False))
    user_roles = set(auth.roles)
    is_admin = bool(user_roles.intersection(ADMIN_ACCESS_ROLES))

    if require_user_id and not auth.user_id:
        log_auth_decision(
            resource_type="tool",
            resource_id=resource_id,
            action="execute",
            decision="deny",
            reason="user_id_required",
            auth_context=auth,
            request_context=context,
        )
        raise AuthorizationDeniedError(f"Tool '{tool_name}' requires user_id")
    if not allow_anonymous and not auth.user_id and not required_roles:
        log_auth_decision(
            resource_type="tool",
            resource_id=resource_id,
            action="execute",
            decision="deny",
            reason="anonymous_not_allowed",
            auth_context=auth,
            request_context=context,
        )
        raise AuthorizationDeniedError(f"Tool '{tool_name}' does not allow anonymous access")
    if required_roles and not user_roles.intersection(required_roles):
        roles = ", ".join(required_roles)
        log_auth_decision(
            resource_type="tool",
            resource_id=resource_id,
            action="execute",
            decision="deny",
            reason=f"required_roles={roles}",
            auth_context=auth,
            request_context=context,
        )
        raise AuthorizationDeniedError(f"Tool '{tool_name}' requires one of roles: {roles}")
    if allowed_channels and str(auth.channel or "web").strip().lower() not in allowed_channels:
        channels = ", ".join(allowed_channels)
        log_auth_decision(
            resource_type="tool",
            resource_id=resource_id,
            action="execute",
            decision="deny",
            reason=f"allowed_channels={channels}",
            auth_context=auth,
            request_context=context,
        )
        raise AuthorizationDeniedError(f"Tool '{tool_name}' is not allowed on channel(s): {channels}")
    if requires_tenant_match and not is_admin and auth.tenant_id:
        target_tenant = str((arguments or {}).get("tenant_id") or "").strip()
        if target_tenant and target_tenant != str(auth.tenant_id).strip():
            log_auth_decision(
                resource_type="tool",
                resource_id=resource_id,
                action="execute",
                decision="deny",
                reason="tenant_scope_mismatch",
                auth_context=auth,
                request_context=context,
            )
            raise AuthorizationDeniedError(f"Tool '{tool_name}' requires tenant scope match")
        target_org = str((arguments or {}).get("org_id") or "").strip()
        if target_org and auth.org_id and target_org != str(auth.org_id).strip():
            log_auth_decision(
                resource_type="tool",
                resource_id=resource_id,
                action="execute",
                decision="deny",
                reason="org_scope_mismatch",
                auth_context=auth,
                request_context=context,
            )
            raise AuthorizationDeniedError(f"Tool '{tool_name}' requires org scope match")
    log_auth_decision(
        resource_type="tool",
        resource_id=resource_id,
        action="execute",
        decision="allow",
        reason="tool_access_granted",
        auth_context=auth,
        request_context=context,
    )
