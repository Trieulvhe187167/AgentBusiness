"""Security policy helpers for the MCP adapter."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

from fastapi import Request

from app.config import settings
from app.database import execute_sync, utcnow_iso
from app.models import AuthContext

RISK_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
HIGH_RISK_LEVELS = {"high", "critical"}
CLIENT_ID_HEADER = "X-MCP-Client-Id"
SCOPES_HEADER = "X-MCP-Scopes"


@dataclass(frozen=True)
class McpClientContext:
    client_id: str | None
    session_id: str | None
    scopes: set[str]


@dataclass(frozen=True)
class McpToolDecision:
    allowed: bool
    reason: str
    required_scopes: set[str]
    granted_scopes: set[str]


def _normalize_token(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _parse_scope_header(raw: str | None) -> set[str]:
    if not raw:
        return set()
    normalized = str(raw).replace(",", " ")
    return {_normalize_token(item) for item in normalized.split() if _normalize_token(item)}


def mcp_client_context(request: Request) -> McpClientContext:
    client_id = (request.headers.get(CLIENT_ID_HEADER) or "").strip() or None
    session_id = (request.headers.get("Mcp-Session-Id") or "").strip() or None
    return McpClientContext(
        client_id=client_id,
        session_id=session_id,
        scopes=_parse_scope_header(request.headers.get(SCOPES_HEADER)),
    )


def tool_required_scopes(tool_name: str, auth_policy: dict[str, Any]) -> set[str]:
    scope = _normalize_token(auth_policy.get("scope") or "general")
    risk = _normalize_token(auth_policy.get("risk_level") or "low")
    return {f"tool:{tool_name}", f"scope:{scope}", f"risk:{risk}"}


def scope_allows(required_scopes: set[str], granted_scopes: set[str]) -> bool:
    if "*" in granted_scopes or "mcp:*" in granted_scopes:
        return True
    return bool(required_scopes.intersection(granted_scopes))


def role_allows(required_roles: list[str], auth: AuthContext) -> bool:
    required = {_normalize_token(role) for role in required_roles if _normalize_token(role)}
    if not required:
        return True
    roles = {_normalize_token(role) for role in auth.roles}
    return bool(required.intersection(roles))


def risk_allows(tool_name: str, risk_level: str) -> bool:
    risk = _normalize_token(risk_level) or "low"
    if risk not in HIGH_RISK_LEVELS:
        return True
    return tool_name in settings.mcp_high_risk_tool_names


def decide_tool_exposure(tool_definition, *, auth: AuthContext, client: McpClientContext) -> McpToolDecision:
    policy = tool_definition.auth_policy
    tool_name = tool_definition.name
    risk_level = _normalize_token(policy.get("risk_level") or "low")
    required_scopes = tool_required_scopes(tool_name, policy)
    if not risk_allows(tool_name, risk_level):
        return McpToolDecision(False, "high_risk_denied_by_default", required_scopes, client.scopes)
    if not role_allows(policy.get("required_roles") or [], auth):
        return McpToolDecision(False, "role_not_allowed", required_scopes, client.scopes)
    if settings.mcp_require_tool_scopes and not scope_allows(required_scopes, client.scopes):
        return McpToolDecision(False, "scope_not_granted", required_scopes, client.scopes)
    return McpToolDecision(True, "allowed", required_scopes, client.scopes)


def sign_tool_manifest(tools: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "server": settings.mcp_server_name,
        "version": settings.mcp_server_version,
        "tools": tools,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    secret = settings.mcp_manifest_signing_secret.strip() or settings.gateway_shared_secret.strip() or "dev-mcp-manifest-secret"
    signature = hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return {
        "algorithm": "HMAC-SHA256",
        "payload_hash": payload_hash,
        "signature": signature,
        "signed_at": utcnow_iso(),
        "key_hint": "mcp_manifest_signing_secret" if settings.mcp_manifest_signing_secret.strip() else "dev_or_gateway_secret",
    }


def log_mcp_audit(
    *,
    request_id: str | None,
    auth: AuthContext,
    client: McpClientContext,
    method: str,
    decision: str,
    reason: str | None = None,
    tool_name: str | None = None,
    resource_uri: str | None = None,
    required_scopes: set[str] | None = None,
    risk_level: str | None = None,
    tool_scope: str | None = None,
) -> None:
    execute_sync(
        """
        INSERT INTO mcp_audit_logs (
            request_id, mcp_client_id, mcp_session_id, user_id, roles_json,
            channel, tenant_id, org_id, method, tool_name, resource_uri,
            required_scopes_json, granted_scopes_json, risk_level, tool_scope,
            decision, reason, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request_id,
            client.client_id,
            client.session_id,
            auth.user_id,
            json.dumps(auth.roles, ensure_ascii=False),
            auth.channel,
            auth.tenant_id,
            auth.org_id,
            method,
            tool_name,
            resource_uri,
            json.dumps(sorted(required_scopes or []), ensure_ascii=False),
            json.dumps(sorted(client.scopes), ensure_ascii=False),
            risk_level,
            tool_scope,
            decision,
            reason,
            utcnow_iso(),
        ),
    )
