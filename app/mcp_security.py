"""Security policy helpers for the MCP adapter."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request

from app.config import settings
from app.database import execute_sync, fetch_all_sync, utcnow_iso
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


@dataclass(frozen=True)
class McpQuotaDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_after_seconds: int
    reason: str = "allowed"


@dataclass(slots=True)
class _QuotaBucket:
    count: int
    reset_at: float


_quota_buckets: dict[str, _QuotaBucket] = {}


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


def _token_matches(configured: str, presented: str) -> bool:
    configured = str(configured or "").strip()
    presented = str(presented or "").strip()
    if not configured or not presented:
        return False
    if configured.startswith("sha256:"):
        expected = configured.removeprefix("sha256:").strip().lower()
        digest = hashlib.sha256(presented.encode("utf-8")).hexdigest()
        return hmac.compare_digest(expected, digest)
    return hmac.compare_digest(configured, presented)


def validate_mcp_client_token(request: Request) -> None:
    if not settings.mcp_require_client_token:
        return
    client = mcp_client_context(request)
    if not client.client_id:
        raise HTTPException(status_code=401, detail="MCP client id is required")
    configured = settings.mcp_client_token_map.get(client.client_id)
    if not configured:
        raise HTTPException(status_code=401, detail="MCP client is not registered")
    header_name = settings.mcp_client_token_header.strip() or "X-MCP-Client-Token"
    presented = (request.headers.get(header_name) or "").strip()
    if not _token_matches(configured, presented):
        raise HTTPException(status_code=401, detail="Invalid MCP client token")


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


def _quota_limit_for(client: McpClientContext, tool_name: str) -> int:
    rules = settings.mcp_tool_quota_rules
    client_id = client.client_id or "anonymous"
    for key in ((client_id, tool_name), (client_id, "*"), ("*", tool_name), ("*", "*")):
        if key in rules:
            return int(rules[key])
    return max(0, int(settings.mcp_default_tool_quota_per_window or 0))


def mcp_tool_quota_status(client: McpClientContext, tool_name: str) -> McpQuotaDecision:
    limit = _quota_limit_for(client, tool_name)
    if limit <= 0:
        return McpQuotaDecision(True, 0, 0, 0, "quota_disabled")

    window_seconds = max(1, int(settings.mcp_tool_quota_window_seconds or 60))
    now = time.monotonic()
    client_id = client.client_id or "anonymous"
    bucket_key = f"{client_id}:{tool_name}"
    bucket = _quota_buckets.get(bucket_key)
    if bucket is None or bucket.reset_at <= now:
        bucket = _QuotaBucket(count=0, reset_at=now + window_seconds)
        _quota_buckets[bucket_key] = bucket
        _cleanup_quota_buckets(now)

    reset_after = max(1, int(bucket.reset_at - now))
    if bucket.count >= limit:
        return McpQuotaDecision(False, limit, 0, reset_after, "tool_quota_exceeded")
    return McpQuotaDecision(True, limit, max(0, limit - bucket.count), reset_after)


def consume_mcp_tool_quota(client: McpClientContext, tool_name: str) -> McpQuotaDecision:
    decision = mcp_tool_quota_status(client, tool_name)
    if not decision.allowed or decision.limit <= 0:
        return decision
    bucket_key = f"{client.client_id or 'anonymous'}:{tool_name}"
    bucket = _quota_buckets.get(bucket_key)
    if bucket is not None:
        bucket.count += 1
        return McpQuotaDecision(
            True,
            decision.limit,
            max(0, decision.remaining - 1),
            decision.reset_after_seconds,
        )
    return decision


def reset_mcp_quota_buckets() -> None:
    _quota_buckets.clear()


def _cleanup_quota_buckets(now: float) -> None:
    if len(_quota_buckets) <= settings.rate_limit_max_buckets:
        return
    expired = [key for key, bucket in _quota_buckets.items() if bucket.reset_at <= now]
    for key in expired:
        _quota_buckets.pop(key, None)


def list_mcp_quota_dashboard() -> list[dict[str, Any]]:
    now = time.monotonic()
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for (client_id, tool_name), limit in settings.mcp_tool_quota_rules.items():
        rows[(client_id, tool_name)] = {
            "client_id": client_id,
            "tool_name": tool_name,
            "limit": int(limit),
            "used": 0,
            "remaining": int(limit),
            "reset_after_seconds": 0,
            "source": "configured",
        }
    for bucket_key, bucket in list(_quota_buckets.items()):
        if bucket.reset_at <= now:
            _quota_buckets.pop(bucket_key, None)
            continue
        client_id, sep, tool_name = bucket_key.partition(":")
        if not sep:
            continue
        limit = _quota_limit_for(McpClientContext(client_id=client_id, session_id=None, scopes=set()), tool_name)
        if limit <= 0:
            continue
        key = (client_id, tool_name)
        rows[key] = {
            "client_id": client_id,
            "tool_name": tool_name,
            "limit": limit,
            "used": int(bucket.count),
            "remaining": max(0, limit - int(bucket.count)),
            "reset_after_seconds": max(1, int(bucket.reset_at - now)),
            "source": "active_window",
        }
    return sorted(rows.values(), key=lambda item: (str(item["client_id"]), str(item["tool_name"])))


def list_mcp_recent_denies(limit: int = 20) -> list[dict[str, Any]]:
    rows = fetch_all_sync(
        """
        SELECT id, request_id, mcp_client_id, mcp_session_id, user_id, roles_json,
               method, tool_name, resource_uri, required_scopes_json, granted_scopes_json,
               risk_level, tool_scope, decision, reason, created_at
        FROM mcp_audit_logs
        WHERE decision = 'deny'
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (max(1, min(int(limit), 100)),),
    )
    items = []
    for row in rows:
        items.append(
            {
                "id": int(row["id"]),
                "request_id": row.get("request_id"),
                "client_id": row.get("mcp_client_id"),
                "session_id": row.get("mcp_session_id"),
                "user_id": row.get("user_id"),
                "roles": json.loads(row.get("roles_json") or "[]"),
                "method": row.get("method"),
                "tool_name": row.get("tool_name"),
                "resource_uri": row.get("resource_uri"),
                "required_scopes": json.loads(row.get("required_scopes_json") or "[]"),
                "granted_scopes": json.loads(row.get("granted_scopes_json") or "[]"),
                "risk_level": row.get("risk_level"),
                "tool_scope": row.get("tool_scope"),
                "decision": row.get("decision"),
                "reason": row.get("reason"),
                "created_at": row.get("created_at"),
            }
        )
    return items


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


def preview_tool_risk(tool_definition, *, auth: AuthContext, client: McpClientContext) -> dict[str, Any]:
    decision = decide_tool_exposure(tool_definition, auth=auth, client=client)
    quota = mcp_tool_quota_status(client, tool_definition.name)
    policy = tool_definition.auth_policy
    return {
        "tool_name": tool_definition.name,
        "allowed": decision.allowed and quota.allowed,
        "decision_reason": quota.reason if decision.allowed and not quota.allowed else decision.reason,
        "risk_level": policy.get("risk_level"),
        "scope": policy.get("scope"),
        "required_roles": policy.get("required_roles") or [],
        "required_scopes": sorted(decision.required_scopes),
        "granted_scopes": sorted(decision.granted_scopes),
        "quota": {
            "enabled": quota.limit > 0,
            "limit": quota.limit,
            "remaining": quota.remaining,
            "reset_after_seconds": quota.reset_after_seconds,
        },
    }


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


def record_mcp_session_activity(
    *,
    auth: AuthContext,
    client: McpClientContext,
    method: str,
    decision: str = "allow",
    reason: str | None = None,
    is_tool_call: bool = False,
    count_request: bool = True,
) -> None:
    client_id = client.client_id or "anonymous"
    session_id = client.session_id or "no-session"
    now = utcnow_iso()
    allowed_inc = 1 if decision == "allow" else 0
    denied_inc = 1 if decision == "deny" else 0
    tool_inc = 1 if is_tool_call else 0
    request_inc = 1 if count_request else 0
    execute_sync(
        """
        INSERT INTO mcp_sessions (
            mcp_client_id, mcp_session_id, user_id, roles_json, channel,
            tenant_id, org_id, first_seen_at, last_seen_at, last_method,
            last_decision, last_reason, request_count, tool_call_count,
            allowed_count, denied_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(mcp_client_id, mcp_session_id) DO UPDATE SET
            user_id = excluded.user_id,
            roles_json = excluded.roles_json,
            channel = excluded.channel,
            tenant_id = excluded.tenant_id,
            org_id = excluded.org_id,
            last_seen_at = excluded.last_seen_at,
            last_method = excluded.last_method,
            last_decision = excluded.last_decision,
            last_reason = excluded.last_reason,
            request_count = request_count + excluded.request_count,
            tool_call_count = tool_call_count + excluded.tool_call_count,
            allowed_count = allowed_count + excluded.allowed_count,
            denied_count = denied_count + excluded.denied_count
        """,
        (
            client_id,
            session_id,
            auth.user_id,
            json.dumps(auth.roles, ensure_ascii=False),
            auth.channel,
            auth.tenant_id,
            auth.org_id,
            now,
            now,
            method,
            decision,
            reason,
            request_inc,
            tool_inc,
            allowed_inc,
            denied_inc,
        ),
    )


def list_mcp_sessions(limit: int = 20) -> list[dict[str, Any]]:
    rows = fetch_all_sync(
        """
        SELECT *
        FROM mcp_sessions
        ORDER BY last_seen_at DESC, id DESC
        LIMIT ?
        """,
        (max(1, min(int(limit), 100)),),
    )
    return [
        {
            "id": int(row["id"]),
            "client_id": row["mcp_client_id"],
            "session_id": row["mcp_session_id"],
            "user_id": row.get("user_id"),
            "roles": json.loads(row.get("roles_json") or "[]"),
            "channel": row.get("channel"),
            "tenant_id": row.get("tenant_id"),
            "org_id": row.get("org_id"),
            "first_seen_at": row.get("first_seen_at"),
            "last_seen_at": row.get("last_seen_at"),
            "last_method": row.get("last_method"),
            "last_decision": row.get("last_decision"),
            "last_reason": row.get("last_reason"),
            "request_count": int(row.get("request_count") or 0),
            "tool_call_count": int(row.get("tool_call_count") or 0),
            "allowed_count": int(row.get("allowed_count") or 0),
            "denied_count": int(row.get("denied_count") or 0),
        }
        for row in rows
    ]
