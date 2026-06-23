"""
Minimal MCP JSON-RPC adapter for the internal ToolRegistry.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from app.auth import get_request_auth
from app.authorization import can_manage_kb
from app.background_jobs import list_background_jobs
from app.case_timeline import build_case_timeline
from app.config import settings
from app.database import fetch_all, fetch_one
from app.evaluations import get_agent_eval_run, list_agent_eval_runs
from app.kb_service import list_accessible_kbs, open_db, resolve_kb_scope
from app.mcp_security import (
    McpClientContext,
    consume_mcp_tool_quota,
    decide_tool_exposure,
    list_mcp_quota_dashboard,
    list_mcp_recent_denies,
    list_mcp_sessions,
    log_mcp_audit,
    mcp_client_context,
    preview_tool_risk,
    record_mcp_session_activity,
    scope_allows,
    sign_tool_manifest,
    tool_required_scopes,
    validate_mcp_client_token,
)
from app.models import AuthContext, RequestContext
from app.observability import trace_span
from app.tools import tool_registry
from app.tools.registry import ToolAuthorizationError, ToolExecutionError, ToolValidationError
from app.vector_store import vector_store

JSONRPC_VERSION = "2.0"
logger = logging.getLogger(__name__)

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
TOOL_ERROR = -32000


def _success(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str, data: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": code, "message": message},
    }
    if data is not None:
        payload["error"]["data"] = data
    return payload


def _tool_to_mcp(definition) -> dict[str, Any]:
    payload = definition.model_dump()
    auth_policy = payload["auth_policy"]
    return {
        "name": payload["name"],
        "description": payload["description"],
        "inputSchema": payload["input_schema"],
        "annotations": {
            "idempotentHint": payload["idempotent"],
            "riskLevel": auth_policy.get("risk_level"),
            "scope": auth_policy.get("scope"),
            "requiredScopes": sorted(tool_required_scopes(payload["name"], auth_policy)),
            "requiredRoles": auth_policy.get("required_roles") or [],
        },
    }


def _resource(uri: str, name: str, description: str) -> dict[str, Any]:
    return {
        "uri": uri,
        "name": name,
        "description": description,
        "mimeType": "application/json",
    }


def _resource_template(uri_template: str, name: str, description: str) -> dict[str, Any]:
    return {
        "uriTemplate": uri_template,
        "name": name,
        "description": description,
        "mimeType": "application/json",
    }


def _json_resource_content(uri: str, payload: dict[str, Any] | list[Any]) -> dict[str, Any]:
    return {
        "uri": uri,
        "mimeType": "application/json",
        "text": json.dumps(payload, ensure_ascii=False),
    }


def _is_exposed_tool(name: str) -> bool:
    exposed = settings.mcp_exposed_tool_names
    return not exposed or name in exposed


def _exposed_tool_definitions():
    return [item for item in tool_registry.list_definitions() if _is_exposed_tool(item.name)]


def _visible_tool_definitions(request: Request):
    auth = get_request_auth(request)
    client = mcp_client_context(request)
    visible = []
    for item in _exposed_tool_definitions():
        decision = decide_tool_exposure(item, auth=auth, client=client)
        if decision.allowed:
            visible.append(item)
    return visible


def _tool_security_report(definitions, *, auth=None, client=None) -> list[dict[str, Any]]:
    rows = []
    for item in definitions:
        policy = item.auth_policy
        required_scopes = tool_required_scopes(item.name, policy)
        if client is not None and auth is not None:
            decision = decide_tool_exposure(item, auth=auth, client=client)
            quota = preview_tool_risk(item, auth=auth, client=client)["quota"]
            allowed = decision.allowed
            reason = decision.reason
        else:
            quota = {"enabled": False, "limit": 0, "remaining": 0, "reset_after_seconds": 0}
            allowed = True
            reason = "configured"
        rows.append(
            {
                "name": item.name,
                "description": item.description,
                "idempotent": item.idempotent,
                "risk_level": policy.get("risk_level"),
                "scope": policy.get("scope"),
                "required_roles": policy.get("required_roles") or [],
                "required_scopes": sorted(required_scopes),
                "exposed": allowed,
                "reason": reason,
                "quota": quota,
            }
        )
    return rows


def _validate_mcp_origin(request: Request) -> None:
    if not settings.mcp_validate_origin:
        return
    origin = (request.headers.get("Origin") or "").strip()
    if not origin:
        return
    allowed = settings.mcp_allowed_origin_values
    if "*" in allowed or origin in allowed:
        return
    raise HTTPException(status_code=403, detail="MCP Origin is not allowed")


def _validate_mcp_auth(request: Request) -> None:
    if not settings.mcp_require_auth:
        return
    auth = get_request_auth(request)
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="MCP authentication required")


def _validate_mcp_protocol_header(request: Request) -> None:
    protocol_version = (request.headers.get("MCP-Protocol-Version") or "").strip()
    if protocol_version and protocol_version != settings.mcp_protocol_version:
        raise HTTPException(status_code=400, detail="Unsupported MCP protocol version")


def _resource_scope(uri: str) -> str:
    if uri.startswith("kb://"):
        return "resource:kb"
    if uri.startswith("support://"):
        return "resource:support"
    if uri.startswith("eval://"):
        return "resource:eval"
    if uri == "jobs://recent":
        return "resource:jobs"
    if uri == "audit://recent":
        return "resource:audit"
    return "resource:*"


def _resource_required_scopes(resource_uri: str | None = None) -> set[str]:
    if not resource_uri:
        return {"resource:*"}
    return {"resource:*", _resource_scope(resource_uri)}


def _resource_scope_allowed(client: McpClientContext, resource_uri: str) -> bool:
    if not settings.mcp_require_resource_scopes:
        return True
    return scope_allows(_resource_required_scopes(resource_uri), client.scopes)


def _require_resource_admin(
    request: Request,
    *,
    method: str,
    request_id: Any,
    resource_uri: str | None = None,
    enforce_scope: bool = True,
) -> None:
    auth = get_request_auth(request)
    client = mcp_client_context(request)
    required_scopes = _resource_required_scopes(resource_uri)
    if not can_manage_kb(auth):
        log_mcp_audit(
            request_id=str(request_id) if request_id is not None else None,
            auth=auth,
            client=client,
            method=method,
            resource_uri=resource_uri,
            required_scopes=required_scopes,
            decision="deny",
            reason="admin_role_required_for_resources",
        )
        raise ToolAuthorizationError("Admin role required for MCP resources")
    if enforce_scope and settings.mcp_require_resource_scopes and not scope_allows(required_scopes, client.scopes):
        log_mcp_audit(
            request_id=str(request_id) if request_id is not None else None,
            auth=auth,
            client=client,
            method=method,
            resource_uri=resource_uri,
            required_scopes=required_scopes,
            decision="deny",
            reason="resource_scope_not_granted",
        )
        raise ToolAuthorizationError("MCP resource scope not granted")
    log_mcp_audit(
        request_id=str(request_id) if request_id is not None else None,
        auth=auth,
        client=client,
        method=method,
        resource_uri=resource_uri,
        required_scopes=required_scopes,
        decision="allow",
        reason="resource_access",
    )


def _request_context(request: Request, params: dict[str, Any] | None = None) -> RequestContext:
    params = params or {}
    raw_context = params.get("context") if isinstance(params.get("context"), dict) else {}
    auth = get_request_auth(request)
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-Id") or "mcp"
    return RequestContext(
        request_id=str(raw_context.get("request_id") or request_id),
        session_id=raw_context.get("session_id") or request.headers.get("Mcp-Session-Id"),
        kb_id=raw_context.get("kb_id"),
        kb_key=raw_context.get("kb_key"),
        auth=auth,
    )


def _parse_roles_json(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _matches_explicit_auth_scope(item, auth: AuthContext) -> bool:
    tenant_id = getattr(item, "tenant_id", None)
    org_id = getattr(item, "org_id", None)
    if tenant_id and tenant_id != auth.tenant_id:
        return False
    if org_id and org_id != auth.org_id:
        return False
    return True


async def _list_kbs_payload(auth: AuthContext | None = None) -> dict[str, Any]:
    db = await open_db()
    try:
        items = await list_accessible_kbs(db, auth_context=auth) if auth else await list_accessible_kbs(db)
    finally:
        await db.close()
    if auth:
        items = [item for item in items if _matches_explicit_auth_scope(item, auth)]
    payloads = [item.model_dump() for item in items]
    return {"total": len(payloads), "items": payloads}


async def _resolve_mcp_kb(kb_id: int, auth: AuthContext):
    db = await open_db()
    try:
        try:
            kb = await resolve_kb_scope(db, kb_id=kb_id, auth_context=auth)
        except HTTPException as err:
            if err.status_code == 404:
                raise KeyError(f"Unknown KB resource: {kb_id}") from err
            raise ToolAuthorizationError(str(err.detail)) from err
    finally:
        await db.close()
    if not _matches_explicit_auth_scope(kb, auth):
        raise ToolAuthorizationError("MCP Knowledge Base scope mismatch")
    return kb


async def _kb_stats_payload(kb_id: int, auth: AuthContext) -> dict[str, Any]:
    kb = await _resolve_mcp_kb(kb_id, auth)
    row = await fetch_one(
        """
        SELECT
            COUNT(*) AS total_files,
            COALESCE(SUM(CASE WHEN status = 'ingested' THEN 1 ELSE 0 END), 0) AS ingested_files
        FROM kb_files
        WHERE kb_id = ?
        """,
        (kb.id,),
    )
    where = {"kb_id": int(kb_id)}
    total_vectors = vector_store.count_by_where(where)
    return {
        "scope": "kb",
        "kb_id": kb.id,
        "kb_key": kb.key,
        "kb_name": kb.name,
        "kb_version": kb.kb_version,
        "is_default": kb.is_default,
        "total_files": int(row.get("total_files") or 0),
        "ingested_files": int(row.get("ingested_files") or 0),
        "total_chunks": total_vectors,
        "total_vectors": total_vectors,
        "sources": vector_store.get_sources(where),
    }


async def _kb_sources_payload(kb_id: int, auth: AuthContext) -> dict[str, Any]:
    await _resolve_mcp_kb(kb_id, auth)
    return {
        "kb_id": int(kb_id),
        "items": vector_store.get_source_stats({"kb_id": int(kb_id)}),
    }


async def _kb_source_health_payload(kb_id: int, auth: AuthContext) -> dict[str, Any]:
    kb = await _resolve_mcp_kb(kb_id, auth)
    rows = await fetch_all(
        """
        SELECT
            kf.id AS kb_file_id,
            kf.status AS kb_status,
            kf.chunk_count,
            kf.last_ingest_at,
            kf.stale_detected_at,
            uf.id AS file_id,
            uf.original_name,
            uf.status AS file_status,
            uf.error_message,
            uf.ingested_at,
            uf.pages_or_rows
        FROM kb_files kf
        JOIN uploaded_files uf ON uf.id = kf.file_id
        WHERE kf.kb_id = ?
        ORDER BY
            CASE
                WHEN kf.status = 'failed' OR uf.status = 'failed' THEN 0
                WHEN COALESCE(kf.chunk_count, 0) = 0 THEN 1
                WHEN kf.stale_detected_at IS NOT NULL THEN 2
                ELSE 3
            END,
            uf.original_name ASC
        LIMIT 100
        """,
        (kb.id,),
    )
    items = []
    for row in rows:
        status = "healthy"
        reasons: list[str] = []
        if row.get("kb_status") == "failed" or row.get("file_status") == "failed":
            status = "failed"
            reasons.append("failed_ingest")
        if int(row.get("chunk_count") or 0) == 0:
            status = "needs_attention" if status == "healthy" else status
            reasons.append("zero_chunks")
        if row.get("stale_detected_at"):
            status = "stale" if status == "healthy" else status
            reasons.append("stale_source")
        items.append(
            {
                "kb_file_id": row.get("kb_file_id"),
                "file_id": row.get("file_id"),
                "filename": row.get("original_name"),
                "status": status,
                "kb_status": row.get("kb_status"),
                "file_status": row.get("file_status"),
                "chunk_count": int(row.get("chunk_count") or 0),
                "pages_or_rows": row.get("pages_or_rows"),
                "last_ingest_at": row.get("last_ingest_at") or row.get("ingested_at"),
                "stale_detected_at": row.get("stale_detected_at"),
                "error_message": row.get("error_message"),
                "reasons": reasons,
            }
        )
    return {
        "scope": "kb",
        "kb_id": kb.id,
        "kb_key": kb.key,
        "total": len(items),
        "attention_count": sum(1 for item in items if item["status"] != "healthy"),
        "items": items,
    }


def _ticket_scope_matches(row: dict[str, Any], auth: AuthContext) -> bool:
    if auth.tenant_id and row.get("tenant_id") not in (None, auth.tenant_id):
        return False
    if auth.org_id and row.get("org_id") not in (None, auth.org_id):
        return False
    return True


async def _support_tickets_recent_payload(auth: AuthContext) -> dict[str, Any]:
    where, params = _scope_clause(auth)
    rows = await fetch_all(
        f"""
        SELECT id, ticket_code, issue_type, status, workflow_status, priority,
               assigned_user_id, tenant_id, org_id, kb_id, kb_key,
               intent, sla_due_at, sla_breached_at, created_at, updated_at
        FROM support_tickets
        {where}
        ORDER BY
            CASE priority WHEN 'P0' THEN 0 WHEN 'P1' THEN 1 WHEN 'P2' THEN 2 WHEN 'P3' THEN 3 ELSE 4 END,
            COALESCE(sla_due_at, updated_at) ASC,
            id DESC
        LIMIT 25
        """,
        params,
    )
    return {
        "total": len(rows),
        "items": [
            {
                "id": row.get("id"),
                "ticket_code": row.get("ticket_code"),
                "issue_type": row.get("issue_type"),
                "status": row.get("status"),
                "workflow_status": row.get("workflow_status"),
                "priority": row.get("priority"),
                "assigned_user_id": row.get("assigned_user_id"),
                "kb_id": row.get("kb_id"),
                "kb_key": row.get("kb_key"),
                "intent": row.get("intent"),
                "sla_due_at": row.get("sla_due_at"),
                "sla_breached_at": row.get("sla_breached_at"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            }
            for row in rows
        ],
    }


async def _support_ticket_timeline_payload(ticket_id: int, auth: AuthContext) -> dict[str, Any]:
    row = await fetch_one("SELECT tenant_id, org_id FROM support_tickets WHERE id = ?", (int(ticket_id),))
    if not row:
        raise KeyError(f"Unknown support ticket resource: {ticket_id}")
    if not _ticket_scope_matches(row, auth):
        raise ToolAuthorizationError("MCP support ticket scope mismatch")
    return build_case_timeline(int(ticket_id))


async def _eval_runs_recent_payload(auth: AuthContext) -> dict[str, Any]:
    where, params = _scope_clause(auth)
    rows = await fetch_all(
        f"""
        SELECT id, name, status, source, kb_id, kb_key, period_days, sample_size,
               pass_count, warn_count, fail_count, avg_score, gate_status,
               created_by_user_id, created_at, completed_at
        FROM agent_eval_runs
        {where}
        ORDER BY created_at DESC, id DESC
        LIMIT 20
        """,
        params,
    )
    return {
        "total": len(rows),
        "items": [
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "status": row.get("status"),
                "source": row.get("source"),
                "kb_id": row.get("kb_id"),
                "kb_key": row.get("kb_key"),
                "period_days": row.get("period_days"),
                "sample_size": row.get("sample_size"),
                "pass_count": row.get("pass_count"),
                "warn_count": row.get("warn_count"),
                "fail_count": row.get("fail_count"),
                "avg_score": row.get("avg_score"),
                "gate_status": row.get("gate_status"),
                "created_by_user_id": row.get("created_by_user_id"),
                "created_at": row.get("created_at"),
                "completed_at": row.get("completed_at"),
            }
            for row in rows
        ],
    }


async def _eval_run_detail_payload(run_id: int, auth: AuthContext) -> dict[str, Any]:
    row = await fetch_one("SELECT tenant_id, org_id FROM agent_eval_runs WHERE id = ?", (int(run_id),))
    if not row:
        raise KeyError(f"Unknown eval run resource: {run_id}")
    if auth.tenant_id and row.get("tenant_id") not in (None, auth.tenant_id):
        raise ToolAuthorizationError("MCP eval run scope mismatch")
    if auth.org_id and row.get("org_id") not in (None, auth.org_id):
        raise ToolAuthorizationError("MCP eval run scope mismatch")
    return get_agent_eval_run(int(run_id), limit=50)


def _scope_clause(auth: AuthContext) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    params: list[Any] = []
    if auth.tenant_id:
        clauses.append("(tenant_id IS NULL OR tenant_id = ?)")
        params.append(auth.tenant_id)
    if auth.org_id:
        clauses.append("(org_id IS NULL OR org_id = ?)")
        params.append(auth.org_id)
    return (f"WHERE {' AND '.join(clauses)}" if clauses else "", tuple(params))


async def _audit_recent_payload(auth: AuthContext, limit: int = 20) -> dict[str, Any]:
    where, scope_params = _scope_clause(auth)
    tool_rows = await fetch_all(
        f"""
        SELECT id, tool_call_id, request_id, session_id, user_id, roles_json, channel,
               tenant_id, org_id, kb_id, kb_key, tool_name, args_json, result_summary,
               tool_status, latency_ms, error_message, created_at
        FROM tool_audit_logs
        {where}
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (*scope_params, limit),
    )
    auth_rows = await fetch_all(
        f"""
        SELECT id, request_id, user_id, roles_json, channel,
               tenant_id, org_id, resource_type, resource_id, action,
               decision, reason, created_at
        FROM auth_audit_logs
        {where}
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (*scope_params, limit),
    )
    return {
        "tool_audit": [
            {
                "id": row["id"],
                "tool_call_id": row.get("tool_call_id"),
                "request_id": row.get("request_id"),
                "session_id": row.get("session_id"),
                "user_id": row.get("user_id"),
                "roles": _parse_roles_json(row.get("roles_json")),
                "channel": row.get("channel"),
                "kb_id": row.get("kb_id"),
                "kb_key": row.get("kb_key"),
                "tool_name": row.get("tool_name"),
                "tool_status": row.get("tool_status"),
                "result_summary": row.get("result_summary"),
                "latency_ms": row.get("latency_ms"),
                "error_message": row.get("error_message"),
                "created_at": row.get("created_at"),
            }
            for row in tool_rows
        ],
        "auth_audit": [
            {
                "id": row["id"],
                "request_id": row.get("request_id"),
                "user_id": row.get("user_id"),
                "roles": _parse_roles_json(row.get("roles_json")),
                "channel": row.get("channel"),
                "resource_type": row.get("resource_type"),
                "resource_id": row.get("resource_id"),
                "action": row.get("action"),
                "decision": row.get("decision"),
                "reason": row.get("reason"),
                "created_at": row.get("created_at"),
            }
            for row in auth_rows
        ],
    }


async def _list_resources(auth: AuthContext | None = None) -> list[dict[str, Any]]:
    payload = await _list_kbs_payload(auth)
    resources = [
        _resource("kb://list", "Knowledge Bases", "List Knowledge Bases visible to admin operations."),
        _resource("jobs://recent", "Recent Background Jobs", "Recent background job queue state."),
        _resource("audit://recent", "Recent Audit Logs", "Recent tool and authorization audit entries."),
        _resource("support://tickets/recent", "Recent Support Tickets", "Recent support tickets scoped to the caller."),
        _resource("eval://runs/recent", "Recent Evaluation Runs", "Recent agent evaluation runs scoped to the caller."),
    ]
    for item in payload["items"]:
        kb_id = item["id"]
        label = item.get("key") or str(kb_id)
        resources.append(_resource(f"kb://{kb_id}/stats", f"KB {label} Stats", "KB ingest and vector statistics."))
        resources.append(_resource(f"kb://{kb_id}/sources", f"KB {label} Sources", "KB source distribution by ingested file."))
        resources.append(_resource(f"kb://{kb_id}/source-health", f"KB {label} Source Health", "KB source quality, stale, failed, and zero-chunk status."))
    return resources


def _list_resource_templates() -> list[dict[str, Any]]:
    return [
        _resource_template("kb://{kb_id}/stats", "KB Stats", "KB ingest and vector statistics by KB ID."),
        _resource_template("kb://{kb_id}/sources", "KB Sources", "KB source distribution by KB ID."),
        _resource_template("kb://{kb_id}/source-health", "KB Source Health", "KB source quality, stale, failed, and zero-chunk status by KB ID."),
        _resource_template("support://tickets/{ticket_id}/timeline", "Support Ticket Timeline", "Support ticket timeline and case context by ticket ID."),
        _resource_template("eval://runs/{run_id}", "Evaluation Run Detail", "Agent evaluation run detail and failed cases by run ID."),
    ]


async def _visible_resources(request: Request) -> list[dict[str, Any]]:
    client = mcp_client_context(request)
    resources = await _list_resources(get_request_auth(request))
    return [item for item in resources if _resource_scope_allowed(client, item["uri"])]


def _visible_resource_templates(request: Request) -> list[dict[str, Any]]:
    client = mcp_client_context(request)
    return [
        item
        for item in _list_resource_templates()
        if _resource_scope_allowed(client, item["uriTemplate"])
    ]


async def build_mcp_status() -> dict[str, Any]:
    definitions = tool_registry.list_definitions()
    exposed = _exposed_tool_definitions()
    status_auth = AuthContext(user_id="admin-status", roles=["admin"], channel="admin")
    status_client = McpClientContext(client_id="admin-status", session_id=None, scopes={"mcp:*"})
    security_rows = _tool_security_report(exposed, auth=status_auth, client=status_client)
    effective_exposed = [item for item in security_rows if item["exposed"]]
    blocked_by_policy = [item for item in security_rows if not item["exposed"]]
    manifest_tools = [_tool_to_mcp(item) for item in exposed if any(row["name"] == item.name and row["exposed"] for row in security_rows)]
    resources = await _list_resources(status_auth)
    resource_templates = _list_resource_templates()
    return {
        "enabled": settings.mcp_server_enabled,
        "endpoint_path": "/mcp",
        "protocol_version": settings.mcp_protocol_version,
        "server": {
            "name": settings.mcp_server_name,
            "version": settings.mcp_server_version,
        },
        "security": {
            "require_auth": settings.mcp_require_auth,
            "validate_origin": settings.mcp_validate_origin,
            "allowed_origins": sorted(settings.mcp_allowed_origin_values),
            "require_tool_scopes": settings.mcp_require_tool_scopes,
            "require_resource_scopes": settings.mcp_require_resource_scopes,
            "scope_header": "X-MCP-Scopes",
            "client_id_header": "X-MCP-Client-Id",
            "require_client_token": settings.mcp_require_client_token,
            "client_token_header": settings.mcp_client_token_header,
            "registered_clients": sorted(settings.mcp_client_token_map.keys()),
            "default_tool_quota_per_window": settings.mcp_default_tool_quota_per_window,
            "tool_quota_window_seconds": settings.mcp_tool_quota_window_seconds,
            "tool_quota_rules": [
                {"client_id": client_id, "tool_name": tool_name, "limit": limit}
                for (client_id, tool_name), limit in sorted(settings.mcp_tool_quota_rules.items())
            ],
            "high_risk_allowed_tools": sorted(settings.mcp_high_risk_tool_names),
            "manifest_signature": sign_tool_manifest(manifest_tools),
            "quota_dashboard": list_mcp_quota_dashboard(),
            "recent_denies": list_mcp_recent_denies(limit=20),
        },
        "tools": {
            "registered_count": len(definitions),
            "configured_count": len(exposed),
            "exposed_count": len(effective_exposed),
            "exposed": effective_exposed,
            "blocked_by_policy": blocked_by_policy,
            "hidden": sorted(item.name for item in definitions if not _is_exposed_tool(item.name)),
        },
        "resources": {
            "count": len(resources),
            "items": resources,
            "template_count": len(resource_templates),
            "templates": resource_templates,
        },
        "capabilities": {
            "tools": True,
            "resources": True,
            "resource_templates": True,
            "risk_preview": True,
            "tool_dry_run": True,
            "session_audit": True,
            "quota_dashboard": True,
            "deny_audit": True,
        },
        "sessions": {
            "recent": list_mcp_sessions(limit=20),
        },
    }


async def _read_resource(uri: str, auth: AuthContext) -> dict[str, Any]:
    parsed = urlparse(uri)
    if uri == "kb://list":
        return _json_resource_content(uri, await _list_kbs_payload(auth))
    if uri == "jobs://recent":
        return _json_resource_content(
            uri,
            list_background_jobs(tenant_id=auth.tenant_id, org_id=auth.org_id, limit=50),
        )
    if uri == "audit://recent":
        return _json_resource_content(uri, await _audit_recent_payload(auth))
    if uri == "support://tickets/recent":
        return _json_resource_content(uri, await _support_tickets_recent_payload(auth))
    if uri == "eval://runs/recent":
        return _json_resource_content(uri, await _eval_runs_recent_payload(auth))
    if parsed.scheme == "kb" and parsed.netloc:
        try:
            kb_id = int(parsed.netloc)
        except ValueError as err:
            raise KeyError(f"Invalid KB resource URI: {uri}") from err
        if parsed.path == "/stats":
            return _json_resource_content(uri, await _kb_stats_payload(kb_id, auth))
        if parsed.path == "/sources":
            return _json_resource_content(uri, await _kb_sources_payload(kb_id, auth))
        if parsed.path == "/source-health":
            return _json_resource_content(uri, await _kb_source_health_payload(kb_id, auth))
    if parsed.scheme == "support" and parsed.netloc == "tickets" and parsed.path.endswith("/timeline"):
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 2 and parts[1] == "timeline":
            try:
                ticket_id = int(parts[0])
            except ValueError as err:
                raise KeyError(f"Invalid support ticket resource URI: {uri}") from err
            return _json_resource_content(uri, await _support_ticket_timeline_payload(ticket_id, auth))
    if parsed.scheme == "eval" and parsed.netloc == "runs" and parsed.path:
        run_id_text = parsed.path.strip("/")
        if run_id_text and run_id_text != "recent":
            try:
                run_id = int(run_id_text)
            except ValueError as err:
                raise KeyError(f"Invalid eval run resource URI: {uri}") from err
            return _json_resource_content(uri, await _eval_run_detail_payload(run_id, auth))
    raise KeyError(f"Unknown MCP resource: {uri}")


async def _handle_method(message: dict[str, Any], request: Request) -> dict[str, Any] | None:
    request_id = message.get("id")
    is_notification = "id" not in message
    method = message.get("method")
    params = message.get("params", {})
    if params is None:
        params = {}

    if (
        message.get("jsonrpc") != JSONRPC_VERSION
        or not isinstance(method, str)
        or (not is_notification and (isinstance(request_id, bool) or not isinstance(request_id, (str, int))))
    ):
        return None if is_notification else _error(request_id, INVALID_REQUEST, "Invalid JSON-RPC request")
    if params is not None and not isinstance(params, dict):
        return None if is_notification else _error(request_id, INVALID_PARAMS, "params must be an object")

    try:
        auth_for_session = get_request_auth(request)
        client_for_session = mcp_client_context(request)
        record_mcp_session_activity(
            auth=auth_for_session,
            client=client_for_session,
            method=method,
            decision="seen",
            reason="method_received",
        )

        with trace_span(
            "mcp.method",
            {
                "rpc.system": "jsonrpc",
                "rpc.method": method,
                "mcp.client_id": client_for_session.client_id,
                "mcp.session_id": client_for_session.session_id,
                "app.request_id": getattr(request.state, "request_id", None),
            },
        ) as span:
            if method == "initialize":
                requested_protocol_version = str(params.get("protocolVersion") or "")
                protocol_version = (
                    requested_protocol_version
                    if requested_protocol_version == settings.mcp_protocol_version
                    else settings.mcp_protocol_version
                )
                result = {
                    "protocolVersion": protocol_version,
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {
                        "name": settings.mcp_server_name,
                        "version": settings.mcp_server_version,
                    },
                }
                span.set_attribute("mcp.protocol_version", protocol_version)
                return None if is_notification else _success(request_id, result)

            if method == "notifications/initialized":
                return None

            if method == "ping":
                return None if is_notification else _success(request_id, {})

            if method == "tools/list":
                tools = [_tool_to_mcp(item) for item in _visible_tool_definitions(request)]
                result = {"tools": tools, "manifest": sign_tool_manifest(tools)}
                span.set_attribute("mcp.tools.count", len(tools))
                return None if is_notification else _success(request_id, result)

            if method == "tools/manifest":
                tools = [_tool_to_mcp(item) for item in _visible_tool_definitions(request)]
                result = {"tools": tools, "manifest": sign_tool_manifest(tools)}
                span.set_attribute("mcp.tools.count", len(tools))
                return None if is_notification else _success(request_id, result)

            if method == "tools/riskPreview":
                name = params.get("name")
                if not isinstance(name, str) or not name.strip():
                    return None if is_notification else _error(request_id, INVALID_PARAMS, "tools/riskPreview requires params.name")
                span.set_attribute("mcp.tool.name", name)
                if not _is_exposed_tool(name):
                    return None if is_notification else _error(request_id, TOOL_ERROR, "Tool is not exposed through MCP", {"reason": "tool_not_exposed"})
                definition = tool_registry.get(name).summary()
                auth = get_request_auth(request)
                client = mcp_client_context(request)
                result = preview_tool_risk(definition, auth=auth, client=client)
                span.set_attribute("mcp.tool.allowed", bool(result.get("allowed")))
                span.set_attribute("mcp.tool.risk_level", result.get("risk_level"))
                return None if is_notification else _success(request_id, result)

            if method == "tools/dryRun":
                name = params.get("name")
                arguments = params.get("arguments") or {}
                if not isinstance(name, str) or not name.strip():
                    return None if is_notification else _error(request_id, INVALID_PARAMS, "tools/dryRun requires params.name")
                if not isinstance(arguments, dict):
                    return None if is_notification else _error(request_id, INVALID_PARAMS, "params.arguments must be an object")
                span.set_attribute("mcp.tool.name", name)
                if not _is_exposed_tool(name):
                    return None if is_notification else _error(
                        request_id,
                        TOOL_ERROR,
                        "Tool is not exposed through MCP",
                        {"reason": "tool_not_exposed"},
                    )
                definition = tool_registry.get(name).summary()
                auth = get_request_auth(request)
                client = mcp_client_context(request)
                decision = decide_tool_exposure(definition, auth=auth, client=client)
                quota = preview_tool_risk(definition, auth=auth, client=client)["quota"]
                dry_run_result: dict[str, Any] = {
                    "tool_name": name,
                    "allowed": decision.allowed,
                    "decision_reason": decision.reason,
                    "would_execute": False,
                    "valid": False,
                    "required_scopes": sorted(decision.required_scopes),
                    "granted_scopes": sorted(decision.granted_scopes),
                    "quota": quota,
                    "risk_level": definition.auth_policy.get("risk_level"),
                    "scope": definition.auth_policy.get("scope"),
                }
                audit_decision = "deny"
                audit_reason = decision.reason
                if decision.allowed:
                    validation = tool_registry.dry_run(
                        name,
                        arguments,
                        request_context=_request_context(request, params),
                    )
                    dry_run_result.update(validation.model_dump())
                    audit_decision = "allow" if validation.valid else "deny"
                    audit_reason = "tool_dry_run_valid" if validation.valid else "tool_dry_run_invalid"
                    if validation.valid and quota.get("enabled") and int(quota.get("remaining") or 0) <= 0:
                        dry_run_result["allowed"] = False
                        dry_run_result["would_execute"] = False
                        dry_run_result["decision_reason"] = "tool_quota_exceeded"
                        audit_decision = "deny"
                        audit_reason = "tool_quota_exceeded"
                span.set_attribute("mcp.decision", audit_decision)
                span.set_attribute("mcp.reason", audit_reason)
                log_mcp_audit(
                    request_id=str(request_id) if request_id is not None else None,
                    auth=auth,
                    client=client,
                    method=method,
                    tool_name=name,
                    required_scopes=decision.required_scopes,
                    risk_level=definition.auth_policy.get("risk_level"),
                    tool_scope=definition.auth_policy.get("scope"),
                    decision=audit_decision,
                    reason=audit_reason,
                )
                return None if is_notification else _success(request_id, dry_run_result)

            if method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments") or {}
                if not isinstance(name, str) or not name.strip():
                    return None if is_notification else _error(request_id, INVALID_PARAMS, "tools/call requires params.name")
                span.set_attribute("mcp.tool.name", name)
                if not _is_exposed_tool(name):
                    span.set_attribute("mcp.decision", "deny")
                    span.set_attribute("mcp.reason", "tool_not_exposed")
                    return None if is_notification else _error(request_id, TOOL_ERROR, "Tool is not exposed through MCP", name)
                if not isinstance(arguments, dict):
                    return None if is_notification else _error(request_id, INVALID_PARAMS, "params.arguments must be an object")
                definition = tool_registry.get(name).summary()
                auth = get_request_auth(request)
                client = mcp_client_context(request)
                decision = decide_tool_exposure(definition, auth=auth, client=client)
                span.set_attribute("mcp.tool.risk_level", definition.auth_policy.get("risk_level"))
                span.set_attribute("mcp.tool.scope", definition.auth_policy.get("scope"))
                if not decision.allowed:
                    span.set_attribute("mcp.decision", "deny")
                    span.set_attribute("mcp.reason", decision.reason)
                    record_mcp_session_activity(
                        auth=auth,
                        client=client,
                        method=method,
                        decision="deny",
                        reason=decision.reason,
                        is_tool_call=True,
                        count_request=False,
                    )
                    log_mcp_audit(
                        request_id=str(request_id) if request_id is not None else None,
                        auth=auth,
                        client=client,
                        method=method,
                        tool_name=name,
                        required_scopes=decision.required_scopes,
                        risk_level=definition.auth_policy.get("risk_level"),
                        tool_scope=definition.auth_policy.get("scope"),
                        decision="deny",
                        reason=decision.reason,
                    )
                    return None if is_notification else _error(
                        request_id,
                        TOOL_ERROR,
                        "Tool blocked by MCP security policy",
                        {"reason": decision.reason, "required_scopes": sorted(decision.required_scopes)},
                    )

                quota = consume_mcp_tool_quota(client, name)
                if not quota.allowed:
                    span.set_attribute("mcp.decision", "deny")
                    span.set_attribute("mcp.reason", quota.reason)
                    span.set_attribute("mcp.quota.limit", quota.limit)
                    span.set_attribute("mcp.quota.remaining", quota.remaining)
                    record_mcp_session_activity(
                        auth=auth,
                        client=client,
                        method=method,
                        decision="deny",
                        reason=quota.reason,
                        is_tool_call=True,
                        count_request=False,
                    )
                    log_mcp_audit(
                        request_id=str(request_id) if request_id is not None else None,
                        auth=auth,
                        client=client,
                        method=method,
                        tool_name=name,
                        required_scopes=decision.required_scopes,
                        risk_level=definition.auth_policy.get("risk_level"),
                        tool_scope=definition.auth_policy.get("scope"),
                        decision="deny",
                        reason=quota.reason,
                    )
                    return None if is_notification else _error(
                        request_id,
                        TOOL_ERROR,
                        "Tool blocked by MCP quota policy",
                        {
                            "reason": quota.reason,
                            "quota": {
                                "limit": quota.limit,
                                "remaining": quota.remaining,
                                "reset_after_seconds": quota.reset_after_seconds,
                            },
                        },
                    )

                execution = await tool_registry.execute(
                    name,
                    arguments,
                    request_context=_request_context(request, params),
                )
                span.set_attribute("mcp.decision", "allow")
                span.set_attribute("mcp.tool.latency_ms", execution.latency_ms)
                record_mcp_session_activity(
                    auth=auth,
                    client=client,
                    method=method,
                    decision="allow",
                    reason="tool_executed",
                    is_tool_call=True,
                    count_request=False,
                )
                log_mcp_audit(
                    request_id=str(request_id) if request_id is not None else None,
                    auth=auth,
                    client=client,
                    method=method,
                    tool_name=name,
                    required_scopes=decision.required_scopes,
                    risk_level=definition.auth_policy.get("risk_level"),
                    tool_scope=definition.auth_policy.get("scope"),
                    decision="allow",
                    reason="tool_executed",
                )
                output_text = json.dumps(execution.output, ensure_ascii=False)
                result = {
                    "content": [{"type": "text", "text": output_text}],
                    "isError": False,
                    "structuredContent": execution.output,
                }
                return None if is_notification else _success(request_id, result)

            if method == "resources/list":
                _require_resource_admin(request, method=method, request_id=request_id, enforce_scope=False)
                result = {"resources": await _visible_resources(request)}
                span.set_attribute("mcp.resources.count", len(result["resources"]))
                return None if is_notification else _success(request_id, result)

            if method == "resources/templates/list":
                _require_resource_admin(request, method=method, request_id=request_id, enforce_scope=False)
                result = {"resourceTemplates": _visible_resource_templates(request)}
                span.set_attribute("mcp.resource_templates.count", len(result["resourceTemplates"]))
                return None if is_notification else _success(request_id, result)

            if method == "resources/read":
                uri = params.get("uri")
                if not isinstance(uri, str) or not uri.strip():
                    return None if is_notification else _error(request_id, INVALID_PARAMS, "resources/read requires params.uri")
                uri = uri.strip()
                span.set_attribute("mcp.resource.uri", uri)
                _require_resource_admin(request, method=method, request_id=request_id, resource_uri=uri)
                result = {"contents": [await _read_resource(uri, get_request_auth(request))]}
                return None if is_notification else _success(request_id, result)

            return None if is_notification else _error(request_id, METHOD_NOT_FOUND, f"Unknown method: {method}")

    except ToolValidationError as err:
        return None if is_notification else _error(request_id, INVALID_PARAMS, str(err))
    except ToolAuthorizationError as err:
        return None if is_notification else _error(request_id, TOOL_ERROR, "Tool authorization denied", str(err))
    except ToolExecutionError as err:
        return None if is_notification else _error(request_id, TOOL_ERROR, "Tool execution failed", str(err))
    except KeyError as err:
        return None if is_notification else _error(request_id, INVALID_PARAMS, str(err))
    except HTTPException:
        raise
    except Exception as err:
        logger.exception("MCP method failed: %s", method)
        return None if is_notification else _error(request_id, INTERNAL_ERROR, "Internal MCP server error", str(err))


async def handle_mcp_request(request: Request) -> Response:
    if not settings.mcp_server_enabled:
        raise HTTPException(status_code=404, detail="MCP server is disabled")
    _validate_mcp_origin(request)
    _validate_mcp_auth(request)
    _validate_mcp_protocol_header(request)
    validate_mcp_client_token(request)

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(_error(None, PARSE_ERROR, "Invalid JSON"), status_code=400)

    if isinstance(payload, list):
        if not payload:
            return JSONResponse(_error(None, INVALID_REQUEST, "Batch request must not be empty"), status_code=400)
        responses = []
        for item in payload:
            if not isinstance(item, dict):
                responses.append(_error(None, INVALID_REQUEST, "Invalid JSON-RPC request"))
                continue
            response = await _handle_method(item, request)
            if response is not None:
                responses.append(response)
        if not responses:
            return Response(status_code=202)
        return JSONResponse(responses)

    if not isinstance(payload, dict):
        return JSONResponse(_error(None, INVALID_REQUEST, "Invalid JSON-RPC request"), status_code=400)

    response = await _handle_method(payload, request)
    if response is None:
        return Response(status_code=202)
    return JSONResponse(response)
