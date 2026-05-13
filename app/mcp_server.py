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
from app.config import settings
from app.database import fetch_all, fetch_one
from app.kb_service import KB_SELECT, open_db, row_to_kb_summary
from app.mcp_security import (
    McpClientContext,
    decide_tool_exposure,
    log_mcp_audit,
    mcp_client_context,
    sign_tool_manifest,
    tool_required_scopes,
)
from app.models import AuthContext, RequestContext
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
            allowed = decision.allowed
            reason = decision.reason
        else:
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


def _require_resource_admin(request: Request, *, method: str, request_id: Any, resource_uri: str | None = None) -> None:
    auth = get_request_auth(request)
    client = mcp_client_context(request)
    if not can_manage_kb(auth):
        log_mcp_audit(
            request_id=str(request_id) if request_id is not None else None,
            auth=auth,
            client=client,
            method=method,
            resource_uri=resource_uri,
            required_scopes={"resource:*"},
            decision="deny",
            reason="admin_role_required_for_resources",
        )
        raise ToolAuthorizationError("Admin role required for MCP resources")
    log_mcp_audit(
        request_id=str(request_id) if request_id is not None else None,
        auth=auth,
        client=client,
        method=method,
        resource_uri=resource_uri,
        required_scopes={"resource:*"},
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


async def _list_kbs_payload() -> dict[str, Any]:
    db = await open_db()
    try:
        cursor = await db.execute(
            KB_SELECT
            + """
            GROUP BY
                kb.id, kb.key, kb.name, kb.description, kb.status, kb.access_level, kb.tenant_id, kb.org_id,
                kb.is_default, kb.kb_version, kb.created_at, kb.updated_at
            ORDER BY kb.is_default DESC, kb.created_at ASC
            """
        )
        rows = await cursor.fetchall()
        items = [row_to_kb_summary(dict(row)).model_dump() for row in rows]
    finally:
        await db.close()
    return {"total": len(items), "items": items}


async def _kb_stats_payload(kb_id: int) -> dict[str, Any]:
    row = await fetch_one(
        """
        SELECT
            kb.id, kb.key, kb.name, kb.kb_version, kb.is_default,
            COUNT(kf.id) AS total_files,
            COALESCE(SUM(CASE WHEN kf.status = 'ingested' THEN 1 ELSE 0 END), 0) AS ingested_files
        FROM knowledge_bases kb
        LEFT JOIN kb_files kf ON kf.kb_id = kb.id
        WHERE kb.id = ?
        GROUP BY kb.id, kb.key, kb.name, kb.kb_version, kb.is_default
        """,
        (kb_id,),
    )
    if not row:
        raise KeyError(f"Unknown KB resource: {kb_id}")
    where = {"kb_id": int(kb_id)}
    total_vectors = vector_store.count_by_where(where)
    return {
        "scope": "kb",
        "kb_id": int(row["id"]),
        "kb_key": row["key"],
        "kb_name": row["name"],
        "kb_version": row["kb_version"],
        "is_default": bool(row["is_default"]),
        "total_files": int(row.get("total_files") or 0),
        "ingested_files": int(row.get("ingested_files") or 0),
        "total_chunks": total_vectors,
        "total_vectors": total_vectors,
        "sources": vector_store.get_sources(where),
    }


async def _kb_sources_payload(kb_id: int) -> dict[str, Any]:
    exists = await fetch_one("SELECT id FROM knowledge_bases WHERE id = ?", (kb_id,))
    if not exists:
        raise KeyError(f"Unknown KB resource: {kb_id}")
    return {
        "kb_id": int(kb_id),
        "items": vector_store.get_source_stats({"kb_id": int(kb_id)}),
    }


async def _audit_recent_payload(limit: int = 20) -> dict[str, Any]:
    tool_rows = await fetch_all(
        """
        SELECT id, tool_call_id, request_id, session_id, user_id, roles_json, channel,
               tenant_id, org_id, kb_id, kb_key, tool_name, args_json, result_summary,
               tool_status, latency_ms, error_message, created_at
        FROM tool_audit_logs
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )
    auth_rows = await fetch_all(
        """
        SELECT id, request_id, user_id, roles_json, channel,
               tenant_id, org_id, resource_type, resource_id, action,
               decision, reason, created_at
        FROM auth_audit_logs
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
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


async def _list_resources() -> list[dict[str, Any]]:
    payload = await _list_kbs_payload()
    resources = [
        _resource("kb://list", "Knowledge Bases", "List Knowledge Bases visible to admin operations."),
        _resource("jobs://recent", "Recent Background Jobs", "Recent background job queue state."),
        _resource("audit://recent", "Recent Audit Logs", "Recent tool and authorization audit entries."),
    ]
    for item in payload["items"]:
        kb_id = item["id"]
        label = item.get("key") or str(kb_id)
        resources.append(_resource(f"kb://{kb_id}/stats", f"KB {label} Stats", "KB ingest and vector statistics."))
        resources.append(_resource(f"kb://{kb_id}/sources", f"KB {label} Sources", "KB source distribution by ingested file."))
    return resources


def _list_resource_templates() -> list[dict[str, Any]]:
    return [
        _resource_template("kb://{kb_id}/stats", "KB Stats", "KB ingest and vector statistics by KB ID."),
        _resource_template("kb://{kb_id}/sources", "KB Sources", "KB source distribution by KB ID."),
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
    resources = await _list_resources()
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
            "scope_header": "X-MCP-Scopes",
            "client_id_header": "X-MCP-Client-Id",
            "high_risk_allowed_tools": sorted(settings.mcp_high_risk_tool_names),
            "manifest_signature": sign_tool_manifest(manifest_tools),
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
        },
    }


async def _read_resource(uri: str) -> dict[str, Any]:
    parsed = urlparse(uri)
    if uri == "kb://list":
        return _json_resource_content(uri, await _list_kbs_payload())
    if uri == "jobs://recent":
        return _json_resource_content(uri, list_background_jobs(limit=50))
    if uri == "audit://recent":
        return _json_resource_content(uri, await _audit_recent_payload())
    if parsed.scheme == "kb" and parsed.netloc:
        try:
            kb_id = int(parsed.netloc)
        except ValueError as err:
            raise KeyError(f"Invalid KB resource URI: {uri}") from err
        if parsed.path == "/stats":
            return _json_resource_content(uri, await _kb_stats_payload(kb_id))
        if parsed.path == "/sources":
            return _json_resource_content(uri, await _kb_sources_payload(kb_id))
    raise KeyError(f"Unknown MCP resource: {uri}")


async def _handle_method(message: dict[str, Any], request: Request) -> dict[str, Any] | None:
    request_id = message.get("id")
    is_notification = "id" not in message
    method = message.get("method")
    params = message.get("params") or {}

    if message.get("jsonrpc") != JSONRPC_VERSION or not isinstance(method, str):
        return None if is_notification else _error(request_id, INVALID_REQUEST, "Invalid JSON-RPC request")
    if params is not None and not isinstance(params, dict):
        return None if is_notification else _error(request_id, INVALID_PARAMS, "params must be an object")

    try:
        if method == "initialize":
            protocol_version = str(params.get("protocolVersion") or settings.mcp_protocol_version)
            result = {
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {
                    "name": settings.mcp_server_name,
                    "version": settings.mcp_server_version,
                },
            }
            return None if is_notification else _success(request_id, result)

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return None if is_notification else _success(request_id, {})

        if method == "tools/list":
            tools = [_tool_to_mcp(item) for item in _visible_tool_definitions(request)]
            result = {"tools": tools, "manifest": sign_tool_manifest(tools)}
            return None if is_notification else _success(request_id, result)

        if method == "tools/manifest":
            tools = [_tool_to_mcp(item) for item in _visible_tool_definitions(request)]
            result = {"tools": tools, "manifest": sign_tool_manifest(tools)}
            return None if is_notification else _success(request_id, result)

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str) or not name.strip():
                return None if is_notification else _error(request_id, INVALID_PARAMS, "tools/call requires params.name")
            if not _is_exposed_tool(name):
                return None if is_notification else _error(request_id, TOOL_ERROR, "Tool is not exposed through MCP", name)
            if not isinstance(arguments, dict):
                return None if is_notification else _error(request_id, INVALID_PARAMS, "params.arguments must be an object")
            definition = tool_registry.get(name).summary()
            auth = get_request_auth(request)
            client = mcp_client_context(request)
            decision = decide_tool_exposure(definition, auth=auth, client=client)
            if not decision.allowed:
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
                return None if is_notification else _error(request_id, TOOL_ERROR, "Tool blocked by MCP security policy", decision.reason)

            execution = await tool_registry.execute(
                name,
                arguments,
                request_context=_request_context(request, params),
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
            _require_resource_admin(request, method=method, request_id=request_id)
            result = {"resources": await _list_resources()}
            return None if is_notification else _success(request_id, result)

        if method == "resources/templates/list":
            _require_resource_admin(request, method=method, request_id=request_id)
            result = {"resourceTemplates": _list_resource_templates()}
            return None if is_notification else _success(request_id, result)

        if method == "resources/read":
            uri = params.get("uri")
            if not isinstance(uri, str) or not uri.strip():
                return None if is_notification else _error(request_id, INVALID_PARAMS, "resources/read requires params.uri")
            uri = uri.strip()
            _require_resource_admin(request, method=method, request_id=request_id, resource_uri=uri)
            result = {"contents": [await _read_resource(uri)]}
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
