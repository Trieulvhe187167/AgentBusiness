from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient

from app.background_jobs import enqueue_background_job
from app.config import settings
from app.database import execute_sync, fetch_one_sync, utcnow_iso
from app.kb import create_knowledge_base
from app.models import KnowledgeBaseCreate, RequestContext
from tests.conftest import admin_headers, auth_headers, isolated_client, run


def _rpc(
    client: TestClient,
    method: str,
    params: dict | None = None,
    *,
    request_id: int = 1,
    headers: dict | None = None,
):
    return client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}},
        headers={
            **admin_headers(),
            "X-MCP-Client-Id": "pytest-mcp-client",
            "Mcp-Session-Id": "mcp-session-1",
            "X-MCP-Scopes": "mcp:*",
            **(headers or {}),
        },
    )


def test_mcp_initialize_and_ping(isolated_client: TestClient):
    initialize = _rpc(
        isolated_client,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "1.0"},
        },
    )
    assert initialize.status_code == 200, initialize.text
    payload = initialize.json()
    assert payload["jsonrpc"] == "2.0"
    assert payload["result"]["protocolVersion"] == "2025-06-18"
    assert payload["result"]["capabilities"]["tools"] == {}
    assert payload["result"]["capabilities"]["resources"] == {}
    assert payload["result"]["serverInfo"]["name"]

    ping = _rpc(isolated_client, "ping", request_id=2)
    assert ping.status_code == 200, ping.text
    assert ping.json()["result"] == {}


def test_mcp_initialize_negotiates_server_version_for_unsupported_client(isolated_client: TestClient):
    initialize = _rpc(
        isolated_client,
        "initialize",
        {
            "protocolVersion": "2099-01-01",
            "capabilities": {},
            "clientInfo": {"name": "future-client", "version": "1.0"},
        },
    )

    assert initialize.status_code == 200, initialize.text
    assert initialize.json()["result"]["protocolVersion"] == settings.mcp_protocol_version


def test_mcp_protocol_version_header_rejects_unsupported_version(isolated_client: TestClient):
    blocked = _rpc(isolated_client, "ping", headers={"MCP-Protocol-Version": "2099-01-01"})
    allowed = _rpc(isolated_client, "ping", headers={"MCP-Protocol-Version": settings.mcp_protocol_version})

    assert blocked.status_code == 400, blocked.text
    assert allowed.status_code == 200, allowed.text


def test_mcp_json_rpc_batch_and_request_validation(isolated_client: TestClient):
    headers = {
        **admin_headers(),
        "X-MCP-Client-Id": "pytest-mcp-client",
        "X-MCP-Scopes": "mcp:*",
    }
    batch = isolated_client.post(
        "/mcp",
        json=[
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ],
        headers=headers,
    )
    null_id = isolated_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": None, "method": "ping"},
        headers=headers,
    )
    invalid_params = isolated_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "ping", "params": []},
        headers=headers,
    )

    assert batch.status_code == 200, batch.text
    assert batch.json() == [{"jsonrpc": "2.0", "id": 1, "result": {}}]
    assert null_id.json()["error"]["code"] == -32600
    assert invalid_params.json()["error"]["code"] == -32602


def test_mcp_tools_list_maps_internal_registry(isolated_client: TestClient):
    response = _rpc(isolated_client, "tools/list")
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    tools = result["tools"]
    names = {item["name"] for item in tools}
    assert "search_kb" in names
    assert "list_customer_tickets" in names
    assert "list_kbs" not in names
    assert "delete_google_drive_source" not in names
    assert "send_email_reply" not in names
    assert result["manifest"]["algorithm"] == "HMAC-SHA256"
    assert result["manifest"]["signature"]

    search_kb = next(item for item in tools if item["name"] == "search_kb")
    assert search_kb["inputSchema"]["type"] == "object"
    assert search_kb["annotations"]["scope"] == "kb"
    assert "scope:kb" in search_kb["annotations"]["requiredScopes"]


def test_mcp_tools_list_minimizes_exposure_to_granted_tool_scope(isolated_client: TestClient):
    response = _rpc(isolated_client, "tools/list", headers={"X-MCP-Scopes": "tool:search_kb"})

    assert response.status_code == 200, response.text
    assert {tool["name"] for tool in response.json()["result"]["tools"]} == {"search_kb"}


def test_mcp_tools_call_executes_registry_tool(isolated_client: TestClient):
    response = _rpc(
        isolated_client,
        "tools/call",
        {"name": "list_customer_tickets", "arguments": {}, "context": {"session_id": "mcp-test"}},
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"
    assert result["structuredContent"]["total"] >= 0

    audit = fetch_one_sync(
        """
        SELECT mcp_client_id, mcp_session_id, tool_name, decision, granted_scopes_json
        FROM mcp_audit_logs
        WHERE tool_name = 'list_customer_tickets'
        ORDER BY id DESC
        LIMIT 1
        """
    )
    assert audit
    assert audit["mcp_client_id"] == "pytest-mcp-client"
    assert audit["mcp_session_id"] == "mcp-session-1"
    assert audit["decision"] == "allow"


def test_mcp_notification_returns_accepted(isolated_client: TestClient):
    response = isolated_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers=admin_headers(),
    )
    assert response.status_code == 202


def test_mcp_invalid_method_returns_json_rpc_error(isolated_client: TestClient):
    response = _rpc(isolated_client, "missing/method")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["error"]["code"] == -32601


def test_mcp_requires_authenticated_caller(isolated_client: TestClient):
    response = isolated_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    assert response.status_code == 401


def test_mcp_blocks_non_exposed_tool_call(isolated_client: TestClient):
    response = _rpc(
        isolated_client,
        "tools/call",
        {"name": "delete_google_drive_source", "arguments": {"source_id": 1, "mode": "unlink"}},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["error"]["code"] == -32000
    assert payload["error"]["message"] == "Tool is not exposed through MCP"


def test_mcp_denies_high_risk_tools_by_default(isolated_client: TestClient):
    response = _rpc(
        isolated_client,
        "tools/call",
        {"name": "list_kbs", "arguments": {}},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["error"]["code"] == -32000
    assert payload["error"]["message"] == "Tool blocked by MCP security policy"
    assert payload["error"]["data"]["reason"] == "high_risk_denied_by_default"


def test_mcp_requires_tool_scopes(isolated_client: TestClient):
    response = _rpc(
        isolated_client,
        "tools/call",
        {"name": "search_kb", "arguments": {"query": "shipping"}},
        headers={"X-MCP-Scopes": "scope:support"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["error"]["code"] == -32000
    assert payload["error"]["data"]["reason"] == "scope_not_granted"
    assert "scope:kb" in payload["error"]["data"]["required_scopes"]


def test_mcp_requires_registered_client_token_when_enabled(isolated_client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "mcp_require_client_token", True)
    monkeypatch.setattr(settings, "mcp_client_tokens", "pytest-mcp-client:secret-token")

    missing = _rpc(isolated_client, "ping")
    assert missing.status_code == 401, missing.text

    allowed = _rpc(isolated_client, "ping", headers={"X-MCP-Client-Token": "secret-token"})
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["result"] == {}


def test_mcp_accepts_hashed_registered_client_token(isolated_client: TestClient, monkeypatch):
    secret = "hashed-secret-token"
    monkeypatch.setattr(settings, "mcp_require_client_token", True)
    monkeypatch.setattr(
        settings,
        "mcp_client_tokens",
        f"pytest-mcp-client:sha256:{hashlib.sha256(secret.encode('utf-8')).hexdigest()}",
    )

    allowed = _rpc(isolated_client, "ping", headers={"X-MCP-Client-Token": secret})

    assert allowed.status_code == 200, allowed.text


def test_mcp_tool_quota_blocks_after_client_limit(isolated_client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "mcp_tool_quotas", "pytest-mcp-client:list_customer_tickets:1")

    first = _rpc(
        isolated_client,
        "tools/call",
        {"name": "list_customer_tickets", "arguments": {}, "context": {"session_id": "mcp-quota"}},
    )
    blocked = _rpc(
        isolated_client,
        "tools/call",
        {"name": "list_customer_tickets", "arguments": {}, "context": {"session_id": "mcp-quota"}},
        request_id=2,
    )

    assert first.status_code == 200, first.text
    assert first.json()["result"]["isError"] is False
    assert blocked.status_code == 200, blocked.text
    payload = blocked.json()
    assert payload["error"]["message"] == "Tool blocked by MCP quota policy"
    assert payload["error"]["data"]["reason"] == "tool_quota_exceeded"
    assert payload["error"]["data"]["quota"]["limit"] == 1


def test_mcp_tool_quota_isolated_by_client(isolated_client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "mcp_tool_quotas", "*:list_customer_tickets:1")
    params = {"name": "list_customer_tickets", "arguments": {}}

    first_a = _rpc(isolated_client, "tools/call", params, headers={"X-MCP-Client-Id": "client-a"})
    blocked_a = _rpc(isolated_client, "tools/call", params, request_id=2, headers={"X-MCP-Client-Id": "client-a"})
    first_b = _rpc(isolated_client, "tools/call", params, request_id=3, headers={"X-MCP-Client-Id": "client-b"})

    assert first_a.json()["result"]["isError"] is False
    assert blocked_a.json()["error"]["data"]["reason"] == "tool_quota_exceeded"
    assert first_b.json()["result"]["isError"] is False


def test_mcp_risk_preview_reports_policy_and_quota(isolated_client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "mcp_tool_quotas", "pytest-mcp-client:search_kb:5")
    response = _rpc(isolated_client, "tools/riskPreview", {"name": "search_kb"})
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["tool_name"] == "search_kb"
    assert result["risk_level"] == "low"
    assert "scope:kb" in result["required_scopes"]
    assert result["quota"]["enabled"] is True
    assert result["quota"]["limit"] == 5


def test_mcp_origin_validation(isolated_client: TestClient, monkeypatch):
    blocked = isolated_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}},
        headers={**admin_headers(), "Origin": "https://evil.example"},
    )
    assert blocked.status_code == 403

    monkeypatch.setattr(settings, "mcp_allowed_origins", "https://client.example")
    allowed = isolated_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}},
        headers={**admin_headers(), "Origin": "https://client.example"},
    )
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["result"] == {}


def test_mcp_resources_list_and_read_kb_resources(isolated_client: TestClient):
    listed = _rpc(isolated_client, "resources/list")
    assert listed.status_code == 200, listed.text
    resources = listed.json()["result"]["resources"]
    uris = {item["uri"] for item in resources}
    assert "kb://list" in uris
    assert "jobs://recent" in uris
    assert "audit://recent" in uris

    kb_list = _rpc(isolated_client, "resources/read", {"uri": "kb://list"}, request_id=2)
    assert kb_list.status_code == 200, kb_list.text
    kb_payload = kb_list.json()["result"]["contents"][0]
    assert kb_payload["mimeType"] == "application/json"
    assert '"key": "default"' in kb_payload["text"]

    default_id = kb_list.json()["result"]["contents"][0]["text"]
    assert default_id
    stats = _rpc(isolated_client, "resources/read", {"uri": "kb://1/stats"}, request_id=3)
    assert stats.status_code == 200, stats.text
    assert '"scope": "kb"' in stats.json()["result"]["contents"][0]["text"]


def test_mcp_resource_scope_minimizes_discovery_and_reads(isolated_client: TestClient):
    listed = _rpc(isolated_client, "resources/list", headers={"X-MCP-Scopes": "resource:kb"})
    blocked = _rpc(
        isolated_client,
        "resources/read",
        {"uri": "audit://recent"},
        request_id=2,
        headers={"X-MCP-Scopes": "resource:kb"},
    )

    assert listed.status_code == 200, listed.text
    uris = {item["uri"] for item in listed.json()["result"]["resources"]}
    assert "kb://list" in uris
    assert "jobs://recent" not in uris
    assert "audit://recent" not in uris
    assert blocked.json()["error"]["message"] == "Tool authorization denied"


def test_mcp_resource_templates_list(isolated_client: TestClient):
    response = _rpc(isolated_client, "resources/templates/list")
    assert response.status_code == 200, response.text
    templates = response.json()["result"]["resourceTemplates"]
    uri_templates = {item["uriTemplate"] for item in templates}
    assert "kb://{kb_id}/stats" in uri_templates
    assert "kb://{kb_id}/sources" in uri_templates
    assert all(item["mimeType"] == "application/json" for item in templates)


def test_mcp_resources_read_jobs_and_audit(isolated_client: TestClient):
    jobs = _rpc(isolated_client, "resources/read", {"uri": "jobs://recent"})
    assert jobs.status_code == 200, jobs.text
    assert '"items"' in jobs.json()["result"]["contents"][0]["text"]

    audit = _rpc(isolated_client, "resources/read", {"uri": "audit://recent"}, request_id=2)
    assert audit.status_code == 200, audit.text
    text = audit.json()["result"]["contents"][0]["text"]
    assert '"tool_audit"' in text
    assert '"auth_audit"' in text


def test_mcp_resources_require_admin_role(isolated_client: TestClient):
    response = isolated_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "resources/list", "params": {}},
        headers=auth_headers(user_id="user-1", roles=["staff"], channel="web"),
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["error"]["code"] == -32000
    assert payload["error"]["message"] == "Tool authorization denied"


def test_mcp_unknown_resource_returns_json_rpc_error(isolated_client: TestClient):
    response = _rpc(isolated_client, "resources/read", {"uri": "kb://999999/stats"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["error"]["code"] == -32602


def test_mcp_resources_and_tools_enforce_tenant_isolation(isolated_client: TestClient):
    tenant_a = run(
        create_knowledge_base(
            KnowledgeBaseCreate(
                name="Tenant A KB",
                key="tenant-a-mcp",
                access_level="internal",
                tenant_id="tenant-a",
                org_id="org-a",
            )
        )
    )
    tenant_b = run(
        create_knowledge_base(
            KnowledgeBaseCreate(
                name="Tenant B KB",
                key="tenant-b-mcp",
                access_level="internal",
                tenant_id="tenant-b",
                org_id="org-b",
            )
        )
    )
    scoped_admin = admin_headers(tenant_id="tenant-a", org_id="org-a")
    listed = _rpc(isolated_client, "resources/list", headers={**scoped_admin, "X-MCP-Scopes": "resource:kb"})
    denied_resource = _rpc(
        isolated_client,
        "resources/read",
        {"uri": f"kb://{tenant_b.id}/stats"},
        request_id=2,
        headers={**scoped_admin, "X-MCP-Scopes": "resource:kb"},
    )
    denied_tool = _rpc(
        isolated_client,
        "tools/call",
        {"name": "search_kb", "arguments": {"query": "private", "kb_id": tenant_b.id}},
        request_id=3,
        headers={
            **auth_headers(
                user_id="tenant-a-user",
                roles=["employee"],
                channel="web",
                tenant_id="tenant-a",
                org_id="org-a",
            ),
            "X-MCP-Scopes": "tool:search_kb",
        },
    )

    assert listed.status_code == 200, listed.text
    uris = {item["uri"] for item in listed.json()["result"]["resources"]}
    assert f"kb://{tenant_a.id}/stats" in uris
    assert f"kb://{tenant_b.id}/stats" not in uris
    assert denied_resource.json()["error"]["message"] == "Tool authorization denied"
    assert denied_tool.json()["error"]["message"] == "Tool execution failed"


def test_mcp_jobs_and_audit_resources_enforce_tenant_isolation(isolated_client: TestClient):
    context_a = RequestContext(
        request_id="mcp-tenant-a",
        auth={"user_id": "admin-a", "roles": ["admin"], "channel": "admin", "tenant_id": "tenant-a", "org_id": "org-a"},
    )
    context_b = RequestContext(
        request_id="mcp-tenant-b",
        auth={"user_id": "admin-b", "roles": ["admin"], "channel": "admin", "tenant_id": "tenant-b", "org_id": "org-b"},
    )
    job_a = enqueue_background_job(job_type="knowledge_gap_report", payload={}, context=context_a)
    job_b = enqueue_background_job(job_type="knowledge_gap_report", payload={}, context=context_b)
    now = utcnow_iso()
    for context in (context_a, context_b):
        execute_sync(
            """
            INSERT INTO tool_audit_logs (
                tool_call_id, request_id, user_id, roles_json, channel, tenant_id, org_id,
                tool_name, tool_status, created_at
            ) VALUES (?, ?, ?, '["admin"]', 'admin', ?, ?, 'search_kb', 'success', ?)
            """,
            (
                f"tool-{context.auth.tenant_id}",
                context.request_id,
                context.auth.user_id,
                context.auth.tenant_id,
                context.auth.org_id,
                now,
            ),
        )

    headers = {**admin_headers(tenant_id="tenant-a", org_id="org-a"), "X-MCP-Scopes": "resource:*"}
    jobs = _rpc(isolated_client, "resources/read", {"uri": "jobs://recent"}, headers=headers)
    audit = _rpc(isolated_client, "resources/read", {"uri": "audit://recent"}, request_id=2, headers=headers)

    jobs_payload = json.loads(jobs.json()["result"]["contents"][0]["text"])
    audit_payload = json.loads(audit.json()["result"]["contents"][0]["text"])
    assert job_a["job_id"] in {item["job_id"] for item in jobs_payload["items"]}
    assert job_b["job_id"] not in {item["job_id"] for item in jobs_payload["items"]}
    assert {item["request_id"] for item in audit_payload["tool_audit"]} == {"mcp-tenant-a"}


def test_admin_mcp_status_endpoint(isolated_client: TestClient):
    response = isolated_client.get("/api/admin/mcp/status", headers=admin_headers())
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["endpoint_path"] == "/mcp"
    assert payload["capabilities"]["tools"] is True
    assert payload["capabilities"]["risk_preview"] is True
    assert payload["capabilities"]["session_audit"] is True
    assert payload["tools"]["registered_count"] >= payload["tools"]["exposed_count"] >= 1
    assert payload["security"]["require_tool_scopes"] is True
    assert payload["security"]["require_resource_scopes"] is True
    assert payload["security"]["require_client_token"] is False
    assert payload["security"]["default_tool_quota_per_window"] >= 0
    assert payload["security"]["manifest_signature"]["signature"]
    assert any(item["name"] == "search_kb" for item in payload["tools"]["exposed"])
    assert any(item["name"] == "list_kbs" for item in payload["tools"]["blocked_by_policy"])
    assert any(item["uri"] == "kb://list" for item in payload["resources"]["items"])
    assert any(item["uriTemplate"] == "kb://{kb_id}/stats" for item in payload["resources"]["templates"])


def test_admin_mcp_status_requires_admin(isolated_client: TestClient):
    response = isolated_client.get(
        "/api/admin/mcp/status",
        headers=auth_headers(user_id="user-1", roles=["staff"], channel="web"),
    )
    assert response.status_code == 403
