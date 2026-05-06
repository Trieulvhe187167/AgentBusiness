from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
from tests.conftest import admin_headers, auth_headers, isolated_client


def _rpc(client: TestClient, method: str, params: dict | None = None, *, request_id: int = 1):
    return client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}},
        headers=admin_headers(),
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


def test_mcp_tools_list_maps_internal_registry(isolated_client: TestClient):
    response = _rpc(isolated_client, "tools/list")
    assert response.status_code == 200, response.text
    tools = response.json()["result"]["tools"]
    names = {item["name"] for item in tools}
    assert "search_kb" in names
    assert "list_kbs" in names
    assert "delete_google_drive_source" not in names
    assert "send_email_reply" not in names

    list_kbs = next(item for item in tools if item["name"] == "list_kbs")
    assert list_kbs["inputSchema"]["type"] == "object"
    assert list_kbs["annotations"]["scope"] == "admin"


def test_mcp_tools_call_executes_registry_tool(isolated_client: TestClient):
    response = _rpc(
        isolated_client,
        "tools/call",
        {"name": "list_kbs", "arguments": {}, "context": {"session_id": "mcp-test"}},
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"
    assert result["structuredContent"]["total"] >= 1
    assert result["structuredContent"]["items"][0]["key"] == "default"


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


def test_admin_mcp_status_endpoint(isolated_client: TestClient):
    response = isolated_client.get("/api/admin/mcp/status", headers=admin_headers())
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["endpoint_path"] == "/mcp"
    assert payload["capabilities"] == {"tools": True, "resources": True, "resource_templates": True}
    assert payload["tools"]["registered_count"] >= payload["tools"]["exposed_count"] >= 1
    assert any(item["name"] == "search_kb" for item in payload["tools"]["exposed"])
    assert any(item["uri"] == "kb://list" for item in payload["resources"]["items"])
    assert any(item["uriTemplate"] == "kb://{kb_id}/stats" for item in payload["resources"]["templates"])


def test_admin_mcp_status_requires_admin(isolated_client: TestClient):
    response = isolated_client.get(
        "/api/admin/mcp/status",
        headers=auth_headers(user_id="user-1", roles=["staff"], channel="web"),
    )
    assert response.status_code == 403
