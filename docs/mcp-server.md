# MCP Server

Phase 1 exposes the internal `ToolRegistry` through a minimal MCP-compatible JSON-RPC endpoint.

Endpoint:

```text
POST /mcp
```

Admin status endpoint:

```text
GET /api/admin/mcp/status
```

Supported methods:

- `initialize`
- `ping`
- `tools/list`
- `tools/call`
- `resources/list`
- `resources/templates/list`
- `resources/read`
- `notifications/initialized`

Local configuration:

```env
RAG_MCP_SERVER_ENABLED=true
RAG_MCP_PROTOCOL_VERSION=2025-06-18
RAG_MCP_SERVER_NAME=AgentBusiness MCP
RAG_MCP_SERVER_VERSION=0.1.0
RAG_MCP_REQUIRE_AUTH=true
RAG_MCP_VALIDATE_ORIGIN=true
RAG_MCP_ALLOWED_ORIGINS=
RAG_MCP_REQUIRE_TOOL_SCOPES=true
RAG_MCP_REQUIRE_RESOURCE_SCOPES=true
RAG_MCP_EXPOSED_TOOLS=search_kb,list_kbs,get_kb_stats,list_google_drive_sources,get_google_drive_sync_status,list_support_emails,read_email_thread
```

Production note: keep `RAG_MCP_SERVER_ENABLED=false` until the gateway protects `/mcp`.

Phase 2 safety controls:

- `RAG_MCP_REQUIRE_AUTH=true` requires an authenticated caller before any JSON-RPC method runs.
- `RAG_MCP_VALIDATE_ORIGIN=true` rejects browser-origin requests unless `Origin` is present in `RAG_MCP_ALLOWED_ORIGINS`.
- `RAG_MCP_EXPOSED_TOOLS` is a comma-separated allowlist. The default exposes read/observe tools only.
- Hidden tools return a JSON-RPC tool error and are not executable through MCP.
- `RAG_MCP_REQUIRE_TOOL_SCOPES=true` minimizes tool discovery and execution using `tool:{name}`, `scope:{domain}`, `risk:{level}`, or `mcp:*`.
- `RAG_MCP_REQUIRE_RESOURCE_SCOPES=true` minimizes resource discovery and reads using `resource:kb`, `resource:jobs`, `resource:audit`, `resource:*`, or `mcp:*`.
- Tool calls still run through `ToolRegistry.execute`, so existing auth policy, validation, timeout, and audit logging apply.
- MCP resources are read-only and require an admin caller because they expose operational data.
- Tenant/org-scoped admin callers only receive global or matching tenant/org resource data. This applies to KB discovery, KB stats, jobs, and audit resources.
- When clients send `MCP-Protocol-Version`, unsupported values return HTTP `400`. Missing headers remain accepted for backwards compatibility.

Built-in resources:

- `kb://list`
- `kb://{kb_id}/stats`
- `kb://{kb_id}/sources`
- `jobs://recent`
- `audit://recent`

Built-in resource templates:

- `kb://{kb_id}/stats`
- `kb://{kb_id}/sources`

Example `tools/list`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

Example `tools/call`:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "list_kbs",
    "arguments": {},
    "context": {
      "session_id": "mcp-client-1"
    }
  }
}
```

The endpoint reuses the existing HTTP auth headers and `ToolRegistry.execute`, so tool validation, authorization policy, timeout, and audit logging still apply.

Recommended scoped headers:

```http
X-MCP-Client-Id: portal-agent
X-MCP-Scopes: tool:search_kb resource:kb
MCP-Protocol-Version: 2025-06-18
```

Example `resources/list`:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "resources/list",
  "params": {}
}
```

Example `resources/templates/list`:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "resources/templates/list",
  "params": {}
}
```

Example `resources/read`:

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "resources/read",
  "params": {
    "uri": "kb://1/stats"
  }
}
```

The Admin dashboard includes an `MCP Server` tab that displays endpoint URL, security settings, exposed tools, hidden tools, resources, and resource templates.

Compatibility tests cover JSON-RPC request validation, batch requests, initialization negotiation, initialized notifications, ping, tool discovery/calls, resource discovery/reads, scope minimization, quotas, origin validation, client tokens, and tenant isolation.

Future expansion can add an MCP client adapter for consuming external MCP servers.
