# Website Gateway Integration

Use this mode when the chatbot is embedded in a real website and the website backend already knows the logged-in user.

## Goal

Do not let the browser send `admin`, `employee`, or other privileged roles directly to the agent.

Instead:

1. the browser sends its normal request to your website backend
2. the website backend verifies the session
3. the website backend forwards the request to the agent with trusted `X-Auth-*` headers

## Agent env

```dotenv
RAG_AUTH_MODE=gateway
RAG_GATEWAY_SHARED_SECRET=change-me
RAG_GATEWAY_SECRET_HEADER=X-Auth-Gateway-Secret
RAG_GATEWAY_USER_ID_HEADER=X-Auth-User-Id
RAG_GATEWAY_ROLES_HEADER=X-Auth-Roles
RAG_GATEWAY_CHANNEL_HEADER=X-Auth-Channel
RAG_GATEWAY_TENANT_ID_HEADER=X-Auth-Tenant-Id
RAG_GATEWAY_ORG_ID_HEADER=X-Auth-Org-Id
```

## Expected forwarded headers

```http
X-Auth-Gateway-Secret: change-me
X-Auth-User-Id: customer-001
X-Auth-Roles: customer
X-Auth-Channel: web
X-Auth-Tenant-Id: tenant-a
X-Auth-Org-Id: org-a
```

## Chat request to agent

```http
POST /api/chat
Content-Type: application/json
X-Auth-Gateway-Secret: change-me
X-Auth-User-Id: customer-001
X-Auth-Roles: customer
X-Auth-Channel: web
```

```json
{
  "message": "Đơn hàng của tôi đang ở đâu?",
  "session_id": "sess-customer-001",
  "lang": "vi",
  "kb_id": 2
}
```

## Node.js proxy example

```js
app.post("/website/chat", async (req, res) => {
  const sessionUser = req.user;

  const upstream = await fetch("http://agent-service:8080/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Auth-Gateway-Secret": process.env.AGENT_GATEWAY_SECRET,
      "X-Auth-User-Id": sessionUser.id,
      "X-Auth-Roles": sessionUser.roles.join(","),
      "X-Auth-Channel": "web",
      "X-Auth-Tenant-Id": sessionUser.tenantId ?? "",
      "X-Auth-Org-Id": sessionUser.orgId ?? "",
    },
    body: JSON.stringify(req.body),
  });

  res.status(upstream.status);
  upstream.body.pipe(res);
});
```

## Security notes

- Do not expose `RAG_GATEWAY_SHARED_SECRET` to the browser.
- Do not let the browser call the agent directly in this mode.
- Keep the agent behind the website backend or a reverse proxy that injects the trusted headers.
