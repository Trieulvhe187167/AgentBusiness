# External Business Integrations

This phase adds three live-data tools on top of the agent layer:

- `get_order_status(order_code, user_id=None)`
- `find_recent_orders(user_id=None, limit=5)`
- `get_online_member_count(alliance_id, server_id=None)`

The model still does not call your production API directly. The backend executes the tool, validates permissions, writes audit logs, and stores snapshots in SQLite.

## Architecture

The flow is:

1. Agent routes the user message to a business tool.
2. Tool handler validates input and `auth_context`.
3. Integration adapter checks SQLite snapshot/cache first.
4. If cache is missing or stale and an external API is configured, the adapter calls the API.
5. The adapter normalizes the payload and upserts SQLite snapshot tables.
6. The tool returns structured JSON to the agent.
7. The agent composes a natural-language answer.

## SQLite intermediate database

Two new tables are used as the intermediate data layer:

- `order_status_cache`
  - keeps the latest known order status per `order_code`
  - indexed by `order_code`
  - also indexed by `user_id` for recent-order suggestions
- `game_online_cache`
  - keeps the latest online count per `alliance_id + server_id`
  - indexed by `alliance_id` and `server_scope`

Why keep this middle layer:

- reduces repeated API traffic
- gives you a fallback when the upstream API is down
- makes audit/debug easier
- lets you pre-sync business data from a worker or cron later

## Expected external API contracts

The code is intentionally flexible, but these payloads are the safest shape to expose.

### Order status

Request:

```http
GET /orders/status?order_code=DH12345&user_id=user-1
Authorization: Bearer <token>
```

Response:

```json
{
  "order_code": "DH12345",
  "user_id": "user-1",
  "status": "dang_giao",
  "last_update": "2026-03-15T14:20:00+07:00",
  "tracking_code": "GHN-9988",
  "carrier": "GHN"
}
```

### Recent orders

Request:

```http
GET /orders/recent?user_id=user-1&limit=5
Authorization: Bearer <token>
```

Response:

```json
{
  "orders": [
    {
      "order_code": "DH12345",
      "user_id": "user-1",
      "status": "dang_giao",
      "last_update": "2026-03-15T14:20:00+07:00",
      "tracking_code": "GHN-9988",
      "carrier": "GHN"
    },
    {
      "order_code": "DH12346",
      "user_id": "user-1",
      "status": "cho_xac_nhan",
      "last_update": "2026-03-15T10:05:00+07:00",
      "tracking_code": null,
      "carrier": null
    }
  ]
}
```

### Game online count

Request:

```http
GET /alliances/online?alliance_id=LM01&server_id=S1
Authorization: Bearer <token>
```

Response:

```json
{
  "alliance_id": "LM01",
  "server_id": "S1",
  "online_count": 128,
  "observed_at": "2026-03-15T14:21:00+07:00"
}
```

## Environment variables

```dotenv
RAG_INTEGRATION_CACHE_TTL_SECONDS=120
RAG_INTEGRATION_HTTP_TIMEOUT_SECONDS=15

RAG_ORDER_API_BASE_URL=http://127.0.0.1:9001
RAG_ORDER_API_KEY=
RAG_ORDER_API_STATUS_PATH=/orders/status
RAG_ORDER_API_RECENT_PATH=/orders/recent

RAG_GAME_API_BASE_URL=http://127.0.0.1:9002
RAG_GAME_API_KEY=
RAG_GAME_API_ONLINE_PATH=/alliances/online
```

## How to connect your real API

### Option A: direct REST integration

This is the fastest path.

1. Stand up your order service and game service with the HTTP routes above.
2. Set the base URLs in `.env`.
3. Restart the FastAPI app.
4. Ask the chatbot:
   - `Đơn hàng của tôi tới đâu rồi?`
   - `Kiểm tra đơn DH12345`
   - `Liên minh LM01 có bao nhiêu người online?`

### Option B: sync worker + SQLite snapshots

This is the safer production path if your source systems are slow or unstable.

1. Write a background worker that pulls from your source DB/API.
2. Upsert normalized rows into `order_status_cache` and `game_online_cache`.
3. Keep the chatbot read-only against SQLite.
4. Only let the worker talk to your production DB directly.

This pattern is better when:

- your order DB is private
- game metrics are high-frequency
- you need strict rate limiting
- you want predictable chatbot latency

## Recommended production split

- Chatbot backend:
  - reads SQLite cache
  - calls public internal APIs only when needed
  - enforces auth and audit
- Worker / integration service:
  - reads source DB or upstream APIs
  - normalizes data
  - writes SQLite snapshot/cache
- Source systems:
  - order management
  - game telemetry
  - payment/invoice systems

## Security rules

- Never let the LLM generate SQL and run it directly.
- Keep user-to-order authorization in backend code.
- Require `user_id` for order tools.
- Only let admin users query other users' orders.
- Log every tool call in `tool_audit_logs`.
