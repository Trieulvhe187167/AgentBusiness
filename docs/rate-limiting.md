# Rate Limiting

The API process includes an in-memory fixed-window rate limiter for production safety. It protects expensive or risky endpoints before request validation/auth dependencies run.

Protected policy buckets:

```text
chat    -> POST /api/chat
mcp     -> POST /mcp
upload  -> POST /api/upload, POST /api/admin/upload
sync    -> sync, ingest, reindex, sync-schedule, and background-job mutations
admin   -> /api/admin/* and KB/file mutations
default -> all other non-exempt routes
```

Identity key order:

```text
tenant/org/user headers -> bearer token hash -> X-Forwarded-For/client IP
```

Response headers:

```text
X-RateLimit-Policy
X-RateLimit-Limit
X-RateLimit-Remaining
X-RateLimit-Reset
Retry-After        # only on 429
```

Environment settings:

```env
RAG_RATE_LIMIT_ENABLED=true
RAG_RATE_LIMIT_WINDOW_SECONDS=60
RAG_RATE_LIMIT_DEFAULT_REQUESTS_PER_WINDOW=600
RAG_RATE_LIMIT_CHAT_REQUESTS_PER_WINDOW=60
RAG_RATE_LIMIT_MCP_REQUESTS_PER_WINDOW=120
RAG_RATE_LIMIT_UPLOAD_REQUESTS_PER_WINDOW=20
RAG_RATE_LIMIT_ADMIN_REQUESTS_PER_WINDOW=300
RAG_RATE_LIMIT_SYNC_REQUESTS_PER_WINDOW=60
RAG_RATE_LIMIT_EXEMPT_PATHS=/health,/api/system
```

For multi-instance production, keep this in-app limiter but also add gateway/WAF or Redis-backed rate limiting. The in-app limiter does not share counters between API replicas.
