# Feedback Backend

Feedback V1 records thumbs-up/thumbs-down quality signals for chat answers.

User endpoint:

```text
POST /api/feedback/chat
```

Payload:

```json
{
  "request_id": "req-123",
  "chat_log_id": 12,
  "rating": "up",
  "reason_code": "good_answer",
  "comment": "Optional note"
}
```

Use either `request_id` or `chat_log_id`. If both are supplied, `chat_log_id` is used. Users can update their own prior feedback for the same chat log; the API upserts by `(chat_log_id, created_by_user_id)`.

Admin endpoints:

```text
GET /api/admin/feedback?rating=down&kb_id=1&limit=50
GET /api/admin/feedback/summary
GET /api/admin/chat-logs
```

`/api/admin/chat-logs` includes `feedback_up` and `feedback_down` aggregates for each chat log.

Authorization:

- Admin can submit feedback for any chat log.
- Non-admin users can submit feedback only for chat logs where `chat_logs.user_id` matches their authenticated user id.
- Anonymous chat logs can only be feedbacked by admin in V1.
