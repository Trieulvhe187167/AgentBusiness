# Support Email Action Tools

Email is implemented as backend action tools. The agent only decides intent and calls a tool; the backend owns provider credentials, mailbox access, ticket creation, outbound sending, and audit logging.

## Runtime Flow

1. User asks for an email action, for example `list support emails` or `tao ticket tu email 12`.
2. `app.agent` routes to one of these tools: `list_support_emails`, `read_email_thread`, `create_ticket_from_email`, `send_email_reply`.
3. `app.tools.registry` validates input, checks admin authorization, executes the handler, and writes `tool_audit_logs`.
4. Mailbox sync is queued as `support_email_sync`; the worker calls `app.integrations.support_email` and stores snapshots in SQLite.
5. Ticket creation writes to `support_tickets` and links `support_email_messages.ticket_code`.
6. Email replies create a `pending_actions` draft. The SMTP send happens only after admin approval and execution.

## Database Tables

- `support_email_messages`: inbound/outbound message snapshot, thread id, sender/recipient metadata, body text, status, linked ticket code.
- `support_email_sync_runs`: mailbox sync run status and counts.
- `support_tickets`: existing ticket table reused by `create_ticket_from_email`.
- `pending_actions`: draft/approve/execute workflow for outbound email replies.
- `tool_audit_logs`: existing audit table records every tool call, success, failure, and latency.

## Tool Names

- `list_support_emails`: list inbound support emails from the local snapshot. Admin UI uses `POST /api/admin/support-email/sync` when a mailbox refresh is needed.
- `read_email_thread`: read the stored thread for an email id or thread id.
- `create_ticket_from_email`: create a support ticket from email content and extract an order code if present.
- `send_email_reply`: create a pending action draft for a plain-text reply. Approve and execute it before SMTP sends the email.

## Configuration

The first provider adapter is `imap_smtp`, compatible with Gmail and Outlook when IMAP/SMTP credentials are available.

```env
RAG_EMAIL_INTEGRATION_ENABLED=true
RAG_EMAIL_PROVIDER=imap_smtp
RAG_EMAIL_IMAP_HOST=imap.gmail.com
RAG_EMAIL_IMAP_PORT=993
RAG_EMAIL_IMAP_USERNAME=support@example.com
RAG_EMAIL_IMAP_PASSWORD=...
RAG_EMAIL_SMTP_HOST=smtp.gmail.com
RAG_EMAIL_SMTP_PORT=587
RAG_EMAIL_SMTP_USERNAME=support@example.com
RAG_EMAIL_SMTP_PASSWORD=...
RAG_EMAIL_FROM_ADDRESS=support@example.com
```

For Outlook, use `outlook.office365.com` and `smtp.office365.com`. Gmail/Outlook API or Microsoft Graph can be added later behind the same tool names by replacing the adapter, without changing agent routing or UI.

## Admin API

- `GET /api/admin/support-email/messages?limit=30`
- `POST /api/admin/support-email/sync`
- `POST /api/admin/sync-schedules` with `schedule_type=support_email_sync`
- `GET /api/admin/sync-schedules`
- `GET /api/admin/support-email/messages/{email_id}/thread`
- `POST /api/admin/support-email/messages/{email_id}/ticket`
- `POST /api/admin/support-email/messages/{email_id}/reply`
- `GET /api/admin/pending-actions`
- `POST /api/admin/pending-actions/{action_id}/approve`
- `POST /api/admin/pending-actions/{action_id}/execute`
- `GET /api/admin/background-jobs/{job_id}`

All routes require admin auth in the current project policy.
