# Action Safety

High-risk actions use a pending approval workflow:

```text
draft -> approved -> executed
      -> rejected
      -> failed
```

## Covered Actions

- `send_email_reply`: always creates a draft pending action. The email is not sent until an admin approves and executes it.
- `delete_google_drive_source`: unlink and purge both create draft pending actions.
- `sync_google_drive_source`: normal changed-file sync enqueues a background job; `force_full=true` creates a pending action first because it can re-import many files and queue ingest jobs.

## Admin API

- `GET /api/admin/pending-actions`
- `POST /api/admin/pending-actions/{action_id}/approve`
- `POST /api/admin/pending-actions/{action_id}/reject`
- `POST /api/admin/pending-actions/{action_id}/execute`

All routes require admin auth. Execution enqueues a `pending_action_execute` background job. The worker dispatches to the existing internal action functions, stores the result in `pending_actions.result_json`, and marks the action as `executed` or `failed`.

## Database

`pending_actions` stores:

- action type and risk level
- draft payload
- approval/execution users and timestamps
- result or error message

This keeps destructive or outbound operations reviewable before they happen.
