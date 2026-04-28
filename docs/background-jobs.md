# Background Jobs

Long-running operations now run through the SQLite-backed `background_jobs` queue. API handlers enqueue work and return a `job_id`; the in-process worker claims queued jobs and writes result/error state back to the database.

## Covered Jobs

- `google_drive_sync`: scans a configured Drive source and imports changed files.
- `support_email_sync`: fetches the support mailbox into `support_email_messages`.
- `kb_ingest`: ingests stale files for one Knowledge Base.
- `kb_reindex`: re-ingests every file attached to one Knowledge Base.
- `kb_file_ingest`: ingests one file in one Knowledge Base.
- `pending_action_execute`: executes an approved pending action such as email reply, Drive delete/purge, or force full sync.

## Admin API

- `GET /api/admin/background-jobs?status=queued|running|done|failed`
- `GET /api/admin/background-jobs/{job_id}`
- `POST /api/admin/google-drive/sources/{source_id}/sync`
- `POST /api/admin/support-email/sync`
- `POST /api/kbs/{kb_id}/ingest`
- `POST /api/kbs/{kb_id}/reindex`
- `POST /api/kbs/{kb_id}/files/{file_id}/ingest`
- `POST /api/admin/pending-actions/{action_id}/execute`

`force_full=true` Drive sync still creates a pending action first. After approval, execution is queued as `pending_action_execute`.

## Database

`background_jobs` stores:

- stable `job_id`, `job_type`, and `status`
- input payload in `payload_json`
- result or error in `result_json` / `error_message`
- progress, attempts, timestamps, and KB/user scope metadata

The current worker is in-process with FastAPI for simple deployment. Because job state is persisted in SQLite, it can later be moved to a separate worker process without changing the public API contract.
