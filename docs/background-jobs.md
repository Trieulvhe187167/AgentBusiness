# Background Jobs

Long-running operations now run through the SQLite-backed `background_jobs` queue. API handlers enqueue work and return a `job_id`; the in-process worker claims queued jobs and writes result/error state back to the database.

## Status Flow

```text
queued -> running -> done
queued -> running -> retrying -> running
queued -> running -> failed
queued/retrying -> cancelled
running -> cancelling -> cancelled
```

Retry uses exponential backoff through `retry_after`. Jobs default to 3 attempts unless the caller sets another `max_attempts`. Running jobs write `worker_id` and `heartbeat_at`; stale jobs are recovered automatically when no heartbeat is seen within `RAG_BACKGROUND_WORKER_STALE_SECONDS`.

## Covered Jobs

- `google_drive_sync`: scans a configured Drive source and imports changed files.
- `support_email_sync`: fetches the support mailbox into `support_email_messages`.
- `kb_ingest`: ingests stale files for one Knowledge Base.
- `kb_reindex`: re-ingests every file attached to one Knowledge Base.
- `kb_file_ingest`: ingests one file in one Knowledge Base.
- `pending_action_execute`: executes an approved pending action such as email reply, Drive delete/purge, or force full sync.

## Scheduled Sync

`sync_schedules` can automatically enqueue:

- `google_drive_sync` for one configured Drive source.
- `support_email_sync` for the support mailbox snapshot.

The scheduler loop runs with the background worker. It checks due schedules, skips enqueue when an equivalent job is already `queued`, `retrying`, `running`, or `cancelling`, and advances `next_run_at`.

## Admin API

- `GET /api/admin/background-jobs?status=queued|running|done|failed`
- `GET /api/admin/background-jobs/{job_id}`
- `POST /api/admin/background-jobs/{job_id}/cancel`
- `POST /api/admin/background-jobs/{job_id}/retry`
- `GET /api/admin/sync-schedules`
- `POST /api/admin/sync-schedules`
- `PATCH /api/admin/sync-schedules/{schedule_id}`
- `DELETE /api/admin/sync-schedules/{schedule_id}`
- `POST /api/admin/google-drive/sources/{source_id}/sync`
- `POST /api/admin/support-email/sync`
- `POST /api/kbs/{kb_id}/ingest`
- `POST /api/kbs/{kb_id}/reindex`
- `POST /api/kbs/{kb_id}/files/{file_id}/ingest`
- `POST /api/admin/pending-actions/{action_id}/execute`

`force_full=true` Drive sync still creates a pending action first. After approval, execution is queued as `pending_action_execute`.

## Retry And Cancel

- Failed transient jobs move to `retrying` while `attempts < max_attempts`.
- Admins can cancel `queued`, `retrying`, or `running` jobs from the Background Jobs UI.
- Admins can manually retry `failed` or `cancelled` jobs.
- Long ingest jobs check cancellation between files.
- Google Drive sync checks cancellation between scanned/downloaded files.
- Email sync and approved action execution can be marked as `cancelling`, but cancellation is only applied after the current provider call returns.

## Worker Process

Local/dev defaults to an in-process worker inside FastAPI:

```dotenv
RAG_BACKGROUND_WORKER_ENABLED=true
```

Production can disable the API worker and run a separate process:

```dotenv
RAG_BACKGROUND_WORKER_ENABLED=false
RAG_BACKGROUND_WORKER_POLL_INTERVAL_SECONDS=0.5
RAG_BACKGROUND_WORKER_HEARTBEAT_INTERVAL_SECONDS=5
RAG_BACKGROUND_WORKER_STALE_SECONDS=60
RAG_SCHEDULED_SYNC_ENABLED=true
RAG_SCHEDULED_SYNC_POLL_INTERVAL_SECONDS=10
```

```powershell
python -m app.worker
```

`docker-compose.prod.yml` runs the API with `RAG_BACKGROUND_WORKER_ENABLED=false` and starts a dedicated `worker` service with `python -m app.worker`.

## Database

`background_jobs` stores:

- stable `job_id`, `job_type`, and `status`
- input payload in `payload_json`
- result or error in `result_json` / `error_message`
- progress, attempts, timestamps, and KB/user scope metadata
- retry scheduling through `retry_after`
- cancellation metadata through `cancel_requested_at`, `cancelled_at`, and `cancel_reason`
- worker ownership and liveness through `worker_id` and `heartbeat_at`

Because job state is persisted in SQLite, the API and worker can run as separate processes while keeping the same public API contract.
