# Schema

## Phase ownership

| Phase | Tables / scope | Purpose |
| --- | --- | --- |
| Phase 0 / MVP | `uploaded_files`, `ingest_jobs`, `knowledge_bases`, `kb_files`, `chat_sessions`, `chat_logs` | Upload, ingest, KB mapping, base chat logs |
| Phase 1 | extra fields on `chat_logs`, `tool_audit_logs` | Request context and tool audit |
| Phase 2 | `support_tickets` | Support workflow via tool execution |
| Phase 5 | `chat_sessions.slots_json` | Session slot memory |
| Phase 19 | `order_status_cache`, `game_online_cache` | External integration cache snapshots |
| Phase 26 | `google_drive_sources`, `google_drive_files`, `google_drive_sync_runs` | Google Drive -> KB sync |
| Phase 27 | `support_email_messages`, `support_email_sync_runs` | Support mailbox snapshots |
| Phase 28 | `pending_actions` | Draft -> approve -> execute safety flow |
| Phase 29 | `background_jobs` | DB-backed queue for sync/ingest/action execution |
| Phase 30 | extra fields on `background_jobs` | Retry scheduling and cancellation controls |
| Phase 31 | extra fields on `background_jobs` | Worker ownership and heartbeat-based stale recovery |
| Phase 32 | `sync_schedules` | Fixed-interval Drive/email sync scheduling |
| Phase 35 | extra fields on `support_tickets` | Support case lifecycle, classification, SLA, action plan, and escalation package |
| Phase 36 | `support_ticket_notes`, extra fields on `support_tickets` | End-to-end support operations: notes, assignment, SLA breach tracking, and workflow jobs |
| Phase 38 | `chat_feedback` | User/admin 👍/👎 feedback for chat answers |

## Ownership view

- Upload domain: `uploaded_files`, `kb_files`, `ingest_jobs`
- KB domain: `knowledge_bases`, `kb_files`
- Chat domain: `chat_sessions`, `chat_logs`
- Tooling domain: `tool_audit_logs`, `support_tickets`, `support_ticket_notes`
- Integration domain: `order_status_cache`, `game_online_cache`
- Sync/job domain: `google_drive_sources`, `google_drive_files`, `google_drive_sync_runs`, `support_email_messages`, `support_email_sync_runs`, `background_jobs`, `sync_schedules`
- Safety domain: `pending_actions`
- Feedback domain: `chat_feedback`

## Recommendation

Keep MVP reporting and onboarding focused on the Phase 0 tables. Treat later-phase tables as advanced runtime capabilities, not as the primary identity of the repo.

## Support workflow fields

`support_tickets` includes workflow lifecycle columns:

- `intent`, `intent_confidence`, `sentiment`, `risk_level`
- `priority`, `sla_due_at`, `sla_breached_at`, `assigned_team`, `assigned_user_id`
- `workflow_status`, `workflow_updated_at`
- `resolution_summary`
- `classification_json`, `context_summary_json`, `action_plan_json`, `escalation_package_json`

These fields let the support workflow layer classify, enrich, plan, auto-resolve low-risk cases, create pending review actions for high-risk cases, and package escalations for human support.

`support_ticket_notes` records operational case history:

- `note_type`: internal, assignment, status_change, escalation, sla_breach.
- `visibility`: defaults to `internal`.
- `body`, `metadata_json`, `created_by_user_id`, `roles_json`, `created_at`.

Support workflows can run synchronously through admin APIs or asynchronously through `background_jobs` using `support_ticket_workflow`, `support_email_workflow`, and `support_sla_monitor`.

## Chat feedback

`chat_feedback` stores one feedback row per `(chat_log_id, created_by_user_id)`:

- `rating`: `up` or `down`.
- `reason_code`, `comment`: optional qualitative signal.
- `created_by_user_id`, `roles_json`, `channel`, `tenant_id`, `org_id`: feedback actor context.
- `request_id`: copied from the target chat log for API lookup and analytics.

Non-admin users can only feedback their own `chat_logs.user_id`; admin can feedback any chat log, including anonymous logs.
