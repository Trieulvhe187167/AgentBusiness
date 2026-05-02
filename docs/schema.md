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

## Ownership view

- Upload domain: `uploaded_files`, `kb_files`, `ingest_jobs`
- KB domain: `knowledge_bases`, `kb_files`
- Chat domain: `chat_sessions`, `chat_logs`
- Tooling domain: `tool_audit_logs`, `support_tickets`
- Integration domain: `order_status_cache`, `game_online_cache`
- Sync/job domain: `google_drive_sources`, `google_drive_files`, `google_drive_sync_runs`, `support_email_messages`, `support_email_sync_runs`, `background_jobs`, `sync_schedules`
- Safety domain: `pending_actions`

## Recommendation

Keep MVP reporting and onboarding focused on the Phase 0 tables. Treat later-phase tables as advanced runtime capabilities, not as the primary identity of the repo.
