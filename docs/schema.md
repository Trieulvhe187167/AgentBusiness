# Schema

## Phase ownership

| Phase | Tables / scope | Purpose |
| --- | --- | --- |
| Phase 0 / MVP | `uploaded_files`, `ingest_jobs`, `knowledge_bases`, `kb_files`, `chat_sessions`, `chat_logs` | Upload, ingest, KB mapping, base chat logs |
| Phase 1 | extra fields on `chat_logs`, `tool_audit_logs` | Request context and tool audit |
| Phase 2 | `support_tickets` | Support workflow via tool execution |
| Phase 5 | `chat_sessions.slots_json` | Session slot memory |
| Phase 19 | `order_status_cache`, `game_online_cache` | External integration cache snapshots |

## Ownership view

- Upload domain: `uploaded_files`, `kb_files`, `ingest_jobs`
- KB domain: `knowledge_bases`, `kb_files`
- Chat domain: `chat_sessions`, `chat_logs`
- Tooling domain: `tool_audit_logs`, `support_tickets`
- Integration domain: `order_status_cache`, `game_online_cache`

## Recommendation

Keep MVP reporting and onboarding focused on the Phase 0 tables. Treat later-phase tables as advanced runtime capabilities, not as the primary identity of the repo.
