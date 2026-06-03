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
| Phase 49 | `workflow_runs`, `workflow_steps` | Durable checkpoints for longer-lived business workflows |
| Phase 51 | `file_versions`, `file_version_ingests` | Uploaded-file version history, binary snapshots, and active ingest tracking |
| Phase 52 | `file_versions`, `file_version_ingests`, `kb_files` | Snapshot rollback, vector invalidation, and rollback re-ingest activation |
| Phase 53 | `file_versions` snapshots | Read-only version compare API with unified text diff |
| Phase 54 | `eval_golden_dataset`, extra fields on `agent_eval_results` | Golden Q&A regression evaluation and quality-drop alerting |
| Phase 56 | extra fields on `eval_golden_dataset`, `agent_eval_results`, `agent_eval_runs` | Extended retrieval metrics, baseline comparison, and evaluation gate status |
| Phase 57 | `agent_runs`, `agent_run_steps`, `pending_actions.agent_run_id` | Durable agent orchestration checkpoints and approval resume flow |
| Phase 58 | extra fields on `chat_logs` | Persisted OpenAI input, output, total, and cached token usage |
| Phase 60 | extra fields on `agent_eval_results` | Optional LLM-as-judge scores and explanations for golden evaluation |

## Ownership view

- Upload domain: `uploaded_files`, `kb_files`, `ingest_jobs`, `file_versions`, `file_version_ingests`
- KB domain: `knowledge_bases`, `kb_files`
- Chat domain: `chat_sessions`, `chat_logs`
- Tooling domain: `tool_audit_logs`, `agent_runs`, `agent_run_steps`, `support_tickets`, `support_ticket_notes`
- Integration domain: `order_status_cache`, `game_online_cache`
- Sync/job domain: `google_drive_sources`, `google_drive_files`, `google_drive_sync_runs`, `support_email_messages`, `support_email_sync_runs`, `background_jobs`, `sync_schedules`
- Safety domain: `pending_actions`
- Feedback/evaluation domain: `chat_feedback`, `agent_eval_runs`, `agent_eval_results`, `eval_golden_dataset`

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

## LLM usage

`chat_logs` stores OpenAI Responses usage totals when available:

- `llm_input_tokens`, `llm_output_tokens`, `llm_total_tokens`
- `llm_cached_tokens`: the cached input-token count reported by OpenAI

The analytics dashboard aggregates these fields and reports cached-input reuse.

## File versioning

`file_versions` stores immutable metadata for each uploaded-file content version:

- `file_id`, `version_number`, `file_hash`, `file_size`, `filename`, `original_name`, `file_type`, `parser_type`.
- `snapshot_path`: optional binary snapshot path under `RAG_FILE_VERSIONING_SNAPSHOT_DIR`.
- `pages_or_rows`, `chunk_count`, `ingest_signature`: filled as upload/ingest progresses.
- `created_by_user_id`, `created_at`, `change_summary`.

`file_version_ingests` records which version was activated for each `(kb_id, file_id)` ingest. Phase 51 exposes history through `GET /api/files/{file_id}/versions`. Phase 52 adds `POST /api/files/{file_id}/versions/{version_number}/rollback`, which restores a retained snapshot as a new current version, invalidates existing vectors for attached KBs, and can queue re-ingest. Phase 53 adds `GET /api/files/{file_id}/versions/{from_version}/diff/{to_version}` for read-only snapshot comparison.

## Golden evaluation

`eval_golden_dataset` stores benchmark Q&A rows for continuous RAG quality checks:

- `kb_id`, `question`, `expected_answer`.
- Optional `expected_source_file_id` for retrieval/citation validation.
- Optional accepted answer variants, multiple expected source IDs, expected chunk IDs, and expected categories.
- Optional `expected_keywords_json` and `tags_json`.
- `active`, actor context, tenant/org scope, and timestamps.

Golden runs reuse `agent_eval_runs` with `source='golden_dataset'`. `agent_eval_results` includes nullable golden metrics: `golden_item_id`, `expected_answer`, `answer_similarity`, `recall_at_k`, `mrr`, `citation_accuracy`, and source/chunk/category matches. Results also retain compact retrieval and citation previews for failed-example inspection.

When optional LLM-as-judge is enabled, `agent_eval_results` also stores `judge_provider`, `judge_model`, `judge_score`, `judge_verdict`, `judge_metrics_json`, `judge_reason`, `judge_latency_ms`, and `judge_error`. Judge metrics cover correctness, groundedness, completeness, citation support, and hallucination risk.

Each golden run stores aggregate metrics, an optional baseline run ID, comparison deltas, and `gate_status`. By default, the previous matching golden run is used as baseline. A metric regression beyond `max_metric_drop` marks the gate as failed.
