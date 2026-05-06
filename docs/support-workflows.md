# Support Workflows

Support workflows turn tickets and inbound emails into case lifecycle records. The workflow layer sits above the existing tools, pending actions, jobs, and audit logs.

Lifecycle:

```text
new -> classified -> enriched -> planned -> waiting_customer
                                      -> waiting_approval
                                      -> escalated
                                      -> resolved
                                      -> closed
```

Current behavior:

- Classifies ticket/email text into structured intent, entities, sentiment, auth requirement, and risk.
- Assigns deterministic priority and SLA using rule-based logic.
- Enriches context with linked email thread, previous tickets, and order status when an order code is present.
- Builds an action plan with low-risk steps and human-review steps.
- Auto-resolves low-risk high-confidence cases when enough context is available.
- Creates `support_case_review` pending actions for high-risk cases such as refund/cancel requests.
- Creates escalation packages with summary, findings, tools used, suggested next action, draft reply, and transcript.
- Supports case operations: list/get tickets, assign owner/team, add internal notes, update lifecycle status, and view full case context.
- Runs support workflows and SLA monitoring through `background_jobs`, so the API can enqueue and return a job id.
- Admin dashboard includes a `Support Cases` module for handle, assign, escalate, note, context, SLA monitor, and SLA schedule operations.

Admin endpoints:

```text
POST /api/admin/support-workflows/tickets/{ticket_id}/classify
POST /api/admin/support-workflows/tickets/{ticket_id}/handle
POST /api/admin/support-workflows/tickets/{ticket_id}/enqueue
POST /api/admin/support-workflows/emails/{email_id}/handle
POST /api/admin/support-workflows/emails/{email_id}/enqueue
GET  /api/admin/support-workflows/summary
POST /api/admin/support-workflows/sla/monitor
POST /api/admin/support-workflows/sla/enqueue
GET  /api/admin/support-tickets
GET  /api/admin/support-tickets/{ticket_id}
GET  /api/admin/support-tickets/{ticket_id}/notes
POST /api/admin/support-tickets/{ticket_id}/notes
POST /api/admin/support-tickets/{ticket_id}/assign
POST /api/admin/support-tickets/{ticket_id}/status
GET  /api/admin/support-tickets/{ticket_id}/context
POST /api/admin/support-tickets/{ticket_id}/escalate
```

Agent-facing support tools:

```text
create_support_ticket
list_customer_tickets
update_ticket_status
assign_ticket
add_ticket_internal_note
```

The workflow stores structured JSON snapshots on `support_tickets`:

- `classification_json`
- `context_summary_json`
- `action_plan_json`
- `escalation_package_json`
- `assigned_team`, `assigned_user_id`, `sla_breached_at`

`support_ticket_notes` stores internal handoff notes, status-change notes, assignment notes, SLA breach notes, and escalation notes.

High-risk actions should remain draft-first through `pending_actions`; the LLM should not directly execute refunds, cancellations, destructive data changes, or outbound email replies.
