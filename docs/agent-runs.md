# Durable agent runs

`agent_runs` checkpoints each agent orchestration turn without replacing the existing tool registry, MCP layer, or support workflow engine.

## Ownership

- `agent_runs`: one chat orchestration turn, route, current status, actor scope, and optional pending action link.
- `agent_run_steps`: route, tool, approval, and resume checkpoints.
- `workflow_runs`: longer-lived business workflows such as support case handling.
- `pending_actions`: reviewed execution boundary for destructive or outbound actions.

## Status flow

Normal routes complete immediately:

```text
running -> completed
        -> failed
```

Risky tool routes pause until the linked pending action reaches a terminal state:

```text
running -> paused -> completed   # approved and executed
                  -> cancelled   # rejected
                  -> failed      # execution failed
```

## Admin API

- `GET /api/admin/agent-runs`
- `GET /api/admin/agent-runs/{agent_run_id}`
- `POST /api/admin/agent-runs/{agent_run_id}/cancel`
- `POST /api/admin/agent-runs/{agent_run_id}/resume`
- `GET /api/admin/pending-actions/{pending_action_id}/events`

List filters:

- `status`
- `session_id`
- `pending_action_id`

Pending actions continue to use the existing approve, reject, and execute endpoints. Successful execution or rejection auto-resumes the linked agent run.

The pending action Events button in Admin UI calls the approval events endpoint. It assembles a timeline from `pending_actions`, notifications, linked `agent_run_steps`, and support workflow checkpoints so operators can audit who approved/executed/rejected an action and whether resume used idempotency metadata.

## Retry/resume without repeated side effects

Agent steps can now be run through an idempotent checkpoint helper. The helper stores:

- `idempotency_key`
- `side_effect`
- `attempt_count`
- `last_attempt_at`
- `output_json`

When a completed step is retried or resumed, the stored output is returned and the step function is not called again. Side-effect steps must provide an idempotency key. This protects actions such as sending email, creating tickets, writing external notes, or terminal approval resume from accidental duplicate execution.

Example key shape:

```text
agent-run:{agent_run_id}:pending-action:{pending_action_id}:terminal:executed
```

Failed non-side-effect steps can be retried automatically and increment `attempt_count`. Failed side-effect steps require an explicit manual retry path unless the caller opts into retrying with an idempotency key.

## Framework decision

This layer keeps orchestration state explicit and inspectable before introducing a framework migration. Evaluate OpenAI Agents SDK or LangGraph only after production traces show requirements that this persisted ledger and the existing workflow engine do not cover.
