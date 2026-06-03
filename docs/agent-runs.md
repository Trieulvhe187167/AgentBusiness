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

List filters:

- `status`
- `session_id`
- `pending_action_id`

Pending actions continue to use the existing approve, reject, and execute endpoints. Successful execution or rejection auto-resumes the linked agent run.

## Framework decision

This layer keeps orchestration state explicit and inspectable before introducing a framework migration. Evaluate OpenAI Agents SDK or LangGraph only after production traces show requirements that this persisted ledger and the existing workflow engine do not cover.
