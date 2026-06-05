# OpenAI Responses API

The optional `openai` provider uses the Responses API for generated RAG answers.
Its stream path consumes server-sent events directly and emits text from
`response.output_text.delta` events.

## Configuration

```dotenv
RAG_LLM_PROVIDER=openai
RAG_OPENAI_API_KEY=
RAG_OPENAI_MODEL=gpt-4o-mini
RAG_OPENAI_BASE_URL=https://api.openai.com/v1
RAG_OPENAI_PROMPT_CACHE_KEY=
RAG_OPENAI_PROMPT_CACHE_RETENTION=
RAG_OPENAI_RESPONSES_TOOL_CONTINUATION_ENABLED=false
RAG_OPENAI_RESPONSES_TOOL_MAX_STEPS=4
RAG_OPENAI_RESPONSES_TOOL_OUTPUT_MAX_CHARS=12000
```

`RAG_OPENAI_PROMPT_CACHE_KEY` is optional. Set a stable value only when requests
share a long static prompt prefix. `RAG_OPENAI_PROMPT_CACHE_RETENTION` is also
optional; leave it empty for the API default, or set `in-memory` or `24h`.
Support for extended `24h` retention depends on the selected OpenAI model.

## Observability

When OpenTelemetry is enabled, completed Responses streams record:

- `gen_ai.usage.input_tokens`
- `gen_ai.usage.output_tokens`
- `gen_ai.usage.total_tokens`
- `gen_ai.usage.cached_tokens`
- `gen_ai.request.prompt_cache_key_configured`
- `gen_ai.request.prompt_cache_retention`

The same usage totals are written to the application log. Cached token counts
come from `usage.input_tokens_details.cached_tokens` on the completed response.
RAG chat turns also persist these totals in `chat_logs` so the admin analytics
dashboard can report cached-input reuse over time.

## Tool-result continuation

When `RAG_OPENAI_RESPONSES_TOOL_CONTINUATION_ENABLED=true`, the agent can use
Responses API tool loops:

```text
Responses call with tools
-> model emits function_call
-> ToolRegistry executes the tool
-> app sends function_call_output with previous_response_id
-> model returns the final answer
```

Safety controls:

- `RAG_OPENAI_RESPONSES_TOOL_MAX_STEPS` caps model/tool loop depth.
- `RAG_OPENAI_RESPONSES_TOOL_OUTPUT_MAX_CHARS` truncates tool output before it is sent back to the model.
- ToolRegistry authorization, audit logging, and pending-action safety remain the source of truth.
- Agent run steps record `responses_initial`, `responses_tool_call:*`, `responses_tool_result:*`, and `responses_continuation`.

## References

- [OpenAI streaming responses](https://platform.openai.com/docs/guides/streaming-responses)
- [OpenAI prompt caching](https://platform.openai.com/docs/guides/prompt-caching)
- [OpenAI Responses API reference](https://platform.openai.com/docs/api-reference/responses/create)
