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

## Scope

Native OpenAI tool routing still uses the Chat Completions-compatible path.
Responses API tool execution should be introduced separately after tool-result
continuation and approval behavior have dedicated regression tests.

## References

- [OpenAI streaming responses](https://platform.openai.com/docs/guides/streaming-responses)
- [OpenAI prompt caching](https://platform.openai.com/docs/guides/prompt-caching)
- [OpenAI Responses API reference](https://platform.openai.com/docs/api-reference/responses/create)
