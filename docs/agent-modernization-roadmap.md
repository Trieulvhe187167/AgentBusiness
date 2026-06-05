# Agent Modernization Roadmap

Date: 2026-06-02

This document captures a practical modernization plan for the current Agent for Business codebase. It is based on the existing local architecture and current public documentation for agent SDKs, retrieval systems, semantic caching, MCP, and evaluation tooling.

## Current Baseline

The project is already more than a simple RAG chatbot:

- `app/rag.py` owns KB-scoped retrieval, query expansion, reranking, citations, answer generation, fallback behavior, and chat logging.
- `app/agent.py` owns route selection across RAG, tools, memory, clarification, and fallback.
- `app/tools/registry.py` gives tools typed input/output schemas, authorization, timeouts, execution audit logs, and OpenAI-compatible function tool schemas.
- `app/mcp_server.py` exposes selected internal tools/resources over MCP-style JSON-RPC with security policy, quotas, manifest signing, and structured tool output.
- `app/evaluations.py` already has golden dataset and chat-log evaluation, but scoring is still mostly rule/token based.
- `app/cache.py` and `app/rag.py` integrate exact and feature-flagged semantic retrieval/response caches with strict KB, version, backend, model, and auth scoping.
- `app/vector_store.py` includes an opt-in Qdrant dense/hybrid backend with sparse BM25 inference and RRF fusion.
- `app/agent_runs.py` checkpoints orchestration turns, routes, tool calls, and approval transitions.
- `app/llm_client.py` streams OpenAI Responses SSE events directly and traces input, output, total, and cached token usage.
- `chat_logs` and the analytics dashboard persist and aggregate OpenAI token usage and cached-input reuse.
- `app/provider_capabilities.py` resolves active provider capabilities for streaming, tool calling, prompt cache controls, usage reporting, and structured-output support.
- `app/observability.py` has optional OpenTelemetry spans for LLM, retrieval, tools, MCP, and workflows.

The main opportunity is not a rewrite. The best path is to strengthen the existing architecture with better cache policy, retrieval quality, evaluation gates, and controlled agent orchestration.

## External Findings

### OpenAI platform direction

OpenAI positions the Agents SDK for code-first agent apps where the server owns orchestration, tool execution, state, and approvals. This matches the current project better than a hosted visual builder.

Relevant docs:

- https://developers.openai.com/api/docs/guides/agents
- https://developers.openai.com/api/docs/guides/tools
- https://developers.openai.com/api/docs/guides/tools-file-search
- https://developers.openai.com/api/docs/guides/prompt-caching

Implications for this repo:

- Keep the current backend-owned tool registry and authorization model.
- Add an optional OpenAI Agents SDK adapter only after the current route/tool contracts are stable.
- Improve OpenAI Responses support in `app/llm_client.py`, especially streaming, usage logging, prompt cache metrics, and tool result handling.
- Use OpenAI hosted file search only as an optional backend, not as a replacement for tenant-scoped local KB logic.

### Semantic and prompt caching

OpenAI prompt caching is automatic for repeated prompt prefixes on newer models, but it only helps exact prefix reuse and requires prompt structure discipline. RedisVL provides explicit semantic cache primitives with vector similarity, TTL, filters, and async checks.

Relevant docs:

- https://developers.openai.com/api/docs/guides/prompt-caching
- https://redis.io/docs/latest/develop/ai/redisvl/api/cache/
- https://redis.io/docs/latest/integrate/google-adk/semantic-caching/

Implications for this repo:

- Add scoped semantic retrieval cache first.
- Add scoped semantic response cache second, guarded by feature flags.
- Keep exact-match cache as the fastest first lookup.
- Never cache tool/live-data routes semantically by default.
- Log cache hit type, semantic score, cached query, latency saved, and whether the answer came from LLM or cache.

### Retrieval modernization

Current Chroma/numpy retrieval is acceptable for MVP, but production search quality should move toward hybrid retrieval: dense vectors for semantics, sparse/BM25 for exact terms, followed by reranking.

Relevant docs:

- https://qdrant.tech/documentation/search/hybrid-queries/
- https://qdrant.tech/documentation/tutorials-basics/reranking-hybrid-search/
- https://docs.weaviate.io/weaviate/concepts/search/hybrid-search
- https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/

Implications for this repo:

- Keep the `VectorStoreFacade` boundary and add a Qdrant backend as an optional production backend.
- Add hybrid search support rather than replacing the current Chroma backend immediately.
- Add a reranker abstraction with measurable latency and quality budgets.
- Improve ingestion idempotency and transform caching for large document sets.

### Agent state, approvals, and durability

LangGraph checkpointing and interrupts are useful references for durable multi-step work, human approval, and state replay. The project already has pending actions, support workflows, background jobs, and tool audit logs, so a full LangGraph migration is not required immediately.

Relevant docs:

- https://docs.langchain.com/oss/python/langgraph/persistence
- https://docs.langchain.com/oss/python/langgraph/interrupts

Implications for this repo:

- Model risky tool execution as explicit pending actions before considering a full graph runtime.
- Add resumable run records for agent workflows that span multiple steps or require approval.
- Keep the current direct RAG path simple for fast support Q&A.

### MCP and external tool interoperability

The MCP 2025-06-18 spec separates base protocol, lifecycle, authorization, tools, resources, prompts, and client features. Security guidance emphasizes least privilege, incremental scopes, and server-side authorization.

Relevant docs:

- https://modelcontextprotocol.io/specification/2025-06-18/basic/index
- https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization
- https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices

Implications for this repo:

- Continue exposing only selected tools via `RAG_MCP_EXPOSED_TOOLS`.
- Add protocol-version compatibility tests.
- Add resource and tool scope tests for tenant/org isolation.
- Avoid wildcard scopes and keep the server-side authorization checks as the source of truth.

### Evaluation and observability

Ragas and Phoenix provide stronger evaluation and trace workflows than the current rule-based scoring alone. Phoenix also aligns with the current OpenTelemetry foundation.

Relevant docs:

- https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
- https://arize.com/docs/phoenix/evaluation/llm-evals

Implications for this repo:

- Keep current deterministic checks, but add optional LLM-as-judge metrics for golden datasets.
- Track retrieval metrics separately from answer metrics.
- Use OpenTelemetry traces as the spine for debugging cache, retrieval, LLM, and tool execution.

## Roadmap

### Phase 1: Measurement and cache safety

Goal: reduce cost/latency without changing answer behavior.

Tasks:

- Add cache config flags: semantic cache enabled, threshold, max entries per scope, response cache enabled.
- Add exact response cache integration only for safe RAG answers.
- Add semantic retrieval cache scoped by KB, KB version, embedding model, backend, top_k, auth scope, tenant, org, and filters.
- Disable semantic cache when hashing embeddings are active.
- Add cache metrics in chat logs or trace spans.
- Add tests for tenant/org/KB isolation and cache invalidation on KB version change.

Success metrics:

- Cache hit rate by type: exact retrieval, semantic retrieval, exact response, semantic response.
- P50/P95 latency reduction.
- LLM call reduction.
- Zero cross-tenant/cache-scope leaks in tests.

### Phase 2: Evaluation quality gate

Goal: make every retrieval/cache/model change measurable.

Tasks:

- Expand golden dataset fields to include expected source/chunk/category and allowed answer variants.
- Add retrieval metrics: recall_at_k, MRR, citation source match.
- Add optional Ragas-style or Phoenix-backed faithfulness and answer relevance scoring.
- Add eval run comparison before/after config changes.
- Add a small CI smoke eval that can run without external LLM dependencies.

Success metrics:

- Every retrieval/cache change has a before/after eval run.
- Regression alerts trigger on meaningful golden score drops.
- Admin UI can inspect failed examples by query, retrieval results, answer, and citations.

### Phase 3: Retrieval backend upgrade path

Goal: improve answer quality for exact terms, product codes, policy names, and multilingual paraphrases.

Tasks:

- Add a Qdrant backend behind `VectorStoreFacade`.
- Add optional hybrid dense+sparse retrieval mode.
- Add sparse model configuration, initially BM25/BM42-style through Qdrant/FastEmbed if adopted.
- Add reranker abstraction with configurable provider.
- Add per-KB retrieval strategy: numpy/chroma for local MVP, qdrant-hybrid for production.

Success metrics:

- Higher recall_at_k on golden set.
- Better citation accuracy for exact IDs, policy names, and mixed Vietnamese/English queries.
- P95 retrieval latency stays within agreed budget.

### Phase 4: LLM client and prompt caching improvements

Goal: improve provider compatibility and reduce OpenAI request cost when OpenAI is used.

Implemented baseline:

- Log response usage where available, including cached prompt tokens.
- Persist OpenAI input, output, total, and cached token usage on RAG chat logs.
- Support `prompt_cache_key` and optional 24h prompt cache retention for OpenAI Responses when configured.
- Stream OpenAI Responses properly instead of chunking a non-streaming response.
- Keep static system/tool instructions before dynamic KB/user context.

Tasks:

- Add OpenAI Responses tool-result continuation after approval behavior has dedicated regression tests.

Success metrics:

- Usage logs include input tokens, output tokens, cached tokens, model, provider, and latency.
- Prompt cache hit rate is visible for OpenAI runs.
- No behavior change for local/openai-compatible providers.

### Phase 5: Agent workflow durability and approvals

Goal: make multi-step business actions safer and resumable.

Implemented baseline:

- Persisted `agent_runs` and `agent_run_steps` now checkpoint agent routes, tool calls, and approval transitions.
- Agent-created `pending_actions` link back through `agent_run_id`.
- Pending action execution, rejection, or failure automatically completes, cancels, or fails the paused agent run.
- Operations APIs expose list, detail, cancel, and safe resume controls.

Tasks:

- Add retry/resume behavior for background workflows. Agent run steps now carry idempotency metadata so completed side-effect steps return stored outputs on retry/resume.
- Add human approval UI events to the existing admin/internal surfaces.
- Evaluate whether LangGraph or OpenAI Agents SDK adds enough value after the internal workflow model is clean.

Success metrics:

- Risky operations are auditable and resumable.
- Failed workflows can be resumed without repeating successful side effects.
- Tool call approval behavior is covered by tests.

### Phase 6: MCP hardening and ecosystem compatibility

Goal: make the internal tools safe and useful to external MCP clients.

Implemented baseline:

- Added protocol compatibility tests for JSON-RPC validation, initialization negotiation, notifications, ping, batch requests, tools, and resources.
- Added tool and resource scope minimization tests.
- Added quota isolation, origin validation, plain/hashed client token, and tenant/org isolation tests.
- Scoped MCP KB, job, and audit resources to global or matching tenant/org data.

Tasks:

- Add tool descriptions optimized for agent selection, not marketing copy.
- Add prompt/resource templates only where they are operationally useful.

Success metrics:

- External MCP clients can discover only allowed tools.
- High-risk tools remain hidden unless explicitly allowed.
- Security tests cover tool/resource access boundaries.

## Recommended Immediate Next Sprint

The highest-leverage next sprint is:

1. Add an optional LLM-as-judge evaluation adapter behind feature flags while keeping deterministic metrics as the default CI gate.
2. Add retry/resume behavior for background workflows without repeating successful side effects.
3. Add approval events to the internal admin UI.
4. Evaluate Responses API tool-result continuation against the current Chat Completions-compatible native tool route.
5. Add per-provider cost reporting once pricing/config inputs are available.

This makes cost, cache effectiveness, and workflow reliability inspectable before introducing a larger orchestration framework.

## Deferred Decisions

- Do not migrate fully to OpenAI Agents SDK yet. Keep it as an adapter candidate.
- Do not replace Chroma/numpy immediately. Add Qdrant as an optional backend after cache/eval gates are in place.
- Do not cache tool/live-data responses semantically until each tool has explicit freshness, idempotency, and security metadata suitable for caching.
- Do not introduce LangGraph until there is a concrete workflow that needs checkpoint replay or graph-level human interrupts.
