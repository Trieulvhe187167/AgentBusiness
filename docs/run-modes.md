# Run modes

## Summary

| Mode | Install | Default env posture | When to use |
| --- | --- | --- | --- |
| MVP / easiest local | `requirements-core.txt` | `RAG_VECTOR_BACKEND=numpy`, `RAG_LLM_PROVIDER=none` | First run, demos, local-only validation |
| Local RAG with Chroma | `requirements-rag.txt` | `RAG_VECTOR_BACKEND=chroma` | Larger KBs, closer-to-prod local retrieval |
| Advanced agent mode | `requirements-rag.txt` plus model server | `RAG_LLM_PROVIDER=openai_compatible` | Tool routing, runtime contract, audit logs |
| Local CPU RAG upgrade | `requirements-rag.txt` | `RAG_DEPLOYMENT_PROFILE=local_cpu` | Embedding/reranker experiments on constrained machines |
| Local GPU or service RAG | `requirements-rag.txt` plus model/embedding services | `RAG_DEPLOYMENT_PROFILE=local_gpu` or `service` | Qwen/BGE embedding and neural reranker with latency budgets |
| Website gateway auth | same as your app mode | `RAG_AUTH_MODE=gateway` | Integrating the agent behind a website backend or reverse proxy |

## MVP / easiest local

Characteristics:

- Windows/Linux friendly
- no model server required
- no Chroma required
- uses `numpy` vector fallback and extractive answers

Recommended env:

```dotenv
RAG_VECTOR_BACKEND=numpy
RAG_LLM_PROVIDER=none
```

## Local RAG with Chroma

Characteristics:

- better fit for larger datasets
- same upload and KB flow as MVP
- no model server required
- can use persistent local Chroma or external Chroma HTTP

Recommended env:

```dotenv
RAG_VECTOR_BACKEND=chroma
# RAG_CHROMA_HTTP_URL=http://127.0.0.1:8000
RAG_LLM_PROVIDER=none
```

## Advanced agent mode

Characteristics:

- requires an LLM provider
- keeps tool execution in backend code
- adds audit logs, session memory, and external integration paths

Recommended baseline:

```dotenv
RAG_LLM_PROVIDER=openai_compatible
RAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
RAG_LLM_API_KEY=EMPTY
RAG_LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507
RAG_AGENT_SERVING_STACK=vllm
RAG_AGENT_TOOL_PROTOCOL=manual_json
RAG_AGENT_NATIVE_TOOL_CALLING=false
```

Native tool rollout remains opt-in.

## Deployment Budget Profiles

Use these when enabling heavier embedding or reranker models. `custom` keeps
explicit env settings unchanged. Named profiles clamp expensive steps so the
agent remains usable under load.

Local CPU:

```dotenv
RAG_DEPLOYMENT_PROFILE=local_cpu
RAG_RERANKER_PROVIDER=bm25_lite
RAG_RUNTIME_MAX_RERANK_CANDIDATES=20
RAG_RUNTIME_MAX_ANSWER_CHUNKS=3
RAG_RUNTIME_RETRIEVAL_LATENCY_BUDGET_MS=2500
```

Local GPU:

```dotenv
RAG_DEPLOYMENT_PROFILE=local_gpu
RAG_RERANKER_PROVIDER=cross_encoder
RAG_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
RAG_RUNTIME_MAX_RERANK_CANDIDATES=80
RAG_RUNTIME_MAX_ANSWER_CHUNKS=5
```

Service mode:

```dotenv
RAG_DEPLOYMENT_PROFILE=service
RAG_EMBEDDING_PROVIDER=tei
RAG_EMBEDDING_BASE_URL=http://127.0.0.1:8081
RAG_LLM_PROVIDER=openai_compatible
RAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
RAG_RUNTIME_MAX_RERANK_CANDIDATES=120
RAG_RUNTIME_LLM_LATENCY_BUDGET_MS=15000
```

Overload controls:

```dotenv
RAG_RUNTIME_DISABLE_RERANKER=false
RAG_RUNTIME_DISABLE_NEURAL_RERANKER=false
RAG_RUNTIME_DISABLE_CORRECTIVE_RAG=false
```

For one request, `/api/chat` also accepts `disable_reranker` and
`disable_corrective_rag`. Chat SSE `start` and `done` events include
`runtime_budget` and `latency_breakdown` with embedding, vector query, reranker,
LLM, and retrieval cache metadata.

## Website gateway auth

Characteristics:

- the browser does not send roles directly to the agent
- a trusted website backend or reverse proxy injects identity headers
- the agent accepts only `X-Auth-*` headers plus an internal shared secret

Recommended baseline:

```dotenv
RAG_AUTH_MODE=gateway
RAG_GATEWAY_SHARED_SECRET=change-me
RAG_GATEWAY_SECRET_HEADER=X-Auth-Gateway-Secret
RAG_GATEWAY_USER_ID_HEADER=X-Auth-User-Id
RAG_GATEWAY_ROLES_HEADER=X-Auth-Roles
RAG_GATEWAY_CHANNEL_HEADER=X-Auth-Channel
```

Expected flow:

1. User logs in to your website.
2. Your website backend resolves the user's identity and roles.
3. The backend or reverse proxy forwards the request to the agent service with trusted `X-Auth-*` headers.
4. The agent ignores client `X-User-Id` and `X-Roles` in this mode.
