# Run modes

## Summary

| Mode | Install | Default env posture | When to use |
| --- | --- | --- | --- |
| MVP / easiest local | `requirements-core.txt` | `RAG_VECTOR_BACKEND=numpy`, `RAG_LLM_PROVIDER=none` | First run, demos, local-only validation |
| Local RAG with Chroma | `requirements-rag.txt` | `RAG_VECTOR_BACKEND=chroma` | Larger KBs, closer-to-prod local retrieval |
| Advanced agent mode | `requirements-rag.txt` plus model server | `RAG_LLM_PROVIDER=openai_compatible` | Tool routing, runtime contract, audit logs |
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
