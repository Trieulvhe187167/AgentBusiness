# Local RAG Knowledge Base Chatbot

A FastAPI-based local RAG app for document upload, KB-scoped ingestion, retrieval, and chat.

The repo also contains an advanced agent runtime for backend tool execution, support tickets, and external business integrations. That advanced path is real, but it is not the first thing new users should run.

## What should I run first?

Start with the MVP path unless you explicitly need tool calling or a model server.

| Run mode | Best for | OS | Model required | Chroma required | Feature level |
| --- | --- | --- | --- | --- | --- |
| MVP / easiest local mode | First run, offline-ish local KB chatbot, smoke testing | Windows or Linux | No | No | Upload, ingest, retrieve, extractive answers |
| Local RAG with Chroma | Larger local datasets, more realistic persistence | Windows or Linux | No | Yes | MVP plus Chroma backend and optional `.xls` support |
| Advanced agent mode | vLLM/OpenAI-compatible server, tool routing, native tool rollout | Windows or Linux | Yes | Optional | RAG plus tool routing, audit logs, session memory, integrations |
| Website gateway auth | Real website/backend integration | Windows or Linux | Optional | Optional | Advanced mode plus trusted upstream auth |

## Quick start

### 1. MVP / easiest local mode

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-core.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

This path uses the default local-first config:

- `RAG_VECTOR_BACKEND=numpy`
- `RAG_LLM_PROVIDER=none`
- extractive fallback answers when no LLM is configured

### 2. Local RAG with Chroma

```powershell
pip install -r requirements-rag.txt
```

or:

```powershell
pip install -r requirements.txt
```

`requirements.txt` is a convenience alias for `requirements-rag.txt`.

To enable Chroma locally:

```dotenv
RAG_VECTOR_BACKEND=chroma
# Optional external server
# RAG_CHROMA_HTTP_URL=http://127.0.0.1:8000
```

### 3. Advanced agent mode

Use this only when you already have an OpenAI-compatible server such as vLLM.

```dotenv
RAG_LLM_PROVIDER=openai_compatible
RAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
RAG_LLM_API_KEY=EMPTY
RAG_LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507
RAG_AGENT_SERVING_STACK=vllm
RAG_AGENT_TOOL_PROTOCOL=manual_json
RAG_AGENT_NATIVE_TOOL_CALLING=false
```

Native tool calling is still opt-in:

```dotenv
RAG_AGENT_TOOL_PROTOCOL=openai_tools
RAG_AGENT_NATIVE_TOOL_CALLING=true
RAG_AGENT_TOOL_CHOICE_MODE=auto
# RAG_AGENT_TOOL_PARSER=qwen3_coder
```

### 4. Website gateway auth

Use this when your website backend or reverse proxy will call the agent on behalf of the logged-in user.

```dotenv
RAG_AUTH_MODE=gateway
RAG_GATEWAY_SHARED_SECRET=change-me
RAG_GATEWAY_SECRET_HEADER=X-Auth-Gateway-Secret
RAG_GATEWAY_USER_ID_HEADER=X-Auth-User-Id
RAG_GATEWAY_ROLES_HEADER=X-Auth-Roles
RAG_GATEWAY_CHANNEL_HEADER=X-Auth-Channel
```

In this mode:

- the website backend injects trusted `X-Auth-*` headers
- the agent ignores browser-supplied `X-User-Id` and `X-Roles`
- `/api/me` and chat KB visibility reflect the forwarded upstream identity

## Dependency profiles

| File | Purpose |
| --- | --- |
| `requirements-core.txt` | MVP / easiest local mode |
| `requirements-rag.txt` | Core plus Chroma and richer tabular parsing |
| `requirements.txt` | Alias to `requirements-rag.txt` for Docker and full local RAG installs |
| `requirements-dev.txt` | Full local RAG dependencies plus `pytest` |

Important dependency notes:

- `chromadb` is optional for MVP and required only for the Chroma backend.
- PDF parsing currently uses `pdfminer.six`; OCR is not included.
- Legacy `.xls` parsing requires `pandas` plus `xlrd`.
- `.docx` parsing uses `python-docx`.

## Supported uploads

Supported file types:

- `.pdf`, `.xlsx`, `.xls`, `.csv`, `.html`, `.htm`, `.txt`, `.md`, `.docx`, `.json`, `.jsonl`

Implementation notes:

- PDF parsing is text-only via `pdfminer.six`; no OCR pipeline is included.
- HTML parsing is static HTML only; dynamic pages are not rendered.
- `.xlsx` works from the core install via `openpyxl`.
- `.xls` needs the richer fallback parser from `requirements-rag.txt`.
- JSON and JSONL are treated as structured text records; see `docs/parser-support.md` for the exact expectations.

## Docker Compose

`docker-compose.yml` is now a Chroma-enabled demo mode:

- app container
- Chroma container
- `RAG_LLM_PROVIDER=none` by default

Run it with:

```powershell
docker compose up --build
```

This gives you a local KB app with Chroma, but without requiring an external model server.

## Main endpoints

- Admin UI: `http://127.0.0.1:8080/admin`
- Chat UI: `http://127.0.0.1:8080/chat`
- Chat with default KB: `http://127.0.0.1:8080/chat?kb=default`
- Swagger: `http://127.0.0.1:8080/docs`
- Health: `http://127.0.0.1:8080/health`

## Docs

- `docs/run-modes.md`
- `docs/parser-support.md`
- `docs/architecture.md`
- `docs/vector-backends.md`
- `docs/schema.md`
- `docs/windows-setup.md`
- `docs/external-integrations.md`
- `docs/website-gateway-integration.md`

## Tests

Run the full suite:

```powershell
pytest tests -q
```

Useful smoke paths:

```powershell
pytest tests/api/test_phase9_smoke_api.py tests/api/test_upload_security.py -q
```
