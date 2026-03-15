# Local RAG Knowledge Base Chatbot

This project is a FastAPI-based RAG chatbot that supports document upload, Knowledge Base (KB) management, ingestion into a vector store, and chat scoped to each KB.

It now also supports backend tool execution for live business data, including:

- recent order lookup for signed-in users
- order status lookup by `order_code`
- online member count lookup for a game alliance/group

## Phase 0 runtime contract

The current agent upgrade baseline is fixed to:

- serving stack: `vLLM`
- API mode: `openai_compatible`
- default model target: `Qwen/Qwen3-4B-Instruct-2507`
- tool protocol: `manual_json`
- native tool calling: `disabled`

This keeps Phase 0 intentionally conservative. The model is only expected to return structured JSON actions in later phases; backend code remains responsible for validation, authorization, execution, and audit logging. `Qwen-Agent` stays a supported alternative for experiments, but it is not the default runtime contract for this repo.

## What the system does

- Upload documents: `.pdf`, `.xlsx`, `.xls`, `.csv`, `.html`, `.htm`, `.txt`, `.md`, `.docx`, `.json`, `.jsonl`
- Automatically creates a `Default KB` on first startup
- Attaches one file to one or many KBs
- Supports KB-scoped ingest, reindex, and chat by `kb_id` or `kb_key`
- Retrieval is isolated per KB, so results do not leak across KBs
- Includes an Admin UI for managing KBs, files, ingest jobs, stats, and retrieval debugging
- Includes a Chat UI with KB selection or `?kb=default`
- Has runtime fallbacks when models or vector backends are unavailable:
  - vector store: `chroma` or fallback `numpy`
  - embeddings: `sentence-transformers` or fallback hashing
  - answer mode: generative when an LLM is available, extractive otherwise

## Main runtime flow

1. A client uploads a file to `POST /api/upload`
2. The file is stored in `data/raw` and metadata is written to SQLite
3. New files are automatically attached to `Default KB`
4. When a KB is ingested:
   - the system loads files attached to that KB
   - parses content by file type
   - chunks the text
   - creates embeddings for each chunk
   - upserts chunks into the vector store with `kb_id` metadata
5. When a chat request is sent:
   - the request resolves `kb_id` or `kb_key`
   - retrieval runs only inside that KB
   - the system returns the answer, citations, and chat logs

## Directory layout

- `app/`: main FastAPI code for upload, ingest, retrieval, and KB management
- `static/`: `/admin` and `/chat` frontends
- `scripts/`: utility scripts such as folder ingest and threshold calibration
- `tests/`: KB isolation, smoke API, and admin/debug tests
- `data/`: runtime data such as SQLite, cache, uploads, and vector store files
- `models/`: local embedding or GGUF model directory for offline runs

## Environment requirements

Recommended:

- Python `3.11` or `3.13`
- `pip`
- Windows or Linux

Python `3.14` can still run the project, but FastAPI and Starlette currently emit deprecation warnings in tests.

## Installation
### 0. Quick start (local)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```
### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

If you have multiple Python versions installed:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install runtime dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install dev dependencies for tests

```powershell
pip install -r requirements-dev.txt
```

## Libraries used

### Main runtime dependencies

Packages in `requirements.txt`:

- `fastapi==0.115.6`
- `uvicorn[standard]==0.34.0`
- `python-multipart==0.0.20`
- `sse-starlette==2.2.1`
- `httpx==0.28.1`
- `openpyxl==3.1.5`
- `pdfminer.six==20231228`
- `beautifulsoup4==4.12.3`
- `numpy`
- `diskcache==5.6.3`
- `aiosqlite==0.20.0`
- `pydantic-settings==2.7.1`
- `sentence-transformers==3.3.1`
- `python-docx>=1.1.0`

### Dev and test

- `pytest==9.0.2` in `requirements-dev.txt`

### Optional dependencies

To use real Chroma instead of the `numpy` fallback:

```powershell
pip install chromadb==0.5.20
```

To run local `llama.cpp`:

```powershell
pip install llama-cpp-python --prefer-binary
```

To parse legacy `.xls` via the fallback path:

```powershell
pip install pandas xlrd
```

If your environment does not already provide a compatible PyTorch build for `sentence-transformers`, install PyTorch separately before using a real embedding model.

## `.env` configuration

Copy the example file:

```powershell
Copy-Item .env.example .env
```

Important settings:

```dotenv
RAG_VECTOR_BACKEND=chroma
RAG_CHROMA_HTTP_URL=
RAG_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
RAG_LLM_PROVIDER=openai_compatible
RAG_ANSWER_MODE=auto
RAG_OPENAI_API_KEY=
RAG_GEMINI_API_KEY=
RAG_OLLAMA_BASE_URL=
RAG_AGENT_SERVING_STACK=vllm
RAG_AGENT_TOOL_PROTOCOL=manual_json
RAG_AGENT_NATIVE_TOOL_CALLING=false
RAG_AGENT_TOOL_CHOICE_MODE=auto
RAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
RAG_LLM_API_KEY=EMPTY
RAG_LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507
RAG_LLM_MODEL_PATH=
RAG_ORDER_API_BASE_URL=
RAG_ORDER_API_STATUS_PATH=/orders/status
RAG_ORDER_API_RECENT_PATH=/orders/recent
RAG_GAME_API_BASE_URL=
RAG_GAME_API_ONLINE_PATH=/alliances/online
```

For the full external API and SQLite middle-layer setup, see [docs/external-integrations.md](/D:/Projects/Projects/Agent_for_business/docs/external-integrations.md).

## Common run modes

### 1. Lightest local mode

No Chroma and no LLM required.

- `RAG_VECTOR_BACKEND=numpy`
- `RAG_LLM_PROVIDER=none`

In this mode the app still works with:

- hashing embedding fallback
- `numpy` vector store fallback
- extractive answers

### 2. OpenAI-compatible server

Useful when you have a local server that exposes an OpenAI-style API, such as vLLM.

```dotenv
RAG_LLM_PROVIDER=openai_compatible
RAG_AGENT_SERVING_STACK=vllm
RAG_AGENT_TOOL_PROTOCOL=manual_json
RAG_AGENT_NATIVE_TOOL_CALLING=false
RAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
RAG_LLM_API_KEY=EMPTY
RAG_LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

For the current roadmap, this is the recommended local runtime. Native tool calling is intentionally left off until the manual JSON action loop is implemented and validated.

### 2b. OpenAI-compatible native tool calling rollout

This is now available as an opt-in path for Phase 6b. The backend still executes all tools itself; the model only selects tools and arguments.

```dotenv
RAG_LLM_PROVIDER=openai_compatible
RAG_AGENT_SERVING_STACK=vllm
RAG_AGENT_TOOL_PROTOCOL=openai_tools
RAG_AGENT_NATIVE_TOOL_CALLING=true
RAG_AGENT_TOOL_CHOICE_MODE=auto
RAG_AGENT_TOOL_PARSER=qwen3_coder
RAG_LLM_BASE_URL=http://127.0.0.1:8000/v1
RAG_LLM_API_KEY=EMPTY
RAG_LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

Notes:

- `manual_json` is still the default and safest rollout path.
- `/api/system` now reports `native_tool_status`, `native_tool_ready`, `native_tool_reason`, and `tool_choice_mode`.
- If your serving stack requires an explicit parser for auto tool choice, set `RAG_AGENT_TOOL_PARSER`.

### 3. Ollama

Two valid patterns:

Pattern 1, use the dedicated `ollama` provider:

```dotenv
RAG_LLM_PROVIDER=ollama
RAG_OLLAMA_BASE_URL=http://localhost:11434
RAG_OLLAMA_MODEL=qwen2.5:3b
```

Pattern 2, if Ollama is exposed through an OpenAI-compatible endpoint, use `openai_compatible` instead.

### 4. OpenAI

```dotenv
RAG_LLM_PROVIDER=openai
RAG_OPENAI_API_KEY=your_key
RAG_OPENAI_MODEL=gpt-4o-mini
```

### 5. Gemini

```dotenv
RAG_LLM_PROVIDER=gemini
RAG_GEMINI_API_KEY=your_key
RAG_GEMINI_MODEL=gemini-2.0-flash
```

### 6. Local llama.cpp

```dotenv
RAG_LLM_PROVIDER=llama_cpp
RAG_LLM_MODEL_PATH=models/your-model.gguf
```

### 7. External business APIs

The agent can call business tools for live data while still keeping execution in backend code.

Examples:

- `Đơn hàng của tôi tới đâu rồi?`
  - if `user_id` exists in `auth_context`, the agent can call `find_recent_orders`
- `Kiểm tra đơn DH12345`
  - the agent can call `get_order_status`
- `Liên minh LM01 có bao nhiêu người online?`
  - the agent can call `get_online_member_count`

Recommended rollout:

- start with SQLite snapshots in `order_status_cache` and `game_online_cache`
- then connect `RAG_ORDER_API_BASE_URL` and `RAG_GAME_API_BASE_URL`
- keep direct source-DB access out of the chatbot process

## Run the application

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Main endpoints:

- Admin UI: `http://127.0.0.1:8080/admin`
- Chat UI: `http://127.0.0.1:8080/chat`
- Chat with the default KB: `http://127.0.0.1:8080/chat?kb=default`
- Swagger: `http://127.0.0.1:8080/docs`
- Health: `http://127.0.0.1:8080/health`

## Recommended usage flow

### Fast path with Default KB

1. Upload a file through the Admin UI or `POST /api/upload`
2. The file is automatically attached to `Default KB`
3. Trigger ingest for `Default KB`
4. Chat using that KB's `kb_id` or open `/chat?kb=default`

### Multi-KB flow

1. Create a new KB through the Admin UI or `POST /api/kbs`
2. Upload files into the source library
3. Attach files to the KB you want
4. Call `POST /api/kbs/{id}/ingest`
5. Chat using `kb_id` or `kb_key`

### Delete semantics

- `DELETE /api/kbs/{id}/files/{file_id}`
  - detaches the file only from that KB
  - does not delete the source file if another KB still uses it
- `DELETE /api/files/{file_id}`
  - deletes the source file from the system
  - if the file is attached to multiple KBs, `force=true` is required

## Important APIs

### Knowledge Base

- `GET /api/kbs`
- `POST /api/kbs`
- `GET /api/kbs/default`
- `GET /api/kbs/{id}`
- `PATCH /api/kbs/{id}`
- `DELETE /api/kbs/{id}`

### KB file mapping

- `GET /api/kbs/{id}/files`
- `POST /api/kbs/{id}/files/{file_id}`
- `DELETE /api/kbs/{id}/files/{file_id}`

### Ingest

- `POST /api/kbs/{id}/ingest`
- `POST /api/kbs/{id}/reindex`
- `POST /api/kbs/{id}/files/{file_id}/ingest`
- `GET /api/jobs/{job_id}`

### Chat and retrieval debugging

- `POST /api/chat`
- `GET /api/debug/similarity?query=...&kb_id=...`
- `GET /api/debug/retrieval?query=...&kb_id=...`

### Stats and admin/debug

- `GET /api/kb/stats?kb_id=...`
- `GET /api/kb/sources?kb_id=...`
- `GET /api/sources/stats?kb_id=...`
- `GET /api/system?kb_id=...`
- `GET /api/admin/chat-logs?limit=50`
- `GET /api/cache/stats`
- `POST /api/cache/clear`

## Example chat request

```json
{
  "session_id": "demo-session",
  "message": "What is the shipping fee?",
  "lang": "vi",
  "kb_id": 1
}
```

## Run tests

### Current regression and smoke suite

```powershell
pytest tests -q
```

### Manual smoke scripts

```powershell
python test_upload.py --file kb_sample.csv --base-url http://127.0.0.1:8080
python test_flow.py --base-url http://127.0.0.1:8080
```

## Utility scripts

### Calibrate retrieval thresholds

```powershell
python scripts/eval_calibrate_thresholds.py --queries .\queries.txt
```

### Ingest a folder through the API

```powershell
python scripts/ingest_folder.py --input .\data\raw --base-url http://127.0.0.1:8080
```

## Docker

The project includes both `Dockerfile` and `docker-compose.yml`.

Run with Docker Compose:

```powershell
docker compose up --build
```

Current compose services:

- `app` on port `8080`
- `chroma` on port `8000`

Notes:

- `Dockerfile` uses `python:3.11-slim`
- the app container installs only the packages listed in `requirements.txt`
- compose sets `RAG_CHROMA_HTTP_URL=http://chroma:8000`
- app data and Chroma data are stored in separate volumes

## Operational notes

- `Default KB` is bootstrapped automatically on first startup
- metadata is stored in SQLite at `data/metadata.db`
- vector metadata always includes `kb_id`, so retrieval can be filtered by KB
- retrieval and response cache are scoped by KB and KB version
- the Admin UI is currently the most complete way to manage KBs in this project
