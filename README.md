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

### Access management

The app includes an internal user directory for Admin UI operations:

- observed users are recorded in `app_users`
- admins can edit display metadata, fallback roles, and active/inactive status
- inactive users are blocked even if they present otherwise valid dev/JWT/gateway identity
- role mode defaults to `fallback`, meaning internal roles are used only when upstream auth sends no roles

This is not a password/login system. For production login, keep using JWT or trusted gateway auth and use Access Management as an operational control layer.

## Dependency profiles

| File | Purpose |
| --- | --- |
| `requirements-core.txt` | MVP / easiest local mode |
| `requirements-rag.txt` | Core plus Chroma and richer tabular parsing |
| `requirements.txt` | Alias to `requirements-rag.txt` for Docker and full local RAG installs |
| `requirements-dev.txt` | Full local RAG dependencies plus `pytest` |

Important dependency notes:

- `chromadb` is optional for MVP and required only for the Chroma backend.
- PDF parsing uses `pdfminer.six` first, extracts structured table rows via `pdfplumber`, and can fall back to local OCR for scanned PDFs via `pdf2image` + `pytesseract`.
- Image uploads use Pillow validation, OCR through `pytesseract`, and OpenCV cleanup for detected table/document regions.
- Legacy `.xls` parsing requires `pandas` plus `xlrd`.
- `.docx` parsing uses `python-docx`.
- File versioning keeps immutable metadata and binary snapshots for uploaded-file updates.

## Supported uploads

Supported file types:

- `.pdf`, `.xlsx`, `.xls`, `.csv`, `.html`, `.htm`, `.txt`, `.md`, `.docx`, `.json`, `.jsonl`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.webp`, `.bmp`

Implementation notes:

- PDF parsing is text-first via `pdfminer.six`; detected tables are added as row-level records via `pdfplumber`; scanned/image-only pages can fall back to OCR when Poppler and Tesseract are installed.
- Image parsing validates the real image format with Pillow, OCRs the image with Tesseract, and adds cleaner OCR records for detected perspective/table regions.
- HTML parsing is static HTML only; dynamic pages are not rendered.
- `.xlsx` works from the core install via `openpyxl`.
- `.xls` needs the richer fallback parser from `requirements-rag.txt`.
- JSON and JSONL are treated as structured text records; see `docs/parser-support.md` for the exact expectations.

### File versioning

Each upload creates an initial file version. Updating an existing file record snapshots the previous binary before overwriting it, then creates the next version record for the new content. Version history is available at:

```http
GET /api/files/{file_id}/versions
```

Rollback restores a retained snapshot as a new current version and can queue re-ingest for affected KBs:

```http
POST /api/files/{file_id}/versions/{version_number}/rollback
```

Version diff compares two retained snapshots and returns a unified text diff:

```http
GET /api/files/{file_id}/versions/{from_version}/diff/{to_version}
```

The admin Knowledge Workspace exposes the same workflow with `Versions`, `Replace`, `Rollback`, and `V-Diff` actions. `Replace` uses `POST /api/files/{file_id}/content` to create the next version for an existing source file.

Relevant settings:

```dotenv
RAG_FILE_VERSIONING_KEEP_SNAPSHOTS=true
RAG_FILE_VERSIONING_SNAPSHOT_DIR=data/raw/versions
RAG_FILE_VERSIONING_RETENTION_COUNT=5
```

### Continuous RAG evaluation

Admins can store golden Q&A pairs and run regression evaluation against the live RAG pipeline:

```http
POST /api/admin/evaluations/golden-dataset
POST /api/admin/evaluations/golden-dataset/upload
POST /api/admin/evaluations/runs
```

Use `source=golden_dataset` in eval runs. Golden eval records `answer_similarity`, `recall_at_k`, and `citation_accuracy`; if the average score drops beyond `alert_drop_threshold`, the app creates an `evaluation.quality_drop` notification. Scheduled runs use `schedule_type=agent_eval_run` through the existing sync schedule API; set `interval_seconds=86400` and `next_run_at` to the next 2AM timestamp for nightly regression.

### OCR for scanned documents and images

The OCR path is local-first. For PDFs it only runs when extracted PDF text is too short; for image uploads it runs directly on the image.

Python dependencies are in `requirements-core.txt`, but the OCR engines are system tools:

- Install Poppler so `pdf2image` can render PDF pages.
- Install Tesseract OCR so `pytesseract` can read rendered page images.
- For Vietnamese PDFs, install the Vietnamese Tesseract language pack and set `RAG_PDF_OCR_LANGUAGE=vie` or `eng+vie`.

Windows `.env` examples:

```dotenv
RAG_PDF_OCR_ENABLED=true
RAG_PDF_OCR_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
RAG_PDF_OCR_POPPLER_PATH=C:\poppler\Library\bin
RAG_PDF_OCR_LANGUAGE=eng
```

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

## Production gateway deploy

Use `docker-compose.prod.yml` when deploying the agent behind a website backend
or trusted reverse proxy. This profile defaults auth to `gateway` and disables
dev header auth.

```powershell
$env:RAG_GATEWAY_SHARED_SECRET="your-long-random-secret"
docker compose -f docker-compose.prod.yml up --build
```

In this mode, browsers must not call the agent directly. Your website backend
authenticates the user and forwards trusted headers such as `X-Auth-User-Id`,
`X-Auth-Roles`, and `X-Auth-Gateway-Secret`. See
`docs/website-gateway-integration.md` and `.env.production.example`.

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
- `docs/google-drive-sync.md`
- `docs/action-safety.md`
- `docs/background-jobs.md`
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
