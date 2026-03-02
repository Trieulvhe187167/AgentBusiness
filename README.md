# CampusRAG-style Local Agent

Product-like RAG assistant with:
- Upload + ingest pipeline (PDF/TXT/MD/HTML/CSV/XLSX)
- 3-mode answer logic (`answer` / `clarify` / `fallback`)
- Citation-first chat UX with SSE streaming
- Pluggable LLM providers (`OpenAI`, `Gemini`, `Ollama`, `llama.cpp`)
- Pluggable vector backend (`Chroma` preferred, `numpy` fallback)
- Persistent clarify session state + chat logs in SQLite
- Admin dashboard and deployment-ready Docker setup

CSV/XLSX schema tips:
- Best schema: `title,content,category,keywords`
- FAQ schema also supported: `question,answer,category,tags`

## 1) Quick start (local)

```powershell
cd C:\Users\dmx\Projects\Agent_for_business
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Recommended Python on Windows: `3.13` (or `3.9` as stable fallback).  
If your default Python is `3.14`, create venv with:

```powershell
py -3.13 -m venv .venv
```

Open:
- `http://127.0.0.1:8080/admin`
- `http://127.0.0.1:8080/chat`
- `http://127.0.0.1:8080/docs`

Optional quality stack (recommended when your environment supports it):

```powershell
pip install sentence-transformers chromadb
```

Without these extras, the app still runs:
- embeddings fallback to hashing mode
- vector backend auto-fallbacks to numpy

## 2) Product path vs Local path

### Product path (online demo)
Use cloud LLM provider, keep ingestion/retrieval local on server:

```dotenv
RAG_LLM_PROVIDER=openai
RAG_OPENAI_API_KEY=...
RAG_OPENAI_MODEL=gpt-4o-mini

# or
RAG_LLM_PROVIDER=gemini
RAG_GEMINI_API_KEY=...
```

### Local path (offline / no external API key)

```dotenv
RAG_LLM_PROVIDER=ollama
RAG_OLLAMA_BASE_URL=http://localhost:11434
RAG_OLLAMA_MODEL=qwen2.5:3b

# or llama.cpp
RAG_LLM_PROVIDER=llama_cpp
RAG_LLM_MODEL_PATH=models/your-model.gguf
```

If no LLM is configured, the system automatically uses extractive answer mode.

## Embedding model options

- Default (`all-MiniLM-L6-v2`): fastest, lightest, weaker multilingual quality.
- `intfloat/multilingual-e5-large`: better multilingual retrieval, much heavier.
- `BAAI/bge-m3`: strong multilingual retrieval, heavy memory/compute usage.

For E5/BGE, the app auto-applies query/passages prefixes in `app/embeddings.py`.

## 3) Vector DB setup

Default:
```dotenv
RAG_VECTOR_BACKEND=chroma
```

Fallback:
```dotenv
RAG_VECTOR_BACKEND=numpy
```

Optional external Chroma server:
```dotenv
RAG_CHROMA_HTTP_URL=http://localhost:8000
```

## 4) Docker deployment

```bash
docker compose up --build
```

The provided `docker-compose.yml` runs:
- `app` (FastAPI)
- `chroma` (vector DB)

with persistent volumes for app data and vector storage.

## 5) Key API endpoints

- `POST /api/upload`
- `POST /api/ingest/all`
- `POST /api/ingest/{file_id}`
- `GET /api/jobs/{job_id}`
- `POST /api/chat` (SSE)
- `GET /api/documents`
- `GET /api/admin/chat-logs?limit=50`
- `GET /api/system`
- `GET /api/debug/similarity?query=...`

### Chat request example
```json
{
  "session_id": "s1",
  "message": "Đơn bao nhiêu thì miễn phí ship?",
  "lang": "vi"
}
```

### SSE events
- `start`
- `token`
- `citations`
- `done`
- `error`

## 6) Admin features

- Upload docs and trigger ingest
- Monitor KB/vector/cache stats
- Re-ingest or delete files
- Inspect chat logs (`mode`, `score`, `latency`, `question`)
- Inspect runtime config (`provider`, thresholds, backend)

## 7) Threshold calibration

Use debug API:
```bash
python scripts/eval_calibrate_thresholds.py --queries ./queries.txt
```

Suggested starter values:
```dotenv
RAG_THRESHOLD_GOOD=0.60
RAG_THRESHOLD_LOW=0.40
RAG_MIN_SIMILARITY_THRESHOLD=0.30
```

## 8) Batch ingest helper

```bash
python scripts/ingest_folder.py --input ./data/uploads --base-url http://127.0.0.1:8080
```

## 9) Notes

- Clarify session state and chat logs are persisted in SQLite (`data/metadata.db`).
- Upload validation includes extension + size + basic magic-bytes checks.
- For multilingual retrieval quality, prefer multilingual embedding models.
