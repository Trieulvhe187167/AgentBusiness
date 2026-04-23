# Windows setup

## Recommended path

1. Install Python 3.11 or 3.13.
2. Open PowerShell in the repo root.
3. Create a virtual environment.
4. Install the dependency profile you need.
5. Copy `.env.example` to `.env`.
6. Start the app with Uvicorn.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-core.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

## If you want Chroma

```powershell
pip install -r requirements-rag.txt
```

Then set:

```dotenv
RAG_VECTOR_BACKEND=chroma
```

## If you want advanced agent mode

Use the RAG profile and point the app to an OpenAI-compatible server such as vLLM.

## Notes

- `requirements-core.txt` is the least fragile starting point.
- PDF parsing is text-only; scanned PDFs still need OCR outside this repo.
- Legacy `.xls` support needs the richer dependency profile.
