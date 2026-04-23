# Architecture

## Core request flow

1. Upload file to `/api/upload`.
2. Store raw file in `data/raw` and metadata in SQLite.
3. Attach file to `Default KB` by default.
4. Trigger ingest for a KB.
5. Parse -> chunk -> embed -> add to vector backend.
6. Query `/api/chat` against one KB scope.

## Module ownership

- `app/upload.py`: upload endpoints and source-file lifecycle
- `app/upload_validation.py`: upload validation and content checks
- `app/ingest.py`: KB-scoped ingest jobs
- `app/parsers.py`, `app/parsers_docx.py`: format-specific extraction
- `app/vector_store.py`: Chroma or numpy backend facade
- `app/rag.py`: retrieval, reranking, prompt building, citations
- `app/agent.py`: advanced route selection, tool routing, slot/session behavior
- `app/database.py`: SQLite schema + migrations
- `app/main.py`: app startup, middleware, admin/debug/system endpoints

## Product layers

### MVP core

- upload
- ingest
- retrieval
- KB scoping
- admin UI
- chat UI

### Advanced runtime

- manual JSON tool routing
- native tool rollout for OpenAI-compatible stacks
- tool audit logs
- support tickets
- external integration cache tables
