"""
FastAPI application entrypoint.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.database import fetch_all, init_db
from app.ingest import router as ingest_router
from app.models import (
    CacheStats,
    ChatLogItem,
    ChatRequest,
    DocumentSummary,
    HealthResponse,
    KBSource,
    KBStats,
)
from app.upload import router as upload_router

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Local RAG Agent starting")

    settings.ensure_dirs()
    await init_db()

    app.state.embeddings_loaded = False
    try:
        from app.embeddings import get_dimension
        from app.vector_store import vector_store

        dim = get_dimension()
        app.state.embeddings_loaded = True
        vector_store.initialize(expected_dim=dim)
        app.state.vector_store_ready = True
        logger.info("Vector store ready: backend=%s dim=%s", vector_store.backend_name, dim)
    except Exception as err:
        app.state.vector_store_ready = False
        logger.error("Vector store init failed: %s", err, exc_info=True)

    try:
        from app.llm_client import is_llm_ready

        app.state.llm_loaded = is_llm_ready()
    except Exception:
        app.state.llm_loaded = False

    logger.info("Startup complete")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Local RAG Agent",
    description="Product-like Local RAG with pluggable vector DB and LLM providers",
    version="0.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    start = datetime.now(timezone.utc)
    response = await call_next(request)
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    return response


app.include_router(upload_router)
app.include_router(ingest_router)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    return HealthResponse(
        status="ok",
        llm_loaded=getattr(request.app.state, "llm_loaded", False),
        embeddings_loaded=getattr(request.app.state, "embeddings_loaded", False),
        vector_store_ready=getattr(request.app.state, "vector_store_ready", False),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/chat")
async def chat(req: ChatRequest):
    from app.rag import rag_stream

    async def event_generator():
        for event in rag_stream(
            query=req.message,
            session_id=req.resolved_session_id,
            lang=req.lang,
        ):
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"], ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@app.get("/api/kb/stats", response_model=KBStats)
async def kb_stats():
    from app.vector_store import vector_store

    files = await fetch_all("SELECT * FROM uploaded_files")
    total_files = len(files)
    ingested_files = len([item for item in files if item["status"] == "ingested"])

    vs_stats = vector_store.get_stats()
    sources = vector_store.get_sources()

    return KBStats(
        total_files=total_files,
        ingested_files=ingested_files,
        total_chunks=vs_stats["total_vectors"],
        total_vectors=vs_stats["total_vectors"],
        sources=sources,
    )


@app.get("/api/kb/sources", response_model=list[KBSource])
async def kb_sources():
    rows = await fetch_all("SELECT * FROM uploaded_files WHERE status='ingested' ORDER BY ingested_at DESC")
    return [
        KBSource(
            source_id=row["id"],
            filename=row["original_name"],
            file_type=row["file_type"],
            chunk_count=row.get("pages_or_rows") or 0,
            ingested_at=row.get("ingested_at"),
        )
        for row in rows
    ]


@app.get("/api/documents", response_model=list[DocumentSummary])
async def documents():
    rows = await fetch_all("SELECT * FROM uploaded_files ORDER BY created_at DESC")
    docs = []
    for row in rows:
        docs.append(
            DocumentSummary(
                doc_id=row["id"],
                file_name=row["original_name"],
                status=row["status"],
                chunks=row.get("pages_or_rows") or 0,
                kb_version=None,
                created_at=row["created_at"],
                ingested_at=row.get("ingested_at"),
            )
        )
    return docs


@app.get("/api/admin/chat-logs", response_model=list[ChatLogItem])
async def admin_chat_logs(limit: int = Query(default=settings.chat_log_limit_default, ge=1, le=500)):
    rows = await fetch_all(
        """
        SELECT id, session_id, mode, top_score, latency_ms, llm_provider,
               user_message, answer_text, created_at
        FROM chat_logs
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    return [
        ChatLogItem(
            id=row["id"],
            session_id=row.get("session_id") or "",
            mode=row["mode"],
            top_score=row.get("top_score"),
            latency_ms=row.get("latency_ms"),
            llm_provider=row.get("llm_provider"),
            user_message=row.get("user_message") or "",
            answer_text=row.get("answer_text") or "",
            created_at=row.get("created_at") or "",
        )
        for row in rows
    ]


@app.post("/api/cache/clear")
async def cache_clear():
    from app.cache import clear_cache

    clear_cache()
    return {"message": "Cache cleared"}


@app.get("/api/cache/stats", response_model=CacheStats)
async def cache_stats_endpoint():
    from app.cache import get_stats

    stats = get_stats()
    return CacheStats(**stats)


@app.get("/api/system")
async def system_info(request: Request):
    from app.cache import get_stats as get_cache_stats
    from app.llm_client import active_provider_name
    from app.embeddings import using_hashing_fallback
    from app.vector_store import vector_store

    cache = get_cache_stats()
    vs = vector_store.get_stats()
    hashing = using_hashing_fallback()

    effective_threshold_good = settings.threshold_good
    effective_threshold_low = settings.threshold_low
    effective_min_similarity = settings.min_similarity_threshold
    if hashing:
        effective_threshold_good = min(effective_threshold_good, settings.hashing_threshold_good)
        effective_threshold_low = min(effective_threshold_low, settings.hashing_threshold_low)
        effective_min_similarity = min(effective_min_similarity, settings.hashing_min_similarity_threshold)

    return {
        "embedding_model": settings.embedding_model,
        "embedding_source": settings.effective_embedding_source,
        "embedding_backend": "hashing" if hashing else "sentence-transformers",
        "vector_backend_config": settings.normalized_vector_backend,
        "vector_backend_active": vs.get("backend"),
        "collection_name": vs.get("collection_name"),
        "total_vectors": vs.get("total_vectors"),
        "top_k": settings.top_k,
        "threshold_good": settings.threshold_good,
        "threshold_low": settings.threshold_low,
        "min_similarity_threshold": settings.min_similarity_threshold,
        "effective_threshold_good": effective_threshold_good,
        "effective_threshold_low": effective_threshold_low,
        "effective_min_similarity_threshold": effective_min_similarity,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "answer_mode_config": settings.normalized_answer_mode,
        "llm_provider_config": settings.normalized_llm_provider,
        "llm_provider_active": active_provider_name(),
        "llm_loaded": getattr(request.app.state, "llm_loaded", False),
        "cache_entries": cache.get("total_entries", 0),
        "cache_size_mb": cache.get("size_mb", 0),
        "vector_store_ready": getattr(request.app.state, "vector_store_ready", False),
        "embeddings_loaded": getattr(request.app.state, "embeddings_loaded", False),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/debug/similarity")
async def debug_similarity(query: str, top_k: int = 10):
    from app.rag import decide_mode, retrieve

    results = retrieve(query, top_k=top_k)
    top_score = float(results[0].get("similarity", 0.0)) if results else 0.0
    return {
        "query": query,
        "top_score": round(top_score, 4),
        "predicted_mode": decide_mode(top_score),
        "threshold_good": settings.threshold_good,
        "threshold_low": settings.threshold_low,
        "results": [
            {
                "rank": idx + 1,
                "similarity": round(float(item.get("similarity", 0.0)), 4),
                "filename": item.get("filename", ""),
                "row_num": item.get("row_num"),
                "category": item.get("category", ""),
                "preview": item.get("text", "")[:200],
            }
            for idx, item in enumerate(results)
        ],
    }


static_dir = settings.data_dir.parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    html_path = static_dir / "admin.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Admin UI not found</h1><p>Place admin.html in /static/</p>")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    html_path = static_dir / "chat.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Chat UI not found</h1><p>Place chat.html in /static/</p>")


@app.get("/")
async def root():
    return {
        "message": "Local RAG Agent API",
        "admin": "/admin",
        "chat": "/chat",
        "health": "/health",
        "system": "/api/system",
        "docs": "/docs",
    }
