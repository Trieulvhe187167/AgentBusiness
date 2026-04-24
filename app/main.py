"""
FastAPI application entrypoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import AppStatus, EventSourceResponse

from app.auth import get_request_auth, require_admin
from app.config import settings
from app.database import fetch_all, fetch_one, init_db
from app.ingest import router as ingest_router
from app.kb import router as kb_router
from app.kb_service import list_accessible_kbs, open_db, resolve_kb_scope
from app.models import (
    AuthAuditLogItem,
    CacheStats,
    ChatLogItem,
    ChatRequest,
    CurrentUserProfile,
    DocumentSummary,
    HealthResponse,
    KnowledgeBaseSummary,
    KBSource,
    KBStats,
    RequestContext,
    SystemRuntime,
    ToolAuditLogItem,
)
from app.upload import router as upload_router
from app.tools.drive_tools import (
    CreateGoogleDriveSourceInput,
    CreateGoogleDriveSourceOutput,
    DeleteGoogleDriveSourceOutput,
    GetGoogleDriveSyncStatusOutput,
    ListGoogleDriveSourcesOutput,
    SyncGoogleDriveSourceOutput,
)
from app.tools.email_tools import (
    CreateTicketFromEmailRequest,
    CreateTicketFromEmailOutput,
    ListSupportEmailsOutput,
    ReadEmailThreadOutput,
    SendEmailReplyRequest,
    SendEmailReplyOutput,
)

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")


def _safe_request_id(raw: str | None) -> str:
    value = (raw or "").strip()
    return value[:80] if value else uuid.uuid4().hex[:8]


def _parse_roles_json(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _reject_body_auth_override_in_jwt_mode(req: ChatRequest) -> None:
    if settings.normalized_auth_mode != "jwt":
        return

    forbidden_fields = {"user_id", "roles", "channel", "tenant_id", "org_id"}
    supplied = sorted(field for field in req.model_fields_set if field in forbidden_fields)
    if supplied:
        raise HTTPException(
            status_code=400,
            detail=f"Auth fields are not allowed in chat body when AUTH_MODE=jwt: {', '.join(supplied)}",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Local RAG Agent starting")

    settings.ensure_dirs()
    await init_db()

    # ── Vector store (needs embedding dim first via hashing estimate) ──────────
    try:
        from app.embeddings import get_dimension
        from app.vector_store import vector_store

        dim = get_dimension()
        vector_store.initialize(expected_dim=dim)
        app.state.vector_store_ready = True
        logger.info("Vector store ready: backend=%s dim=%s", vector_store.backend_name, dim)
    except Exception as err:
        app.state.vector_store_ready = False
        logger.error("Vector store init failed: %s", err, exc_info=True)

    # ── Embedding model warm-up (async, non-blocking) ─────────────────────────
    from app.embeddings import warm_up_model
    asyncio.create_task(asyncio.to_thread(warm_up_model))
    logger.info("Embedding warm-up started in background thread")

    # ── LLM readiness ─────────────────────────────────────────────────────────
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
    request_id = _safe_request_id(request.headers.get("X-Request-ID"))
    start = datetime.now(timezone.utc)
    request.state.request_id = request_id
    response = await call_next(request)
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    return response


app.include_router(upload_router)
app.include_router(ingest_router)
app.include_router(kb_router)


async def _resolve_optional_kb_scope(
    kb_id: int | None = None,
    kb_key: str | None = None,
):
    if kb_id is None and not kb_key:
        return None

    db = await open_db()
    try:
        return await resolve_kb_scope(db, kb_id=kb_id, kb_key=kb_key)
    finally:
        await db.close()


async def _build_stats_response(
    kb_id: int | None = None,
    kb_key: str | None = None,
) -> KBStats:
    from app.vector_store import vector_store

    kb_scope = await _resolve_optional_kb_scope(kb_id=kb_id, kb_key=kb_key)
    if kb_scope:
        row = await fetch_one(
            """
            SELECT
                COUNT(*) AS total_files,
                COALESCE(SUM(CASE WHEN status = 'ingested' THEN 1 ELSE 0 END), 0) AS ingested_files
            FROM kb_files
            WHERE kb_id = ?
            """,
            (kb_scope.id,),
        )
        where = {"kb_id": kb_scope.id}
        total_vectors = vector_store.count_by_where(where)
        return KBStats(
            total_files=int((row or {}).get("total_files") or 0),
            ingested_files=int((row or {}).get("ingested_files") or 0),
            total_chunks=total_vectors,
            total_vectors=total_vectors,
            sources=vector_store.get_sources(where),
            scope="kb",
            kb_id=kb_scope.id,
            kb_key=kb_scope.key,
            kb_name=kb_scope.name,
            kb_version=kb_scope.kb_version,
            is_default=kb_scope.is_default,
        )

    row = await fetch_one(
        """
        SELECT
            COUNT(*) AS total_files,
            COALESCE(SUM(CASE WHEN status = 'ingested' THEN 1 ELSE 0 END), 0) AS ingested_files
        FROM uploaded_files
        """
    )
    vs_stats = vector_store.get_stats()
    return KBStats(
        total_files=int((row or {}).get("total_files") or 0),
        ingested_files=int((row or {}).get("ingested_files") or 0),
        total_chunks=int(vs_stats["total_vectors"]),
        total_vectors=int(vs_stats["total_vectors"]),
        sources=vector_store.get_sources(),
        scope="global",
        is_default=None,
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    from app.embeddings import is_embeddings_ready, using_hashing_fallback

    hashing = using_hashing_fallback()
    return HealthResponse(
        status="ok",
        llm_loaded=getattr(request.app.state, "llm_loaded", False),
        embeddings_loaded=not hashing,
        embeddings_backend="hashing" if hashing else "sentence-transformers",
        embeddings_ready=is_embeddings_ready(),
        vector_store_ready=getattr(request.app.state, "vector_store_ready", False),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    auth=Depends(get_request_auth),
):
    from app.agent import agent_stream

    _reject_body_auth_override_in_jwt_mode(req)

    request_context = RequestContext(
        request_id=req.request_id or getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
        session_id=req.resolved_session_id,
        kb_id=req.kb_id,
        kb_key=req.kb_key,
        auth=auth,
    )
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None

    async def event_generator():
        async for event in agent_stream(
            query=req.message,
            session_id=req.resolved_session_id,
            lang=req.lang,
            kb_id=req.kb_id,
            kb_key=req.kb_key,
            request_context=request_context,
        ):
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"], ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@app.get("/api/chat/kbs", response_model=list[KnowledgeBaseSummary])
async def chat_visible_kbs(request: Request, auth=Depends(get_request_auth)):
    db = await open_db()
    try:
        return await list_accessible_kbs(
            db,
            auth,
            request_context={"request_id": getattr(request.state, "request_id", None), "auth": auth},
        )
    finally:
        await db.close()


@app.get("/api/me", response_model=CurrentUserProfile)
async def current_user_profile(auth=Depends(get_request_auth)):
    return CurrentUserProfile(
        authenticated=bool(auth.user_id or auth.roles),
        auth_mode=settings.normalized_auth_mode,
        debug_auth_inputs_enabled=settings.normalized_auth_mode == "dev" and settings.allow_header_auth_in_dev,
        user_id=auth.user_id,
        roles=auth.roles,
        channel=auth.channel,
        tenant_id=auth.tenant_id,
        org_id=auth.org_id,
    )


@app.get("/api/kb/stats", response_model=KBStats)
async def kb_stats(
    kb_id: int | None = Query(default=None, ge=1),
    kb_key: str | None = Query(default=None),
    _=Depends(require_admin),
):
    return await _build_stats_response(kb_id=kb_id, kb_key=kb_key)


@app.get("/api/kb/sources", response_model=list[KBSource])
async def kb_sources(
    kb_id: int | None = Query(default=None, ge=1),
    kb_key: str | None = Query(default=None),
    _=Depends(require_admin),
):
    kb_scope = await _resolve_optional_kb_scope(kb_id=kb_id, kb_key=kb_key)
    if kb_scope:
        rows = await fetch_all(
            """
            SELECT
                uf.id,
                uf.original_name,
                uf.file_type,
                kf.chunk_count,
                kf.last_ingest_at
            FROM kb_files kf
            JOIN uploaded_files uf ON uf.id = kf.file_id
            WHERE kf.kb_id = ? AND kf.status = 'ingested'
            ORDER BY kf.last_ingest_at DESC
            """,
            (kb_scope.id,),
        )
        return [
            KBSource(
                source_id=row["id"],
                filename=row["original_name"],
                file_type=row["file_type"],
                chunk_count=int(row.get("chunk_count") or 0),
                ingested_at=row.get("last_ingest_at"),
            )
            for row in rows
        ]

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


@app.get("/api/sources/stats")
async def api_sources_stats(
    kb_id: int | None = Query(default=None, ge=1),
    kb_key: str | None = Query(default=None),
    _=Depends(require_admin),
):
    from app.vector_store import vector_store

    kb_scope = await _resolve_optional_kb_scope(kb_id=kb_id, kb_key=kb_key)
    if kb_scope:
        return vector_store.get_source_stats({"kb_id": kb_scope.id})
    return vector_store.get_source_stats()


@app.get("/api/documents", response_model=list[DocumentSummary])
async def documents(_=Depends(require_admin)):
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
async def admin_chat_logs(
    limit: int = Query(default=settings.chat_log_limit_default, ge=1, le=500),
    _=Depends(require_admin),
):
    rows = await fetch_all(
        """
        SELECT id, session_id, request_id, user_id, roles_json, channel,
               tenant_id, org_id, kb_id, kb_key, mode, top_score, latency_ms, llm_provider,
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
            request_id=row.get("request_id"),
            user_id=row.get("user_id"),
            roles=_parse_roles_json(row.get("roles_json")),
            channel=row.get("channel"),
            tenant_id=row.get("tenant_id"),
            org_id=row.get("org_id"),
            kb_id=row.get("kb_id"),
            kb_key=row.get("kb_key"),
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


@app.get("/api/admin/tool-audit-logs", response_model=list[ToolAuditLogItem])
async def admin_tool_audit_logs(
    limit: int = Query(default=settings.chat_log_limit_default, ge=1, le=500),
    _=Depends(require_admin),
):
    rows = await fetch_all(
        """
        SELECT id, tool_call_id, request_id, session_id, user_id, roles_json, channel,
               tenant_id, org_id, kb_id, kb_key, tool_name, args_json, result_summary,
               tool_status, latency_ms, error_message, created_at
        FROM tool_audit_logs
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )

    return [
        ToolAuditLogItem(
            id=row["id"],
            tool_call_id=row.get("tool_call_id") or "",
            request_id=row.get("request_id"),
            session_id=row.get("session_id"),
            user_id=row.get("user_id"),
            roles=_parse_roles_json(row.get("roles_json")),
            channel=row.get("channel"),
            tenant_id=row.get("tenant_id"),
            org_id=row.get("org_id"),
            kb_id=row.get("kb_id"),
            kb_key=row.get("kb_key"),
            tool_name=row.get("tool_name") or "",
            tool_status=row.get("tool_status") or "",
            args_json=row.get("args_json"),
            result_summary=row.get("result_summary"),
            latency_ms=row.get("latency_ms"),
            error_message=row.get("error_message"),
            created_at=row.get("created_at") or "",
        )
        for row in rows
    ]


@app.get("/api/admin/auth-audit-logs", response_model=list[AuthAuditLogItem])
async def admin_auth_audit_logs(
    limit: int = Query(default=settings.chat_log_limit_default, ge=1, le=500),
    _=Depends(require_admin),
):
    rows = await fetch_all(
        """
        SELECT id, request_id, user_id, roles_json, channel,
               tenant_id, org_id, resource_type, resource_id, action,
               decision, reason, created_at
        FROM auth_audit_logs
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )

    return [
        AuthAuditLogItem(
            id=row["id"],
            request_id=row.get("request_id"),
            user_id=row.get("user_id"),
            roles=_parse_roles_json(row.get("roles_json")),
            channel=row.get("channel"),
            tenant_id=row.get("tenant_id"),
            org_id=row.get("org_id"),
            resource_type=row.get("resource_type") or "",
            resource_id=row.get("resource_id"),
            action=row.get("action") or "",
            decision=row.get("decision") or "",
            reason=row.get("reason"),
            created_at=row.get("created_at") or "",
        )
        for row in rows
    ]


@app.get("/api/admin/tools")
async def admin_tools_registry(_=Depends(require_admin)):
    from app.tools import tool_registry

    return [item.model_dump() for item in tool_registry.list_definitions()]


@app.get("/api/admin/google-drive/sources", response_model=ListGoogleDriveSourcesOutput)
async def admin_list_google_drive_sources(_=Depends(require_admin)):
    from app.drive_sync import list_google_drive_sources

    return list_google_drive_sources()


@app.post("/api/admin/google-drive/sources", response_model=CreateGoogleDriveSourceOutput)
async def admin_create_google_drive_source(payload: CreateGoogleDriveSourceInput, auth=Depends(require_admin)):
    from app.drive_sync import create_google_drive_source

    return await create_google_drive_source(
        kb_id=payload.kb_id,
        kb_key=payload.kb_key,
        name=payload.name,
        folder_id=payload.folder_id,
        shared_drive_id=payload.shared_drive_id,
        recursive=payload.recursive,
        include_patterns=payload.include_patterns,
        exclude_patterns=payload.exclude_patterns,
        supported_mime_types=payload.supported_mime_types,
        delete_policy=payload.delete_policy,
        auth=auth,
    )


@app.post("/api/admin/google-drive/sources/{source_id}/sync", response_model=SyncGoogleDriveSourceOutput)
async def admin_sync_google_drive_source(
    source_id: int,
    force_full: bool = Query(default=False),
    auth=Depends(require_admin),
):
    from app.drive_sync import sync_google_drive_source

    return await sync_google_drive_source(
        source_id,
        triggered_by_user_id=auth.user_id,
        trigger_mode="route",
        force_full=force_full,
    )


@app.get("/api/admin/google-drive/sources/{source_id}/status", response_model=GetGoogleDriveSyncStatusOutput)
async def admin_get_google_drive_sync_status(source_id: int, _=Depends(require_admin)):
    from app.drive_sync import get_google_drive_sync_status

    return get_google_drive_sync_status(source_id)


@app.delete("/api/admin/google-drive/sources/{source_id}", response_model=DeleteGoogleDriveSourceOutput)
async def admin_delete_google_drive_source(
    source_id: int,
    mode: str = Query(default="unlink"),
    _=Depends(require_admin),
):
    from app.drive_sync import delete_google_drive_source

    return delete_google_drive_source(source_id, mode=mode)


@app.get("/api/admin/support-email/messages", response_model=ListSupportEmailsOutput)
async def admin_list_support_email_messages(
    limit: int = Query(default=settings.email_fetch_limit, ge=1, le=100),
    unread_only: bool = Query(default=False),
    sync_first: bool = Query(default=False),
    _=Depends(require_admin),
):
    from app.integrations.support_email import list_support_emails

    return await list_support_emails(limit=limit, unread_only=unread_only, sync_first=sync_first)


@app.get("/api/admin/support-email/messages/{email_id}/thread", response_model=ReadEmailThreadOutput)
async def admin_read_support_email_thread(email_id: int, _=Depends(require_admin)):
    from app.integrations.support_email import read_support_email_thread

    return read_support_email_thread(email_id=email_id)


@app.post("/api/admin/support-email/messages/{email_id}/ticket", response_model=CreateTicketFromEmailOutput)
async def admin_create_ticket_from_email(
    email_id: int,
    payload: CreateTicketFromEmailRequest,
    request: Request,
    auth=Depends(require_admin),
):
    from app.integrations.support_email import create_ticket_from_email

    return create_ticket_from_email(
        email_id=email_id,
        issue_type=payload.issue_type,
        message_override=payload.message_override,
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.post("/api/admin/support-email/messages/{email_id}/reply", response_model=SendEmailReplyOutput)
async def admin_send_support_email_reply(
    email_id: int,
    payload: SendEmailReplyRequest,
    request: Request,
    auth=Depends(require_admin),
):
    from app.integrations.support_email import send_email_reply

    return await send_email_reply(
        email_id=email_id,
        body=payload.body,
        to_address=payload.to_address,
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.post("/api/cache/clear")
async def cache_clear(_=Depends(require_admin)):
    from app.cache import clear_cache

    clear_cache()
    return {"message": "Cache cleared"}


@app.get("/api/cache/stats", response_model=CacheStats)
async def cache_stats_endpoint(_=Depends(require_admin)):
    from app.cache import get_stats

    stats = get_stats()
    return CacheStats(**stats)


@app.get("/api/system", response_model=SystemRuntime)
async def system_info(
    request: Request,
    kb_id: int | None = Query(default=None, ge=1),
    kb_key: str | None = Query(default=None),
    _=Depends(require_admin),
):
    from app.cache import get_stats as get_cache_stats
    from app.llm_client import active_provider_name
    from app.embeddings import using_hashing_fallback, is_embeddings_ready
    from app.vector_store import vector_store

    stats = await _build_stats_response(kb_id=kb_id, kb_key=kb_key)
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
        "scope": {
            "type": stats.scope,
            "kb_id": stats.kb_id,
            "kb_key": stats.kb_key,
            "kb_name": stats.kb_name,
            "kb_version": stats.kb_version,
            "is_default": stats.is_default,
        },
        "agent_runtime": {
            "serving_stack": settings.normalized_agent_serving_stack,
            "brain_mode": settings.normalized_agent_brain_mode,
            "tool_protocol": settings.normalized_agent_tool_protocol,
            "native_tool_calling": settings.agent_native_tool_calling,
            "tool_choice_mode": settings.normalized_agent_tool_choice_mode,
            "native_tool_status": settings.agent_native_tool_status,
            "native_tool_ready": settings.agent_native_tool_ready,
            "native_tool_reason": settings.agent_native_tool_reason,
            "native_tool_warning": settings.agent_native_tool_warning,
            "tool_parser": settings.agent_tool_parser or None,
            "target_model": settings.effective_chat_model or None,
        },
        "embedding_model": settings.embedding_model,
        "embedding_source": settings.effective_embedding_source,
        "embedding_backend": "hashing" if hashing else "sentence-transformers",
        "vector_backend": vs.get("backend"),
        "vector_backend_config": settings.normalized_vector_backend,
        "vector_backend_active": vs.get("backend"),
        "collection_name": vs.get("collection_name"),
        "total_files": stats.total_files,
        "ingested_files": stats.ingested_files,
        "source_count": len(stats.sources),
        "total_vectors": stats.total_vectors,
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
        "llm_model": settings.effective_chat_model,
        "llm_loaded": getattr(request.app.state, "llm_loaded", False),
        "cache_entries": cache.get("total_entries", 0),
        "cache_size_mb": cache.get("size_mb", 0),
        "vector_store_ready": getattr(request.app.state, "vector_store_ready", False),
        "embeddings_loaded": getattr(request.app.state, "embeddings_loaded", False),
        "embeddings_ready": is_embeddings_ready(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/debug/similarity")
async def debug_similarity(
    query: str,
    top_k: int = 10,
    kb_id: int | None = Query(default=None, ge=1),
    kb_key: str | None = Query(default=None),
    _=Depends(require_admin),
):
    from app.rag import _resolve_kb_scope, decide_mode, retrieve

    try:
        kb_scope = _resolve_kb_scope(kb_id=kb_id, kb_key=kb_key)
    except ValueError as err:
        raise HTTPException(404, str(err)) from err
    results = retrieve(query, top_k=top_k, kb_id=kb_scope["id"])
    top_score = float(results[0].get("similarity", 0.0)) if results else 0.0
    return {
        "query": query,
        "kb": kb_scope,
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
                "lang": item.get("lang", ""),
                "preview": item.get("text", "")[:200],
            }
            for idx, item in enumerate(results)
        ],
    }


@app.get("/api/debug/retrieval")
async def debug_retrieval(
    query: str,
    top_k: int = 10,
    kb_id: int | None = Query(default=None, ge=1),
    kb_key: str | None = Query(default=None),
    _=Depends(require_admin),
):
    """
    Rich retrieval debug: returns top_k results with score, lang, category,
    source, bm25_score, and content snippet. Useful for threshold calibration.
    """
    from app.rag import _resolve_kb_scope, decide_mode, retrieve
    from app.lang import detect_language

    try:
        kb_scope = _resolve_kb_scope(kb_id=kb_id, kb_key=kb_key)
    except ValueError as err:
        raise HTTPException(404, str(err)) from err
    results = retrieve(query, top_k=top_k, kb_id=kb_scope["id"])
    top_score = float(results[0].get("similarity", 0.0)) if results else 0.0
    detected_lang = detect_language(query)

    return {
        "query": query,
        "kb": kb_scope,
        "detected_lang": detected_lang,
        "top_score": round(top_score, 4),
        "predicted_mode": decide_mode(top_score),
        "thresholds": {
            "good": settings.threshold_good,
            "low": settings.threshold_low,
            "min": settings.min_similarity_threshold,
        },
        "results": [
            {
                "rank": idx + 1,
                "similarity": round(float(item.get("similarity", 0.0)), 4),
                "bm25_score": round(float(item.get("bm25_score", 0.0)), 4),
                "lang": item.get("lang", ""),
                "category": item.get("category", ""),
                "filename": item.get("filename", ""),
                "source": f"{item.get('filename', '')} row {item.get('row_num', '')}".strip(),
                "snippet": (item.get("text") or "")[:300],
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
