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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import AppStatus, EventSourceResponse

from app.access_management import (
    AppUserItem,
    ListAppUsersOutput,
    ProductionReadinessOutput,
    UpdateAppUserInput,
    UpsertAppUserInput,
    build_production_readiness,
    list_app_users,
    update_app_user,
    upsert_app_user,
)
from app.analytics import build_analytics_dashboard
from app.auth import (
    get_request_auth,
    require_admin,
    require_analytics_role,
    require_approver_role,
    require_audit_role,
    require_integration_role,
    require_knowledge_role,
    require_operations_role,
    require_support_role,
    require_system_role,
)
from app.background_jobs import (
    BackgroundJobDecisionInput,
    BackgroundJobItem,
    ListBackgroundJobsOutput,
    background_worker_loop,
    cancel_background_job,
    enqueue_background_job,
    get_background_job,
    list_background_jobs,
    retry_background_job,
)
from app.case_timeline import CaseTimelineOutput, build_case_timeline
from app.config import settings
from app.database import fetch_all, fetch_one, init_db
from app.evaluations import create_agent_eval_run, get_agent_eval_run, list_agent_eval_runs
from app.feedback import feedback_summary, list_chat_feedback, submit_chat_feedback
from app.ingest import router as ingest_router
from app.kb import router as kb_router
from app.kb_service import list_accessible_kbs, open_db, resolve_kb_scope
from app.models import (
    AnalyticsDashboardOutput,
    AgentEvalRunDetail,
    AuthAuditLogItem,
    CacheStats,
    ChatFeedbackItem,
    ChatLogItem,
    ChatRequest,
    CreateAgentEvalRunInput,
    CurrentUserProfile,
    DocumentSummary,
    FeedbackSummaryOutput,
    HealthResponse,
    KnowledgeBaseSummary,
    KBSource,
    KBStats,
    ListAgentEvalRunsOutput,
    ListChatFeedbackOutput,
    RequestContext,
    SubmitChatFeedbackInput,
    SystemRuntime,
    ToolAuditLogItem,
)
from app.mcp_server import build_mcp_status, handle_mcp_request
from app.notifications import (
    CreateNotificationInput,
    ListNotificationsOutput,
    ListWebhookDeliveriesOutput,
    ListWebhookSubscriptionsOutput,
    NotificationItem,
    UpsertWebhookSubscriptionInput,
    WebhookDeliveryItem,
    WebhookSubscriptionItem,
    create_webhook_subscription,
    create_notification,
    delete_webhook_subscription,
    deliver_due_webhooks_once,
    list_notifications,
    list_webhook_deliveries,
    list_webhook_subscriptions,
    mark_all_notifications_read,
    mark_notification_read,
    retry_webhook_delivery,
    test_webhook_subscription,
    update_webhook_subscription,
)
from app.observability import configure_tracing, shutdown_tracing, trace_span, tracing_status
from app.pending_actions import (
    CreatePendingActionInput,
    ListPendingActionsOutput,
    PendingActionDecisionInput,
    PendingActionItem,
    approve_pending_action,
    create_pending_action,
    draft_drive_delete_action,
    draft_drive_full_sync_action,
    draft_email_reply_action,
    execute_pending_action,
    list_pending_actions,
    reject_pending_action,
)
from app.rate_limit import rate_limit_error_payload, rate_limit_headers, rate_limiter
from app.scheduled_sync import (
    ListSyncSchedulesOutput,
    SyncScheduleItem,
    UpdateSyncScheduleInput,
    UpsertSyncScheduleInput,
    delete_sync_schedule,
    list_sync_schedules,
    scheduled_sync_loop,
    update_sync_schedule,
    upsert_sync_schedule,
)
from app.support_workflows import (
    add_ticket_note,
    assign_ticket,
    classify_ticket,
    escalate_ticket,
    get_support_ticket,
    get_support_ticket_by_code,
    get_ticket_context,
    handle_email_case,
    handle_ticket_case,
    list_support_tickets,
    list_ticket_notes,
    process_sla_breaches,
    update_ticket_status,
    workflow_summary,
)
from app.support_workflows.schemas import (
    AddTicketNoteInput,
    AssignTicketInput,
    CaseClassification,
    CreateSupportTicketInput,
    ListSupportTicketNotesOutput,
    ListSupportTicketsOutput,
    SlaMonitorResult,
    SupportTicketItem,
    SupportTicketNoteItem,
    UpdateTicketStatusInput,
    WorkflowResult,
)
from app.support_ticket_service import create_support_ticket
from app.support_drafts import SupportDraftReplyInput, SupportDraftReplyOutput, generate_support_draft_reply
from app.upload import router as upload_router
from app.workflow_engine import (
    ListWorkflowRunsOutput,
    WorkflowDecisionInput,
    WorkflowRunDetail,
    cancel_workflow_run,
    get_workflow_run,
    list_workflow_runs,
    resume_workflow_run,
    retry_workflow_run,
)
from app.tools.drive_tools import (
    CreateGoogleDriveSourceInput,
    CreateGoogleDriveSourceOutput,
    GetGoogleDriveSyncStatusOutput,
    ListGoogleDriveSourcesOutput,
)
from app.tools.email_tools import (
    CreateTicketFromEmailRequest,
    CreateTicketFromEmailOutput,
    ListSupportEmailsOutput,
    ReadEmailThreadOutput,
    SendEmailReplyRequest,
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
    settings.validate_runtime_settings()
    configure_tracing()
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

    app.state.background_worker_task = None
    app.state.scheduled_sync_task = None
    if settings.background_worker_enabled:
        app.state.background_worker_task = asyncio.create_task(background_worker_loop())
        logger.info("Background job worker task started")
        if settings.scheduled_sync_enabled:
            app.state.scheduled_sync_task = asyncio.create_task(scheduled_sync_loop())
            logger.info("Scheduled sync task started")
    else:
        logger.info("Background job worker disabled for this API process")

    logger.info("Startup complete")
    logger.info("=" * 60)

    try:
        yield
    finally:
        worker_task = getattr(app.state, "background_worker_task", None)
        scheduler_task = getattr(app.state, "scheduled_sync_task", None)
        if scheduler_task:
            scheduler_task.cancel()
            try:
                await scheduler_task
            except asyncio.CancelledError:
                pass
        if worker_task:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
        shutdown_tracing()
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
    with trace_span(
        "http.request",
        {
            "http.request.method": request.method,
            "url.path": request.url.path,
            "url.scheme": request.url.scheme,
            "server.address": request.url.hostname,
            "app.request_id": request_id,
        },
        carrier=dict(request.headers),
    ) as span:
        rate_limit_decision = rate_limiter.check(request)
        if rate_limit_decision is not None:
            span.set_attribute("app.rate_limit.policy", rate_limit_decision.policy)
            span.set_attribute("app.rate_limit.remaining", rate_limit_decision.remaining)
        if rate_limit_decision is not None and not rate_limit_decision.allowed:
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            response = JSONResponse(
                status_code=429,
                content=rate_limit_error_payload(rate_limit_decision),
                headers=rate_limit_headers(rate_limit_decision),
            )
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
            span.set_attribute("http.response.status_code", 429)
            span.set_attribute("app.response_time_ms", int(elapsed * 1000))
            span.set_attribute("app.rate_limit.blocked", True)
            return response

        response = await call_next(request)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
        for header, value in rate_limit_headers(rate_limit_decision).items():
            response.headers[header] = value
        span.set_attribute("http.response.status_code", response.status_code)
        span.set_attribute("app.response_time_ms", int(elapsed * 1000))
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
    readiness = build_production_readiness()
    issues = [
        {
            "level": "error" if item.status == "fail" else "warn",
            "component": item.key,
            "message": item.message,
            "fix": item.fix,
        }
        for item in readiness.checks
        if item.status in {"warn", "fail"}
    ]
    return HealthResponse(
        status="ok",
        llm_loaded=getattr(request.app.state, "llm_loaded", False),
        embeddings_loaded=not hashing,
        embeddings_backend="hashing" if hashing else "sentence-transformers",
        embeddings_ready=is_embeddings_ready(),
        vector_store_ready=getattr(request.app.state, "vector_store_ready", False),
        ready_for_chat=readiness.ready_for_chat,
        ready_for_production=readiness.ready_for_production,
        setup_complete=readiness.setup_complete,
        issues=issues,
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
        debug_auth_inputs_enabled=(
            settings.normalized_auth_mode == "dev"
            and settings.allow_header_auth_in_dev
            and settings.auth_header_debug_enabled
        ),
        access_management_enabled=settings.access_management_enabled,
        access_role_mode=settings.normalized_access_management_role_mode,
        user_id=auth.user_id,
        roles=auth.roles,
        channel=auth.channel,
        tenant_id=auth.tenant_id,
        org_id=auth.org_id,
    )


@app.get("/api/admin/access/users", response_model=ListAppUsersOutput)
async def admin_list_app_users(
    query: str | None = Query(default=None, max_length=200),
    status: str = Query(default="all", pattern="^(all|active|inactive)$"),
    limit: int = Query(default=100, ge=1, le=500),
    _=Depends(require_admin),
):
    return list_app_users(query=query, status=status, limit=limit)


@app.post("/api/admin/access/users", response_model=AppUserItem)
async def admin_upsert_app_user(payload: UpsertAppUserInput, auth=Depends(require_admin)):
    return upsert_app_user(payload, auth=auth)


@app.patch("/api/admin/access/users/{user_id}", response_model=AppUserItem)
async def admin_update_app_user(user_id: str, payload: UpdateAppUserInput, auth=Depends(require_admin)):
    return update_app_user(user_id=user_id, payload=payload, auth=auth)


@app.get("/api/admin/readiness", response_model=ProductionReadinessOutput)
async def admin_production_readiness(_=Depends(require_system_role)):
    return build_production_readiness()


@app.get("/api/kb/stats", response_model=KBStats)
async def kb_stats(
    kb_id: int | None = Query(default=None, ge=1),
    kb_key: str | None = Query(default=None),
    _=Depends(require_knowledge_role),
):
    return await _build_stats_response(kb_id=kb_id, kb_key=kb_key)


@app.get("/api/kb/sources", response_model=list[KBSource])
async def kb_sources(
    kb_id: int | None = Query(default=None, ge=1),
    kb_key: str | None = Query(default=None),
    _=Depends(require_knowledge_role),
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
    _=Depends(require_knowledge_role),
):
    from app.vector_store import vector_store

    kb_scope = await _resolve_optional_kb_scope(kb_id=kb_id, kb_key=kb_key)
    if kb_scope:
        return vector_store.get_source_stats({"kb_id": kb_scope.id})
    return vector_store.get_source_stats()


@app.get("/api/documents", response_model=list[DocumentSummary])
async def documents(_=Depends(require_knowledge_role)):
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


@app.post("/api/feedback/chat", response_model=ChatFeedbackItem)
async def api_submit_chat_feedback(
    payload: SubmitChatFeedbackInput,
    request: Request,
    auth=Depends(get_request_auth),
):
    return submit_chat_feedback(
        payload,
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


def _require_internal_user(auth) -> None:
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="Authentication required")


def _is_admin_auth(auth) -> bool:
    return any(str(role).lower() == "admin" for role in auth.roles)


def _assert_ticket_visible_to_user(ticket: dict, auth) -> None:
    if _is_admin_auth(auth):
        return
    if ticket.get("created_by_user_id") != auth.user_id:
        raise HTTPException(status_code=403, detail="Support ticket access denied")


@app.get("/api/support-tickets", response_model=ListSupportTicketsOutput)
async def list_my_support_tickets(
    status: str | None = Query(default=None),
    workflow_status: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    auth=Depends(get_request_auth),
):
    _require_internal_user(auth)
    return list_support_tickets(
        status=status,
        workflow_status=workflow_status,
        created_by_user_id=None if _is_admin_auth(auth) else auth.user_id,
        limit=limit,
    )


@app.post("/api/support-tickets", response_model=SupportTicketItem)
async def create_my_support_ticket(
    payload: CreateSupportTicketInput,
    request: Request,
    auth=Depends(get_request_auth),
):
    _require_internal_user(auth)
    result = create_support_ticket(
        issue_type=payload.issue_type,
        message=payload.message,
        contact=payload.contact,
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            kb_id=payload.kb_id,
            kb_key=payload.kb_key,
            auth=auth,
        ),
    )
    return get_support_ticket_by_code(result["ticket_code"])


@app.get("/api/support-tickets/{ticket_id}", response_model=SupportTicketItem)
async def get_my_support_ticket(ticket_id: int, auth=Depends(get_request_auth)):
    _require_internal_user(auth)
    try:
        ticket = get_support_ticket(ticket_id)
    except ValueError as err:
        raise HTTPException(status_code=404, detail="Support ticket not found") from err
    _assert_ticket_visible_to_user(ticket, auth)
    return ticket


@app.get("/api/support-tickets/{ticket_id}/notes", response_model=ListSupportTicketNotesOutput)
async def list_my_support_ticket_notes(ticket_id: int, auth=Depends(get_request_auth)):
    _require_internal_user(auth)
    try:
        ticket = get_support_ticket(ticket_id)
    except ValueError as err:
        raise HTTPException(status_code=404, detail="Support ticket not found") from err
    _assert_ticket_visible_to_user(ticket, auth)
    return list_ticket_notes(ticket_id, visibility="public")


@app.post("/api/support-tickets/{ticket_id}/notes", response_model=SupportTicketNoteItem)
async def add_my_support_ticket_note(
    ticket_id: int,
    payload: AddTicketNoteInput,
    request: Request,
    auth=Depends(get_request_auth),
):
    _require_internal_user(auth)
    try:
        ticket = get_support_ticket(ticket_id)
    except ValueError as err:
        raise HTTPException(status_code=404, detail="Support ticket not found") from err
    _assert_ticket_visible_to_user(ticket, auth)
    if not _is_admin_auth(auth) and (ticket.get("workflow_status") or ticket.get("status")) == "closed":
        raise HTTPException(status_code=409, detail="Support ticket is closed")
    context = RequestContext(
        request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
        auth=auth,
    )
    note = add_ticket_note(
        ticket_id,
        AddTicketNoteInput(
            body=payload.body,
            note_type="customer_reply",
            visibility="public",
            metadata={**payload.metadata, "source": "portal"},
        ),
        context=context,
    )
    try:
        create_notification(
            event_type="support.employee_replied",
            severity="warning",
            title=f"Employee replied: {ticket.get('ticket_code') or ticket_id}",
            message=payload.body[:500],
            entity_type="support_ticket",
            entity_id=ticket_id,
            payload={"ticket_code": ticket.get("ticket_code")},
            context=context,
        )
    except Exception:
        logger.exception("Failed to create employee reply notification")
    if (ticket.get("workflow_status") or ticket.get("status")) in {"waiting_customer", "resolved", "closed"}:
        update_ticket_status(
            ticket_id,
            UpdateTicketStatusInput(
                status="open",
                note="Employee replied in portal; case reopened for support review.",
            ),
            context=context,
        )
    return note


@app.get("/api/admin/notifications", response_model=ListNotificationsOutput)
async def admin_list_notifications(
    status: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    _=Depends(require_operations_role),
):
    return list_notifications(status=status, severity=severity, limit=limit)


@app.post("/api/admin/notifications", response_model=NotificationItem)
async def admin_create_notification(
    payload: CreateNotificationInput,
    request: Request,
    auth=Depends(require_operations_role),
):
    return create_notification(
        event_type=payload.event_type,
        severity=payload.severity,
        title=payload.title,
        message=payload.message,
        entity_type=payload.entity_type,
        entity_id=payload.entity_id,
        payload=payload.payload,
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.post("/api/admin/notifications/{notification_id}/read", response_model=NotificationItem)
async def admin_mark_notification_read(notification_id: int, auth=Depends(require_operations_role)):
    try:
        return mark_notification_read(notification_id, auth=auth)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@app.post("/api/admin/notifications/read-all", response_model=ListNotificationsOutput)
async def admin_mark_all_notifications_read(auth=Depends(require_operations_role)):
    return mark_all_notifications_read(auth=auth)


@app.get("/api/admin/webhook-subscriptions", response_model=ListWebhookSubscriptionsOutput)
async def admin_list_webhook_subscriptions(
    include_disabled: bool = Query(default=True),
    limit: int = Query(default=100, ge=1, le=200),
    _=Depends(require_operations_role),
):
    return list_webhook_subscriptions(include_disabled=include_disabled, limit=limit)


@app.post("/api/admin/webhook-subscriptions", response_model=WebhookSubscriptionItem)
async def admin_create_webhook_subscription(payload: UpsertWebhookSubscriptionInput, auth=Depends(require_operations_role)):
    return create_webhook_subscription(payload, auth=auth)


@app.put("/api/admin/webhook-subscriptions/{subscription_id}", response_model=WebhookSubscriptionItem)
async def admin_update_webhook_subscription(
    subscription_id: int,
    payload: UpsertWebhookSubscriptionInput,
    _=Depends(require_operations_role),
):
    try:
        return update_webhook_subscription(subscription_id, payload)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@app.delete("/api/admin/webhook-subscriptions/{subscription_id}")
async def admin_delete_webhook_subscription(subscription_id: int, _=Depends(require_operations_role)):
    try:
        return delete_webhook_subscription(subscription_id)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@app.post("/api/admin/webhook-subscriptions/{subscription_id}/test")
async def admin_test_webhook_subscription(subscription_id: int, auth=Depends(require_operations_role)):
    try:
        return test_webhook_subscription(subscription_id, auth=auth)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@app.get("/api/admin/webhook-deliveries", response_model=ListWebhookDeliveriesOutput)
async def admin_list_webhook_deliveries(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    _=Depends(require_operations_role),
):
    return list_webhook_deliveries(status=status, limit=limit)


@app.post("/api/admin/webhook-deliveries/run")
async def admin_run_webhook_deliveries(_=Depends(require_operations_role)):
    delivered = await deliver_due_webhooks_once(limit=25)
    return {"delivered": delivered}


@app.post("/api/admin/webhook-deliveries/{delivery_id}/retry", response_model=WebhookDeliveryItem)
async def admin_retry_webhook_delivery(delivery_id: int, _=Depends(require_operations_role)):
    try:
        return await retry_webhook_delivery(delivery_id)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@app.get("/api/admin/chat-logs", response_model=list[ChatLogItem])
async def admin_chat_logs(
    limit: int = Query(default=settings.chat_log_limit_default, ge=1, le=500),
    _=Depends(require_audit_role),
):
    rows = await fetch_all(
        """
        SELECT cl.id, cl.session_id, cl.request_id, cl.user_id, cl.roles_json, cl.channel,
               cl.tenant_id, cl.org_id, cl.kb_id, cl.kb_key, cl.mode, cl.top_score, cl.latency_ms, cl.llm_provider,
               cl.user_message, cl.answer_text, cl.created_at,
               COALESCE(SUM(CASE WHEN cf.rating = 'up' THEN 1 ELSE 0 END), 0) AS feedback_up,
               COALESCE(SUM(CASE WHEN cf.rating = 'down' THEN 1 ELSE 0 END), 0) AS feedback_down
        FROM chat_logs cl
        LEFT JOIN chat_feedback cf ON cf.chat_log_id = cl.id
        GROUP BY cl.id
        ORDER BY cl.created_at DESC, cl.id DESC
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
            feedback_up=int(row.get("feedback_up") or 0),
            feedback_down=int(row.get("feedback_down") or 0),
        )
        for row in rows
    ]


@app.get("/api/admin/feedback", response_model=ListChatFeedbackOutput)
async def admin_list_chat_feedback(
    rating: str | None = Query(default=None),
    kb_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=500),
    _=Depends(require_analytics_role),
):
    return list_chat_feedback(rating=rating, kb_id=kb_id, limit=limit)


@app.get("/api/admin/feedback/summary", response_model=FeedbackSummaryOutput)
async def admin_chat_feedback_summary(_=Depends(require_analytics_role)):
    return feedback_summary()


@app.get("/api/admin/analytics", response_model=AnalyticsDashboardOutput)
async def admin_analytics_dashboard(
    days: int = Query(default=7, ge=1, le=90),
    kb_id: int | None = Query(default=None, ge=1),
    _=Depends(require_analytics_role),
):
    return await build_analytics_dashboard(days=days, kb_id=kb_id)


@app.post("/api/admin/evaluations/runs", response_model=AgentEvalRunDetail)
async def admin_create_agent_eval_run(
    payload: CreateAgentEvalRunInput,
    auth=Depends(require_analytics_role),
):
    return create_agent_eval_run(payload, auth=auth)


@app.get("/api/admin/evaluations/runs", response_model=ListAgentEvalRunsOutput)
async def admin_list_agent_eval_runs(
    limit: int = Query(default=20, ge=1, le=100),
    _=Depends(require_analytics_role),
):
    return list_agent_eval_runs(limit=limit)


@app.get("/api/admin/evaluations/runs/{run_id}", response_model=AgentEvalRunDetail)
async def admin_get_agent_eval_run(
    run_id: int,
    limit: int = Query(default=100, ge=1, le=500),
    _=Depends(require_analytics_role),
):
    return get_agent_eval_run(run_id, limit=limit)


@app.get("/api/admin/tool-audit-logs", response_model=list[ToolAuditLogItem])
async def admin_tool_audit_logs(
    limit: int = Query(default=settings.chat_log_limit_default, ge=1, le=500),
    _=Depends(require_audit_role),
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
    _=Depends(require_audit_role),
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
async def admin_tools_registry(_=Depends(require_system_role)):
    from app.tools import tool_registry

    return [item.model_dump() for item in tool_registry.list_definitions()]


@app.post("/mcp")
async def mcp_json_rpc_endpoint(request: Request):
    return await handle_mcp_request(request)


@app.get("/api/admin/mcp/status")
async def admin_mcp_status(_=Depends(require_integration_role)):
    return await build_mcp_status()


@app.get("/api/admin/google-drive/sources", response_model=ListGoogleDriveSourcesOutput)
async def admin_list_google_drive_sources(_=Depends(require_integration_role)):
    from app.drive_sync import list_google_drive_sources

    return list_google_drive_sources()


@app.post("/api/admin/google-drive/sources", response_model=CreateGoogleDriveSourceOutput)
async def admin_create_google_drive_source(payload: CreateGoogleDriveSourceInput, auth=Depends(require_integration_role)):
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


@app.post("/api/admin/google-drive/sources/{source_id}/sync")
async def admin_sync_google_drive_source(
    source_id: int,
    request: Request,
    force_full: bool = Query(default=False),
    auth=Depends(require_integration_role),
):
    if force_full:
        return draft_drive_full_sync_action(
            source_id=source_id,
            force_full=force_full,
            context=RequestContext(
                request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
                auth=auth,
            ),
        )

    return enqueue_background_job(
        job_type="google_drive_sync",
        payload={"source_id": source_id, "force_full": force_full},
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.get("/api/admin/google-drive/sources/{source_id}/status", response_model=GetGoogleDriveSyncStatusOutput)
async def admin_get_google_drive_sync_status(source_id: int, _=Depends(require_integration_role)):
    from app.drive_sync import get_google_drive_sync_status

    return get_google_drive_sync_status(source_id)


@app.delete("/api/admin/google-drive/sources/{source_id}", response_model=PendingActionItem)
async def admin_delete_google_drive_source(
    source_id: int,
    request: Request,
    mode: str = Query(default="unlink"),
    auth=Depends(require_integration_role),
):
    return draft_drive_delete_action(
        source_id=source_id,
        mode=mode,
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.get("/api/admin/support-email/messages", response_model=ListSupportEmailsOutput)
async def admin_list_support_email_messages(
    request: Request,
    limit: int = Query(default=settings.email_fetch_limit, ge=1, le=100),
    unread_only: bool = Query(default=False),
    sync_first: bool = Query(default=False),
    auth=Depends(require_support_role),
):
    from app.integrations.support_email import list_support_emails

    payload = await list_support_emails(limit=limit, unread_only=unread_only, sync_first=False)
    if sync_first:
        payload["sync_job"] = enqueue_background_job(
            job_type="support_email_sync",
            payload={"limit": limit, "unread_only": unread_only},
            context=RequestContext(
                request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
                auth=auth,
            ),
        )
    return payload


@app.post("/api/admin/support-email/sync", response_model=BackgroundJobItem)
async def admin_sync_support_email_messages(
    request: Request,
    limit: int = Query(default=settings.email_fetch_limit, ge=1, le=100),
    unread_only: bool = Query(default=False),
    auth=Depends(require_support_role),
):
    return enqueue_background_job(
        job_type="support_email_sync",
        payload={"limit": limit, "unread_only": unread_only},
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.get("/api/admin/sync-schedules", response_model=ListSyncSchedulesOutput)
async def admin_list_sync_schedules(
    schedule_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=200),
    _=Depends(require_operations_role),
):
    return list_sync_schedules(schedule_type=schedule_type, limit=limit)


@app.post("/api/admin/sync-schedules", response_model=SyncScheduleItem)
async def admin_upsert_sync_schedule(payload: UpsertSyncScheduleInput, auth=Depends(require_integration_role)):
    return upsert_sync_schedule(payload, auth=auth)


@app.patch("/api/admin/sync-schedules/{schedule_id}", response_model=SyncScheduleItem)
async def admin_update_sync_schedule(
    schedule_id: int,
    payload: UpdateSyncScheduleInput,
    _=Depends(require_integration_role),
):
    return update_sync_schedule(schedule_id, payload)


@app.delete("/api/admin/sync-schedules/{schedule_id}", response_model=SyncScheduleItem)
async def admin_delete_sync_schedule(schedule_id: int, _=Depends(require_integration_role)):
    return delete_sync_schedule(schedule_id)


@app.get("/api/admin/support-email/messages/{email_id}/thread", response_model=ReadEmailThreadOutput)
async def admin_read_support_email_thread(email_id: int, _=Depends(require_support_role)):
    from app.integrations.support_email import read_support_email_thread

    return read_support_email_thread(email_id=email_id)


@app.post("/api/admin/support-email/messages/{email_id}/ticket", response_model=CreateTicketFromEmailOutput)
async def admin_create_ticket_from_email(
    email_id: int,
    payload: CreateTicketFromEmailRequest,
    request: Request,
    auth=Depends(require_support_role),
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


@app.post("/api/admin/support-email/messages/{email_id}/reply", response_model=PendingActionItem)
async def admin_send_support_email_reply(
    email_id: int,
    payload: SendEmailReplyRequest,
    request: Request,
    auth=Depends(require_support_role),
):
    return draft_email_reply_action(
        email_id=email_id,
        body=payload.body,
        to_address=payload.to_address,
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.get("/api/admin/support-workflows/summary")
async def admin_support_workflow_summary(_=Depends(require_support_role)):
    return workflow_summary()


@app.post("/api/admin/support-workflows/sla/monitor", response_model=SlaMonitorResult)
async def admin_monitor_support_sla(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    auth=Depends(require_support_role),
):
    return process_sla_breaches(
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
        limit=limit,
    )


@app.post("/api/admin/support-workflows/sla/enqueue", response_model=BackgroundJobItem)
async def admin_enqueue_support_sla_monitor(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    auth=Depends(require_support_role),
):
    return enqueue_background_job(
        job_type="support_sla_monitor",
        payload={"limit": limit},
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.get("/api/admin/support-tickets", response_model=ListSupportTicketsOutput)
async def admin_list_support_tickets(
    status: str | None = Query(default=None),
    workflow_status: str | None = Query(default=None),
    priority: str | None = Query(default=None),
    assigned_user_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    _=Depends(require_support_role),
):
    return list_support_tickets(
        status=status,
        workflow_status=workflow_status,
        priority=priority,
        assigned_user_id=assigned_user_id,
        limit=limit,
    )


@app.get("/api/admin/support-tickets/{ticket_id}", response_model=SupportTicketItem)
async def admin_get_support_ticket(ticket_id: int, _=Depends(require_support_role)):
    return get_support_ticket(ticket_id)


@app.get("/api/admin/support-tickets/{ticket_id}/notes", response_model=ListSupportTicketNotesOutput)
async def admin_list_support_ticket_notes(ticket_id: int, _=Depends(require_support_role)):
    return list_ticket_notes(ticket_id)


@app.post("/api/admin/support-tickets/{ticket_id}/notes", response_model=SupportTicketNoteItem)
async def admin_add_support_ticket_note(
    ticket_id: int,
    payload: AddTicketNoteInput,
    request: Request,
    auth=Depends(require_support_role),
):
    return add_ticket_note(
        ticket_id,
        payload,
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.post("/api/admin/support-tickets/{ticket_id}/assign", response_model=SupportTicketItem)
async def admin_assign_support_ticket(
    ticket_id: int,
    payload: AssignTicketInput,
    request: Request,
    auth=Depends(require_support_role),
):
    return assign_ticket(
        ticket_id,
        payload,
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.post("/api/admin/support-tickets/{ticket_id}/status", response_model=SupportTicketItem)
async def admin_update_support_ticket_status(
    ticket_id: int,
    payload: UpdateTicketStatusInput,
    request: Request,
    auth=Depends(require_support_role),
):
    return update_ticket_status(
        ticket_id,
        payload,
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.post("/api/admin/support-workflows/tickets/{ticket_id}/classify", response_model=CaseClassification)
async def admin_classify_support_ticket(ticket_id: int, _=Depends(require_support_role)):
    return classify_ticket(ticket_id)


@app.post("/api/admin/support-workflows/tickets/{ticket_id}/handle", response_model=WorkflowResult)
async def admin_handle_support_ticket_workflow(ticket_id: int, request: Request, auth=Depends(require_support_role)):
    return await handle_ticket_case(
        ticket_id,
        RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.post("/api/admin/support-workflows/tickets/{ticket_id}/enqueue", response_model=BackgroundJobItem)
async def admin_enqueue_support_ticket_workflow(ticket_id: int, request: Request, auth=Depends(require_support_role)):
    return enqueue_background_job(
        job_type="support_ticket_workflow",
        payload={"ticket_id": ticket_id},
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.post("/api/admin/support-workflows/emails/{email_id}/handle", response_model=WorkflowResult)
async def admin_handle_support_email_workflow(email_id: int, request: Request, auth=Depends(require_support_role)):
    return await handle_email_case(
        email_id,
        RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.post("/api/admin/support-workflows/emails/{email_id}/enqueue", response_model=BackgroundJobItem)
async def admin_enqueue_support_email_workflow(email_id: int, request: Request, auth=Depends(require_support_role)):
    return enqueue_background_job(
        job_type="support_email_workflow",
        payload={"email_id": email_id},
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.get("/api/admin/support-tickets/{ticket_id}/context")
async def admin_get_support_ticket_context(ticket_id: int, _=Depends(require_support_role)):
    return get_ticket_context(ticket_id)


@app.get("/api/admin/support-tickets/{ticket_id}/timeline", response_model=CaseTimelineOutput)
async def admin_get_support_ticket_timeline(ticket_id: int, _=Depends(require_support_role)):
    try:
        return build_case_timeline(ticket_id)
    except ValueError as err:
        raise HTTPException(status_code=404, detail="Support ticket not found") from err


@app.post("/api/admin/support-tickets/{ticket_id}/draft-reply", response_model=SupportDraftReplyOutput)
async def admin_generate_support_ticket_draft_reply(
    ticket_id: int,
    payload: SupportDraftReplyInput,
    request: Request,
    auth=Depends(require_support_role),
):
    try:
        return generate_support_draft_reply(
            ticket_id,
            payload,
            context=RequestContext(
                request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
                auth=auth,
            ),
        )
    except ValueError as err:
        raise HTTPException(status_code=404, detail="Support ticket not found") from err


@app.post("/api/admin/support-tickets/{ticket_id}/escalate")
async def admin_escalate_support_ticket(
    ticket_id: int,
    request: Request,
    payload: PendingActionDecisionInput | None = None,
    auth=Depends(require_support_role),
):
    return escalate_ticket(
        ticket_id,
        reason=(payload.note if payload else None) or "Manual escalation requested by admin.",
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.get("/api/admin/pending-actions", response_model=ListPendingActionsOutput)
async def admin_list_pending_actions(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    _=Depends(require_operations_role),
):
    return list_pending_actions(status=status, limit=limit)


@app.get("/api/admin/background-jobs", response_model=ListBackgroundJobsOutput)
async def admin_list_background_jobs(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    _=Depends(require_operations_role),
):
    return list_background_jobs(status=status, limit=limit)


@app.get("/api/admin/background-jobs/{job_id}", response_model=BackgroundJobItem)
async def admin_get_background_job(job_id: str, _=Depends(require_operations_role)):
    return get_background_job(job_id)


@app.post("/api/admin/background-jobs/{job_id}/cancel", response_model=BackgroundJobItem)
async def admin_cancel_background_job(
    job_id: str,
    payload: BackgroundJobDecisionInput | None = None,
    _=Depends(require_operations_role),
):
    return cancel_background_job(job_id, reason=payload.reason if payload else None)


@app.post("/api/admin/background-jobs/{job_id}/retry", response_model=BackgroundJobItem)
async def admin_retry_background_job(job_id: str, _=Depends(require_operations_role)):
    return retry_background_job(job_id)


@app.get("/api/admin/workflows/runs", response_model=ListWorkflowRunsOutput)
async def admin_list_workflow_runs(
    status: str | None = Query(default=None),
    entity_type: str | None = Query(default=None),
    entity_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    _=Depends(require_operations_role),
):
    return list_workflow_runs(status=status, entity_type=entity_type, entity_id=entity_id, limit=limit)


@app.get("/api/admin/workflows/runs/{run_id}", response_model=WorkflowRunDetail)
async def admin_get_workflow_run(run_id: int, _=Depends(require_operations_role)):
    try:
        return get_workflow_run(run_id)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@app.post("/api/admin/workflows/runs/{run_id}/cancel", response_model=WorkflowRunDetail)
async def admin_cancel_workflow_run(
    run_id: int,
    payload: WorkflowDecisionInput | None = None,
    auth=Depends(require_operations_role),
):
    try:
        return cancel_workflow_run(run_id, reason=payload.reason if payload else None, auth=auth)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


@app.post("/api/admin/workflows/runs/{run_id}/retry", response_model=WorkflowRunDetail)
async def admin_retry_workflow_run(run_id: int, request: Request, auth=Depends(require_operations_role)):
    try:
        return await retry_workflow_run(
            run_id,
            context=RequestContext(
                request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
                auth=auth,
            ),
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


@app.post("/api/admin/workflows/runs/{run_id}/resume", response_model=WorkflowRunDetail)
async def admin_resume_workflow_run(run_id: int, request: Request, auth=Depends(require_operations_role)):
    try:
        return await resume_workflow_run(
            run_id,
            context=RequestContext(
                request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
                auth=auth,
            ),
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


@app.post("/api/admin/pending-actions", response_model=PendingActionItem)
async def admin_create_pending_action(
    payload: CreatePendingActionInput,
    request: Request,
    auth=Depends(require_approver_role),
):
    return create_pending_action(
        action_type=payload.action_type,
        risk_level=payload.risk_level,
        title=payload.title,
        summary=payload.summary,
        payload=payload.payload,
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.post("/api/admin/pending-actions/{action_id}/approve", response_model=PendingActionItem)
async def admin_approve_pending_action(
    action_id: int,
    _payload: PendingActionDecisionInput | None = None,
    auth=Depends(require_approver_role),
):
    return approve_pending_action(action_id, auth=auth)


@app.post("/api/admin/pending-actions/{action_id}/reject", response_model=PendingActionItem)
async def admin_reject_pending_action(
    action_id: int,
    payload: PendingActionDecisionInput | None = None,
    auth=Depends(require_approver_role),
):
    return reject_pending_action(action_id, auth=auth, note=payload.note if payload else None)


@app.post("/api/admin/pending-actions/{action_id}/execute", response_model=BackgroundJobItem)
async def admin_execute_pending_action(
    action_id: int,
    request: Request,
    auth=Depends(require_approver_role),
):
    return enqueue_background_job(
        job_type="pending_action_execute",
        payload={"action_id": action_id},
        context=RequestContext(
            request_id=getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8],
            auth=auth,
        ),
    )


@app.post("/api/cache/clear")
async def cache_clear(_=Depends(require_system_role)):
    from app.cache import clear_cache

    clear_cache()
    return {"message": "Cache cleared"}


@app.get("/api/cache/stats", response_model=CacheStats)
async def cache_stats_endpoint(_=Depends(require_system_role)):
    from app.cache import get_stats

    stats = get_stats()
    return CacheStats(**stats)


@app.get("/api/system", response_model=SystemRuntime)
async def system_info(
    request: Request,
    kb_id: int | None = Query(default=None, ge=1),
    kb_key: str | None = Query(default=None),
    _=Depends(require_system_role),
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
        "observability": tracing_status(),
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
    _=Depends(require_knowledge_role),
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
    _=Depends(require_knowledge_role),
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


@app.get("/portal", response_class=HTMLResponse)
async def internal_portal_page():
    html_path = static_dir / "internal.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Internal Portal not found</h1><p>Place internal.html in /static/</p>")


@app.get("/")
async def root():
    return {
        "message": "Local RAG Agent API",
        "admin": "/admin",
        "chat": "/chat",
        "portal": "/portal",
        "health": "/health",
        "system": "/api/system",
        "docs": "/docs",
    }
