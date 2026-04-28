"""
Database-backed background job queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.models import RequestContext

logger = logging.getLogger(__name__)

BackgroundJobType = Literal[
    "google_drive_sync",
    "support_email_sync",
    "kb_ingest",
    "kb_reindex",
    "kb_file_ingest",
    "pending_action_execute",
]


class BackgroundJobItem(BaseModel):
    id: int
    job_id: str
    job_type: str
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error_message: str | None = None
    progress: float
    created_by_user_id: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    kb_id: int | None = None
    kb_key: str | None = None
    attempts: int
    max_attempts: int
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None


class ListBackgroundJobsOutput(BaseModel):
    total: int
    items: list[BackgroundJobItem]


def _parse_json(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else None


def _serialize_row(row: dict[str, Any]) -> BackgroundJobItem:
    return BackgroundJobItem(
        id=int(row["id"]),
        job_id=row["job_id"],
        job_type=row["job_type"],
        status=row["status"],
        payload=_parse_json(row.get("payload_json")) or {},
        result=_parse_json(row.get("result_json")),
        error_message=row.get("error_message"),
        progress=float(row.get("progress") or 0.0),
        created_by_user_id=row.get("created_by_user_id"),
        tenant_id=row.get("tenant_id"),
        org_id=row.get("org_id"),
        kb_id=row.get("kb_id"),
        kb_key=row.get("kb_key"),
        attempts=int(row.get("attempts") or 0),
        max_attempts=int(row.get("max_attempts") or 1),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
    )


def get_background_job(job_id: str) -> dict[str, Any]:
    row = fetch_one_sync("SELECT * FROM background_jobs WHERE job_id = ?", (job_id,))
    if not row:
        raise ValueError("Background job not found")
    return _serialize_row(row).model_dump()


def list_background_jobs(*, status: str | None = None, limit: int = 50) -> dict[str, Any]:
    if status:
        rows = fetch_all_sync(
            """
            SELECT * FROM background_jobs
            WHERE status = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (status, max(1, min(limit, 200))),
        )
    else:
        rows = fetch_all_sync(
            """
            SELECT * FROM background_jobs
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        )
    items = [_serialize_row(row).model_dump() for row in rows]
    return {"total": len(items), "items": items}


def enqueue_background_job(
    *,
    job_type: str,
    payload: dict[str, Any],
    context: RequestContext,
    max_attempts: int = 1,
) -> dict[str, Any]:
    now = utcnow_iso()
    job_id = f"BGJ-{uuid.uuid4().hex[:12].upper()}"
    execute_sync(
        """
        INSERT INTO background_jobs (
            job_id, job_type, status, payload_json, progress,
            created_by_user_id, tenant_id, org_id, kb_id, kb_key,
            max_attempts, created_at, updated_at
        ) VALUES (?, ?, 'queued', ?, 0.0, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            job_type,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            context.auth.user_id,
            context.auth.tenant_id,
            context.auth.org_id,
            context.kb_id,
            context.kb_key,
            max(1, max_attempts),
            now,
            now,
        ),
    )
    return get_background_job(job_id)


def mark_background_job_progress(job_id: str, progress: float) -> None:
    execute_sync(
        "UPDATE background_jobs SET progress = ?, updated_at = ? WHERE job_id = ?",
        (max(0.0, min(1.0, progress)), utcnow_iso(), job_id),
    )


def _claim_next_job() -> dict[str, Any] | None:
    row = fetch_one_sync(
        """
        SELECT * FROM background_jobs
        WHERE status = 'queued'
        ORDER BY created_at ASC, id ASC
        LIMIT 1
        """
    )
    if not row:
        return None
    now = utcnow_iso()
    execute_sync(
        """
        UPDATE background_jobs
        SET status = 'running',
            attempts = attempts + 1,
            started_at = COALESCE(started_at, ?),
            updated_at = ?
        WHERE job_id = ? AND status = 'queued'
        """,
        (now, now, row["job_id"]),
    )
    claimed = fetch_one_sync("SELECT * FROM background_jobs WHERE job_id = ?", (row["job_id"],))
    if not claimed or claimed["status"] != "running":
        return None
    return _serialize_row(claimed).model_dump()


async def background_worker_loop(*, poll_interval_seconds: float = 0.5) -> None:
    logger.info("Background job worker started")
    try:
        while True:
            job = _claim_next_job()
            if not job:
                await asyncio.sleep(poll_interval_seconds)
                continue
            await _run_background_job(job)
    except asyncio.CancelledError:
        logger.info("Background job worker stopped")
        raise


async def run_due_background_jobs_once() -> bool:
    job = _claim_next_job()
    if not job:
        return False
    await _run_background_job(job)
    return True


async def _run_background_job(job: dict[str, Any]) -> None:
    job_id = job["job_id"]
    try:
        result = await _dispatch_job(job)
    except Exception as err:
        logger.exception("Background job %s failed", job_id)
        now = utcnow_iso()
        execute_sync(
            """
            UPDATE background_jobs
            SET status = 'failed',
                error_message = ?,
                progress = 1.0,
                finished_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (str(err), now, now, job_id),
        )
        return

    now = utcnow_iso()
    execute_sync(
        """
        UPDATE background_jobs
        SET status = 'done',
            result_json = ?,
            progress = 1.0,
            finished_at = ?,
            updated_at = ?
        WHERE job_id = ?
        """,
        (json.dumps(result or {}, ensure_ascii=False, sort_keys=True), now, now, job_id),
    )


def _context_from_job(job: dict[str, Any]) -> RequestContext:
    return RequestContext(
        request_id=job["job_id"],
        kb_id=job.get("kb_id"),
        kb_key=job.get("kb_key"),
        auth={
            "user_id": job.get("created_by_user_id"),
            "roles": ["admin"],
            "channel": "admin",
            "tenant_id": job.get("tenant_id"),
            "org_id": job.get("org_id"),
        },
    )


async def _dispatch_job(job: dict[str, Any]) -> dict[str, Any]:
    job_type = job["job_type"]
    payload = job.get("payload") or {}
    context = _context_from_job(job)

    if job_type == "google_drive_sync":
        from app.drive_sync import sync_google_drive_source

        return await sync_google_drive_source(
            int(payload["source_id"]),
            triggered_by_user_id=context.auth.user_id,
            trigger_mode="background_job",
            force_full=bool(payload.get("force_full")),
        )

    if job_type == "support_email_sync":
        from app.integrations.support_email import sync_support_emails

        return await sync_support_emails(
            limit=int(payload.get("limit") or 20),
            unread_only=bool(payload.get("unread_only")),
        )

    if job_type == "pending_action_execute":
        from app.pending_actions import execute_pending_action

        return await execute_pending_action(int(payload["action_id"]), context=context)

    if job_type in {"kb_ingest", "kb_reindex", "kb_file_ingest"}:
        return await _dispatch_ingest_job(job_type, payload, job_id=job["job_id"])

    raise RuntimeError(f"Unsupported background job type: {job_type}")


async def _dispatch_ingest_job(job_type: str, payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    from app.ingest import _fetch_kb_file, _list_kb_files, _needs_ingest, _run_ingest, queue_ingest_job
    from app.kb_service import get_kb_or_404, open_db

    kb_id = int(payload["kb_id"])
    db = await open_db()
    try:
        await get_kb_or_404(db, kb_id)
    finally:
        await db.close()

    if job_type == "kb_file_ingest":
        file_id = int(payload["file_id"])
        row = await _fetch_kb_file(kb_id, file_id)
        if not row:
            raise ValueError("File is not attached to this Knowledge Base")
        rows = [row]
    elif job_type == "kb_reindex":
        rows = await _list_kb_files(kb_id)
    else:
        rows = [row for row in await _list_kb_files(kb_id) if _needs_ingest(row, kb_id)]

    ingest_jobs: list[dict[str, Any]] = []
    total = len(rows)
    if total == 0:
        return {"message": "No files to ingest", "jobs": []}

    for index, row in enumerate(rows, start=1):
        file_id = int(row["file_id"])
        ingest_job = await queue_ingest_job(kb_id, file_id, start_immediately=False)
        ingest_jobs.append(ingest_job)
        await _run_ingest(ingest_job["job_id"], kb_id, file_id)
        mark_background_job_progress(job_id, index / total)

    return {"message": f"Processed {len(ingest_jobs)} ingest job(s)", "jobs": ingest_jobs}
