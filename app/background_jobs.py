"""
Database-backed background job queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config import settings
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
    retry_after: str | None = None
    cancel_requested_at: str | None = None
    cancelled_at: str | None = None
    cancel_reason: str | None = None
    worker_id: str | None = None
    heartbeat_at: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None


class ListBackgroundJobsOutput(BaseModel):
    total: int
    items: list[BackgroundJobItem]


class BackgroundJobDecisionInput(BaseModel):
    reason: str | None = Field(default=None, max_length=1000)


class BackgroundJobCancelledError(RuntimeError):
    pass


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
        retry_after=row.get("retry_after"),
        cancel_requested_at=row.get("cancel_requested_at"),
        cancelled_at=row.get("cancelled_at"),
        cancel_reason=row.get("cancel_reason"),
        worker_id=row.get("worker_id"),
        heartbeat_at=row.get("heartbeat_at"),
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


def _next_retry_after(attempts: int) -> str:
    delay_seconds = min(300, 15 * (2 ** max(0, attempts - 1)))
    return (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat()


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"


def _utcnow_minus(seconds: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=max(1, seconds))).isoformat()


def is_background_job_cancel_requested(job_id: str) -> bool:
    row = fetch_one_sync(
        "SELECT status, cancel_requested_at FROM background_jobs WHERE job_id = ?",
        (job_id,),
    )
    if not row:
        return False
    return bool(row.get("cancel_requested_at")) or str(row.get("status") or "") in {"cancelling", "cancelled"}


def ensure_background_job_not_cancelled(job_id: str) -> None:
    if is_background_job_cancel_requested(job_id):
        raise BackgroundJobCancelledError(f"Background job {job_id} was cancelled")


def cancel_background_job(job_id: str, *, reason: str | None = None) -> dict[str, Any]:
    item = get_background_job(job_id)
    status = item["status"]
    if status in {"done", "failed", "cancelled"}:
        raise ValueError(f"Cannot cancel background job in status '{status}'")

    now = utcnow_iso()
    if status in {"queued", "retrying"}:
        execute_sync(
            """
            UPDATE background_jobs
            SET status = 'cancelled',
                cancel_requested_at = COALESCE(cancel_requested_at, ?),
                cancelled_at = ?,
                cancel_reason = ?,
                error_message = ?,
                finished_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (now, now, reason, reason or "Cancelled by admin", now, now, job_id),
        )
    else:
        execute_sync(
            """
            UPDATE background_jobs
            SET status = 'cancelling',
                cancel_requested_at = COALESCE(cancel_requested_at, ?),
                cancel_reason = ?,
                error_message = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (now, reason, reason or "Cancellation requested by admin", now, job_id),
        )
    return get_background_job(job_id)


def retry_background_job(job_id: str) -> dict[str, Any]:
    item = get_background_job(job_id)
    status = item["status"]
    if status not in {"failed", "cancelled"}:
        raise ValueError(f"Only failed or cancelled background jobs can be retried, got '{status}'")

    now = utcnow_iso()
    execute_sync(
        """
        UPDATE background_jobs
        SET status = 'queued',
            attempts = 0,
            progress = 0.0,
            result_json = NULL,
            error_message = NULL,
            retry_after = NULL,
            cancel_requested_at = NULL,
            cancelled_at = NULL,
            cancel_reason = NULL,
            worker_id = NULL,
            heartbeat_at = NULL,
            started_at = NULL,
            finished_at = NULL,
            updated_at = ?
        WHERE job_id = ?
        """,
        (now, job_id),
    )
    return get_background_job(job_id)


def enqueue_background_job(
    *,
    job_type: str,
    payload: dict[str, Any],
    context: RequestContext,
    max_attempts: int = 3,
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
    return claim_next_background_job(worker_id=_default_worker_id())


def recover_stale_background_jobs(*, stale_seconds: int | None = None) -> int:
    cutoff = _utcnow_minus(stale_seconds or settings.background_worker_stale_seconds)
    rows = fetch_all_sync(
        """
        SELECT job_id, status, attempts, max_attempts
        FROM background_jobs
        WHERE status IN ('running', 'cancelling')
          AND (
            heartbeat_at IS NULL
            OR heartbeat_at < ?
          )
        ORDER BY updated_at ASC, id ASC
        """,
        (cutoff,),
    )
    recovered = 0
    now = utcnow_iso()
    for row in rows:
        job_id = str(row["job_id"])
        status = str(row["status"])
        if status == "cancelling":
            execute_sync(
                """
                UPDATE background_jobs
                SET status = 'cancelled',
                    error_message = COALESCE(error_message, 'Cancelled after stale worker heartbeat'),
                    cancelled_at = COALESCE(cancelled_at, ?),
                    finished_at = COALESCE(finished_at, ?),
                    worker_id = NULL,
                    updated_at = ?
                WHERE job_id = ? AND status = 'cancelling'
                """,
                (now, now, now, job_id),
            )
            recovered += 1
            continue

        attempts = int(row.get("attempts") or 0)
        max_attempts = int(row.get("max_attempts") or 1)
        if attempts < max_attempts:
            execute_sync(
                """
                UPDATE background_jobs
                SET status = 'retrying',
                    error_message = 'Worker heartbeat stale; job returned to retry queue',
                    retry_after = ?,
                    worker_id = NULL,
                    heartbeat_at = NULL,
                    updated_at = ?
                WHERE job_id = ? AND status = 'running'
                """,
                (now, now, job_id),
            )
        else:
            execute_sync(
                """
                UPDATE background_jobs
                SET status = 'failed',
                    error_message = 'Worker heartbeat stale and max attempts exhausted',
                    progress = 1.0,
                    finished_at = ?,
                    worker_id = NULL,
                    updated_at = ?
                WHERE job_id = ? AND status = 'running'
                """,
                (now, now, job_id),
            )
        recovered += 1
    return recovered


def claim_next_background_job(*, worker_id: str) -> dict[str, Any] | None:
    recover_stale_background_jobs()
    now = utcnow_iso()
    row = fetch_one_sync(
        """
        SELECT * FROM background_jobs
        WHERE status IN ('queued', 'retrying')
          AND cancel_requested_at IS NULL
          AND (retry_after IS NULL OR retry_after <= ?)
        ORDER BY created_at ASC, id ASC
        LIMIT 1
        """,
        (now,),
    )
    if not row:
        return None
    execute_sync(
        """
        UPDATE background_jobs
        SET status = 'running',
            attempts = attempts + 1,
            retry_after = NULL,
            worker_id = ?,
            heartbeat_at = ?,
            started_at = COALESCE(started_at, ?),
            updated_at = ?
        WHERE job_id = ?
          AND status IN ('queued', 'retrying')
          AND cancel_requested_at IS NULL
          AND (retry_after IS NULL OR retry_after <= ?)
        """,
        (worker_id, now, now, now, row["job_id"], now),
    )
    claimed = fetch_one_sync("SELECT * FROM background_jobs WHERE job_id = ?", (row["job_id"],))
    if not claimed or claimed["status"] != "running":
        return None
    return _serialize_row(claimed).model_dump()


async def background_worker_loop(
    *,
    poll_interval_seconds: float | None = None,
    worker_id: str | None = None,
) -> None:
    resolved_worker_id = worker_id or _default_worker_id()
    poll_interval = poll_interval_seconds or settings.background_worker_poll_interval_seconds
    logger.info("Background job worker started: worker_id=%s", resolved_worker_id)
    try:
        while True:
            job = claim_next_background_job(worker_id=resolved_worker_id)
            if not job:
                await asyncio.sleep(poll_interval)
                continue
            await _run_background_job(job, worker_id=resolved_worker_id)
    except asyncio.CancelledError:
        logger.info("Background job worker stopped")
        raise


async def run_due_background_jobs_once() -> bool:
    job = _claim_next_job()
    if not job:
        return False
    await _run_background_job(job, worker_id=job.get("worker_id") or "test-worker")
    return True


async def _heartbeat_loop(job_id: str, *, worker_id: str) -> None:
    interval = max(1.0, float(settings.background_worker_heartbeat_interval_seconds))
    try:
        while True:
            execute_sync(
                """
                UPDATE background_jobs
                SET heartbeat_at = ?, updated_at = ?
                WHERE job_id = ?
                  AND worker_id = ?
                  AND status IN ('running', 'cancelling')
                """,
                (utcnow_iso(), utcnow_iso(), job_id, worker_id),
            )
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        raise


async def _stop_heartbeat_task(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _run_background_job(job: dict[str, Any], *, worker_id: str | None = None) -> None:
    job_id = job["job_id"]
    resolved_worker_id = worker_id or job.get("worker_id") or _default_worker_id()
    heartbeat_task = asyncio.create_task(_heartbeat_loop(job_id, worker_id=resolved_worker_id))
    try:
        ensure_background_job_not_cancelled(job_id)
        result = await _dispatch_job(job)
        ensure_background_job_not_cancelled(job_id)
    except BackgroundJobCancelledError as err:
        logger.info("Background job %s cancelled", job_id)
        now = utcnow_iso()
        execute_sync(
            """
            UPDATE background_jobs
            SET status = 'cancelled',
                error_message = ?,
                cancelled_at = COALESCE(cancelled_at, ?),
                finished_at = ?,
                worker_id = NULL,
                updated_at = ?
            WHERE job_id = ?
            """,
            (str(err), now, now, now, job_id),
        )
        await _stop_heartbeat_task(heartbeat_task)
        return
    except Exception as err:
        if is_background_job_cancel_requested(job_id):
            logger.info("Background job %s cancelled after provider error", job_id)
            now = utcnow_iso()
            execute_sync(
                """
                UPDATE background_jobs
                SET status = 'cancelled',
                    error_message = ?,
                    cancelled_at = COALESCE(cancelled_at, ?),
                    finished_at = ?,
                    worker_id = NULL,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (str(err), now, now, now, job_id),
            )
            await _stop_heartbeat_task(heartbeat_task)
            return

        logger.exception("Background job %s failed", job_id)
        latest = get_background_job(job_id)
        now = utcnow_iso()
        if int(latest["attempts"]) < int(latest["max_attempts"]):
            retry_after = _next_retry_after(int(latest["attempts"]))
            execute_sync(
                """
                UPDATE background_jobs
                SET status = 'retrying',
                    error_message = ?,
                    retry_after = ?,
                    worker_id = NULL,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (str(err), retry_after, now, job_id),
            )
            await _stop_heartbeat_task(heartbeat_task)
            return

        execute_sync(
            """
            UPDATE background_jobs
            SET status = 'failed',
                error_message = ?,
                progress = 1.0,
                finished_at = ?,
                worker_id = NULL,
                updated_at = ?
            WHERE job_id = ?
            """,
            (str(err), now, now, job_id),
        )
        await _stop_heartbeat_task(heartbeat_task)
        return

    now = utcnow_iso()
    execute_sync(
        """
        UPDATE background_jobs
        SET status = 'done',
            result_json = ?,
            progress = 1.0,
            finished_at = ?,
            worker_id = NULL,
            updated_at = ?
        WHERE job_id = ?
        """,
        (json.dumps(result or {}, ensure_ascii=False, sort_keys=True), now, now, job_id),
    )
    await _stop_heartbeat_task(heartbeat_task)


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
            cancel_check=lambda: ensure_background_job_not_cancelled(job["job_id"]),
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
        ensure_background_job_not_cancelled(job_id)
        file_id = int(row["file_id"])
        ingest_job = await queue_ingest_job(kb_id, file_id, start_immediately=False)
        ingest_jobs.append(ingest_job)
        await _run_ingest(ingest_job["job_id"], kb_id, file_id)
        mark_background_job_progress(job_id, index / total)
        ensure_background_job_not_cancelled(job_id)

    return {"message": f"Processed {len(ingest_jobs)} ingest job(s)", "jobs": ingest_jobs}
