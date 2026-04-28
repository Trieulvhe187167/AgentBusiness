"""
Ingestion pipeline with KB-scoped background jobs.
Parse -> Chunk -> Embed -> Upsert to vector store.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from app.auth import require_admin
from app.background_jobs import enqueue_background_job
from app.chunker import chunk_records
from app.config import settings
from app.database import execute_with_retry, fetch_all, fetch_one
from app.embeddings import embed_texts
from app.kb_service import attach_file_to_kb, get_default_kb, get_kb_or_404, new_kb_version, open_db
from app.models import AuthContext, IngestJobResponse, JobStatus, RequestContext
from app.parsers import parse_file
from app.vector_store import vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ingest"], dependencies=[Depends(require_admin)])

_ACCESS_LEVEL_ORDER = {"public": 0, "internal": 1, "admin": 2}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_ingest_signature(kb_id: int, file_hash: str, chunk_cfg: str, model_id: str) -> str:
    raw = f"{kb_id}:{file_hash}:{chunk_cfg}:{model_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _expected_ingest_signature(row: dict[str, Any], kb_id: int) -> str:
    chunk_cfg = f"{settings.chunk_size}:{settings.chunk_overlap}"
    return _compute_ingest_signature(
        kb_id=kb_id,
        file_hash=row["file_hash"],
        chunk_cfg=chunk_cfg,
        model_id=settings.effective_embedding_model_id,
    )


def _needs_ingest(row: dict[str, Any], kb_id: int) -> bool:
    if row["kb_status"] in {"queued", "ingesting"}:
        return False
    if row["kb_status"] in {"attached", "failed"}:
        return True
    if not row.get("ingest_signature"):
        return True
    return row["ingest_signature"] != _expected_ingest_signature(row, kb_id)


async def _list_kb_files(kb_id: int) -> list[dict[str, Any]]:
    return await fetch_all(
        """
        SELECT
            kf.kb_id,
            kf.file_id,
            kf.status AS kb_status,
            kf.chunk_count,
            kf.ingest_signature,
            kf.last_job_id,
            kf.attached_at,
            kf.last_ingest_at,
            uf.filename,
            uf.original_name,
            uf.file_type,
            uf.file_size,
            uf.file_hash,
            uf.parser_type,
            uf.status AS upload_status,
            uf.pages_or_rows,
            uf.ingested_at,
            uf.error_message,
            uf.created_at
        FROM kb_files kf
        JOIN uploaded_files uf ON uf.id = kf.file_id
        WHERE kf.kb_id = ?
        ORDER BY kf.attached_at DESC
        """,
        (kb_id,),
    )


async def _fetch_kb_file(kb_id: int, file_id: int) -> dict[str, Any] | None:
    return await fetch_one(
        """
        SELECT
            kf.kb_id,
            kf.file_id,
            kf.status AS kb_status,
            kf.chunk_count,
            kf.ingest_signature,
            kf.last_job_id,
            kf.attached_at,
            kf.last_ingest_at,
            uf.filename,
            uf.original_name,
            uf.file_type,
            uf.file_size,
            uf.file_hash,
            uf.parser_type,
            uf.status AS upload_status,
            uf.pages_or_rows,
            uf.ingested_at,
            uf.error_message,
            uf.created_at
        FROM kb_files kf
        JOIN uploaded_files uf ON uf.id = kf.file_id
        WHERE kf.kb_id = ? AND kf.file_id = ?
        """,
        (kb_id, file_id),
    )


async def _queue_ingest_job(background_tasks: BackgroundTasks, kb_id: int, file_id: int) -> dict[str, Any]:
    return await queue_ingest_job(kb_id, file_id, background_tasks=background_tasks)


async def queue_ingest_job(
    kb_id: int,
    file_id: int,
    *,
    background_tasks: BackgroundTasks | None = None,
    start_immediately: bool = True,
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    await execute_with_retry(
        "INSERT INTO ingest_jobs (job_id, file_id, kb_id, status) VALUES (?, ?, ?, 'queued')",
        (job_id, file_id, kb_id),
    )
    await execute_with_retry(
        """
        UPDATE kb_files
        SET status = 'queued',
            last_job_id = ?
        WHERE kb_id = ? AND file_id = ?
        """,
        (job_id, kb_id, file_id),
    )
    if start_immediately:
        if background_tasks is not None:
            background_tasks.add_task(_run_ingest, job_id, kb_id, file_id)
        else:
            asyncio.create_task(_run_ingest(job_id, kb_id, file_id))
    return {"job_id": job_id, "kb_id": kb_id, "file_id": file_id, "status": "queued"}


async def _update_file_failure_state(file_id: int, error_message: str):
    ingested_count = await fetch_one(
        "SELECT COUNT(*) AS total FROM kb_files WHERE file_id = ? AND status = 'ingested'",
        (file_id,),
    )
    if ingested_count and int(ingested_count["total"]) > 0:
        await execute_with_retry(
            "UPDATE uploaded_files SET error_message = ? WHERE id = ?",
            (error_message[:500], file_id),
        )
        return

    await execute_with_retry(
        "UPDATE uploaded_files SET status = 'failed', error_message = ? WHERE id = ?",
        (error_message[:500], file_id),
    )


async def _run_ingest(job_id: str, kb_id: int, file_id: int):
    now = _utcnow()

    try:
        await execute_with_retry(
            "UPDATE ingest_jobs SET status='running', started_at=?, kb_id=? WHERE job_id=?",
            (now, kb_id, job_id),
        )
        await execute_with_retry(
            """
            UPDATE kb_files
            SET status='ingesting', last_job_id=?
            WHERE kb_id=? AND file_id=?
            """,
            (job_id, kb_id, file_id),
        )
        await execute_with_retry(
            "UPDATE uploaded_files SET status='ingesting' WHERE id=?",
            (file_id,),
        )

        db = await open_db()
        try:
            kb = await get_kb_or_404(db, kb_id)
        finally:
            await db.close()

        file_row = await fetch_one("SELECT * FROM uploaded_files WHERE id=?", (file_id,))
        if not file_row:
            raise ValueError(f"File {file_id} not found")

        file_path = settings.raw_upload_dir / file_row["filename"]
        parser_type = file_row["parser_type"]

        logger.info("[%s] Parsing %s for kb=%s", job_id, file_row["original_name"], kb_id)
        await execute_with_retry("UPDATE ingest_jobs SET progress=0.1 WHERE job_id=?", (job_id,))

        records = parse_file(file_path, parser_type)
        pages_or_rows = len(records)

        logger.info("[%s] Chunking %s records for kb=%s", job_id, pages_or_rows, kb_id)
        await execute_with_retry("UPDATE ingest_jobs SET progress=0.3 WHERE job_id=?", (job_id,))

        chunk_cfg = f"{settings.chunk_size}:{settings.chunk_overlap}"
        ingest_signature = _compute_ingest_signature(
            kb_id=kb_id,
            file_hash=file_row["file_hash"],
            chunk_cfg=chunk_cfg,
            model_id=settings.effective_embedding_model_id,
        )
        next_kb_version = new_kb_version()

        vector_store.delete_by_kb_and_file(kb_id, file_id)
        kb_access_level = str(kb.access_level or "public").lower()
        file_access_level = str(file_row.get("access_level") or "public").lower()
        effective_access_level = (
            kb_access_level
            if _ACCESS_LEVEL_ORDER.get(kb_access_level, 0) >= _ACCESS_LEVEL_ORDER.get(file_access_level, 0)
            else file_access_level
        )
        effective_tenant_id = file_row.get("tenant_id") or kb.tenant_id
        effective_org_id = file_row.get("org_id") or kb.org_id
        effective_owner_user_id = file_row.get("owner_user_id")

        chunks = chunk_records(
            records=records,
            kb_id=kb_id,
            source_id=str(file_id),
            filename=file_row["original_name"],
            file_type=file_row["file_type"],
            file_hash=file_row["file_hash"],
            kb_version=next_kb_version,
            ingest_signature=ingest_signature,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            access_level=effective_access_level,
            tenant_id=effective_tenant_id,
            org_id=effective_org_id,
            owner_user_id=effective_owner_user_id,
        )
        if not chunks:
            raise ValueError("No chunks produced - file may be empty or unparseable")

        logger.info("[%s] Embedding %s chunks for kb=%s", job_id, len(chunks), kb_id)
        await execute_with_retry("UPDATE ingest_jobs SET progress=0.5 WHERE job_id=?", (job_id,))

        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_texts(texts)

        logger.info("[%s] Upserting vectors for kb=%s", job_id, kb_id)
        await execute_with_retry("UPDATE ingest_jobs SET progress=0.8 WHERE job_id=?", (job_id,))
        vector_store.add_chunks(chunks, embeddings)

        finished = _utcnow()
        await execute_with_retry(
            "UPDATE ingest_jobs SET status='done', progress=1.0, finished_at=?, kb_id=? WHERE job_id=?",
            (finished, kb_id, job_id),
        )
        await execute_with_retry(
            """
            UPDATE kb_files
            SET status='ingested',
                chunk_count=?,
                ingest_signature=?,
                last_job_id=?,
                last_ingest_at=?
            WHERE kb_id=? AND file_id=?
            """,
            (len(chunks), ingest_signature, job_id, finished, kb_id, file_id),
        )
        await execute_with_retry(
            """
            UPDATE knowledge_bases
            SET kb_version=?, updated_at=?
            WHERE id=?
            """,
            (next_kb_version, finished, kb_id),
        )
        await execute_with_retry(
            """
            UPDATE uploaded_files
            SET status='ingested', pages_or_rows=?, ingested_at=?, error_message=NULL
            WHERE id=?
            """,
            (pages_or_rows, finished, file_id),
        )
        logger.info(
            "[%s] Ingestion complete for kb=%s file=%s with %s chunks (prev_kb_version=%s new=%s)",
            job_id,
            kb_id,
            file_id,
            len(chunks),
            kb.kb_version,
            next_kb_version,
        )

    except Exception as err:
        logger.error("[%s] Ingestion failed for kb=%s file=%s: %s", job_id, kb_id, file_id, err, exc_info=True)
        finished = _utcnow()
        message = str(err)[:500]
        await execute_with_retry(
            "UPDATE ingest_jobs SET status='failed', error_message=?, finished_at=?, kb_id=? WHERE job_id=?",
            (message, finished, kb_id, job_id),
        )
        await execute_with_retry(
            """
            UPDATE kb_files
            SET status='failed', last_job_id=?
            WHERE kb_id=? AND file_id=?
            """,
            (job_id, kb_id, file_id),
        )
        await _update_file_failure_state(file_id, message)


async def _resolve_default_kb_id() -> int:
    db = await open_db()
    try:
        kb = await get_default_kb(db)
        return kb.id
    finally:
        await db.close()


async def _queue_kb_files(background_tasks: BackgroundTasks, kb_id: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row in rows:
        jobs.append(await _queue_ingest_job(background_tasks, kb_id, int(row["file_id"])))
    return jobs


@router.post("/ingest/all")
async def ingest_all(request: Request, auth=Depends(require_admin)):
    """Queue ingestion for stale files in the default Knowledge Base."""
    kb_id = await _resolve_default_kb_id()
    rows = [row for row in await _list_kb_files(kb_id) if _needs_ingest(row, kb_id)]
    if not rows:
        return {"message": "No files to ingest in the default Knowledge Base", "jobs": []}

    job = _enqueue_ingest_background_job(
        request=request,
        auth=auth,
        kb_id=kb_id,
        job_type="kb_ingest",
        payload={"kb_id": kb_id},
    )
    return {
        **job,
        "background_job_id": job["job_id"],
        "message": f"Queued background ingest for KB {kb_id}",
        "jobs": [job],
    }


@router.post("/ingest/{file_id}")
async def ingest_file(file_id: int, request: Request, auth=Depends(require_admin)):
    """Trigger ingestion for a single file in the default Knowledge Base."""
    kb_id = await _resolve_default_kb_id()
    row = await _fetch_kb_file(kb_id, file_id)
    if not row:
        file_row = await fetch_one("SELECT id FROM uploaded_files WHERE id = ?", (file_id,))
        if not file_row:
            raise HTTPException(404, "File not found")
        db = await open_db()
        try:
            await attach_file_to_kb(db, kb_id, file_id, status="attached")
            await db.commit()
        finally:
            await db.close()
        row = await _fetch_kb_file(kb_id, file_id)
    if not row:
        raise HTTPException(404, "File is not attached to the default Knowledge Base")

    job = _enqueue_ingest_background_job(
        request=request,
        auth=auth,
        kb_id=kb_id,
        job_type="kb_file_ingest",
        payload={"kb_id": kb_id, "file_id": file_id},
    )
    return {**job, "background_job_id": job["job_id"], "file_id": file_id, "kb_id": kb_id}


@router.post("/kbs/{kb_id}/ingest")
async def ingest_kb(kb_id: int, request: Request, auth=Depends(require_admin)):
    """Queue ingestion for stale files attached to a Knowledge Base."""
    db = await open_db()
    try:
        await get_kb_or_404(db, kb_id)
    finally:
        await db.close()

    rows = [row for row in await _list_kb_files(kb_id) if _needs_ingest(row, kb_id)]
    if not rows:
        return {"message": "No stale files to ingest for this Knowledge Base", "jobs": []}

    job = _enqueue_ingest_background_job(
        request=request,
        auth=auth,
        kb_id=kb_id,
        job_type="kb_ingest",
        payload={"kb_id": kb_id},
    )
    return {
        **job,
        "background_job_id": job["job_id"],
        "message": f"Queued background ingest for Knowledge Base {kb_id}",
        "jobs": [job],
    }


@router.post("/kbs/{kb_id}/reindex")
async def reindex_kb(kb_id: int, request: Request, auth=Depends(require_admin)):
    """Queue reindex for all files attached to a Knowledge Base."""
    db = await open_db()
    try:
        await get_kb_or_404(db, kb_id)
    finally:
        await db.close()

    rows = await _list_kb_files(kb_id)
    if not rows:
        return {"message": "No attached files to reindex for this Knowledge Base", "jobs": []}

    job = _enqueue_ingest_background_job(
        request=request,
        auth=auth,
        kb_id=kb_id,
        job_type="kb_reindex",
        payload={"kb_id": kb_id},
    )
    return {
        **job,
        "background_job_id": job["job_id"],
        "message": f"Queued background reindex for Knowledge Base {kb_id}",
        "jobs": [job],
    }


@router.post("/kbs/{kb_id}/files/{file_id}/ingest")
async def ingest_kb_file(kb_id: int, file_id: int, request: Request, auth=Depends(require_admin)):
    """Trigger ingestion for one file attached to one Knowledge Base."""
    db = await open_db()
    try:
        await get_kb_or_404(db, kb_id)
    finally:
        await db.close()

    row = await _fetch_kb_file(kb_id, file_id)
    if not row:
        raise HTTPException(404, "File is not attached to this Knowledge Base")

    job = _enqueue_ingest_background_job(
        request=request,
        auth=auth,
        kb_id=kb_id,
        job_type="kb_file_ingest",
        payload={"kb_id": kb_id, "file_id": file_id},
    )
    return {**job, "background_job_id": job["job_id"], "file_id": file_id, "kb_id": kb_id}


def _enqueue_ingest_background_job(
    *,
    request: Request,
    auth,
    kb_id: int,
    job_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    request_state = getattr(request, "state", None)
    auth_context = auth if isinstance(auth, AuthContext) else AuthContext(user_id="admin-1", roles=["admin"], channel="admin")
    return enqueue_background_job(
        job_type=job_type,
        payload=payload,
        context=RequestContext(
            request_id=getattr(request_state, "request_id", None) or f"ingest-{uuid.uuid4().hex[:8]}",
            kb_id=kb_id,
            auth=auth_context,
        ),
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    row = await fetch_one("SELECT * FROM ingest_jobs WHERE job_id=?", (job_id,))
    if not row:
        raise HTTPException(404, "Job not found")

    return JobStatus(
        job_id=row["job_id"],
        file_id=row["file_id"],
        kb_id=row.get("kb_id"),
        status=row["status"],
        progress=row["progress"],
        error_message=row.get("error_message"),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
    )
