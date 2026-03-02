"""
Ingestion pipeline with background job tracking.
Parse → Chunk → Embed → Upsert to vector store.
"""

import hashlib
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.config import settings
from app.database import execute_with_retry, fetch_all, fetch_one
from app.parsers import parse_file
from app.chunker import chunk_records
from app.embeddings import embed_texts
from app.vector_store import vector_store
from app.models import IngestJobResponse, JobStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ingest"])


def _compute_kb_version(file_hash: str, chunk_cfg: str, model_id: str) -> str:
    """Deterministic KB version from inputs."""
    raw = f"{file_hash}:{chunk_cfg}:{model_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


async def _run_ingest(job_id: str, file_id: int):
    """Background ingest task: parse → chunk → embed → upsert."""
    now = datetime.now(timezone.utc).isoformat()

    try:
        # Mark job as running
        await execute_with_retry(
            "UPDATE ingest_jobs SET status='running', started_at=? WHERE job_id=?",
            (now, job_id)
        )
        await execute_with_retry(
            "UPDATE uploaded_files SET status='ingesting' WHERE id=?",
            (file_id,)
        )

        # Fetch file info
        file_row = await fetch_one("SELECT * FROM uploaded_files WHERE id=?", (file_id,))
        if not file_row:
            raise ValueError(f"File {file_id} not found")

        file_path = settings.raw_upload_dir / file_row["filename"]
        parser_type = file_row["parser_type"]

        # ── Step 1: Parse ──────────────────────────────────
        logger.info(f"[{job_id}] Parsing {file_row['original_name']} ({parser_type})")
        await execute_with_retry(
            "UPDATE ingest_jobs SET progress=0.1 WHERE job_id=?", (job_id,)
        )

        records = parse_file(file_path, parser_type)
        pages_or_rows = len(records)

        # ── Step 2: Chunk ──────────────────────────────────
        logger.info(f"[{job_id}] Chunking {pages_or_rows} records")
        await execute_with_retry(
            "UPDATE ingest_jobs SET progress=0.3 WHERE job_id=?", (job_id,)
        )

        chunk_cfg = f"{settings.chunk_size}:{settings.chunk_overlap}"
        kb_version = _compute_kb_version(
            file_row["file_hash"], chunk_cfg, settings.effective_embedding_model_id
        )

        # Delete old vectors for this source (re-ingest support)
        vector_store.delete_by_source(str(file_id))

        chunks = chunk_records(
            records=records,
            source_id=str(file_id),
            filename=file_row["original_name"],
            file_type=file_row["file_type"],
            kb_version=kb_version,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            raise ValueError("No chunks produced — file may be empty or unparseable")

        # ── Step 3: Embed ──────────────────────────────────
        logger.info(f"[{job_id}] Embedding {len(chunks)} chunks")
        await execute_with_retry(
            "UPDATE ingest_jobs SET progress=0.5 WHERE job_id=?", (job_id,)
        )

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        # ── Step 4: Upsert to vector store ─────────────────
        logger.info(f"[{job_id}] Upserting to vector store")
        await execute_with_retry(
            "UPDATE ingest_jobs SET progress=0.8 WHERE job_id=?", (job_id,)
        )

        vector_store.add_chunks(chunks, embeddings)

        # ── Done ───────────────────────────────────────────
        finished = datetime.now(timezone.utc).isoformat()
        await execute_with_retry(
            "UPDATE ingest_jobs SET status='done', progress=1.0, finished_at=? WHERE job_id=?",
            (finished, job_id)
        )
        await execute_with_retry(
            """UPDATE uploaded_files
               SET status='ingested', pages_or_rows=?, ingested_at=?, error_message=NULL
               WHERE id=?""",
            (pages_or_rows, finished, file_id)
        )
        logger.info(f"[{job_id}] Ingestion complete: {len(chunks)} chunks indexed")

    except Exception as e:
        logger.error(f"[{job_id}] Ingestion failed: {e}", exc_info=True)
        finished = datetime.now(timezone.utc).isoformat()
        await execute_with_retry(
            "UPDATE ingest_jobs SET status='failed', error_message=?, finished_at=? WHERE job_id=?",
            (str(e)[:500], finished, job_id)
        )
        await execute_with_retry(
            "UPDATE uploaded_files SET status='failed', error_message=? WHERE id=?",
            (str(e)[:500], file_id)
        )


@router.post("/ingest/all")
async def ingest_all(background_tasks: BackgroundTasks):
    """Queue ingestion for all uploaded (not yet ingested) files."""
    rows = await fetch_all(
        "SELECT * FROM uploaded_files WHERE status IN ('uploaded', 'failed', 'ingesting')"
    )
    if not rows:
        return {"message": "No files to ingest", "jobs": []}

    jobs = []
    for row in rows:
        job_id = uuid.uuid4().hex[:12]
        await execute_with_retry(
            "INSERT INTO ingest_jobs (job_id, file_id, status) VALUES (?, ?, 'queued')",
            (job_id, row["id"])
        )
        background_tasks.add_task(_run_ingest, job_id, row["id"])
        jobs.append({"job_id": job_id, "file_id": row["id"], "status": "queued"})

    return {"message": f"Queued {len(jobs)} files for ingestion", "jobs": jobs}


@router.post("/ingest/{file_id}", response_model=IngestJobResponse)
async def ingest_file(file_id: int, background_tasks: BackgroundTasks):
    """Trigger ingestion for a single file. Returns job_id for tracking."""
    file_row = await fetch_one("SELECT * FROM uploaded_files WHERE id=?", (file_id,))
    if not file_row:
        raise HTTPException(404, "File not found")

    job_id = uuid.uuid4().hex[:12]
    await execute_with_retry(
        "INSERT INTO ingest_jobs (job_id, file_id, status) VALUES (?, ?, 'queued')",
        (job_id, file_id)
    )

    background_tasks.add_task(_run_ingest, job_id, file_id)

    return IngestJobResponse(job_id=job_id, file_id=file_id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of an ingest job."""
    row = await fetch_one("SELECT * FROM ingest_jobs WHERE job_id=?", (job_id,))
    if not row:
        raise HTTPException(404, "Job not found")

    return JobStatus(
        job_id=row["job_id"],
        file_id=row["file_id"],
        status=row["status"],
        progress=row["progress"],
        error_message=row.get("error_message"),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
    )
