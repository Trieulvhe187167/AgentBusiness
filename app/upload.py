"""Secure file upload endpoints."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.auth import require_admin
from app.background_jobs import enqueue_background_job
from app.config import settings
from app.database import fetch_all, fetch_one, get_db
from app.file_versions import diff_file_versions, list_file_versions, restore_file_version
from app.kb_service import bump_kb_version
from app.models import (
    AuthContext,
    DiffFileVersionsOutput,
    FileInfo,
    FileVersionItem,
    ListFileVersionsOutput,
    RequestContext,
    RollbackFileVersionInput,
    RollbackFileVersionOutput,
    UploadResponse,
)
from app.upload_service import import_content_to_uploaded_file

router = APIRouter(prefix="/api", tags=["upload"], dependencies=[Depends(require_admin)])


def file_row_to_info(row: dict) -> FileInfo:
    return FileInfo(
        id=row["id"],
        filename=row["filename"],
        original_name=row["original_name"],
        file_type=row["file_type"],
        file_size=row["file_size"],
        file_hash=row["file_hash"],
        status=row["status"],
        access_level=row.get("access_level") or "public",
        tenant_id=row.get("tenant_id"),
        org_id=row.get("org_id"),
        owner_user_id=row.get("owner_user_id"),
        parser_type=row.get("parser_type"),
        pages_or_rows=row.get("pages_or_rows"),
        ingested_at=row.get("ingested_at"),
        error_message=row.get("error_message"),
        created_at=row["created_at"],
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file with security validation."""
    content = await file.read()
    row = await import_content_to_uploaded_file(filename=file.filename, content=content)

    return UploadResponse(
        message=f"File '{row['original_name']}' uploaded successfully",
        file=file_row_to_info(row)
    )


@router.post("/admin/upload", response_model=UploadResponse, include_in_schema=False)
async def upload_file_admin(file: UploadFile = File(...)):
    """Admin alias for product-style endpoint naming."""
    return await upload_file(file)


@router.post("/files/{file_id}/content", response_model=UploadResponse)
async def replace_file_content(file_id: int, file: UploadFile = File(...), auth=Depends(require_admin)):
    """Replace an existing source file's binary content and create a new version."""
    existing = await fetch_one("SELECT * FROM uploaded_files WHERE id = ?", (file_id,))
    if not existing:
        raise HTTPException(404, "File not found")

    content = await file.read()
    row = await import_content_to_uploaded_file(
        filename=file.filename,
        content=content,
        existing_file_id=file_id,
        access_level=existing.get("access_level") or "public",
        tenant_id=existing.get("tenant_id"),
        org_id=existing.get("org_id"),
        owner_user_id=getattr(auth, "user_id", None) or existing.get("owner_user_id"),
    )

    return UploadResponse(
        message=f"File '{row['original_name']}' replaced successfully",
        file=file_row_to_info(row),
    )


@router.get("/files", response_model=list[FileInfo])
async def list_files():
    """List all uploaded files with status."""
    rows = await fetch_all(
        "SELECT * FROM uploaded_files ORDER BY created_at DESC"
    )
    return [file_row_to_info(r) for r in rows]


def _version_row_to_item(row: dict) -> FileVersionItem:
    return FileVersionItem(
        id=int(row["id"]),
        file_id=int(row["file_id"]),
        version_number=int(row["version_number"]),
        file_hash=row["file_hash"],
        file_size=int(row["file_size"]),
        filename=row["filename"],
        original_name=row["original_name"],
        file_type=row["file_type"],
        parser_type=row.get("parser_type"),
        pages_or_rows=row.get("pages_or_rows"),
        chunk_count=row.get("chunk_count"),
        ingest_signature=row.get("ingest_signature"),
        has_snapshot=bool(row.get("snapshot_path")),
        change_summary=row.get("change_summary"),
        created_by_user_id=row.get("created_by_user_id"),
        created_at=row["created_at"],
        is_current=bool(row.get("is_current")),
        is_active=bool(row.get("is_active")),
    )


@router.get("/files/{file_id}/versions", response_model=ListFileVersionsOutput)
async def get_file_versions(file_id: int):
    """List immutable uploaded-file versions."""
    row = await fetch_one("SELECT id FROM uploaded_files WHERE id = ?", (file_id,))
    if not row:
        raise HTTPException(404, "File not found")

    versions = [_version_row_to_item(item) for item in await list_file_versions(file_id)]
    current_version = next((item.version_number for item in versions if item.is_current), None)
    return ListFileVersionsOutput(
        file_id=file_id,
        current_version=current_version,
        versions=versions,
    )


@router.get(
    "/files/{file_id}/versions/{from_version}/diff/{to_version}",
    response_model=DiffFileVersionsOutput,
)
async def diff_versions(
    file_id: int,
    from_version: int,
    to_version: int,
    context_lines: int = 3,
    max_diff_lines: int = 500,
):
    """Compare retained binary snapshots for two file versions."""
    row = await fetch_one("SELECT id FROM uploaded_files WHERE id = ?", (file_id,))
    if not row:
        raise HTTPException(404, "File not found")

    try:
        result = await diff_file_versions(
            file_id=file_id,
            from_version_number=from_version,
            to_version_number=to_version,
            context_lines=context_lines,
            max_diff_lines=max_diff_lines,
        )
    except FileNotFoundError as err:
        raise HTTPException(409, str(err)) from err
    except ValueError as err:
        raise HTTPException(404, str(err)) from err

    return DiffFileVersionsOutput(
        file_id=file_id,
        from_version=_version_row_to_item(result["from_version"]),
        to_version=_version_row_to_item(result["to_version"]),
        changed=bool(result["changed"]),
        additions=int(result["additions"]),
        deletions=int(result["deletions"]),
        from_line_count=int(result["from_line_count"]),
        to_line_count=int(result["to_line_count"]),
        diff_lines=result["diff_lines"],
        truncated=bool(result["truncated"]),
    )


async def _attached_kb_ids(file_id: int, kb_id: int | None = None) -> list[int]:
    if kb_id is not None:
        row = await fetch_one(
            "SELECT kb_id FROM kb_files WHERE file_id = ? AND kb_id = ?",
            (file_id, kb_id),
        )
        if not row:
            raise HTTPException(404, "File is not attached to this Knowledge Base")
        return [int(kb_id)]

    rows = await fetch_all(
        "SELECT kb_id FROM kb_files WHERE file_id = ? ORDER BY kb_id ASC",
        (file_id,),
    )
    return [int(row["kb_id"]) for row in rows]


async def _invalidate_kb_file_vectors(file_id: int, kb_ids: list[int]) -> None:
    if not kb_ids:
        return

    try:
        from app.vector_store import vector_store

        for kb_id in kb_ids:
            vector_store.delete_by_kb_and_file(kb_id, file_id)
    except Exception:
        pass

    db = await get_db()
    try:
        for kb_id in kb_ids:
            await db.execute(
                """
                UPDATE kb_files
                SET status = 'attached',
                    chunk_count = 0,
                    ingest_signature = NULL,
                    last_job_id = NULL
                WHERE kb_id = ? AND file_id = ?
                """,
                (kb_id, file_id),
            )
            await bump_kb_version(db, kb_id)
        await db.commit()
    finally:
        await db.close()


def _enqueue_rollback_reingest_jobs(
    *,
    request: Request,
    auth: Any,
    file_id: int,
    kb_ids: list[int],
) -> list[dict[str, Any]]:
    request_state = getattr(request, "state", None)
    auth_context = auth if isinstance(auth, AuthContext) else AuthContext(user_id="admin-1", roles=["admin"], channel="admin")
    jobs: list[dict[str, Any]] = []
    for kb_id in kb_ids:
        jobs.append(
            enqueue_background_job(
                job_type="kb_file_ingest",
                payload={"kb_id": kb_id, "file_id": file_id},
                context=RequestContext(
                    request_id=getattr(request_state, "request_id", None) or f"rollback-{file_id}-{kb_id}",
                    kb_id=kb_id,
                    auth=auth_context,
                ),
            )
        )
    return jobs


@router.post("/files/{file_id}/versions/{version_number}/rollback", response_model=RollbackFileVersionOutput)
async def rollback_file_version(
    file_id: int,
    version_number: int,
    request: Request,
    payload: RollbackFileVersionInput = RollbackFileVersionInput(),
    auth=Depends(require_admin),
):
    """Restore a retained version snapshot as the current file content."""
    row = await fetch_one("SELECT id FROM uploaded_files WHERE id = ?", (file_id,))
    if not row:
        raise HTTPException(404, "File not found")

    try:
        result = await restore_file_version(
            file_id=file_id,
            version_number=version_number,
            created_by_user_id=getattr(auth, "user_id", None),
            change_summary=payload.reason or f"Rollback to version {version_number}",
        )
    except FileNotFoundError as err:
        raise HTTPException(409, str(err)) from err
    except ValueError as err:
        message = str(err)
        status_code = 404 if "not found" in message.lower() else 409
        raise HTTPException(status_code, message) from err

    changed = bool(result["changed"])
    kb_ids = await _attached_kb_ids(file_id, payload.kb_id) if changed else []
    if changed:
        await _invalidate_kb_file_vectors(file_id, kb_ids)

    jobs = (
        _enqueue_rollback_reingest_jobs(
            request=request,
            auth=auth,
            file_id=file_id,
            kb_ids=kb_ids,
        )
        if changed and payload.reingest
        else []
    )

    restored_file = await fetch_one("SELECT * FROM uploaded_files WHERE id = ?", (file_id,))
    if not restored_file:
        raise HTTPException(404, "File not found after rollback")

    return RollbackFileVersionOutput(
        message=(
            f"Rolled back file {file_id} to version {version_number}"
            if changed
            else f"File {file_id} is already at version {version_number}"
        ),
        file=file_row_to_info(restored_file),
        restored_from=_version_row_to_item(result["target_version"]),
        restored_as=_version_row_to_item({**result["restored_version"], "is_current": 1}),
        changed=changed,
        jobs=jobs,
    )


@router.delete("/files/{file_id}")
async def delete_file(file_id: int, force: bool = False):
    """Delete a source file and remove it from any attached Knowledge Bases."""
    row = await fetch_one("SELECT * FROM uploaded_files WHERE id = ?", (file_id,))
    if not row:
        raise HTTPException(404, "File not found")

    mappings = await fetch_all(
        "SELECT kb_id FROM kb_files WHERE file_id = ? ORDER BY kb_id ASC",
        (file_id,),
    )
    affected_kb_ids = [int(item["kb_id"]) for item in mappings]
    if len(affected_kb_ids) > 1 and not force:
        raise HTTPException(
            409,
            "File is attached to multiple Knowledge Bases. Detach it from other KBs first, or retry with force=true.",
        )

    # Delete raw file
    file_path = settings.raw_upload_dir / row["filename"]
    if file_path.exists():
        file_path.unlink()
    for version in await list_file_versions(file_id):
        snapshot_path = version.get("snapshot_path")
        if snapshot_path:
            Path(snapshot_path).unlink(missing_ok=True)

    # Delete from vector store (imported lazily to avoid circular imports)
    try:
        from app.vector_store import vector_store
        vector_store.delete_by_source(str(file_id))
    except Exception:
        pass  # Vector store may not be initialized

    db = await get_db()
    try:
        for kb_id in affected_kb_ids:
            await bump_kb_version(db, kb_id)
        await db.execute("DELETE FROM ingest_jobs WHERE file_id = ?", (file_id,))
        await db.execute("DELETE FROM uploaded_files WHERE id = ?", (file_id,))
        await db.commit()
    finally:
        await db.close()

    return {
        "message": f"File {file_id} deleted",
        "filename": row["original_name"],
        "force": force,
        "detached_kb_ids": affected_kb_ids,
    }
