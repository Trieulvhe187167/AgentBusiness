"""Secure file upload endpoints."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.auth import require_admin
from app.config import settings
from app.database import fetch_all, fetch_one, get_db
from app.kb_service import bump_kb_version
from app.models import FileInfo, UploadResponse
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


@router.get("/files", response_model=list[FileInfo])
async def list_files():
    """List all uploaded files with status."""
    rows = await fetch_all(
        "SELECT * FROM uploaded_files ORDER BY created_at DESC"
    )
    return [file_row_to_info(r) for r in rows]


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
