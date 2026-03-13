"""
Secure file upload endpoint.
- Extension whitelist + magic bytes check (no python-magic-bin)
- SHA256 file hash for dedup
- UUID filename prefix to prevent collisions
- Path traversal protection
"""

import hashlib
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from app.config import settings
from app.database import execute_with_retry, fetch_all, fetch_one, get_db
from app.kb_service import bump_kb_version
from app.models import FileInfo, UploadResponse

router = APIRouter(prefix="/api", tags=["upload"])

# Magic bytes signatures for validation
MAGIC_BYTES = {
    ".pdf":  [b"%PDF"],
    ".xlsx": [b"PK\x03\x04"],          # ZIP-based
    ".xls":  [b"\xd0\xcf\x11\xe0"],    # OLE2
    ".html": [b"<!DOCTYPE", b"<html", b"<!doctype", b"<HTML"],
    ".htm":  [b"<!DOCTYPE", b"<html", b"<!doctype", b"<HTML"],
    ".txt":  [],
    ".md":   [],
    ".csv":  [],                        # No reliable magic bytes for CSV
}


def sanitize_filename(name: str) -> str:
    """Remove path traversal and dangerous characters."""
    name = Path(name).name  # strip directory components
    name = name.replace("..", "").replace("/", "").replace("\\", "")
    # Remove non-ASCII control characters
    name = "".join(c for c in name if ord(c) >= 32)
    return name.strip() or "unnamed"


def check_magic_bytes(content: bytes, extension: str) -> bool:
    """Validate file content matches expected magic bytes."""
    signatures = MAGIC_BYTES.get(extension, [])
    if not signatures:
        return True  # No signature to check (e.g., CSV)
    return any(content[:20].startswith(sig) or sig in content[:100]
               for sig in signatures)


def compute_file_hash(content: bytes) -> str:
    """SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def file_row_to_info(row: dict) -> FileInfo:
    return FileInfo(
        id=row["id"],
        filename=row["filename"],
        original_name=row["original_name"],
        file_type=row["file_type"],
        file_size=row["file_size"],
        file_hash=row["file_hash"],
        status=row["status"],
        parser_type=row.get("parser_type"),
        pages_or_rows=row.get("pages_or_rows"),
        ingested_at=row.get("ingested_at"),
        error_message=row.get("error_message"),
        created_at=row["created_at"],
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file with security validation."""
    # 1. Check filename exists
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    original_name = sanitize_filename(file.filename)

    # 2. Check extension
    ext = Path(original_name).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            400,
            f"File type '{ext}' not allowed. Allowed: {settings.allowed_extensions}"
        )

    # 3. Read content and check size
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty file rejected")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            400,
            f"File too large ({len(content) / 1024 / 1024:.1f}MB). "
            f"Max: {settings.max_upload_size_mb}MB"
        )

    # 4. Magic bytes validation
    if not check_magic_bytes(content, ext):
        raise HTTPException(400, f"File content does not match '{ext}' format")

    # 5. Compute hash for dedup
    file_hash = compute_file_hash(content)

    # 6. Save file with UUID prefix
    safe_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
    save_path = settings.raw_upload_dir / safe_name
    settings.raw_upload_dir.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)

    # 7. Detect parser type from extension
    parser_map = {
        ".xlsx": "excel", ".xls": "excel",
        ".csv": "csv",
        ".pdf": "pdf",
        ".html": "html", ".htm": "html",
        ".txt": "text", ".md": "text",
        ".docx": "docx",
        ".json": "json", ".jsonl": "jsonl",
    }
    parser_type = parser_map.get(ext, "unknown")

    # 8. Insert into database
    cursor = await execute_with_retry(
        """INSERT INTO uploaded_files
           (filename, original_name, file_type, file_size, file_hash, status, parser_type)
           VALUES (?, ?, ?, ?, ?, 'uploaded', ?)""",
        (safe_name, original_name, ext, len(content), file_hash, parser_type)
    )

    # 9. Fetch inserted record
    row = await fetch_one("SELECT * FROM uploaded_files WHERE id = ?",
                          (cursor.lastrowid,))

    db = await get_db()
    try:
        kb_cursor = await db.execute(
            "SELECT id FROM knowledge_bases WHERE is_default = 1 LIMIT 1"
        )
        default_kb = await kb_cursor.fetchone()
        if default_kb:
            await db.execute(
                """
                INSERT OR IGNORE INTO kb_files (
                    kb_id, file_id, status, chunk_count, attached_at
                ) VALUES (?, ?, 'attached', 0, datetime('now'))
                """,
                (default_kb["id"], row["id"]),
            )
            await db.commit()
    finally:
        await db.close()

    return UploadResponse(
        message=f"File '{original_name}' uploaded successfully",
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
