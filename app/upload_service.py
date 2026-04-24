"""
Shared upload/import helpers for HTTP uploads and sync integrations.
"""

from __future__ import annotations

import uuid

from fastapi import HTTPException

from app.config import settings
from app.database import execute_with_retry, fetch_one
from app.kb_service import attach_file_to_kb, get_default_kb, open_db
from app.upload_validation import (
    UploadValidationError,
    compute_file_hash,
    validate_upload,
)


def _upload_error(status_code: int, code: str, message: str, **meta):
    detail = {"code": code, "message": message}
    clean_meta = {key: value for key, value in meta.items() if value is not None}
    if clean_meta:
        detail["meta"] = clean_meta
    raise HTTPException(status_code=status_code, detail=detail)


async def import_content_to_uploaded_file(
    *,
    filename: str,
    content: bytes,
    kb_id: int | None = None,
    existing_file_id: int | None = None,
    access_level: str = "public",
    tenant_id: str | None = None,
    org_id: str | None = None,
    owner_user_id: str | None = None,
) -> dict:
    try:
        validated = validate_upload(
            filename=filename,
            content=content,
            allowed_extensions=settings.allowed_extensions,
            max_upload_bytes=settings.max_upload_bytes,
            max_upload_size_mb=settings.max_upload_size_mb,
        )
    except UploadValidationError as err:
        _upload_error(400, err.code, err.message, **err.meta)

    original_name = validated.original_name
    ext = validated.extension
    file_hash = compute_file_hash(content)

    if existing_file_id is not None:
        existing = await fetch_one("SELECT * FROM uploaded_files WHERE id = ?", (existing_file_id,))
        if not existing:
            raise HTTPException(status_code=404, detail=f"Uploaded file {existing_file_id} not found")
        safe_name = existing["filename"]
    else:
        safe_name = f"{uuid.uuid4().hex[:8]}_{original_name}"

    save_path = settings.raw_upload_dir / safe_name
    settings.raw_upload_dir.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)

    if existing_file_id is not None:
        await execute_with_retry(
            """
            UPDATE uploaded_files
            SET filename = ?,
                original_name = ?,
                file_type = ?,
                file_size = ?,
                file_hash = ?,
                status = 'uploaded',
                access_level = ?,
                tenant_id = ?,
                org_id = ?,
                owner_user_id = ?,
                parser_type = ?,
                pages_or_rows = NULL,
                ingested_at = NULL,
                error_message = NULL
            WHERE id = ?
            """,
            (
                safe_name,
                original_name,
                ext,
                len(content),
                file_hash,
                access_level,
                tenant_id,
                org_id,
                owner_user_id,
                validated.parser_type,
                existing_file_id,
            ),
        )
        row = await fetch_one("SELECT * FROM uploaded_files WHERE id = ?", (existing_file_id,))
    else:
        cursor = await execute_with_retry(
            """
            INSERT INTO uploaded_files (
                filename,
                original_name,
                file_type,
                file_size,
                file_hash,
                status,
                access_level,
                tenant_id,
                org_id,
                owner_user_id,
                parser_type
            ) VALUES (?, ?, ?, ?, ?, 'uploaded', ?, ?, ?, ?, ?)
            """,
            (
                safe_name,
                original_name,
                ext,
                len(content),
                file_hash,
                access_level,
                tenant_id,
                org_id,
                owner_user_id,
                validated.parser_type,
            ),
        )
        row = await fetch_one("SELECT * FROM uploaded_files WHERE id = ?", (cursor.lastrowid,))

    if not row:
        raise HTTPException(status_code=500, detail="Uploaded file record was not persisted")

    db = await open_db()
    try:
        target_kb_id = kb_id
        if target_kb_id is None:
            default_kb = await get_default_kb(db)
            target_kb_id = default_kb.id
        await attach_file_to_kb(db, target_kb_id, row["id"], status="attached")
        await db.commit()
    finally:
        await db.close()

    return row
