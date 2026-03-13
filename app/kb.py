"""
Knowledge Base CRUD and file mapping endpoints.
"""

from __future__ import annotations

import aiosqlite
from fastapi import APIRouter, HTTPException

from app.kb_service import (
    KB_SELECT,
    attach_file_to_kb,
    bump_kb_version,
    fetch_kb_summary,
    get_default_kb,
    get_kb_or_404,
    normalize_kb_key,
    new_kb_version,
    open_db,
    row_to_kb_summary,
)
from app.models import (
    KBFileSummary,
    KBStats,
    KnowledgeBaseCreate,
    KnowledgeBaseDeleteResponse,
    KnowledgeBaseSummary,
    KnowledgeBaseUpdate,
)
from app.vector_store import vector_store

router = APIRouter(prefix="/api/kbs", tags=["knowledge-bases"])

_STATUS_ALLOWED = {"active", "archived"}
_KB_FILE_SELECT = """
SELECT
    kf.id AS mapping_id,
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
    uf.status AS upload_status,
    uf.created_at
FROM kb_files kf
JOIN uploaded_files uf ON uf.id = kf.file_id
"""


def _row_to_kb_file_summary(row: dict) -> KBFileSummary:
    return KBFileSummary(
        kb_id=row["kb_id"],
        file_id=row["file_id"],
        mapping_id=row["mapping_id"],
        filename=row["filename"],
        original_name=row["original_name"],
        file_type=row["file_type"],
        file_size=row["file_size"],
        file_hash=row["file_hash"],
        upload_status=row["upload_status"],
        kb_status=row["kb_status"],
        chunk_count=int(row.get("chunk_count") or 0),
        ingest_signature=row.get("ingest_signature"),
        last_job_id=row.get("last_job_id"),
        attached_at=row["attached_at"],
        last_ingest_at=row.get("last_ingest_at"),
        created_at=row["created_at"],
    )


async def _fetch_kb_file_mapping(db: aiosqlite.Connection, kb_id: int, file_id: int) -> KBFileSummary | None:
    cursor = await db.execute(
        _KB_FILE_SELECT + " WHERE kf.kb_id = ? AND kf.file_id = ?",
        (kb_id, file_id),
    )
    row = await cursor.fetchone()
    return _row_to_kb_file_summary(dict(row)) if row else None


async def _sync_uploaded_file_status(db: aiosqlite.Connection, file_id: int):
    cursor = await db.execute(
        "SELECT COUNT(*) AS total FROM kb_files WHERE file_id = ? AND status = 'ingested'",
        (file_id,),
    )
    ingested_count = int((await cursor.fetchone())["total"])
    if ingested_count > 0:
        await db.execute(
            "UPDATE uploaded_files SET status = 'ingested', error_message = NULL WHERE id = ?",
            (file_id,),
        )
        return

    await db.execute(
        """
        UPDATE uploaded_files
        SET status = 'uploaded',
            ingested_at = NULL,
            error_message = NULL
        WHERE id = ?
        """,
        (file_id,),
    )


async def _build_kb_stats(kb_id: int) -> tuple[KnowledgeBaseSummary, KBStats]:
    db = await open_db()
    try:
        kb = await get_kb_or_404(db, kb_id)
        cursor = await db.execute(
            """
            SELECT
                COUNT(*) AS total_files,
                COALESCE(SUM(CASE WHEN status = 'ingested' THEN 1 ELSE 0 END), 0) AS ingested_files
            FROM kb_files
            WHERE kb_id = ?
            """,
            (kb_id,),
        )
        row = await cursor.fetchone()
    finally:
        await db.close()

    where = {"kb_id": int(kb_id)}
    total_vectors = vector_store.count_by_where(where)
    return kb, KBStats(
        total_files=int(row["total_files"] or 0),
        ingested_files=int(row["ingested_files"] or 0),
        total_chunks=total_vectors,
        total_vectors=total_vectors,
        sources=vector_store.get_sources(where),
        scope="kb",
        kb_id=kb.id,
        kb_key=kb.key,
        kb_name=kb.name,
        kb_version=kb.kb_version,
        is_default=kb.is_default,
    )


@router.get("", response_model=list[KnowledgeBaseSummary])
async def list_knowledge_bases():
    db = await open_db()
    try:
        cursor = await db.execute(
            KB_SELECT
            + """
            GROUP BY
                kb.id, kb.key, kb.name, kb.description, kb.status,
                kb.is_default, kb.kb_version, kb.created_at, kb.updated_at
            ORDER BY kb.is_default DESC, kb.created_at ASC
            """
        )
        rows = await cursor.fetchall()
        return [row_to_kb_summary(dict(row)) for row in rows]
    finally:
        await db.close()


@router.post("", response_model=KnowledgeBaseSummary)
async def create_knowledge_base(payload: KnowledgeBaseCreate):
    normalized_key = normalize_kb_key(payload.key)
    db = await open_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) AS total FROM knowledge_bases")
        total = int((await cursor.fetchone())["total"])
        make_default = payload.is_default or total == 0

        if make_default:
            await db.execute("UPDATE knowledge_bases SET is_default = 0")

        try:
            insert_cursor = await db.execute(
                """
                INSERT INTO knowledge_bases
                    (key, name, description, status, is_default, kb_version, created_at, updated_at)
                VALUES (?, ?, ?, 'active', ?, ?, datetime('now'), datetime('now'))
                """,
                (
                    normalized_key,
                    payload.name.strip(),
                    payload.description.strip() if payload.description else None,
                    1 if make_default else 0,
                    new_kb_version(),
                ),
            )
        except aiosqlite.IntegrityError as err:
            raise HTTPException(409, f"Knowledge Base key already exists: {normalized_key}") from err

        await db.commit()
        created = await fetch_kb_summary(db, "kb.id = ?", (insert_cursor.lastrowid,))
        if not created:
            raise HTTPException(500, "Failed to load created Knowledge Base")
        return created
    finally:
        await db.close()


@router.get("/default", response_model=KnowledgeBaseSummary)
async def get_default_knowledge_base():
    db = await open_db()
    try:
        return await get_default_kb(db)
    finally:
        await db.close()


@router.get("/{kb_id}/stats", response_model=KBStats)
async def get_knowledge_base_stats(kb_id: int):
    _, stats = await _build_kb_stats(kb_id)
    return stats


@router.get("/{kb_id}/sources/stats")
async def get_knowledge_base_source_stats(kb_id: int):
    db = await open_db()
    try:
        await get_kb_or_404(db, kb_id)
    finally:
        await db.close()
    return vector_store.get_source_stats({"kb_id": int(kb_id)})


@router.get("/{kb_id}", response_model=KnowledgeBaseSummary)
async def get_knowledge_base(kb_id: int):
    db = await open_db()
    try:
        return await get_kb_or_404(db, kb_id)
    finally:
        await db.close()


@router.patch("/{kb_id}", response_model=KnowledgeBaseSummary)
async def update_knowledge_base(kb_id: int, payload: KnowledgeBaseUpdate):
    changes = payload.model_dump(exclude_unset=True)
    db = await open_db()
    try:
        current = await get_kb_or_404(db, kb_id)

        assignments: list[str] = []
        params: list[object] = []

        if "key" in changes:
            assignments.append("key = ?")
            params.append(normalize_kb_key(str(changes["key"])))

        if "name" in changes:
            name = str(changes["name"]).strip()
            if not name:
                raise HTTPException(400, "Knowledge Base name cannot be empty")
            assignments.append("name = ?")
            params.append(name)

        if "description" in changes:
            description = changes["description"]
            assignments.append("description = ?")
            params.append(description.strip() if isinstance(description, str) else None)

        if "status" in changes:
            status = str(changes["status"]).strip().lower()
            if status not in _STATUS_ALLOWED:
                raise HTTPException(400, f"Invalid status. Allowed: {sorted(_STATUS_ALLOWED)}")
            assignments.append("status = ?")
            params.append(status)

        if "is_default" in changes:
            requested_default = bool(changes["is_default"])
            if not requested_default and current.is_default:
                raise HTTPException(400, "Cannot unset the default Knowledge Base directly")
            if requested_default:
                await db.execute("UPDATE knowledge_bases SET is_default = 0")
                assignments.append("is_default = 1")

        if not assignments:
            return current

        assignments.append("updated_at = datetime('now')")
        params.append(kb_id)

        try:
            await db.execute(
                f"UPDATE knowledge_bases SET {', '.join(assignments)} WHERE id = ?",
                tuple(params),
            )
        except aiosqlite.IntegrityError as err:
            raise HTTPException(409, "Knowledge Base key already exists") from err

        await db.commit()
        updated = await fetch_kb_summary(db, "kb.id = ?", (kb_id,))
        if not updated:
            raise HTTPException(500, "Failed to load updated Knowledge Base")
        return updated
    finally:
        await db.close()


@router.delete("/{kb_id}", response_model=KnowledgeBaseDeleteResponse)
async def delete_knowledge_base(kb_id: int):
    db = await open_db()
    try:
        current = await get_kb_or_404(db, kb_id)

        cursor = await db.execute("SELECT COUNT(*) AS total FROM knowledge_bases")
        total = int((await cursor.fetchone())["total"])
        if total <= 1:
            raise HTTPException(400, "Cannot delete the only Knowledge Base")
        if current.is_default:
            raise HTTPException(400, "Cannot delete the default Knowledge Base. Set another KB as default first.")

        cursor = await db.execute("SELECT file_id FROM kb_files WHERE kb_id = ?", (kb_id,))
        file_ids = [int(row["file_id"]) for row in await cursor.fetchall()]
        vector_store.delete_by_kb(kb_id)
        await db.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
        for file_id in file_ids:
            await _sync_uploaded_file_status(db, file_id)
        await db.commit()
        return KnowledgeBaseDeleteResponse(
            message=f"Knowledge Base '{current.name}' deleted",
            id=current.id,
            key=current.key,
        )
    finally:
        await db.close()


@router.get("/{kb_id}/files", response_model=list[KBFileSummary])
async def list_kb_files(kb_id: int):
    db = await open_db()
    try:
        await get_kb_or_404(db, kb_id)
        cursor = await db.execute(
            _KB_FILE_SELECT + " WHERE kf.kb_id = ? ORDER BY kf.attached_at DESC",
            (kb_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_kb_file_summary(dict(row)) for row in rows]
    finally:
        await db.close()


@router.post("/{kb_id}/files/{file_id}", response_model=KBFileSummary)
async def attach_kb_file(kb_id: int, file_id: int):
    db = await open_db()
    try:
        await get_kb_or_404(db, kb_id)
        cursor = await db.execute("SELECT id, status FROM uploaded_files WHERE id = ?", (file_id,))
        file_row = await cursor.fetchone()
        if not file_row:
            raise HTTPException(404, "File not found")

        attachment = await attach_file_to_kb(db, kb_id, file_id, status="attached")
        if attachment.get("was_created"):
            await bump_kb_version(db, kb_id)
        await db.commit()

        mapping = await _fetch_kb_file_mapping(db, kb_id, file_id)
        if not mapping:
            raise HTTPException(500, "Failed to load KB file mapping")
        return mapping
    finally:
        await db.close()


@router.delete("/{kb_id}/files/{file_id}")
async def detach_kb_file(kb_id: int, file_id: int):
    db = await open_db()
    try:
        await get_kb_or_404(db, kb_id)
        mapping = await _fetch_kb_file_mapping(db, kb_id, file_id)
        if not mapping:
            raise HTTPException(404, "File is not attached to this Knowledge Base")

        vector_store.delete_by_kb_and_file(kb_id, file_id)
        await db.execute(
            "DELETE FROM kb_files WHERE kb_id = ? AND file_id = ?",
            (kb_id, file_id),
        )
        await _sync_uploaded_file_status(db, file_id)
        await bump_kb_version(db, kb_id)
        await db.commit()
        return {
            "message": "File detached from Knowledge Base",
            "kb_id": kb_id,
            "file_id": file_id,
        }
    finally:
        await db.close()
