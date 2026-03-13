"""
Shared Knowledge Base helpers for CRUD and ingest flows.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import aiosqlite
from fastapi import HTTPException

from app.database import get_db, utcnow_iso
from app.models import KnowledgeBaseSummary

_KEY_SANITIZE_RE = re.compile(r"[^a-z0-9_-]+")

KB_SELECT = """
SELECT
    kb.id,
    kb.key,
    kb.name,
    kb.description,
    kb.status,
    kb.is_default,
    kb.kb_version,
    kb.created_at,
    kb.updated_at,
    COUNT(kf.id) AS file_count,
    COALESCE(SUM(CASE WHEN kf.status = 'ingested' THEN 1 ELSE 0 END), 0) AS ingested_file_count
FROM knowledge_bases kb
LEFT JOIN kb_files kf ON kf.kb_id = kb.id
"""


def new_kb_version() -> str:
    return uuid.uuid4().hex[:12]


def normalize_kb_key(raw: str) -> str:
    key = _KEY_SANITIZE_RE.sub("-", raw.strip().lower()).strip("-_")
    key = re.sub(r"-{2,}", "-", key)
    if not key:
        raise HTTPException(400, "Knowledge Base key is invalid after normalization")
    return key


def row_to_kb_summary(row: dict[str, Any]) -> KnowledgeBaseSummary:
    return KnowledgeBaseSummary(
        id=row["id"],
        key=row["key"],
        name=row["name"],
        description=row.get("description"),
        status=row["status"],
        is_default=bool(row["is_default"]),
        kb_version=row["kb_version"],
        file_count=int(row.get("file_count") or 0),
        ingested_file_count=int(row.get("ingested_file_count") or 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def fetch_kb_summary(db: aiosqlite.Connection, where_sql: str, params: tuple) -> KnowledgeBaseSummary | None:
    cursor = await db.execute(
        KB_SELECT
        + f"""
        WHERE {where_sql}
        GROUP BY
            kb.id, kb.key, kb.name, kb.description, kb.status,
            kb.is_default, kb.kb_version, kb.created_at, kb.updated_at
        """,
        params,
    )
    row = await cursor.fetchone()
    return row_to_kb_summary(dict(row)) if row else None


async def get_kb_or_404(db: aiosqlite.Connection, kb_id: int) -> KnowledgeBaseSummary:
    kb = await fetch_kb_summary(db, "kb.id = ?", (kb_id,))
    if not kb:
        raise HTTPException(404, "Knowledge Base not found")
    return kb


async def get_default_kb(db: aiosqlite.Connection) -> KnowledgeBaseSummary:
    kb = await fetch_kb_summary(db, "kb.is_default = 1", ())
    if not kb:
        raise HTTPException(404, "Default Knowledge Base not found")
    return kb


async def resolve_kb_scope(
    db: aiosqlite.Connection,
    *,
    kb_id: int | None = None,
    kb_key: str | None = None,
) -> KnowledgeBaseSummary:
    if kb_id is not None:
        return await get_kb_or_404(db, kb_id)

    if kb_key:
        normalized_key = normalize_kb_key(kb_key)
        kb = await fetch_kb_summary(db, "kb.key = ?", (normalized_key,))
        if not kb:
            raise HTTPException(404, f"Knowledge Base not found for key '{normalized_key}'")
        return kb

    return await get_default_kb(db)


async def bump_kb_version(db: aiosqlite.Connection, kb_id: int) -> str:
    version = new_kb_version()
    await db.execute(
        "UPDATE knowledge_bases SET kb_version = ?, updated_at = ? WHERE id = ?",
        (version, utcnow_iso(), kb_id),
    )
    return version


async def attach_file_to_kb(
    db: aiosqlite.Connection,
    kb_id: int,
    file_id: int,
    status: str = "attached",
) -> dict[str, Any]:
    now = utcnow_iso()
    await db.execute(
        """
        INSERT OR IGNORE INTO kb_files (
            kb_id, file_id, status, chunk_count, attached_at
        ) VALUES (?, ?, ?, 0, ?)
        """,
        (kb_id, file_id, status, now),
    )
    changes_cursor = await db.execute("SELECT changes() AS total")
    created = int((await changes_cursor.fetchone())["total"] or 0) > 0
    cursor = await db.execute(
        """
        SELECT id, kb_id, file_id, status, chunk_count, ingest_signature, last_job_id,
               attached_at, last_ingest_at
        FROM kb_files
        WHERE kb_id = ? AND file_id = ?
        """,
        (kb_id, file_id),
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(500, "Failed to attach file to Knowledge Base")
    payload = dict(row)
    payload["was_created"] = created
    return payload


async def open_db() -> aiosqlite.Connection:
    return await get_db()
