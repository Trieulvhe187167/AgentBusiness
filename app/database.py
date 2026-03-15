"""
SQLite helpers, schema migrations, and bootstrap data.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from datetime import datetime, timezone

import aiosqlite

from app.config import settings

DB_PATH = str(settings.sqlite_path)

_CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploaded_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    original_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'uploaded',
    parser_type TEXT,
    pages_or_rows INTEGER,
    ingested_at TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id TEXT PRIMARY KEY,
    file_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    progress REAL NOT NULL DEFAULT 0.0,
    error_message TEXT,
    started_at TEXT,
    finished_at TEXT,
    FOREIGN KEY (file_id) REFERENCES uploaded_files(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    pending_clarify_query TEXT,
    pending_clarify_category TEXT,
    pending_clarify_lang TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chat_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    user_message TEXT NOT NULL,
    merged_query TEXT,
    mode TEXT NOT NULL,
    top_score REAL,
    answer_text TEXT,
    citations_json TEXT,
    latency_ms INTEGER,
    llm_provider TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_uploaded_files_status ON uploaded_files(status);
CREATE INDEX IF NOT EXISTS idx_chat_logs_session_time ON chat_logs(session_id, created_at DESC);
"""

_KB_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    is_default INTEGER NOT NULL DEFAULT 0,
    kb_version TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kb_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'attached',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    ingest_signature TEXT,
    last_job_id TEXT,
    attached_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_ingest_at TEXT,
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    FOREIGN KEY (file_id) REFERENCES uploaded_files(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_bases_key
    ON knowledge_bases(key);

CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_bases_default
    ON knowledge_bases(is_default)
    WHERE is_default = 1;

CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_files_kb_file
    ON kb_files(kb_id, file_id);

CREATE INDEX IF NOT EXISTS idx_kb_files_kb_id ON kb_files(kb_id);
CREATE INDEX IF NOT EXISTS idx_kb_files_file_id ON kb_files(file_id);
CREATE INDEX IF NOT EXISTS idx_kb_files_status ON kb_files(status);
"""

_INGEST_JOB_KB_SCHEMA = """
ALTER TABLE ingest_jobs ADD COLUMN kb_id INTEGER;
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_kb_id ON ingest_jobs(kb_id);
"""

_PHASE1_CONTEXT_AND_AUDIT_SCHEMA = """
ALTER TABLE chat_logs ADD COLUMN request_id TEXT;
ALTER TABLE chat_logs ADD COLUMN user_id TEXT;
ALTER TABLE chat_logs ADD COLUMN roles_json TEXT;
ALTER TABLE chat_logs ADD COLUMN channel TEXT;
ALTER TABLE chat_logs ADD COLUMN tenant_id TEXT;
ALTER TABLE chat_logs ADD COLUMN org_id TEXT;
ALTER TABLE chat_logs ADD COLUMN kb_id INTEGER;
ALTER TABLE chat_logs ADD COLUMN kb_key TEXT;
CREATE INDEX IF NOT EXISTS idx_chat_logs_request_id ON chat_logs(request_id);
CREATE INDEX IF NOT EXISTS idx_chat_logs_user_id ON chat_logs(user_id);

CREATE TABLE IF NOT EXISTS tool_audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_call_id TEXT NOT NULL,
    request_id TEXT,
    session_id TEXT,
    user_id TEXT,
    roles_json TEXT,
    channel TEXT,
    tenant_id TEXT,
    org_id TEXT,
    kb_id INTEGER,
    kb_key TEXT,
    tool_name TEXT NOT NULL,
    args_json TEXT,
    result_summary TEXT,
    tool_status TEXT NOT NULL,
    latency_ms INTEGER,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tool_audit_logs_request_id ON tool_audit_logs(request_id);
CREATE INDEX IF NOT EXISTS idx_tool_audit_logs_tool_call_id ON tool_audit_logs(tool_call_id);
CREATE INDEX IF NOT EXISTS idx_tool_audit_logs_session_time ON tool_audit_logs(session_id, created_at DESC);
"""

_PHASE2_SUPPORT_TICKETS_SCHEMA = """
CREATE TABLE IF NOT EXISTS support_tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_code TEXT NOT NULL UNIQUE,
    issue_type TEXT NOT NULL,
    message TEXT NOT NULL,
    contact TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_by_user_id TEXT,
    channel TEXT,
    tenant_id TEXT,
    org_id TEXT,
    kb_id INTEGER,
    kb_key TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_support_tickets_ticket_code ON support_tickets(ticket_code);
CREATE INDEX IF NOT EXISTS idx_support_tickets_user_id ON support_tickets(created_by_user_id);
CREATE INDEX IF NOT EXISTS idx_support_tickets_created_at ON support_tickets(created_at DESC);
"""

_PHASE5_SLOT_MEMORY_SCHEMA = """
ALTER TABLE chat_sessions ADD COLUMN slots_json TEXT;
"""

_PHASE19_EXTERNAL_INTEGRATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS order_status_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_code TEXT NOT NULL,
    user_id TEXT,
    status TEXT NOT NULL,
    last_update TEXT,
    tracking_code TEXT,
    carrier TEXT,
    source TEXT NOT NULL DEFAULT 'snapshot',
    raw_json TEXT,
    cached_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_order_status_cache_code ON order_status_cache(order_code);
CREATE INDEX IF NOT EXISTS idx_order_status_cache_user_time ON order_status_cache(user_id, cached_at DESC);

CREATE TABLE IF NOT EXISTS game_online_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alliance_id TEXT NOT NULL,
    server_id TEXT,
    server_scope TEXT NOT NULL,
    online_count INTEGER NOT NULL,
    observed_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'snapshot',
    raw_json TEXT,
    cached_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_game_online_cache_scope ON game_online_cache(alliance_id, server_scope);
CREATE INDEX IF NOT EXISTS idx_game_online_cache_time ON game_online_cache(cached_at DESC);
"""

MIGRATIONS: list[tuple[str, str]] = [
    ("001_core_schema", _CORE_SCHEMA),
    ("002_knowledge_bases", _KB_SCHEMA),
    ("003_ingest_jobs_kb_id", _INGEST_JOB_KB_SCHEMA),
    ("004_phase1_context_and_audit", _PHASE1_CONTEXT_AND_AUDIT_SCHEMA),
    ("005_phase2_support_tickets", _PHASE2_SUPPORT_TICKETS_SCHEMA),
    ("006_phase5_slot_memory", _PHASE5_SLOT_MEMORY_SCHEMA),
    ("007_phase19_external_integrations", _PHASE19_EXTERNAL_INTEGRATIONS_SCHEMA),
]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_kb_version() -> str:
    return uuid.uuid4().hex[:12]


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=5000")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    settings.ensure_dirs()
    db = await get_db()
    try:
        await _ensure_migrations_table(db)
        applied = await _get_applied_migrations(db)
        for version, sql in MIGRATIONS:
            if version in applied:
                continue
            await db.executescript(sql)
            await db.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, utcnow_iso()),
            )
            await db.commit()

        await _ensure_default_kb(db)
        await _backfill_default_kb_mappings(db)
        await db.commit()
    finally:
        await db.close()


async def execute_with_retry(query: str, params: tuple = (), max_retries: int = 3):
    db = await get_db()
    try:
        for attempt in range(max_retries):
            try:
                cursor = await db.execute(query, params)
                await db.commit()
                return cursor
            except aiosqlite.OperationalError as err:
                if "database is locked" in str(err) and attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
                raise
    finally:
        await db.close()


async def fetch_all(query: str, params: tuple = ()) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


async def fetch_one(query: str, params: tuple = ()) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute(query, params)
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


def _sync_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def execute_sync(query: str, params: tuple = ()) -> int | None:
    conn = _sync_conn()
    try:
        cur = conn.execute(query, params)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def fetch_one_sync(query: str, params: tuple = ()) -> dict | None:
    conn = _sync_conn()
    try:
        cur = conn.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def fetch_all_sync(query: str, params: tuple = ()) -> list[dict]:
    conn = _sync_conn()
    try:
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


async def _ensure_migrations_table(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    await db.commit()


async def _get_applied_migrations(db: aiosqlite.Connection) -> set[str]:
    cursor = await db.execute("SELECT version FROM schema_migrations")
    rows = await cursor.fetchall()
    return {row["version"] for row in rows}


async def _ensure_default_kb(db: aiosqlite.Connection):
    cursor = await db.execute(
        "SELECT id, is_default FROM knowledge_bases ORDER BY id ASC"
    )
    rows = await cursor.fetchall()
    if not rows:
        now = utcnow_iso()
        await db.execute(
            """
            INSERT INTO knowledge_bases
                (key, name, description, status, is_default, kb_version, created_at, updated_at)
            VALUES (?, ?, ?, 'active', 1, ?, ?, ?)
            """,
            (
                "default",
                "Default KB",
                "Auto-created default knowledge base",
                _new_kb_version(),
                now,
                now,
            ),
        )
        return

    default_row = next((row for row in rows if int(row["is_default"]) == 1), None)
    if default_row is None:
        await db.execute("UPDATE knowledge_bases SET is_default = 0")
        await db.execute(
            "UPDATE knowledge_bases SET is_default = 1, updated_at = ? WHERE id = ?",
            (utcnow_iso(), rows[0]["id"]),
        )

    missing_versions = await db.execute(
        "SELECT id FROM knowledge_bases WHERE kb_version = '' OR kb_version IS NULL"
    )
    empty_rows = await missing_versions.fetchall()
    for row in empty_rows:
        await db.execute(
            "UPDATE knowledge_bases SET kb_version = ?, updated_at = ? WHERE id = ?",
            (_new_kb_version(), utcnow_iso(), row["id"]),
        )


async def _backfill_default_kb_mappings(db: aiosqlite.Connection):
    cursor = await db.execute(
        "SELECT id FROM knowledge_bases WHERE is_default = 1 LIMIT 1"
    )
    default_kb = await cursor.fetchone()
    if not default_kb:
        return

    now = utcnow_iso()
    await db.execute(
        """
        INSERT INTO kb_files (
            kb_id,
            file_id,
            status,
            chunk_count,
            attached_at,
            last_ingest_at
        )
        SELECT
            ?,
            uf.id,
            CASE
                WHEN uf.status = 'ingested' THEN 'ingested'
                WHEN uf.status = 'failed' THEN 'failed'
                ELSE 'attached'
            END,
            COALESCE(uf.pages_or_rows, 0),
            COALESCE(uf.created_at, ?),
            uf.ingested_at
        FROM uploaded_files uf
        WHERE NOT EXISTS (
            SELECT 1
            FROM kb_files kf
            WHERE kf.kb_id = ? AND kf.file_id = uf.id
        )
        """,
        (default_kb["id"], now, default_kb["id"]),
    )
