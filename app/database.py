"""
SQLite helpers, schema migrations, and bootstrap data.

Schema ownership by phase:
- Phase 0 / MVP core: uploads, ingest jobs, KB metadata, kb_files, chat logs
- Phase 1+: request context, tool audit, slot memory
- Phase 2+: support tickets
- Phase 19+: external integration caches
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from datetime import datetime, timezone

import aiosqlite

from app.config import settings

DB_PATH = str(settings.sqlite_path)

# ---------------------------------------------------------------------------
# Phase 0 / MVP core schema
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Phase 0.5 / KB-scoped ingest
# ---------------------------------------------------------------------------
_INGEST_JOB_KB_SCHEMA = """
ALTER TABLE ingest_jobs ADD COLUMN kb_id INTEGER;
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_kb_id ON ingest_jobs(kb_id);
"""

# ---------------------------------------------------------------------------
# Phase 1 / request context + audit
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Phase 2 / support tooling
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Phase 5 / session memory
# ---------------------------------------------------------------------------
_PHASE5_SLOT_MEMORY_SCHEMA = """
ALTER TABLE chat_sessions ADD COLUMN slots_json TEXT;
"""

# ---------------------------------------------------------------------------
# Phase 19 / external integration caches
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Phase 22 / KB access levels
# ---------------------------------------------------------------------------
_PHASE22_KB_ACCESS_LEVEL_SCHEMA = """
ALTER TABLE knowledge_bases ADD COLUMN access_level TEXT NOT NULL DEFAULT 'public';
CREATE INDEX IF NOT EXISTS idx_knowledge_bases_access_level ON knowledge_bases(access_level);
"""

# ---------------------------------------------------------------------------
# Phase 23 / KB tenant-org scope + file ACL metadata
# ---------------------------------------------------------------------------
_PHASE23_KB_SCOPE_SCHEMA = """
ALTER TABLE knowledge_bases ADD COLUMN tenant_id TEXT;
ALTER TABLE knowledge_bases ADD COLUMN org_id TEXT;
CREATE INDEX IF NOT EXISTS idx_knowledge_bases_tenant_id ON knowledge_bases(tenant_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_bases_org_id ON knowledge_bases(org_id);
"""

_PHASE23_FILE_ACL_SCHEMA = """
ALTER TABLE uploaded_files ADD COLUMN access_level TEXT NOT NULL DEFAULT 'public';
ALTER TABLE uploaded_files ADD COLUMN tenant_id TEXT;
ALTER TABLE uploaded_files ADD COLUMN org_id TEXT;
ALTER TABLE uploaded_files ADD COLUMN owner_user_id TEXT;
CREATE INDEX IF NOT EXISTS idx_uploaded_files_access_level ON uploaded_files(access_level);
CREATE INDEX IF NOT EXISTS idx_uploaded_files_tenant_id ON uploaded_files(tenant_id);
CREATE INDEX IF NOT EXISTS idx_uploaded_files_org_id ON uploaded_files(org_id);
CREATE INDEX IF NOT EXISTS idx_uploaded_files_owner_user_id ON uploaded_files(owner_user_id);
"""

# ---------------------------------------------------------------------------
# Phase 24 / explicit ACL tables + order scope cache
# ---------------------------------------------------------------------------
_PHASE24_ACL_TABLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_acl (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_id INTEGER NOT NULL,
    subject_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    permission TEXT NOT NULL DEFAULT 'read',
    effect TEXT NOT NULL DEFAULT 'allow',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (kb_id, subject_type, subject_id, permission, effect),
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_kb_acl_scope ON kb_acl(kb_id, subject_type, subject_id);

CREATE TABLE IF NOT EXISTS document_acl (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    subject_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    permission TEXT NOT NULL DEFAULT 'read',
    effect TEXT NOT NULL DEFAULT 'allow',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (file_id, subject_type, subject_id, permission, effect),
    FOREIGN KEY (file_id) REFERENCES uploaded_files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_document_acl_scope ON document_acl(file_id, subject_type, subject_id);
"""

_PHASE24_ORDER_SCOPE_SCHEMA = """
ALTER TABLE order_status_cache ADD COLUMN tenant_id TEXT;
ALTER TABLE order_status_cache ADD COLUMN org_id TEXT;
CREATE INDEX IF NOT EXISTS idx_order_status_cache_tenant_id ON order_status_cache(tenant_id);
CREATE INDEX IF NOT EXISTS idx_order_status_cache_org_id ON order_status_cache(org_id);
"""

# ---------------------------------------------------------------------------
# Phase 25 / authorization audit
# ---------------------------------------------------------------------------
_PHASE25_AUTH_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS auth_audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    user_id TEXT,
    roles_json TEXT,
    channel TEXT,
    tenant_id TEXT,
    org_id TEXT,
    resource_type TEXT NOT NULL,
    resource_id TEXT,
    action TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_auth_audit_logs_request_id ON auth_audit_logs(request_id);
CREATE INDEX IF NOT EXISTS idx_auth_audit_logs_user_time ON auth_audit_logs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_auth_audit_logs_resource ON auth_audit_logs(resource_type, resource_id, created_at DESC);
"""

# ---------------------------------------------------------------------------
# Phase 26 / Google Drive sync sources
# ---------------------------------------------------------------------------
_PHASE26_GOOGLE_DRIVE_SYNC_SCHEMA = """
CREATE TABLE IF NOT EXISTS google_drive_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    folder_id TEXT NOT NULL,
    shared_drive_id TEXT,
    recursive INTEGER NOT NULL DEFAULT 1,
    include_patterns_json TEXT,
    exclude_patterns_json TEXT,
    supported_mime_types_json TEXT,
    delete_policy TEXT NOT NULL DEFAULT 'detach',
    status TEXT NOT NULL DEFAULT 'active',
    tenant_id TEXT,
    org_id TEXT,
    created_by_user_id TEXT,
    last_sync_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_google_drive_sources_kb_id ON google_drive_sources(kb_id);
CREATE INDEX IF NOT EXISTS idx_google_drive_sources_status ON google_drive_sources(status);

CREATE TABLE IF NOT EXISTS google_drive_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    drive_file_id TEXT NOT NULL,
    drive_parent_id TEXT,
    name TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    export_ext TEXT,
    revision_id TEXT,
    md5_checksum TEXT,
    etag TEXT,
    size_bytes INTEGER,
    modified_time TEXT,
    uploaded_file_id INTEGER,
    sync_status TEXT NOT NULL DEFAULT 'pending',
    last_seen_at TEXT,
    last_synced_at TEXT,
    raw_json TEXT,
    UNIQUE(source_id, drive_file_id),
    FOREIGN KEY (source_id) REFERENCES google_drive_sources(id) ON DELETE CASCADE,
    FOREIGN KEY (uploaded_file_id) REFERENCES uploaded_files(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_google_drive_files_source_id ON google_drive_files(source_id, sync_status);
CREATE INDEX IF NOT EXISTS idx_google_drive_files_uploaded_file_id ON google_drive_files(uploaded_file_id);

CREATE TABLE IF NOT EXISTS google_drive_sync_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    triggered_by_user_id TEXT,
    trigger_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    scanned_count INTEGER NOT NULL DEFAULT 0,
    changed_count INTEGER NOT NULL DEFAULT 0,
    imported_count INTEGER NOT NULL DEFAULT 0,
    skipped_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error_message TEXT,
    FOREIGN KEY (source_id) REFERENCES google_drive_sources(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_google_drive_sync_runs_source_time
    ON google_drive_sync_runs(source_id, started_at DESC);
"""

# ---------------------------------------------------------------------------
# Phase 27 / support email tools
# ---------------------------------------------------------------------------
_PHASE27_SUPPORT_EMAIL_SCHEMA = """
CREATE TABLE IF NOT EXISTS support_email_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    mailbox TEXT NOT NULL,
    provider_message_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    message_id_header TEXT,
    in_reply_to TEXT,
    references_header TEXT,
    from_address TEXT,
    from_name TEXT,
    to_addresses_json TEXT,
    cc_addresses_json TEXT,
    subject TEXT NOT NULL DEFAULT '',
    body_text TEXT NOT NULL DEFAULT '',
    snippet TEXT NOT NULL DEFAULT '',
    received_at TEXT,
    direction TEXT NOT NULL DEFAULT 'inbound',
    status TEXT NOT NULL DEFAULT 'new',
    ticket_code TEXT,
    raw_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(provider, mailbox, provider_message_id)
);
CREATE INDEX IF NOT EXISTS idx_support_email_messages_thread_id
    ON support_email_messages(thread_id, received_at, id);
CREATE INDEX IF NOT EXISTS idx_support_email_messages_received_at
    ON support_email_messages(received_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_support_email_messages_ticket_code
    ON support_email_messages(ticket_code);

CREATE TABLE IF NOT EXISTS support_email_sync_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    mailbox TEXT NOT NULL,
    status TEXT NOT NULL,
    scanned_count INTEGER NOT NULL DEFAULT 0,
    imported_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_support_email_sync_runs_time
    ON support_email_sync_runs(started_at DESC, id DESC);
"""

# ---------------------------------------------------------------------------
# Phase 28 / pending action approvals
# ---------------------------------------------------------------------------
_PHASE28_PENDING_ACTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type TEXT NOT NULL,
    risk_level TEXT NOT NULL DEFAULT 'high',
    status TEXT NOT NULL DEFAULT 'draft',
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    result_json TEXT,
    error_message TEXT,
    created_by_user_id TEXT,
    approved_by_user_id TEXT,
    executed_by_user_id TEXT,
    tenant_id TEXT,
    org_id TEXT,
    kb_id INTEGER,
    kb_key TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    approved_at TEXT,
    executed_at TEXT,
    expires_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_pending_actions_status_time
    ON pending_actions(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pending_actions_action_type
    ON pending_actions(action_type, created_at DESC);
"""

# ---------------------------------------------------------------------------
# Phase 29 / background job queue
# ---------------------------------------------------------------------------
_PHASE29_BACKGROUND_JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS background_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    payload_json TEXT NOT NULL,
    result_json TEXT,
    error_message TEXT,
    progress REAL NOT NULL DEFAULT 0.0,
    created_by_user_id TEXT,
    tenant_id TEXT,
    org_id TEXT,
    kb_id INTEGER,
    kb_key TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_background_jobs_status_time
    ON background_jobs(status, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_background_jobs_type_time
    ON background_jobs(job_type, created_at DESC);
"""

MIGRATIONS: list[tuple[str, str]] = [
    ("001_core_schema", _CORE_SCHEMA),
    ("002_knowledge_bases", _KB_SCHEMA),
    ("003_ingest_jobs_kb_id", _INGEST_JOB_KB_SCHEMA),
    ("004_phase1_context_and_audit", _PHASE1_CONTEXT_AND_AUDIT_SCHEMA),
    ("005_phase2_support_tickets", _PHASE2_SUPPORT_TICKETS_SCHEMA),
    ("006_phase5_slot_memory", _PHASE5_SLOT_MEMORY_SCHEMA),
    ("007_phase19_external_integrations", _PHASE19_EXTERNAL_INTEGRATIONS_SCHEMA),
    ("008_phase22_kb_access_level", _PHASE22_KB_ACCESS_LEVEL_SCHEMA),
    ("009_phase23_kb_scope", _PHASE23_KB_SCOPE_SCHEMA),
    ("010_phase23_file_acl", _PHASE23_FILE_ACL_SCHEMA),
    ("011_phase24_acl_tables", _PHASE24_ACL_TABLES_SCHEMA),
    ("012_phase24_order_scope", _PHASE24_ORDER_SCOPE_SCHEMA),
    ("013_phase25_auth_audit", _PHASE25_AUTH_AUDIT_SCHEMA),
    ("014_phase26_google_drive_sync", _PHASE26_GOOGLE_DRIVE_SYNC_SCHEMA),
    ("015_phase27_support_email", _PHASE27_SUPPORT_EMAIL_SCHEMA),
    ("016_phase28_pending_actions", _PHASE28_PENDING_ACTIONS_SCHEMA),
    ("017_phase29_background_jobs", _PHASE29_BACKGROUND_JOBS_SCHEMA),
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
