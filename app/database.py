"""
SQLite helpers and schema for files, ingest jobs, chat sessions, and chat logs.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone

import aiosqlite

from app.config import settings

DB_PATH = str(settings.sqlite_path)

SCHEMA = """
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
    FOREIGN KEY (file_id) REFERENCES uploaded_files(id)
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


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=5000")
    return db


async def init_db():
    settings.ensure_dirs()
    db = await get_db()
    try:
        await db.executescript(SCHEMA)
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
