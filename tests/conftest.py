from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.cache as cache_module
import app.database as database
import app.embeddings as embeddings
import app.main as main
from app.config import settings
from app.kb import create_knowledge_base
from app.kb_service import attach_file_to_kb, get_default_kb, open_db
from app.models import KnowledgeBaseCreate
from app.vector_store import vector_store


def run(coro):
    return asyncio.run(coro)


def configure_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, expected_dim: int = 2):
    data_dir = tmp_path / "data"
    paths = {
        "data_dir": data_dir,
        "raw_upload_dir": data_dir / "raw",
        "processed_dir": data_dir / "processed",
        "vectordb_dir": data_dir / "vectordb",
        "chroma_dir": data_dir / "vectordb" / "chroma",
        "cache_dir": data_dir / "cache",
        "sqlite_path": data_dir / "metadata.db",
    }
    for attr, value in paths.items():
        monkeypatch.setattr(settings, attr, value)

    monkeypatch.setattr(settings, "vector_backend", "numpy")
    monkeypatch.setattr(settings, "llm_provider", "none")
    monkeypatch.setattr(settings, "embedding_model_path", "models/missing-model")
    monkeypatch.setattr(settings, "min_similarity_threshold", 0.0)
    monkeypatch.setattr(database, "DB_PATH", str(paths["sqlite_path"]))
    monkeypatch.setattr(cache_module, "_cache", None)
    monkeypatch.setattr(embeddings, "_model", None)
    monkeypatch.setattr(embeddings, "_embeddings_ready", False)

    settings.ensure_dirs()
    run(database.init_db())
    vector_store.initialize(expected_dim=expected_dim)


def insert_file(original_name: str, *, status: str = "uploaded") -> int:
    safe_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
    file_path = settings.raw_upload_dir / safe_name
    file_path.write_text("name,answer\nshipping,free\n", encoding="utf-8")
    return int(
        database.execute_sync(
            """
            INSERT INTO uploaded_files
                (filename, original_name, file_type, file_size, file_hash, status, parser_type, created_at)
            VALUES (?, ?, '.csv', ?, ?, ?, 'csv', datetime('now'))
            """,
            (safe_name, original_name, file_path.stat().st_size, f"hash-{original_name}", status),
        )
        or 0
    )


def fetch_default_kb():
    async def _fetch():
        db = await open_db()
        try:
            return await get_default_kb(db)
        finally:
            await db.close()

    return run(_fetch())


def create_kb(name: str, key: str):
    return run(create_knowledge_base(KnowledgeBaseCreate(name=name, key=key)))


def fetch_kb_version(kb_id: int) -> str:
    row = database.fetch_one_sync(
        "SELECT kb_version FROM knowledge_bases WHERE id = ?",
        (kb_id,),
    )
    assert row is not None
    return str(row["kb_version"])


def attach_file(kb_id: int, file_id: int, status: str = "attached"):
    async def _attach():
        db = await open_db()
        try:
            await attach_file_to_kb(db, kb_id, file_id, status=status)
            await db.commit()
        finally:
            await db.close()

    run(_attach())


def mark_ingested(kb_id: int, file_id: int, *, chunk_count: int = 1, ingest_signature: str | None = None):
    signature = ingest_signature or f"sig-{kb_id}-{file_id}"
    database.execute_sync(
        """
        UPDATE kb_files
        SET status = 'ingested',
            chunk_count = ?,
            ingest_signature = ?,
            last_ingest_at = datetime('now')
        WHERE kb_id = ? AND file_id = ?
        """,
        (chunk_count, signature, kb_id, file_id),
    )
    database.execute_sync(
        """
        UPDATE uploaded_files
        SET status = 'ingested',
            pages_or_rows = ?,
            ingested_at = datetime('now'),
            error_message = NULL
        WHERE id = ?
        """,
        (chunk_count, file_id),
    )


def add_vector(kb_id: int, file_id: int, text: str, *, filename: str, kb_version: str, chunk_id: str):
    vector_store.add_chunks(
        [
            {
                "chunk_id": chunk_id,
                "kb_id": kb_id,
                "source_id": str(file_id),
                "file_id": file_id,
                "filename": filename,
                "file_type": ".csv",
                "kb_version": kb_version,
                "ingest_signature": f"sig-{kb_id}-{file_id}",
                "content_preview": text,
                "text": text,
            }
        ],
        [[1.0, 0.0]],
    )


def poll_jobs(client: TestClient, jobs: list[dict], timeout_seconds: int = 20):
    import time

    deadline = time.time() + timeout_seconds
    pending = {job["job_id"] for job in jobs}
    while pending and time.time() < deadline:
        for job_id in list(pending):
            response = client.get(f"/api/jobs/{job_id}")
            response.raise_for_status()
            payload = response.json()
            if payload["status"] == "failed":
                raise AssertionError(f"Job {job_id} failed: {payload.get('error_message')}")
            if payload["status"] == "done":
                pending.remove(job_id)
        if pending:
            time.sleep(0.2)

    assert not pending, f"Jobs did not finish before timeout: {sorted(pending)}"


@pytest.fixture()
def isolated_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    configure_test_env(tmp_path, monkeypatch)
    with TestClient(main.app) as client:
        yield client
