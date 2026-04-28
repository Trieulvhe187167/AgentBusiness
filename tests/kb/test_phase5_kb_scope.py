from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.background import BackgroundTasks

import app.database as database
import app.ingest as ingest
import app.kb as kb_api
import app.main as main
import app.rag as rag
import app.upload as upload
from tests.conftest import (
    add_vector,
    attach_file,
    configure_test_env,
    create_kb,
    fetch_default_kb,
    fetch_kb_version,
    insert_file,
    mark_ingested,
    run,
)


def test_scoped_stats_and_system_info(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    configure_test_env(tmp_path, monkeypatch)
    default_kb = fetch_default_kb()
    archive_kb = create_kb("Archive KB", "archive")

    default_file = insert_file("default.csv")
    archive_file = insert_file("archive.csv")
    attach_file(default_kb.id, default_file)
    attach_file(archive_kb.id, archive_file)
    mark_ingested(default_kb.id, default_file)
    mark_ingested(archive_kb.id, archive_file)

    add_vector(
        default_kb.id,
        default_file,
        "Default shipping policy",
        filename="default.csv",
        kb_version=default_kb.kb_version,
        chunk_id="default-chunk",
    )
    add_vector(
        archive_kb.id,
        archive_file,
        "Archive refund policy",
        filename="archive.csv",
        kb_version=archive_kb.kb_version,
        chunk_id="archive-chunk",
    )

    scoped_stats = run(main.kb_stats(kb_id=archive_kb.id, kb_key=None))
    assert scoped_stats.scope == "kb"
    assert scoped_stats.kb_id == archive_kb.id
    assert scoped_stats.total_files == 1
    assert scoped_stats.total_vectors == 1
    assert scoped_stats.sources == ["archive.csv"]

    source_stats = run(main.api_sources_stats(kb_id=archive_kb.id, kb_key=None))
    assert len(source_stats) == 1
    assert source_stats[0]["filename"] == "archive.csv"

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(llm_loaded=False, vector_store_ready=True, embeddings_loaded=False)
        )
    )
    system = run(main.system_info(request, kb_id=archive_kb.id, kb_key=None))
    assert system["scope"]["type"] == "kb"
    assert system["scope"]["kb_id"] == archive_kb.id
    assert system["total_files"] == 1
    assert system["ingested_files"] == 1
    assert system["source_count"] == 1
    assert system["total_vectors"] == 1


def test_duplicate_attach_is_idempotent_and_does_not_bump_kb_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    configure_test_env(tmp_path, monkeypatch)
    archive_kb = create_kb("Archive KB", "archive")
    file_id = insert_file("archive.csv")

    version_before_first_attach = fetch_kb_version(archive_kb.id)
    first_mapping = run(kb_api.attach_kb_file(archive_kb.id, file_id))
    version_after_first_attach = fetch_kb_version(archive_kb.id)
    second_mapping = run(kb_api.attach_kb_file(archive_kb.id, file_id))
    version_after_second_attach = fetch_kb_version(archive_kb.id)

    mapping_count = database.fetch_one_sync(
        "SELECT COUNT(*) AS total FROM kb_files WHERE kb_id = ? AND file_id = ?",
        (archive_kb.id, file_id),
    )

    assert first_mapping.file_id == file_id
    assert second_mapping.file_id == file_id
    assert version_after_first_attach != version_before_first_attach
    assert version_after_second_attach == version_after_first_attach
    assert mapping_count["total"] == 1


def test_detach_only_removes_vectors_for_that_kb_and_delete_requires_single_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    configure_test_env(tmp_path, monkeypatch)
    default_kb = fetch_default_kb()
    archive_kb = create_kb("Archive KB", "archive")

    shared_file = insert_file("shared.csv")
    attach_file(default_kb.id, shared_file)
    attach_file(archive_kb.id, shared_file)
    mark_ingested(default_kb.id, shared_file)
    mark_ingested(archive_kb.id, shared_file)

    add_vector(
        default_kb.id,
        shared_file,
        "Shipping for default KB",
        filename="shared.csv",
        kb_version=default_kb.kb_version,
        chunk_id="shared-default",
    )
    add_vector(
        archive_kb.id,
        shared_file,
        "Shipping for archive KB",
        filename="shared.csv",
        kb_version=archive_kb.kb_version,
        chunk_id="shared-archive",
    )

    with pytest.raises(HTTPException) as exc_info:
        run(upload.delete_file(shared_file))
    assert exc_info.value.status_code == 409

    result = run(kb_api.detach_kb_file(archive_kb.id, shared_file))
    assert result["kb_id"] == archive_kb.id
    assert result["file_id"] == shared_file

    remaining_mappings = database.fetch_one_sync(
        "SELECT COUNT(*) AS total FROM kb_files WHERE file_id = ?",
        (shared_file,),
    )
    assert remaining_mappings["total"] == 1
    assert rag.vector_store.count_by_where({"kb_id": archive_kb.id, "file_id": shared_file}) == 0
    assert rag.vector_store.count_by_where({"kb_id": default_kb.id, "file_id": shared_file}) == 1


def test_force_delete_source_file_cleans_all_kb_mappings_and_bumps_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    configure_test_env(tmp_path, monkeypatch)
    default_kb = fetch_default_kb()
    archive_kb = create_kb("Archive KB", "archive")

    shared_file = insert_file("shared.csv")
    attach_file(default_kb.id, shared_file)
    attach_file(archive_kb.id, shared_file)
    mark_ingested(default_kb.id, shared_file)
    mark_ingested(archive_kb.id, shared_file)

    add_vector(
        default_kb.id,
        shared_file,
        "Shipping for default KB",
        filename="shared.csv",
        kb_version=default_kb.kb_version,
        chunk_id="shared-default-force",
    )
    add_vector(
        archive_kb.id,
        shared_file,
        "Shipping for archive KB",
        filename="shared.csv",
        kb_version=archive_kb.kb_version,
        chunk_id="shared-archive-force",
    )

    default_version_before = fetch_kb_version(default_kb.id)
    archive_version_before = fetch_kb_version(archive_kb.id)

    result = run(upload.delete_file(shared_file, force=True))

    assert result["force"] is True
    assert result["detached_kb_ids"] == [default_kb.id, archive_kb.id]
    assert database.fetch_one_sync("SELECT * FROM uploaded_files WHERE id = ?", (shared_file,)) is None
    assert database.fetch_one_sync("SELECT * FROM kb_files WHERE file_id = ?", (shared_file,)) is None
    assert rag.vector_store.count_by_where({"kb_id": default_kb.id, "file_id": shared_file}) == 0
    assert rag.vector_store.count_by_where({"kb_id": archive_kb.id, "file_id": shared_file}) == 0
    assert fetch_kb_version(default_kb.id) != default_version_before
    assert fetch_kb_version(archive_kb.id) != archive_version_before


def test_retrieve_and_reindex_are_kb_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    configure_test_env(tmp_path, monkeypatch)
    default_kb = fetch_default_kb()
    archive_kb = create_kb("Archive KB", "archive")

    default_file = insert_file("default.csv")
    archive_file = insert_file("archive.csv")
    attach_file(default_kb.id, default_file)
    attach_file(archive_kb.id, archive_file)
    mark_ingested(default_kb.id, default_file)
    mark_ingested(archive_kb.id, archive_file)

    add_vector(
        default_kb.id,
        default_file,
        "Default KB shipping answer",
        filename="default.csv",
        kb_version=default_kb.kb_version,
        chunk_id="chunk-default",
    )
    add_vector(
        archive_kb.id,
        archive_file,
        "Archive KB shipping answer",
        filename="archive.csv",
        kb_version=archive_kb.kb_version,
        chunk_id="chunk-archive",
    )

    monkeypatch.setattr(rag, "expand_query", lambda query: [query])
    monkeypatch.setattr(rag, "embed_query", lambda query: [1.0, 0.0])
    monkeypatch.setattr(rag, "rerank", lambda query, items: items)

    results_default = rag.retrieve("shipping", top_k=5, kb_id=default_kb.id)
    assert len(results_default) == 1
    assert results_default[0]["kb_id"] == default_kb.id
    assert results_default[0]["filename"] == "default.csv"

    results_archive = rag.retrieve("shipping", top_k=5, kb_id=archive_kb.id)
    assert len(results_archive) == 1
    assert results_archive[0]["kb_id"] == archive_kb.id
    assert results_archive[0]["filename"] == "archive.csv"

    class _Request:
        class state:
            request_id = "test-reindex"

    jobs = run(ingest.reindex_kb(archive_kb.id, _Request()))
    assert len(jobs["jobs"]) == 1
    job_id = jobs["jobs"][0]["job_id"]

    job_row = database.fetch_one_sync(
        "SELECT kb_id, job_type, status, payload_json FROM background_jobs WHERE job_id = ?",
        (job_id,),
    )
    assert job_row["kb_id"] == archive_kb.id
    assert job_row["job_type"] == "kb_reindex"
    assert f'"kb_id": {archive_kb.id}' in job_row["payload_json"]
    assert job_row["status"] == "queued"
