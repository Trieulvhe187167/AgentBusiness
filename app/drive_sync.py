"""
Google Drive -> Knowledge Base sync orchestration.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from typing import Any, Callable

from app.background_jobs import enqueue_background_job
from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.integrations.google_drive import GoogleDriveClient, normalize_google_drive_id
from app.kb_service import open_db, resolve_kb_scope
from app.config import settings
from app.models import AuthContext, RequestContext
from app.upload_service import import_content_to_uploaded_file
from app.vector_store import vector_store

logger = logging.getLogger(__name__)

DEFAULT_SUPPORTED_MIME_TYPES = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/csv",
    "text/plain",
    "text/markdown",
    "application/json",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
]


def _json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item).strip() for item in payload if str(item).strip()]


def _serialize_source_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "kb_id": int(row["kb_id"]),
        "name": row["name"],
        "folder_id": row["folder_id"],
        "shared_drive_id": row.get("shared_drive_id"),
        "recursive": bool(row.get("recursive")),
        "include_patterns": _json_list(row.get("include_patterns_json")),
        "exclude_patterns": _json_list(row.get("exclude_patterns_json")),
        "supported_mime_types": _json_list(row.get("supported_mime_types_json")) or DEFAULT_SUPPORTED_MIME_TYPES,
        "delete_policy": row.get("delete_policy") or "detach",
        "status": row.get("status") or "active",
        "tenant_id": row.get("tenant_id"),
        "org_id": row.get("org_id"),
        "created_by_user_id": row.get("created_by_user_id"),
        "last_sync_at": row.get("last_sync_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def list_google_drive_sources() -> dict[str, Any]:
    rows = fetch_all_sync(
        """
        SELECT id, kb_id, name, folder_id, shared_drive_id, recursive,
               include_patterns_json, exclude_patterns_json, supported_mime_types_json,
               delete_policy, status, tenant_id, org_id, created_by_user_id,
               last_sync_at, created_at, updated_at
        FROM google_drive_sources
        ORDER BY created_at DESC, id DESC
        """
    )
    items = [_serialize_source_row(dict(row)) for row in rows]
    return {"total": len(items), "items": items}


async def create_google_drive_source(
    *,
    kb_id: int | None = None,
    kb_key: str | None = None,
    name: str,
    folder_id: str,
    shared_drive_id: str | None = None,
    recursive: bool = True,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    supported_mime_types: list[str] | None = None,
    delete_policy: str = "detach",
    auth: AuthContext | None = None,
) -> dict[str, Any]:
    db = await open_db()
    try:
        kb = await resolve_kb_scope(db, kb_id=kb_id, kb_key=kb_key, auth_context=auth)
    finally:
        await db.close()

    now = utcnow_iso()
    source_id = execute_sync(
        """
        INSERT INTO google_drive_sources (
            kb_id, name, folder_id, shared_drive_id, recursive,
            include_patterns_json, exclude_patterns_json, supported_mime_types_json,
            delete_policy, status, tenant_id, org_id, created_by_user_id,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
        """,
        (
            kb.id,
            " ".join(name.strip().split()),
            normalize_google_drive_id(folder_id),
            normalize_google_drive_id(shared_drive_id) if shared_drive_id else None,
            1 if recursive else 0,
            json.dumps(include_patterns or [], ensure_ascii=False),
            json.dumps(exclude_patterns or [], ensure_ascii=False),
            json.dumps(supported_mime_types or DEFAULT_SUPPORTED_MIME_TYPES, ensure_ascii=False),
            delete_policy.strip() or "detach",
            kb.tenant_id,
            kb.org_id,
            auth.user_id if auth else None,
            now,
            now,
        ),
    )
    row = fetch_one_sync(
        """
        SELECT id, kb_id, name, folder_id, shared_drive_id, recursive,
               include_patterns_json, exclude_patterns_json, supported_mime_types_json,
               delete_policy, status, tenant_id, org_id, created_by_user_id,
               last_sync_at, created_at, updated_at
        FROM google_drive_sources
        WHERE id = ?
        """,
        (source_id,),
    )
    if not row:
        raise RuntimeError("Google Drive source was not persisted")
    return _serialize_source_row(dict(row))


def get_google_drive_sync_status(source_id: int) -> dict[str, Any]:
    source = fetch_one_sync(
        """
        SELECT id, kb_id, name, folder_id, shared_drive_id, recursive,
               include_patterns_json, exclude_patterns_json, supported_mime_types_json,
               delete_policy, status, tenant_id, org_id, created_by_user_id,
               last_sync_at, created_at, updated_at
        FROM google_drive_sources
        WHERE id = ?
        """,
        (source_id,),
    )
    if not source:
        raise ValueError(f"Google Drive source {source_id} not found")

    run = fetch_one_sync(
        """
        SELECT id, status, scanned_count, changed_count, imported_count,
               skipped_count, failed_count, started_at, finished_at, error_message
        FROM google_drive_sync_runs
        WHERE source_id = ?
        ORDER BY started_at DESC, id DESC
        LIMIT 1
        """,
        (source_id,),
    )
    payload = _serialize_source_row(dict(source))
    payload["last_run"] = dict(run) if run else None
    return payload


def _sync_uploaded_file_status_after_detach(file_id: int) -> None:
    row = fetch_one_sync(
        "SELECT COUNT(*) AS total FROM kb_files WHERE file_id = ? AND status = 'ingested'",
        (file_id,),
    )
    ingested_count = int((row or {}).get("total") or 0)
    if ingested_count > 0:
        execute_sync(
            "UPDATE uploaded_files SET status = 'ingested', error_message = NULL WHERE id = ?",
            (file_id,),
        )
        return

    execute_sync(
        """
        UPDATE uploaded_files
        SET status = 'uploaded',
            ingested_at = NULL,
            error_message = NULL
        WHERE id = ?
        """,
        (file_id,),
    )


def delete_google_drive_source(source_id: int, *, mode: str = "unlink") -> dict[str, Any]:
    source_row = fetch_one_sync(
        """
        SELECT id, kb_id, name, folder_id
        FROM google_drive_sources
        WHERE id = ?
        """,
        (source_id,),
    )
    if not source_row:
        raise ValueError(f"Google Drive source {source_id} not found")

    normalized_mode = str(mode or "unlink").strip().lower()
    if normalized_mode not in {"unlink", "purge"}:
        raise ValueError("delete mode must be 'unlink' or 'purge'")

    tracked_rows = fetch_all_sync(
        """
        SELECT uploaded_file_id
        FROM google_drive_files
        WHERE source_id = ? AND uploaded_file_id IS NOT NULL
        """,
        (source_id,),
    )
    tracked_file_ids = sorted({int(row["uploaded_file_id"]) for row in tracked_rows if row.get("uploaded_file_id")})

    detached_count = 0
    deleted_file_count = 0
    preserved_file_count = 0

    if normalized_mode == "purge":
        for file_id in tracked_file_ids:
            vector_store.delete_by_kb_and_file(int(source_row["kb_id"]), file_id)
            execute_sync(
                "DELETE FROM kb_files WHERE kb_id = ? AND file_id = ?",
                (int(source_row["kb_id"]), file_id),
            )
            detached_count += 1

            remaining = fetch_one_sync(
                "SELECT COUNT(*) AS total FROM kb_files WHERE file_id = ?",
                (file_id,),
            )
            remaining_count = int((remaining or {}).get("total") or 0)
            if remaining_count == 0:
                file_row = fetch_one_sync(
                    "SELECT filename FROM uploaded_files WHERE id = ?",
                    (file_id,),
                )
                if file_row and file_row.get("filename"):
                    raw_path = settings.raw_upload_dir / str(file_row["filename"])
                    if raw_path.exists():
                        raw_path.unlink()
                execute_sync("DELETE FROM ingest_jobs WHERE file_id = ?", (file_id,))
                execute_sync("DELETE FROM uploaded_files WHERE id = ?", (file_id,))
                deleted_file_count += 1
            else:
                _sync_uploaded_file_status_after_detach(file_id)
                preserved_file_count += 1

    execute_sync("DELETE FROM google_drive_sources WHERE id = ?", (source_id,))
    return {
        "source_id": int(source_row["id"]),
        "kb_id": int(source_row["kb_id"]),
        "name": source_row["name"],
        "mode": normalized_mode,
        "tracked_file_count": len(tracked_file_ids),
        "detached_file_count": detached_count,
        "deleted_file_count": deleted_file_count,
        "preserved_file_count": preserved_file_count,
        "message": (
            f"Google Drive source {source_row['id']} deleted"
            if normalized_mode == "unlink"
            else f"Google Drive source {source_row['id']} deleted and imported files purged from the KB"
        ),
    }


def _matches_patterns(name: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    normalized = name.lower()
    return any(fnmatch.fnmatch(normalized, pattern.lower()) for pattern in patterns)


def _should_include_file(item: dict[str, Any], source: dict[str, Any]) -> bool:
    name = str(item.get("name") or "").strip()
    mime_type = str(item.get("mimeType") or "").strip()
    if not name or not mime_type:
        return False
    if mime_type not in set(source["supported_mime_types"]):
        return False
    if source["include_patterns"] and not _matches_patterns(name, source["include_patterns"]):
        return False
    if source["exclude_patterns"] and _matches_patterns(name, source["exclude_patterns"]):
        return False
    return True


def _is_changed(existing: dict[str, Any] | None, item: dict[str, Any], *, force_full: bool) -> bool:
    if force_full or not existing:
        return True
    current_revision = str(item.get("version") or "").strip() or None
    current_md5 = str(item.get("md5Checksum") or "").strip() or None
    current_modified = str(item.get("modifiedTime") or "").strip() or None
    return any(
        [
            (existing.get("revision_id") or None) != current_revision,
            (existing.get("md5_checksum") or None) != current_md5,
            (existing.get("modified_time") or None) != current_modified,
        ]
    )


def _upsert_drive_file_row(
    *,
    source_id: int,
    item: dict[str, Any],
    export_ext: str | None,
    uploaded_file_id: int | None,
    sync_status: str,
) -> None:
    now = utcnow_iso()
    execute_sync(
        """
        INSERT INTO google_drive_files (
            source_id, drive_file_id, drive_parent_id, name, mime_type, export_ext,
            revision_id, md5_checksum, etag, size_bytes, modified_time, uploaded_file_id,
            sync_status, last_seen_at, last_synced_at, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id, drive_file_id) DO UPDATE SET
            drive_parent_id=excluded.drive_parent_id,
            name=excluded.name,
            mime_type=excluded.mime_type,
            export_ext=excluded.export_ext,
            revision_id=excluded.revision_id,
            md5_checksum=excluded.md5_checksum,
            etag=excluded.etag,
            size_bytes=excluded.size_bytes,
            modified_time=excluded.modified_time,
            uploaded_file_id=excluded.uploaded_file_id,
            sync_status=excluded.sync_status,
            last_seen_at=excluded.last_seen_at,
            last_synced_at=excluded.last_synced_at,
            raw_json=excluded.raw_json
        """,
        (
            source_id,
            str(item.get("id") or "").strip(),
            str((item.get("parents") or [None])[0] or "").strip() or None,
            str(item.get("name") or "").strip(),
            str(item.get("mimeType") or "").strip(),
            export_ext,
            str(item.get("version") or "").strip() or None,
            str(item.get("md5Checksum") or "").strip() or None,
            str(item.get("etag") or "").strip() or None,
            int(item["size"]) if str(item.get("size") or "").strip().isdigit() else None,
            str(item.get("modifiedTime") or "").strip() or None,
            uploaded_file_id,
            sync_status,
            now,
            now if sync_status == "synced" else None,
            json.dumps(item, ensure_ascii=False),
        ),
    )


def _mark_remote_deleted(source: dict[str, Any], tracked_rows: list[dict[str, Any]], seen_ids: set[str]) -> None:
    if source["delete_policy"] != "detach":
        return
    for row in tracked_rows:
        drive_file_id = str(row.get("drive_file_id") or "").strip()
        if not drive_file_id or drive_file_id in seen_ids:
            continue
        execute_sync(
            """
            UPDATE google_drive_files
            SET sync_status = 'deleted_remote'
            WHERE source_id = ? AND drive_file_id = ?
            """,
            (source["id"], drive_file_id),
        )
        uploaded_file_id = row.get("uploaded_file_id")
        if uploaded_file_id:
            execute_sync(
                "DELETE FROM kb_files WHERE kb_id = ? AND file_id = ?",
                (source["kb_id"], uploaded_file_id),
            )
            vector_store.delete_by_kb_and_file(source["kb_id"], int(uploaded_file_id))


async def sync_google_drive_source(
    source_id: int,
    *,
    triggered_by_user_id: str | None = None,
    trigger_mode: str = "tool",
    force_full: bool = False,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    source_row = fetch_one_sync(
        """
        SELECT id, kb_id, name, folder_id, shared_drive_id, recursive,
               include_patterns_json, exclude_patterns_json, supported_mime_types_json,
               delete_policy, status, tenant_id, org_id, created_by_user_id,
               last_sync_at, created_at, updated_at
        FROM google_drive_sources
        WHERE id = ?
        """,
        (source_id,),
    )
    if not source_row:
        raise ValueError(f"Google Drive source {source_id} not found")
    source = _serialize_source_row(dict(source_row))

    run_started = utcnow_iso()
    run_id = execute_sync(
        """
        INSERT INTO google_drive_sync_runs (
            source_id, triggered_by_user_id, trigger_mode, status, started_at
        ) VALUES (?, ?, ?, 'running', ?)
        """,
        (source_id, triggered_by_user_id, trigger_mode, run_started),
    )

    scanned_count = 0
    changed_count = 0
    imported_count = 0
    skipped_count = 0
    failed_count = 0
    queued_jobs: list[str] = []

    try:
        if cancel_check:
            cancel_check()
        client = GoogleDriveClient()
        tracked_rows = fetch_all_sync(
            """
            SELECT source_id, drive_file_id, revision_id, md5_checksum, modified_time, uploaded_file_id
            FROM google_drive_files
            WHERE source_id = ?
            """,
            (source_id,),
        )
        tracked_by_drive_id = {str(row["drive_file_id"]): dict(row) for row in tracked_rows}

        remote_items = await client.list_files(
            source["folder_id"],
            shared_drive_id=source["shared_drive_id"],
            recursive=source["recursive"],
        )
        seen_ids: set[str] = set()

        for item in remote_items:
            if cancel_check:
                cancel_check()
            if not _should_include_file(item, source):
                continue

            scanned_count += 1
            drive_file_id = str(item.get("id") or "").strip()
            if not drive_file_id:
                skipped_count += 1
                continue
            seen_ids.add(drive_file_id)

            existing = tracked_by_drive_id.get(drive_file_id)
            if not _is_changed(existing, item, force_full=force_full):
                skipped_count += 1
                _upsert_drive_file_row(
                    source_id=source_id,
                    item=item,
                    export_ext=existing.get("export_ext") if existing else None,
                    uploaded_file_id=existing.get("uploaded_file_id") if existing else None,
                    sync_status="synced",
                )
                continue

            changed_count += 1
            try:
                if cancel_check:
                    cancel_check()
                content, filename, export_ext = await client.download_file(item)
                imported = await import_content_to_uploaded_file(
                    filename=filename,
                    content=content,
                    kb_id=source["kb_id"],
                    existing_file_id=int(existing["uploaded_file_id"]) if existing and existing.get("uploaded_file_id") else None,
                    access_level="internal",
                    tenant_id=source.get("tenant_id"),
                    org_id=source.get("org_id"),
                    owner_user_id=triggered_by_user_id or source.get("created_by_user_id"),
                )
                execute_sync(
                    """
                    UPDATE kb_files
                    SET status = 'attached',
                        chunk_count = 0,
                        ingest_signature = NULL,
                        last_job_id = NULL
                    WHERE kb_id = ? AND file_id = ?
                    """,
                    (source["kb_id"], imported["id"]),
                )
                job = enqueue_background_job(
                    job_type="kb_file_ingest",
                    payload={"kb_id": source["kb_id"], "file_id": int(imported["id"])},
                    context=RequestContext(
                        request_id=f"drive-sync-{run_id}-{imported['id']}",
                        kb_id=source["kb_id"],
                        auth=AuthContext(
                            user_id=triggered_by_user_id or source.get("created_by_user_id"),
                            roles=["admin"],
                            channel="worker",
                            tenant_id=source.get("tenant_id"),
                            org_id=source.get("org_id"),
                        ),
                    ),
                )
                queued_jobs.append(job["job_id"])
                imported_count += 1
                _upsert_drive_file_row(
                    source_id=source_id,
                    item=item,
                    export_ext=export_ext,
                    uploaded_file_id=int(imported["id"]),
                    sync_status="synced",
                )
            except Exception as err:
                logger.exception("Google Drive file sync failed for source=%s file=%s", source_id, drive_file_id)
                failed_count += 1
                _upsert_drive_file_row(
                    source_id=source_id,
                    item=item,
                    export_ext=None,
                    uploaded_file_id=int(existing["uploaded_file_id"]) if existing and existing.get("uploaded_file_id") else None,
                    sync_status="failed",
                )

        _mark_remote_deleted(source, tracked_rows, seen_ids)
        finished = utcnow_iso()
        execute_sync(
            """
            UPDATE google_drive_sources
            SET last_sync_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (finished, finished, source_id),
        )
        execute_sync(
            """
            UPDATE google_drive_sync_runs
            SET status = 'success',
                scanned_count = ?,
                changed_count = ?,
                imported_count = ?,
                skipped_count = ?,
                failed_count = ?,
                finished_at = ?
            WHERE id = ?
            """,
            (scanned_count, changed_count, imported_count, skipped_count, failed_count, finished, run_id),
        )
        return {
            "source_id": source_id,
            "kb_id": source["kb_id"],
            "run_id": int(run_id or 0),
            "status": "success",
            "scanned_count": scanned_count,
            "changed_count": changed_count,
            "imported_count": imported_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "queued_job_ids": queued_jobs,
            "last_sync_at": finished,
        }
    except Exception as err:
        finished = utcnow_iso()
        execute_sync(
            """
            UPDATE google_drive_sync_runs
            SET status = 'failed',
                scanned_count = ?,
                changed_count = ?,
                imported_count = ?,
                skipped_count = ?,
                failed_count = ?,
                finished_at = ?,
                error_message = ?
            WHERE id = ?
            """,
            (scanned_count, changed_count, imported_count, skipped_count, failed_count, finished, str(err)[:500], run_id),
        )
        raise
