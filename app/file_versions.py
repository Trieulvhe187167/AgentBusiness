"""
File version tracking and binary snapshots.

Phase 1 keeps immutable version metadata plus optional raw-file snapshots.
Rollback and diff APIs build on these records later.
"""

from __future__ import annotations

import difflib
import logging
import shutil
from pathlib import Path
from typing import Any

from app.config import settings
from app.database import execute_with_retry, fetch_all, fetch_one, utcnow_iso
from app.parsers import parse_file
from app.upload_validation import compute_file_hash

logger = logging.getLogger(__name__)

_RAW_TEXT_DIFF_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".ndjson",
    ".xml",
    ".html",
    ".htm",
}


def _row_dict(row: dict[str, Any] | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def _snapshot_path_for(row: dict[str, Any], version_number: int) -> Path:
    suffix = Path(str(row.get("original_name") or row.get("filename") or "")).suffix
    if not suffix:
        suffix = str(row.get("file_type") or "")
    hash_prefix = str(row.get("file_hash") or "unknown")[:8] or "unknown"
    filename = f"{int(row['id'])}_v{version_number}_{hash_prefix}{suffix}"
    return settings.file_versioning_snapshot_dir / filename


def _copy_snapshot(row: dict[str, Any], version_number: int) -> str | None:
    if not settings.file_versioning_keep_snapshots:
        return None

    source_path = settings.raw_upload_dir / str(row["filename"])
    if not source_path.exists():
        logger.warning(
            "Cannot snapshot file_id=%s version=%s; source missing at %s",
            row.get("id"),
            version_number,
            source_path,
        )
        return None

    settings.file_versioning_snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = _snapshot_path_for(row, version_number)
    shutil.copy2(source_path, snapshot_path)
    return str(snapshot_path)


async def latest_file_version(file_id: int) -> dict[str, Any] | None:
    return _row_dict(
        await fetch_one(
            """
            SELECT *
            FROM file_versions
            WHERE file_id = ?
            ORDER BY version_number DESC
            LIMIT 1
            """,
            (file_id,),
        )
    )


async def list_file_versions(file_id: int) -> list[dict[str, Any]]:
    rows = await fetch_all(
        """
        SELECT
            fv.*,
            CASE
                WHEN fv.id = (
                    SELECT fv_current.id
                    FROM file_versions fv_current
                    WHERE fv_current.file_id = fv.file_id
                      AND fv_current.file_hash = uf.file_hash
                    ORDER BY fv_current.version_number DESC
                    LIMIT 1
                ) THEN 1
                ELSE 0
            END AS is_current,
            COALESCE(
                (
                    SELECT MAX(fvi.is_active)
                    FROM file_version_ingests fvi
                    WHERE fvi.file_version_id = fv.id
                ),
                0
            ) AS is_active
        FROM file_versions fv
        JOIN uploaded_files uf ON uf.id = fv.file_id
        WHERE fv.file_id = ?
        ORDER BY fv.version_number DESC
        """,
        (file_id,),
    )
    return [dict(row) for row in rows]


async def get_file_version(file_id: int, version_number: int) -> dict[str, Any] | None:
    return _row_dict(
        await fetch_one(
            """
            SELECT *
            FROM file_versions
            WHERE file_id = ? AND version_number = ?
            """,
            (int(file_id), int(version_number)),
        )
    )


def _decode_snapshot_bytes(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _snapshot_text(version: dict[str, Any]) -> str:
    snapshot_path = version.get("snapshot_path")
    if not snapshot_path:
        raise FileNotFoundError(f"Version {version['version_number']} has no retained binary snapshot")

    path = Path(str(snapshot_path))
    if not path.exists():
        raise FileNotFoundError(f"Version snapshot is missing: {path}")

    file_type = str(version.get("file_type") or "").lower()
    if file_type in _RAW_TEXT_DIFF_EXTENSIONS:
        return _decode_snapshot_bytes(path)

    parser_type = version.get("parser_type")
    if parser_type:
        try:
            records = parse_file(path, str(parser_type))
            texts = [str(record.get("text") or "").strip() for record in records]
            parsed = "\n\n".join(text for text in texts if text)
            if parsed:
                return parsed
        except Exception:
            logger.warning(
                "Falling back to raw text decode for file_id=%s version=%s",
                version.get("file_id"),
                version.get("version_number"),
                exc_info=True,
            )

    return _decode_snapshot_bytes(path)


async def diff_file_versions(
    *,
    file_id: int,
    from_version_number: int,
    to_version_number: int,
    context_lines: int = 3,
    max_diff_lines: int = 500,
) -> dict[str, Any]:
    from_version = await get_file_version(file_id, from_version_number)
    if not from_version:
        raise ValueError(f"File version {from_version_number} not found for file {file_id}")

    to_version = await get_file_version(file_id, to_version_number)
    if not to_version:
        raise ValueError(f"File version {to_version_number} not found for file {file_id}")

    from_text = _snapshot_text(from_version)
    to_text = _snapshot_text(to_version)
    from_lines = from_text.splitlines()
    to_lines = to_text.splitlines()
    diff_lines = list(
        difflib.unified_diff(
            from_lines,
            to_lines,
            fromfile=f"v{from_version_number}:{from_version['original_name']}",
            tofile=f"v{to_version_number}:{to_version['original_name']}",
            lineterm="",
            n=max(0, int(context_lines)),
        )
    )
    additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    limit = max(1, int(max_diff_lines))

    return {
        "file_id": int(file_id),
        "from_version": from_version,
        "to_version": to_version,
        "changed": bool(diff_lines),
        "additions": additions,
        "deletions": deletions,
        "from_line_count": len(from_lines),
        "to_line_count": len(to_lines),
        "diff_lines": diff_lines[:limit],
        "truncated": len(diff_lines) > limit,
    }


async def create_file_version(
    row: dict[str, Any],
    *,
    version_number: int | None = None,
    created_by_user_id: str | None = None,
    change_summary: str | None = None,
    snapshot: bool = True,
) -> dict[str, Any]:
    file_id = int(row["id"])
    if version_number is None:
        latest = await latest_file_version(file_id)
        version_number = int(latest["version_number"]) + 1 if latest else 1

    snapshot_path = _copy_snapshot(row, version_number) if snapshot else None
    created_at = utcnow_iso()
    cursor = await execute_with_retry(
        """
        INSERT INTO file_versions (
            file_id,
            version_number,
            file_hash,
            file_size,
            filename,
            original_name,
            file_type,
            parser_type,
            pages_or_rows,
            chunk_count,
            ingest_signature,
            snapshot_path,
            change_summary,
            created_by_user_id,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file_id,
            int(version_number),
            row["file_hash"],
            int(row["file_size"]),
            row["filename"],
            row["original_name"],
            row["file_type"],
            row.get("parser_type"),
            row.get("pages_or_rows"),
            row.get("chunk_count"),
            row.get("ingest_signature"),
            snapshot_path,
            change_summary,
            created_by_user_id,
            created_at,
        ),
    )
    version_id = int(cursor.lastrowid)
    await _enforce_snapshot_retention(file_id)
    created = await fetch_one("SELECT * FROM file_versions WHERE id = ?", (version_id,))
    if not created:
        raise RuntimeError(f"File version {version_id} was not persisted")
    return dict(created)


async def ensure_current_file_version(
    row: dict[str, Any],
    *,
    created_by_user_id: str | None = None,
) -> dict[str, Any]:
    latest = await latest_file_version(int(row["id"]))
    if latest and latest.get("file_hash") == row.get("file_hash"):
        return latest
    return await create_file_version(row, created_by_user_id=created_by_user_id)


async def record_uploaded_file_version(
    file_id: int,
    *,
    created_by_user_id: str | None = None,
) -> dict[str, Any]:
    row = await fetch_one("SELECT * FROM uploaded_files WHERE id = ?", (file_id,))
    if not row:
        raise ValueError(f"Uploaded file {file_id} not found")
    return await create_file_version(dict(row), created_by_user_id=created_by_user_id)


async def restore_file_version(
    *,
    file_id: int,
    version_number: int,
    created_by_user_id: str | None = None,
    change_summary: str | None = None,
) -> dict[str, Any]:
    current = await fetch_one("SELECT * FROM uploaded_files WHERE id = ?", (int(file_id),))
    if not current:
        raise ValueError(f"Uploaded file {file_id} not found")
    current_row = dict(current)

    target = await get_file_version(file_id, version_number)
    if not target:
        raise ValueError(f"File version {version_number} not found for file {file_id}")

    current_version = await ensure_current_file_version(
        current_row,
        created_by_user_id=created_by_user_id,
    )
    if str(current_row.get("file_hash")) == str(target.get("file_hash")):
        return {
            "target_version": target,
            "restored_version": current_version,
            "changed": False,
        }

    snapshot_path = target.get("snapshot_path")
    if not snapshot_path:
        raise FileNotFoundError(f"Version {version_number} has no retained binary snapshot")

    snapshot = Path(str(snapshot_path))
    if not snapshot.exists():
        raise FileNotFoundError(f"Version snapshot is missing: {snapshot}")

    snapshot_bytes = snapshot.read_bytes()
    restored_hash = compute_file_hash(snapshot_bytes)
    expected_hash = str(target["file_hash"])
    if restored_hash != expected_hash:
        raise ValueError(
            f"Version snapshot hash mismatch for file {file_id} v{version_number}: "
            f"expected {expected_hash}, got {restored_hash}"
        )

    raw_path = settings.raw_upload_dir / str(current_row["filename"])
    settings.raw_upload_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot, raw_path)

    await execute_with_retry(
        """
        UPDATE uploaded_files
        SET original_name = ?,
            file_type = ?,
            file_size = ?,
            file_hash = ?,
            status = 'uploaded',
            parser_type = ?,
            pages_or_rows = NULL,
            ingested_at = NULL,
            error_message = NULL
        WHERE id = ?
        """,
        (
            target["original_name"],
            target["file_type"],
            int(target["file_size"]),
            target["file_hash"],
            target.get("parser_type"),
            int(file_id),
        ),
    )

    restored_row = await fetch_one("SELECT * FROM uploaded_files WHERE id = ?", (int(file_id),))
    if not restored_row:
        raise RuntimeError(f"Uploaded file {file_id} disappeared during rollback")

    restored_version = await create_file_version(
        dict(restored_row),
        created_by_user_id=created_by_user_id,
        change_summary=change_summary
        or f"Rollback to version {int(target['version_number'])}",
    )
    return {
        "target_version": target,
        "restored_version": restored_version,
        "changed": True,
    }


async def active_file_version_for_current_hash(file_id: int) -> dict[str, Any] | None:
    row = await fetch_one(
        """
        SELECT fv.*
        FROM file_versions fv
        JOIN uploaded_files uf ON uf.id = fv.file_id
        WHERE fv.file_id = ? AND fv.file_hash = uf.file_hash
        ORDER BY fv.version_number DESC
        LIMIT 1
        """,
        (file_id,),
    )
    return _row_dict(row)


async def mark_version_ingested(
    *,
    file_version_id: int,
    file_id: int,
    kb_id: int,
    chunk_count: int,
    ingest_signature: str,
    activated_by_user_id: str | None = None,
) -> None:
    now = utcnow_iso()
    await execute_with_retry(
        """
        UPDATE file_versions
        SET chunk_count = ?,
            ingest_signature = ?
        WHERE id = ?
        """,
        (int(chunk_count), ingest_signature, int(file_version_id)),
    )
    await execute_with_retry(
        """
        UPDATE file_version_ingests
        SET is_active = 0,
            deactivated_at = ?
        WHERE kb_id = ? AND file_id = ? AND is_active = 1
        """,
        (now, int(kb_id), int(file_id)),
    )
    await execute_with_retry(
        """
        INSERT INTO file_version_ingests (
            file_version_id,
            kb_id,
            file_id,
            is_active,
            chunk_count,
            ingest_signature,
            activated_at,
            activated_by_user_id
        ) VALUES (?, ?, ?, 1, ?, ?, ?, ?)
        """,
        (
            int(file_version_id),
            int(kb_id),
            int(file_id),
            int(chunk_count),
            ingest_signature,
            now,
            activated_by_user_id,
        ),
    )


async def _enforce_snapshot_retention(file_id: int) -> None:
    keep = max(0, int(settings.file_versioning_retention_count))
    if keep == 0:
        return

    rows = await fetch_all(
        """
        SELECT id, snapshot_path
        FROM file_versions
        WHERE file_id = ? AND snapshot_path IS NOT NULL
        ORDER BY version_number DESC
        """,
        (file_id,),
    )
    for row in rows[keep:]:
        snapshot_path = row.get("snapshot_path")
        if snapshot_path:
            try:
                Path(snapshot_path).unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to remove old file snapshot %s", snapshot_path, exc_info=True)
        await execute_with_retry(
            "UPDATE file_versions SET snapshot_path = NULL WHERE id = ?",
            (int(row["id"]),),
        )
