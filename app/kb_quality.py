"""Knowledge Base content governance and quality scoring."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.models import AuthContext

LifecycleStatus = Literal["draft", "reviewed", "published", "archived"]


class KbFileLifecycleUpdate(BaseModel):
    lifecycle_status: LifecycleStatus


class KbFileQualityItem(BaseModel):
    kb_id: int
    file_id: int
    mapping_id: int
    filename: str
    original_name: str
    kb_status: str
    lifecycle_status: str
    quality_score: float
    feedback_up: int = 0
    feedback_down: int = 0
    fallback_count: int = 0
    citation_count: int = 0
    stale: bool = False
    stale_reason: str | None = None
    stale_detected_at: str | None = None
    drive_sync_status: str | None = None
    drive_modified_time: str | None = None
    last_ingest_at: str | None = None
    reviewed_by_user_id: str | None = None
    reviewed_at: str | None = None
    published_at: str | None = None
    archived_at: str | None = None


class KbQualityReport(BaseModel):
    kb_id: int
    total_files: int
    draft_count: int = 0
    reviewed_count: int = 0
    published_count: int = 0
    archived_count: int = 0
    stale_count: int = 0
    average_quality_score: float = 0.0
    items: list[KbFileQualityItem] = Field(default_factory=list)


class KbFileDiffOutput(BaseModel):
    kb_id: int
    file_id: int
    mapping_id: int
    original_name: str
    has_drive_source: bool
    changed: bool
    reason: str | None = None
    local: dict[str, Any] = Field(default_factory=dict)
    remote: dict[str, Any] = Field(default_factory=dict)


class KbReviewQueueItem(BaseModel):
    kb_id: int
    file_id: int
    mapping_id: int
    original_name: str
    filename: str
    issue_type: str
    priority: str
    reason: str
    suggested_action: str
    quality_score: float
    lifecycle_status: str
    kb_status: str
    stale_reason: str | None = None
    stale_detected_at: str | None = None
    drive_sync_status: str | None = None
    drive_modified_time: str | None = None
    last_ingest_at: str | None = None
    feedback_down: int = 0
    fallback_count: int = 0


class KbReviewQueueOutput(BaseModel):
    kb_id: int
    total: int
    draft_count: int = 0
    stale_count: int = 0
    drive_changed_count: int = 0
    low_quality_count: int = 0
    items: list[KbReviewQueueItem] = Field(default_factory=list)


_STALE_AFTER_DAYS = 90
_LIFECYCLE_ALLOWED = {"draft", "reviewed", "published", "archived"}
_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _parse_json(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    value = str(raw).strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _rows_for_kb(kb_id: int) -> list[dict[str, Any]]:
    return fetch_all_sync(
        """
        SELECT
            kf.id AS mapping_id,
            kf.kb_id,
            kf.file_id,
            kf.status AS kb_status,
            kf.chunk_count,
            kf.ingest_signature,
            kf.last_job_id,
            kf.lifecycle_status,
            kf.reviewed_by_user_id,
            kf.reviewed_at,
            kf.published_at,
            kf.archived_at,
            kf.quality_score,
            kf.stale_reason,
            kf.stale_detected_at,
            kf.attached_at,
            kf.last_ingest_at,
            uf.filename,
            uf.original_name,
            uf.file_type,
            uf.file_size,
            uf.file_hash,
            uf.status AS upload_status,
            gf.sync_status AS drive_sync_status,
            gf.revision_id AS drive_revision_id,
            gf.md5_checksum AS drive_md5_checksum,
            gf.modified_time AS drive_modified_time,
            gf.etag AS drive_etag,
            gf.name AS drive_name,
            gf.drive_file_id
        FROM kb_files kf
        JOIN uploaded_files uf ON uf.id = kf.file_id
        LEFT JOIN google_drive_files gf ON gf.uploaded_file_id = uf.id
        WHERE kf.kb_id = ?
        ORDER BY kf.attached_at DESC, kf.id DESC
        """,
        (kb_id,),
    )


def _citation_stats(kb_id: int) -> dict[int, dict[str, int]]:
    rows = fetch_all_sync(
        """
        SELECT cl.citations_json, cf.rating, cl.mode
        FROM chat_logs cl
        LEFT JOIN chat_feedback cf ON cf.chat_log_id = cl.id
        WHERE cl.kb_id = ?
        """,
        (kb_id,),
    )
    file_lookup = {
        str(row["filename"]): int(row["file_id"])
        for row in fetch_all_sync(
            """
            SELECT uf.filename, kf.file_id
            FROM kb_files kf
            JOIN uploaded_files uf ON uf.id = kf.file_id
            WHERE kf.kb_id = ?
            """,
            (kb_id,),
        )
    }
    stats: dict[int, dict[str, int]] = {}
    for row in rows:
        citations = _parse_json(row.get("citations_json"), [])
        if not isinstance(citations, list):
            continue
        matched_file_ids: set[int] = set()
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            filename = str(citation.get("filename") or "")
            file_id = file_lookup.get(filename)
            if file_id:
                matched_file_ids.add(file_id)
        for file_id in matched_file_ids:
            bucket = stats.setdefault(file_id, {"citations": 0, "up": 0, "down": 0, "fallback": 0})
            bucket["citations"] += 1
            if row.get("rating") == "up":
                bucket["up"] += 1
            elif row.get("rating") == "down":
                bucket["down"] += 1
            if row.get("mode") in {"fallback", "clarify"}:
                bucket["fallback"] += 1
    return stats


def _stale_reason(row: dict[str, Any], now: datetime) -> str | None:
    if row.get("drive_sync_status") == "deleted_remote":
        return "Drive source file was deleted remotely."
    drive_modified = _parse_dt(row.get("drive_modified_time"))
    last_ingest = _parse_dt(row.get("last_ingest_at"))
    if drive_modified and (not last_ingest or drive_modified > last_ingest):
        return "Drive file changed after last ingest."
    if row.get("kb_status") != "ingested":
        return "File is not currently ingested."
    if last_ingest and now - last_ingest > timedelta(days=_STALE_AFTER_DAYS):
        return f"Last ingest is older than {_STALE_AFTER_DAYS} days."
    if (row.get("lifecycle_status") or "draft") == "draft":
        return "Document is still in draft lifecycle."
    return None


def _quality_score(row: dict[str, Any], stats: dict[str, int], stale_reason: str | None) -> float:
    score = 100.0
    down = int(stats.get("down") or 0)
    up = int(stats.get("up") or 0)
    fallback = int(stats.get("fallback") or 0)
    citations = int(stats.get("citations") or 0)
    score -= min(35, down * 12)
    score += min(10, up * 3)
    score -= min(20, fallback * 8)
    if citations == 0 and row.get("kb_status") == "ingested":
        score -= 5
    if stale_reason:
        score -= 20
    lifecycle = row.get("lifecycle_status") or "draft"
    if lifecycle == "draft":
        score -= 12
    elif lifecycle == "reviewed":
        score -= 4
    elif lifecycle == "archived":
        score -= 25
    if row.get("kb_status") != "ingested":
        score -= 25
    return round(max(0.0, min(100.0, score)), 1)


def build_kb_quality_report(kb_id: int) -> dict[str, Any]:
    kb = fetch_one_sync("SELECT id FROM knowledge_bases WHERE id = ?", (kb_id,))
    if not kb:
        raise ValueError("Knowledge Base not found")
    rows = _rows_for_kb(kb_id)
    stats_by_file = _citation_stats(kb_id)
    now = datetime.now(timezone.utc)
    items: list[KbFileQualityItem] = []
    lifecycle_counts = {"draft": 0, "reviewed": 0, "published": 0, "archived": 0}
    stale_count = 0
    for row in rows:
        lifecycle = row.get("lifecycle_status") or "draft"
        lifecycle_counts[lifecycle if lifecycle in lifecycle_counts else "draft"] += 1
        stats = stats_by_file.get(int(row["file_id"]), {"citations": 0, "up": 0, "down": 0, "fallback": 0})
        reason = _stale_reason(row, now)
        stale = bool(reason)
        stale_count += 1 if stale else 0
        score = _quality_score(row, stats, reason)
        stale_detected_at = row.get("stale_detected_at") or (utcnow_iso() if stale else None)
        if row.get("quality_score") != score or row.get("stale_reason") != reason:
            execute_sync(
                """
                UPDATE kb_files
                SET quality_score = ?,
                    stale_reason = ?,
                    stale_detected_at = CASE WHEN ? IS NULL THEN NULL ELSE COALESCE(stale_detected_at, ?) END
                WHERE id = ?
                """,
                (score, reason, reason, stale_detected_at, row["mapping_id"]),
            )
        items.append(
            KbFileQualityItem(
                kb_id=int(row["kb_id"]),
                file_id=int(row["file_id"]),
                mapping_id=int(row["mapping_id"]),
                filename=row["filename"],
                original_name=row["original_name"],
                kb_status=row["kb_status"],
                lifecycle_status=lifecycle,
                quality_score=score,
                feedback_up=int(stats.get("up") or 0),
                feedback_down=int(stats.get("down") or 0),
                fallback_count=int(stats.get("fallback") or 0),
                citation_count=int(stats.get("citations") or 0),
                stale=stale,
                stale_reason=reason,
                stale_detected_at=stale_detected_at,
                drive_sync_status=row.get("drive_sync_status"),
                drive_modified_time=row.get("drive_modified_time"),
                last_ingest_at=row.get("last_ingest_at"),
                reviewed_by_user_id=row.get("reviewed_by_user_id"),
                reviewed_at=row.get("reviewed_at"),
                published_at=row.get("published_at"),
                archived_at=row.get("archived_at"),
            )
        )
    average = round(sum(item.quality_score for item in items) / len(items), 1) if items else 0.0
    return KbQualityReport(
        kb_id=kb_id,
        total_files=len(items),
        draft_count=lifecycle_counts["draft"],
        reviewed_count=lifecycle_counts["reviewed"],
        published_count=lifecycle_counts["published"],
        archived_count=lifecycle_counts["archived"],
        stale_count=stale_count,
        average_quality_score=average,
        items=items,
    ).model_dump()


def _review_queue_entry(item: dict[str, Any]) -> tuple[str, str, str, str] | None:
    lifecycle = str(item.get("lifecycle_status") or "draft")
    stale_reason = str(item.get("stale_reason") or "")
    score = float(item.get("quality_score") or 0)
    feedback_down = int(item.get("feedback_down") or 0)
    fallback_count = int(item.get("fallback_count") or 0)
    kb_status = str(item.get("kb_status") or "")
    if "Drive file changed" in stale_reason:
        return ("drive_changed", "P1", stale_reason, "Review diff and reingest the changed Drive file.")
    if item.get("drive_sync_status") == "deleted_remote":
        return ("stale", "P1", stale_reason or "Drive source file was deleted remotely.", "Archive or detach the missing Drive document.")
    if kb_status != "ingested":
        return ("stale", "P2", stale_reason or "File is not currently ingested.", "Run ingest before publishing this document.")
    if score < 55 or feedback_down > 0 or fallback_count > 0:
        return ("low_quality", "P2", f"Quality score {score}; {feedback_down} down, {fallback_count} fallback.", "Review source content and improve or replace stale sections.")
    if lifecycle == "draft":
        return ("draft", "P3", "Document is still in draft lifecycle.", "Review and publish when content is approved.")
    if stale_reason:
        return ("stale", "P3", stale_reason, "Review freshness and reingest if needed.")
    return None


def build_kb_review_queue(kb_id: int, issue_type: str | None = None) -> dict[str, Any]:
    report = build_kb_quality_report(kb_id)
    normalized_issue = str(issue_type or "").strip().lower()
    valid_issues = {"draft", "stale", "drive_changed", "low_quality"}
    items: list[KbReviewQueueItem] = []
    counts = {"draft": 0, "stale": 0, "drive_changed": 0, "low_quality": 0}
    for raw in report["items"]:
        entry = _review_queue_entry(raw)
        if not entry:
            continue
        issue, priority, reason, suggested_action = entry
        counts[issue] += 1
        if normalized_issue and normalized_issue in valid_issues and issue != normalized_issue:
            continue
        items.append(
            KbReviewQueueItem(
                kb_id=int(raw["kb_id"]),
                file_id=int(raw["file_id"]),
                mapping_id=int(raw["mapping_id"]),
                original_name=raw["original_name"],
                filename=raw["filename"],
                issue_type=issue,
                priority=priority,
                reason=reason,
                suggested_action=suggested_action,
                quality_score=float(raw["quality_score"]),
                lifecycle_status=raw["lifecycle_status"],
                kb_status=raw["kb_status"],
                stale_reason=raw.get("stale_reason"),
                stale_detected_at=raw.get("stale_detected_at"),
                drive_sync_status=raw.get("drive_sync_status"),
                drive_modified_time=raw.get("drive_modified_time"),
                last_ingest_at=raw.get("last_ingest_at"),
                feedback_down=int(raw.get("feedback_down") or 0),
                fallback_count=int(raw.get("fallback_count") or 0),
            )
        )
    items.sort(key=lambda item: (_PRIORITY_ORDER.get(item.priority, 9), item.quality_score, item.original_name.lower()))
    return KbReviewQueueOutput(
        kb_id=kb_id,
        total=len(items),
        draft_count=counts["draft"],
        stale_count=counts["stale"],
        drive_changed_count=counts["drive_changed"],
        low_quality_count=counts["low_quality"],
        items=items,
    ).model_dump()


def update_kb_file_lifecycle(kb_id: int, file_id: int, lifecycle_status: str, *, auth: AuthContext) -> dict[str, Any]:
    status = lifecycle_status.strip().lower()
    if status not in _LIFECYCLE_ALLOWED:
        raise ValueError(f"Invalid lifecycle status: {lifecycle_status}")
    row = fetch_one_sync("SELECT id FROM kb_files WHERE kb_id = ? AND file_id = ?", (kb_id, file_id))
    if not row:
        raise ValueError("KB file mapping not found")
    now = utcnow_iso()
    reviewed_at = now if status in {"reviewed", "published"} else None
    published_at = now if status == "published" else None
    archived_at = now if status == "archived" else None
    execute_sync(
        """
        UPDATE kb_files
        SET lifecycle_status = ?,
            reviewed_by_user_id = CASE WHEN ? IS NOT NULL THEN ? ELSE reviewed_by_user_id END,
            reviewed_at = COALESCE(?, reviewed_at),
            published_at = CASE WHEN ? IS NOT NULL THEN ? ELSE published_at END,
            archived_at = CASE WHEN ? IS NOT NULL THEN ? ELSE archived_at END
        WHERE kb_id = ? AND file_id = ?
        """,
        (
            status,
            reviewed_at,
            auth.user_id,
            reviewed_at,
            published_at,
            published_at,
            archived_at,
            archived_at,
            kb_id,
            file_id,
        ),
    )
    report = build_kb_quality_report(kb_id)
    for item in report["items"]:
        if int(item["file_id"]) == int(file_id):
            return item
    raise ValueError("Updated KB file not found")


def build_kb_file_diff(kb_id: int, file_id: int) -> dict[str, Any]:
    row = fetch_one_sync(
        """
        SELECT
            kf.id AS mapping_id,
            kf.kb_id,
            kf.file_id,
            kf.status AS kb_status,
            kf.last_ingest_at,
            kf.ingest_signature,
            uf.original_name,
            uf.file_hash,
            uf.file_size,
            uf.status AS upload_status,
            gf.drive_file_id,
            gf.name AS drive_name,
            gf.revision_id,
            gf.md5_checksum,
            gf.etag,
            gf.size_bytes,
            gf.modified_time,
            gf.sync_status
        FROM kb_files kf
        JOIN uploaded_files uf ON uf.id = kf.file_id
        LEFT JOIN google_drive_files gf ON gf.uploaded_file_id = uf.id
        WHERE kf.kb_id = ? AND kf.file_id = ?
        """,
        (kb_id, file_id),
    )
    if not row:
        raise ValueError("KB file mapping not found")
    local = {
        "file_hash": row.get("file_hash"),
        "file_size": row.get("file_size"),
        "upload_status": row.get("upload_status"),
        "last_ingest_at": row.get("last_ingest_at"),
        "ingest_signature": row.get("ingest_signature"),
    }
    remote = {
        "drive_file_id": row.get("drive_file_id"),
        "name": row.get("drive_name"),
        "revision_id": row.get("revision_id"),
        "md5_checksum": row.get("md5_checksum"),
        "etag": row.get("etag"),
        "size_bytes": row.get("size_bytes"),
        "modified_time": row.get("modified_time"),
        "sync_status": row.get("sync_status"),
    }
    has_drive = bool(row.get("drive_file_id"))
    reason = None
    if has_drive:
        reason = _stale_reason(
            {
                "drive_sync_status": row.get("sync_status"),
                "drive_modified_time": row.get("modified_time"),
                "last_ingest_at": row.get("last_ingest_at"),
                "kb_status": row.get("kb_status"),
                "lifecycle_status": "published",
            },
            datetime.now(timezone.utc),
        )
    return KbFileDiffOutput(
        kb_id=kb_id,
        file_id=file_id,
        mapping_id=int(row["mapping_id"]),
        original_name=row["original_name"],
        has_drive_source=has_drive,
        changed=bool(reason),
        reason=reason,
        local=local,
        remote=remote,
    ).model_dump()
