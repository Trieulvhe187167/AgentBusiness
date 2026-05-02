"""
Scheduled sync orchestration for Drive and support email jobs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.background_jobs import enqueue_background_job
from app.config import settings
from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.models import AuthContext, RequestContext

logger = logging.getLogger(__name__)

ScheduleType = Literal["google_drive_sync", "support_email_sync"]


class SyncScheduleItem(BaseModel):
    id: int
    schedule_type: str
    target_id: int | None = None
    name: str
    enabled: bool
    interval_seconds: int
    payload: dict[str, Any] = Field(default_factory=dict)
    created_by_user_id: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    kb_id: int | None = None
    kb_key: str | None = None
    last_job_id: str | None = None
    last_run_at: str | None = None
    next_run_at: str
    created_at: str
    updated_at: str


class ListSyncSchedulesOutput(BaseModel):
    total: int
    items: list[SyncScheduleItem]


class UpsertSyncScheduleInput(BaseModel):
    schedule_type: ScheduleType
    target_id: int | None = Field(default=None, ge=1)
    name: str | None = Field(default=None, max_length=180)
    enabled: bool = True
    interval_seconds: int = Field(default=900, ge=60, le=86400)
    payload: dict[str, Any] = Field(default_factory=dict)


class UpdateSyncScheduleInput(BaseModel):
    enabled: bool | None = None
    interval_seconds: int | None = Field(default=None, ge=60, le=86400)
    name: str | None = Field(default=None, max_length=180)
    payload: dict[str, Any] | None = None


def _parse_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _serialize_row(row: dict[str, Any]) -> SyncScheduleItem:
    return SyncScheduleItem(
        id=int(row["id"]),
        schedule_type=row["schedule_type"],
        target_id=row.get("target_id"),
        name=row["name"],
        enabled=bool(row.get("enabled")),
        interval_seconds=int(row["interval_seconds"]),
        payload=_parse_json(row.get("payload_json")),
        created_by_user_id=row.get("created_by_user_id"),
        tenant_id=row.get("tenant_id"),
        org_id=row.get("org_id"),
        kb_id=row.get("kb_id"),
        kb_key=row.get("kb_key"),
        last_job_id=row.get("last_job_id"),
        last_run_at=row.get("last_run_at"),
        next_run_at=row["next_run_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _json_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _next_run_after(interval_seconds: int, *, from_time: datetime | None = None) -> str:
    base = from_time or datetime.now(timezone.utc)
    return (base + timedelta(seconds=max(60, interval_seconds))).isoformat()


def _default_name(schedule_type: str, target_id: int | None) -> str:
    if schedule_type == "google_drive_sync":
        if target_id is None:
            raise ValueError("Google Drive sync schedule requires target_id")
        row = fetch_one_sync("SELECT name FROM google_drive_sources WHERE id = ?", (target_id,))
        if not row:
            raise ValueError(f"Google Drive source {target_id} not found")
        return f"Drive: {row['name']}"
    return "Support Email"


def _schedule_scope(schedule_type: str, target_id: int | None, auth: AuthContext) -> dict[str, Any]:
    if schedule_type == "google_drive_sync" and target_id is not None:
        row = fetch_one_sync(
            "SELECT kb_id, tenant_id, org_id FROM google_drive_sources WHERE id = ?",
            (target_id,),
        )
        if not row:
            raise ValueError(f"Google Drive source {target_id} not found")
        return {
            "kb_id": row.get("kb_id"),
            "tenant_id": row.get("tenant_id") or auth.tenant_id,
            "org_id": row.get("org_id") or auth.org_id,
        }
    return {"kb_id": None, "tenant_id": auth.tenant_id, "org_id": auth.org_id}


def _normalize_schedule_payload(schedule_type: str, target_id: int | None, payload: dict[str, Any]) -> dict[str, Any]:
    if schedule_type == "google_drive_sync":
        if target_id is None:
            raise ValueError("Google Drive sync schedule requires target_id")
        return {"source_id": int(target_id), "force_full": False}

    if schedule_type == "support_email_sync":
        return {
            "limit": int(payload.get("limit") or settings.email_fetch_limit),
            "unread_only": bool(payload.get("unread_only")),
        }

    raise ValueError(f"Unsupported schedule type: {schedule_type}")


def _active_job_exists(job_type: str, payload: dict[str, Any]) -> bool:
    payload_json = _json_payload(payload)
    row = fetch_one_sync(
        """
        SELECT id FROM background_jobs
        WHERE job_type = ?
          AND payload_json = ?
          AND status IN ('queued', 'retrying', 'running', 'cancelling')
        LIMIT 1
        """,
        (job_type, payload_json),
    )
    return row is not None


def list_sync_schedules(*, schedule_type: str | None = None, limit: int = 100) -> dict[str, Any]:
    if schedule_type:
        rows = fetch_all_sync(
            """
            SELECT * FROM sync_schedules
            WHERE schedule_type = ?
            ORDER BY enabled DESC, next_run_at ASC, id DESC
            LIMIT ?
            """,
            (schedule_type, max(1, min(limit, 200))),
        )
    else:
        rows = fetch_all_sync(
            """
            SELECT * FROM sync_schedules
            ORDER BY enabled DESC, next_run_at ASC, id DESC
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        )
    items = [_serialize_row(row).model_dump() for row in rows]
    return {"total": len(items), "items": items}


def get_sync_schedule(schedule_id: int) -> dict[str, Any]:
    row = fetch_one_sync("SELECT * FROM sync_schedules WHERE id = ?", (schedule_id,))
    if not row:
        raise ValueError("Sync schedule not found")
    return _serialize_row(row).model_dump()


def upsert_sync_schedule(payload: UpsertSyncScheduleInput, *, auth: AuthContext) -> dict[str, Any]:
    normalized_payload = _normalize_schedule_payload(payload.schedule_type, payload.target_id, payload.payload)
    name = payload.name or _default_name(payload.schedule_type, payload.target_id)
    scope = _schedule_scope(payload.schedule_type, payload.target_id, auth)
    now = utcnow_iso()

    existing = fetch_one_sync(
        """
        SELECT id FROM sync_schedules
        WHERE schedule_type = ?
          AND (
            (target_id IS NULL AND ? IS NULL)
            OR target_id = ?
          )
        ORDER BY id DESC
        LIMIT 1
        """,
        (payload.schedule_type, payload.target_id, payload.target_id),
    )
    if existing:
        schedule_id = int(existing["id"])
        execute_sync(
            """
            UPDATE sync_schedules
            SET name = ?,
                enabled = ?,
                interval_seconds = ?,
                payload_json = ?,
                tenant_id = ?,
                org_id = ?,
                kb_id = ?,
                next_run_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                name,
                1 if payload.enabled else 0,
                payload.interval_seconds,
                _json_payload(normalized_payload),
                scope["tenant_id"],
                scope["org_id"],
                scope["kb_id"],
                _next_run_after(payload.interval_seconds),
                now,
                schedule_id,
            ),
        )
        return get_sync_schedule(schedule_id)

    schedule_id = execute_sync(
        """
        INSERT INTO sync_schedules (
            schedule_type, target_id, name, enabled, interval_seconds,
            payload_json, created_by_user_id, tenant_id, org_id,
            kb_id, next_run_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.schedule_type,
            payload.target_id,
            name,
            1 if payload.enabled else 0,
            payload.interval_seconds,
            _json_payload(normalized_payload),
            auth.user_id,
            scope["tenant_id"],
            scope["org_id"],
            scope["kb_id"],
            _next_run_after(payload.interval_seconds),
            now,
            now,
        ),
    )
    return get_sync_schedule(int(schedule_id or 0))


def update_sync_schedule(schedule_id: int, payload: UpdateSyncScheduleInput) -> dict[str, Any]:
    item = get_sync_schedule(schedule_id)
    interval_seconds = payload.interval_seconds or int(item["interval_seconds"])
    name = payload.name if payload.name is not None else item["name"]
    enabled = payload.enabled if payload.enabled is not None else bool(item["enabled"])
    raw_payload = payload.payload if payload.payload is not None else item["payload"]
    normalized_payload = _normalize_schedule_payload(item["schedule_type"], item.get("target_id"), raw_payload)
    now = utcnow_iso()
    execute_sync(
        """
        UPDATE sync_schedules
        SET name = ?,
            enabled = ?,
            interval_seconds = ?,
            payload_json = ?,
            next_run_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            name,
            1 if enabled else 0,
            interval_seconds,
            _json_payload(normalized_payload),
            _next_run_after(interval_seconds),
            now,
            schedule_id,
        ),
    )
    return get_sync_schedule(schedule_id)


def delete_sync_schedule(schedule_id: int) -> dict[str, Any]:
    item = get_sync_schedule(schedule_id)
    execute_sync("DELETE FROM sync_schedules WHERE id = ?", (schedule_id,))
    return item


def run_due_sync_schedules_once(*, limit: int = 20) -> int:
    now = utcnow_iso()
    rows = fetch_all_sync(
        """
        SELECT * FROM sync_schedules
        WHERE enabled = 1
          AND next_run_at <= ?
        ORDER BY next_run_at ASC, id ASC
        LIMIT ?
        """,
        (now, max(1, min(limit, 100))),
    )
    enqueued = 0
    for row in rows:
        schedule = _serialize_row(row).model_dump()
        payload = schedule["payload"]
        job_type = schedule["schedule_type"]
        next_run_at = _next_run_after(int(schedule["interval_seconds"]))

        if _active_job_exists(job_type, payload):
            execute_sync(
                "UPDATE sync_schedules SET next_run_at = ?, updated_at = ? WHERE id = ?",
                (next_run_at, now, schedule["id"]),
            )
            continue

        context = RequestContext(
            request_id=f"sync-schedule-{schedule['id']}",
            kb_id=schedule.get("kb_id"),
            kb_key=schedule.get("kb_key"),
            auth=AuthContext(
                user_id=schedule.get("created_by_user_id") or "scheduler",
                roles=["admin"],
                channel="scheduler",
                tenant_id=schedule.get("tenant_id"),
                org_id=schedule.get("org_id"),
            ),
        )
        job = enqueue_background_job(job_type=job_type, payload=payload, context=context)
        execute_sync(
            """
            UPDATE sync_schedules
            SET last_job_id = ?,
                last_run_at = ?,
                next_run_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (job["job_id"], now, next_run_at, now, schedule["id"]),
        )
        enqueued += 1
    return enqueued


async def scheduled_sync_loop(*, poll_interval_seconds: float | None = None) -> None:
    poll_interval = poll_interval_seconds or settings.scheduled_sync_poll_interval_seconds
    logger.info("Scheduled sync loop started")
    try:
        while True:
            try:
                run_due_sync_schedules_once()
            except Exception:
                logger.exception("Scheduled sync loop iteration failed")
            await asyncio.sleep(max(1.0, poll_interval))
    except asyncio.CancelledError:
        logger.info("Scheduled sync loop stopped")
        raise
