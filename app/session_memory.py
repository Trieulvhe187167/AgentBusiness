"""
Rule-based session slot memory stored alongside chat session state.
"""

from __future__ import annotations

import json
from typing import Any

from app.database import execute_sync, fetch_one_sync, utcnow_iso


def _sanitize_slots(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if item is None:
            continue
        cleaned[key] = item
    return cleaned


def load_slots(session_id: str | None) -> dict[str, Any]:
    if not session_id:
        return {}

    row = fetch_one_sync("SELECT slots_json FROM chat_sessions WHERE session_id = ?", (session_id,))
    raw = (row or {}).get("slots_json")
    if not raw:
        return {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return _sanitize_slots(payload)


def save_slots(session_id: str | None, slots: dict[str, Any]) -> dict[str, Any]:
    if not session_id:
        return {}

    cleaned = _sanitize_slots(slots)
    execute_sync(
        """
        INSERT INTO chat_sessions (
            session_id,
            slots_json,
            updated_at
        ) VALUES (?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            slots_json = excluded.slots_json,
            updated_at = excluded.updated_at
        """,
        (session_id, json.dumps(cleaned, ensure_ascii=False), utcnow_iso()),
    )
    return cleaned


def merge_slots(session_id: str | None, updates: dict[str, Any]) -> dict[str, Any]:
    if not session_id:
        return {}

    current = load_slots(session_id)
    for key, value in (updates or {}).items():
        if not isinstance(key, str) or not key.strip():
            continue
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    return save_slots(session_id, current)

