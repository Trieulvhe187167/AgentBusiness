"""
SQLite-backed conversation memory helpers for prompt construction.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.config import settings
from app.database import fetch_all_sync

_FOLLOWUP_PATTERNS = (
    re.compile(r"\bit\b", re.IGNORECASE),
    re.compile(r"\bthis one\b", re.IGNORECASE),
    re.compile(r"\bthat one\b", re.IGNORECASE),
    re.compile(r"\bthat order\b", re.IGNORECASE),
    re.compile(r"\bthis order\b", re.IGNORECASE),
    re.compile(r"\bwhen will it arrive\b", re.IGNORECASE),
    re.compile(r"\btoo expensive\b", re.IGNORECASE),
    re.compile(r"\btoo cheap\b", re.IGNORECASE),
    re.compile(r"\bprice seems high\b", re.IGNORECASE),
    re.compile(r"\bseems expensive\b", re.IGNORECASE),
)

_FOLLOWUP_ASCII_HINTS = (
    "dat qua",
    "hoi dat",
    "dat nhi",
    "gia hoi cao",
    "cao qua",
    "hoi cao",
    "re qua",
    "cai nay",
    "cai do",
    "mon nay",
    "mon do",
    "san pham nay",
    "san pham do",
    "don nay",
    "don do",
    "lien minh nay",
    "lien minh do",
    "khi nao toi",
    "bao gio toi",
    "con hang khong",
    "con bao nhieu",
)

_SHORT_FOLLOWUP_HINTS = {
    "dat",
    "re",
    "cao",
    "thap",
    "mac",
    "sao",
    "the nao",
}

_OPINION_FOLLOWUP_HINTS = (
    "dat",
    "re",
    "cao",
    "thap",
    "mac",
    "too expensive",
    "too cheap",
    "price seems high",
    "seems expensive",
)

_REACTION_HIGH_HINTS = (
    "dat qua",
    "hoi dat",
    "dat nhi",
    "gia hoi cao",
    "cao qua",
    "hoi cao",
    "too expensive",
    "price seems high",
    "seems expensive",
)

_REACTION_LOW_HINTS = (
    "re qua",
    "qua re",
    "gia re",
    "hoi re",
    "thap qua",
    "too cheap",
    "price seems low",
    "seems cheap",
)


def _clean_text(value: Any, *, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _ascii_hint(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(stripped.split())


def load_recent_turns(session_id: str | None, *, limit: int | None = None) -> list[dict[str, Any]]:
    resolved_limit = limit if limit is not None else settings.conversation_memory_turn_limit
    if not session_id or resolved_limit <= 0:
        return []

    try:
        rows = fetch_all_sync(
            """
            SELECT user_message, answer_text, mode
            FROM chat_logs
            WHERE session_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (session_id, resolved_limit),
        )
    except Exception:
        return []

    return [dict(row) for row in reversed(rows)]


def build_conversation_context(turns: list[dict[str, Any]]) -> str:
    if not turns:
        return ""

    lines = ["Recent conversation history:"]
    for index, turn in enumerate(turns, start=1):
        user_message = _clean_text(turn.get("user_message"))
        answer_text = _clean_text(turn.get("answer_text"))
        if user_message:
            lines.append(f"Turn {index} user: {user_message}")
        if answer_text:
            lines.append(f"Turn {index} assistant: {answer_text}")
    return "\n".join(lines)


def looks_like_followup(query: str) -> bool:
    normalized = " ".join(str(query or "").split()).strip(" .,!?:;").lower()
    if not normalized:
        return False
    ascii_normalized = _ascii_hint(normalized)
    tokens = [token for token in re.split(r"\s+", ascii_normalized) if token]
    return any(pattern.search(normalized) for pattern in _FOLLOWUP_PATTERNS) or any(
        hint in ascii_normalized for hint in _FOLLOWUP_ASCII_HINTS
    ) or (len(tokens) <= 4 and any(hint in ascii_normalized for hint in _SHORT_FOLLOWUP_HINTS))


def _compact_followup_query(last_user: str, last_answer: str, current: str) -> str:
    current_ascii = _ascii_hint(current)
    if any(hint in current_ascii for hint in _OPINION_FOLLOWUP_HINTS):
        return f"{last_user}\nUser follow-up about the same topic: {current}"
    if last_answer:
        return f"{last_user}\nPrevious answer: {last_answer}\nUser follow-up: {current}"
    return f"{last_user}\nUser follow-up: {current}"


def resolve_followup_query(query: str, turns: list[dict[str, Any]]) -> tuple[str, str | None]:
    if not looks_like_followup(query) or not turns:
        return query, None

    latest_turn = turns[-1]
    last_user = _clean_text(latest_turn.get("user_message"), max_chars=140)
    last_answer = _clean_text(latest_turn.get("answer_text"), max_chars=180)
    current = _clean_text(query, max_chars=140)

    if last_user:
        return _compact_followup_query(last_user, last_answer, current), "history_followup_rewrite"

    parts = []
    if last_answer:
        parts.append(f"Previous assistant answer: {last_answer}")
    parts.append(f"Current follow-up: {current}")
    return "\n".join(parts), "history_followup_rewrite"


def detect_followup_reaction(query: str) -> str | None:
    normalized = " ".join(str(query or "").split()).strip(" .,!?:;").lower()
    if not normalized:
        return None

    ascii_normalized = _ascii_hint(normalized)
    if any(hint in ascii_normalized for hint in _REACTION_HIGH_HINTS):
        return "high"
    if any(hint in ascii_normalized for hint in _REACTION_LOW_HINTS):
        return "low"
    return None
