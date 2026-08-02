"""Build review-ready golden dataset candidates from local production-like data.

The script exports a CSV that matches `/api/admin/evaluations/golden-dataset/upload`.
Rows are inactive by default because chat answers, ticket text, and emails are
signals, not authoritative references. A knowledge owner should review expected
answers and source IDs before activating rows for a quality gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "kb_id",
    "question",
    "expected_answer",
    "expected_answers",
    "expected_source_file_id",
    "expected_source_file_ids",
    "expected_chunk_ids",
    "expected_categories",
    "expected_keywords",
    "tags",
    "active",
]

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()/-]{7,}\d)(?!\d)")
TOKEN_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|[A-Fa-f0-9]{32,})\b")
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]{3,}", re.UNICODE)

GENERIC_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "what",
    "when",
    "where",
    "which",
    "how",
    "can",
    "you",
    "are",
    "your",
    "cua",
    "cho",
    "toi",
    "toi",
    "ban",
    "nhu",
    "the",
    "nao",
    "khi",
    "khong",
    "duoc",
    "trong",
    "phai",
    "hay",
    "voi",
    "mot",
    "cac",
    "nhung",
}

ANSWER_LIKE_MODES = {"answer", "tool", "memory"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export golden dataset candidates from local DB")
    parser.add_argument("--db", default="data/metadata.db", help="SQLite database path")
    parser.add_argument("--output", default="data/golden_candidates.csv", help="Output CSV path")
    parser.add_argument("--kb-id", type=int, default=None, help="Only export rows for this KB when available")
    parser.add_argument("--default-kb-id", type=int, default=1, help="KB id for tickets/emails without a KB")
    parser.add_argument("--limit", type=int, default=150, help="Maximum total candidate rows")
    parser.add_argument("--chat-limit", type=int, default=90, help="Maximum chat-derived rows")
    parser.add_argument("--ticket-limit", type=int, default=40, help="Maximum ticket-derived rows")
    parser.add_argument("--email-limit", type=int, default=40, help="Maximum email-derived rows")
    parser.add_argument("--min-question-chars", type=int, default=12, help="Skip shorter questions/messages")
    parser.add_argument(
        "--active",
        action="store_true",
        help="Mark exported rows active. Use only after manual answer/source review.",
    )
    parser.add_argument(
        "--include-chat-answers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use existing chat answers as suggested expected answers.",
    )
    return parser.parse_args()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _clean_text(value: Any, *, max_chars: int = 1200) -> str:
    text = str(value or "")
    text = EMAIL_RE.sub("[email]", text)
    text = PHONE_RE.sub("[phone]", text)
    text = TOKEN_RE.sub("[secret]", text)
    text = SPACE_RE.sub(" ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _normalize_key(value: str) -> str:
    return SPACE_RE.sub(" ", value.strip().lower())


def _safe_json_list(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _filename_to_file_ids(conn: sqlite3.Connection) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    rows = conn.execute("SELECT id, filename, original_name FROM uploaded_files ORDER BY id DESC").fetchall()
    for row in rows:
        for name in (row["filename"], row["original_name"]):
            normalized = _normalize_key(str(name or ""))
            if normalized:
                mapping.setdefault(normalized, []).append(int(row["id"]))
    return mapping


def _source_ids_from_citations(citations: list[dict[str, Any]], file_map: dict[str, list[int]]) -> list[int]:
    source_ids: list[int] = []
    for citation in citations:
        direct = citation.get("file_id") or citation.get("source_file_id")
        if direct:
            try:
                parsed = int(direct)
            except (TypeError, ValueError):
                parsed = 0
            if parsed > 0 and parsed not in source_ids:
                source_ids.append(parsed)
        filename = _normalize_key(str(citation.get("filename") or citation.get("source") or ""))
        for parsed in file_map.get(filename, []):
            if parsed not in source_ids:
                source_ids.append(parsed)
    return source_ids[:5]


def _chunk_ids_from_citations(citations: list[dict[str, Any]]) -> list[str]:
    chunk_ids: list[str] = []
    for citation in citations:
        chunk_id = SPACE_RE.sub(" ", str(citation.get("chunk_id") or "")).strip()
        if len(chunk_id) > 120:
            chunk_id = chunk_id[:117].rstrip() + "..."
        if chunk_id and chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)
    return chunk_ids[:10]


def _keywords(*texts: str, limit: int = 8) -> list[str]:
    seen: list[str] = []
    for text in texts:
        for word in WORD_RE.findall(text.lower()):
            if word in GENERIC_STOPWORDS or len(word) < 3:
                continue
            if word not in seen:
                seen.append(word)
            if len(seen) >= limit:
                return seen
    return seen


def _row(
    *,
    kb_id: int,
    question: str,
    expected_answer: str,
    categories: Iterable[str],
    tags: Iterable[str],
    active: bool,
    source_file_ids: list[int] | None = None,
    chunk_ids: list[str] | None = None,
    keywords: list[str] | None = None,
) -> dict[str, str]:
    source_ids = source_file_ids or []
    chunks = chunk_ids or []
    category_values = [item for item in categories if item]
    tag_values = [item for item in tags if item]
    if not source_ids:
        tag_values.append("needs_source_review")
    if expected_answer.startswith("TODO:"):
        tag_values.append("needs_answer_review")
    return {
        "kb_id": str(kb_id),
        "question": question,
        "expected_answer": expected_answer,
        "expected_answers": "",
        "expected_source_file_id": str(source_ids[0]) if source_ids else "",
        "expected_source_file_ids": ",".join(str(item) for item in source_ids),
        "expected_chunk_ids": ",".join(chunks),
        "expected_categories": ",".join(dict.fromkeys(category_values)),
        "expected_keywords": ",".join(keywords or _keywords(question, expected_answer)),
        "tags": ",".join(dict.fromkeys(tag_values)),
        "active": "true" if active else "false",
    }


def _chat_candidates(
    conn: sqlite3.Connection,
    *,
    kb_id: int | None,
    limit: int,
    min_question_chars: int,
    include_answers: bool,
    active: bool,
    file_map: dict[str, list[int]],
) -> list[dict[str, str]]:
    clauses = ["user_message IS NOT NULL", "TRIM(user_message) != ''"]
    params: list[Any] = []
    if kb_id is not None:
        clauses.append("kb_id = ?")
        params.append(kb_id)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT id, kb_id, user_message, answer_text, citations_json, mode, top_score, created_at
        FROM chat_logs
        WHERE {' AND '.join(clauses)}
        ORDER BY
            CASE WHEN mode IN ('answer', 'tool', 'memory') THEN 0 ELSE 1 END,
            CASE WHEN citations_json IS NOT NULL AND citations_json NOT IN ('', '[]') THEN 0 ELSE 1 END,
            created_at DESC,
            id DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    candidates: list[dict[str, str]] = []
    for row in rows:
        question = _clean_text(row["user_message"], max_chars=900)
        if len(question) < min_question_chars:
            continue
        citations = _safe_json_list(row["citations_json"])
        source_ids = _source_ids_from_citations(citations, file_map)
        chunk_ids = _chunk_ids_from_citations(citations)
        mode = _clean_text(row["mode"], max_chars=80)
        can_reuse_answer = include_answers and mode in ANSWER_LIKE_MODES
        answer = _clean_text(row["answer_text"], max_chars=1600) if can_reuse_answer else ""
        if not answer:
            answer = "TODO: replace with grounded reference answer from reviewed KB source"
        tags = ["candidate", "from_chat_log", f"chat_log:{row['id']}"]
        if mode:
            tags.append(f"mode:{mode}")
        candidates.append(
            _row(
                kb_id=int(row["kb_id"] or kb_id or 1),
                question=question,
                expected_answer=answer,
                source_file_ids=source_ids,
                chunk_ids=chunk_ids,
                categories=["chat_log", mode],
                tags=tags,
                active=active,
            )
        )
    return candidates


def _ticket_candidates(
    conn: sqlite3.Connection,
    *,
    kb_id: int | None,
    default_kb_id: int,
    limit: int,
    min_question_chars: int,
    active: bool,
) -> list[dict[str, str]]:
    clauses = ["message IS NOT NULL", "TRIM(message) != ''"]
    params: list[Any] = []
    if kb_id is not None:
        clauses.append("(kb_id = ? OR kb_id IS NULL)")
        params.append(kb_id)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT id, ticket_code, kb_id, issue_type, intent, priority, status,
               message, resolution_summary, created_at
        FROM support_tickets
        WHERE {' AND '.join(clauses)}
        ORDER BY
            CASE WHEN resolution_summary IS NOT NULL AND TRIM(resolution_summary) != '' THEN 0 ELSE 1 END,
            created_at DESC,
            id DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    candidates: list[dict[str, str]] = []
    for row in rows:
        question = _clean_text(row["message"], max_chars=900)
        if len(question) < min_question_chars:
            continue
        answer = _clean_text(row["resolution_summary"], max_chars=1600)
        if not answer:
            answer = "TODO: replace with approved support resolution from KB or case notes"
        issue_type = _clean_text(row["issue_type"], max_chars=80)
        intent = _clean_text(row["intent"], max_chars=80)
        priority = _clean_text(row["priority"], max_chars=40)
        candidates.append(
            _row(
                kb_id=int(row["kb_id"] or kb_id or default_kb_id),
                question=question,
                expected_answer=answer,
                categories=["support_ticket", issue_type, intent, priority],
                tags=["candidate", "from_support_ticket", f"ticket:{row['ticket_code'] or row['id']}"],
                active=active,
            )
        )
    return candidates


def _email_candidates(
    conn: sqlite3.Connection,
    *,
    kb_id: int | None,
    default_kb_id: int,
    limit: int,
    min_question_chars: int,
    active: bool,
) -> list[dict[str, str]]:
    params: list[Any] = [limit]
    rows = conn.execute(
        """
        SELECT id, subject, snippet, body_text, ticket_code, direction, received_at, created_at
        FROM support_email_messages
        WHERE direction = 'inbound'
          AND COALESCE(TRIM(subject), '') != ''
        ORDER BY received_at DESC, created_at DESC, id DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    candidates: list[dict[str, str]] = []
    for row in rows:
        body = _clean_text(row["snippet"] or row["body_text"], max_chars=600)
        subject = _clean_text(row["subject"], max_chars=220)
        question = subject if not body else f"{subject} - {body}"
        if len(question) < min_question_chars:
            continue
        tags = ["candidate", "from_support_email", f"email:{row['id']}"]
        if row["ticket_code"]:
            tags.append(f"ticket:{row['ticket_code']}")
        candidates.append(
            _row(
                kb_id=int(kb_id or default_kb_id),
                question=question,
                expected_answer="TODO: replace with approved support answer from KB or resolved ticket",
                categories=["support_email"],
                tags=tags,
                active=active,
            )
        )
    return candidates


def build_candidates(
    *,
    db_path: str,
    kb_id: int | None,
    default_kb_id: int,
    limit: int,
    chat_limit: int,
    ticket_limit: int,
    email_limit: int,
    min_question_chars: int,
    include_chat_answers: bool,
    active: bool,
) -> list[dict[str, str]]:
    conn = _connect(db_path)
    file_map = _filename_to_file_ids(conn)
    raw_rows: list[dict[str, str]] = []
    raw_rows.extend(
        _chat_candidates(
            conn,
            kb_id=kb_id,
            limit=chat_limit,
            min_question_chars=min_question_chars,
            include_answers=include_chat_answers,
            active=active,
            file_map=file_map,
        )
    )
    raw_rows.extend(
        _ticket_candidates(
            conn,
            kb_id=kb_id,
            default_kb_id=default_kb_id,
            limit=ticket_limit,
            min_question_chars=min_question_chars,
            active=active,
        )
    )
    raw_rows.extend(
        _email_candidates(
            conn,
            kb_id=kb_id,
            default_kb_id=default_kb_id,
            limit=email_limit,
            min_question_chars=min_question_chars,
            active=active,
        )
    )

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in raw_rows:
        key = _normalize_key(row["question"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def write_csv(rows: list[dict[str, str]], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    rows = build_candidates(
        db_path=args.db,
        kb_id=args.kb_id,
        default_kb_id=args.default_kb_id,
        limit=max(1, args.limit),
        chat_limit=max(0, args.chat_limit),
        ticket_limit=max(0, args.ticket_limit),
        email_limit=max(0, args.email_limit),
        min_question_chars=max(1, args.min_question_chars),
        include_chat_answers=args.include_chat_answers,
        active=args.active,
    )
    write_csv(rows, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "rows": len(rows), "active": args.active}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
