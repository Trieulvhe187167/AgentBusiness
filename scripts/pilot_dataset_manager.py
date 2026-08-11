"""Build and finalize a privacy-safe review queue for an enterprise pilot.

The build command derives inactive candidates from local operational data and
adds review metadata without exposing raw email addresses, phone numbers, or
secret-shaped values. The finalize command only emits upload-ready golden rows
after an identified reviewer approves the row and verifies its evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from scripts import seed_golden_dataset_from_db as seed
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import seed_golden_dataset_from_db as seed


REVIEW_FIELDS = [
    "candidate_id",
    *seed.FIELDNAMES,
    "source_type",
    "source_record_id",
    "language_hint",
    "coverage_buckets",
    "pii_flags",
    "review_decision",
    "reviewer_id",
    "reviewed_at",
    "source_verified",
    "review_notes",
]

APPROVED_DECISIONS = {"approved", "reviewed"}
NEGATIVE_BUCKETS = {"negative", "no_answer", "unknown", "out_of_scope"}
VI_MARKERS = {
    "khong",
    "duoc",
    "chinh",
    "sach",
    "don",
    "hang",
    "bao",
    "nhieu",
    "the",
    "nao",
    "cua",
    "toi",
}
EN_MARKERS = {"what", "when", "where", "which", "how", "policy", "order", "support", "refund"}
VI_DIACRITIC_RE = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.IGNORECASE)
EXACT_ID_RE = re.compile(r"\b(?=[A-Z0-9_-]{4,}\b)(?=[A-Z0-9_-]*\d)[A-Z][A-Z0-9_-]*\b", re.IGNORECASE)
FOLLOWUP_RE = re.compile(r"^(?:and |what about|how about|còn |thế còn|vậy |nếu |trường hợp )", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage enterprise-pilot golden dataset review")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build an inactive, redacted review queue")
    build.add_argument("--db", default="data/metadata.db")
    build.add_argument("--output", default="data/pilot/golden_review_queue.csv")
    build.add_argument("--report-output", default="data/pilot/golden_review_queue.report.json")
    build.add_argument("--kb-id", type=int, required=True)
    build.add_argument("--limit", type=int, default=150)
    build.add_argument("--chat-limit", type=int, default=100)
    build.add_argument("--ticket-limit", type=int, default=40)
    build.add_argument("--email-limit", type=int, default=40)
    build.add_argument("--min-question-chars", type=int, default=12)

    finalize = subparsers.add_parser("finalize", help="Create an upload pack from approved review rows")
    finalize.add_argument("--input", required=True)
    finalize.add_argument("--output", default="data/pilot/golden_upload.csv")
    finalize.add_argument("--report-output", default="data/pilot/golden_finalize.report.json")
    finalize.add_argument("--db", default="data/metadata.db")
    finalize.add_argument("--minimum-reviewed", type=int, default=80)
    return parser.parse_args()


def _read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _split_values(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _normalized_question(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", value.casefold()).strip()


def _candidate_id(row: dict[str, str]) -> str:
    provenance = next(
        (tag for tag in _split_values(row.get("tags")) if tag.startswith(("chat_log:", "ticket:", "email:"))),
        "",
    )
    raw = f"{row.get('kb_id', '')}|{_normalized_question(row.get('question', ''))}|{provenance}"
    return "pilot-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _provenance(row: dict[str, str]) -> tuple[str, str]:
    tags = _split_values(row.get("tags"))
    for tag in tags:
        if tag == "from_chat_log":
            source_type = "chat_log"
            prefix = "chat_log:"
            break
        if tag == "from_support_ticket":
            source_type = "support_ticket"
            prefix = "ticket:"
            break
        if tag == "from_support_email":
            source_type = "support_email"
            prefix = "email:"
            break
    else:
        return "unknown", ""
    source_record_id = next((tag[len(prefix) :] for tag in tags if tag.startswith(prefix)), "")
    return source_type, source_record_id


def _language_hint(text: str) -> str:
    lowered = text.casefold()
    words = set(re.findall(r"[a-z]+", lowered))
    if VI_DIACRITIC_RE.search(lowered) or len(words & VI_MARKERS) >= 2:
        return "vi"
    if words & EN_MARKERS:
        return "en"
    return "unknown"


def _coverage_buckets(row: dict[str, str]) -> list[str]:
    question = str(row.get("question") or "")
    values = {item.casefold() for item in _split_values(row.get("tags")) + _split_values(row.get("expected_categories"))}
    buckets: list[str] = []
    language = _language_hint(question)
    if language in {"vi", "en"}:
        buckets.append(language)
    if row.get("expected_source_file_id") or row.get("expected_source_file_ids"):
        buckets.append("citation")
    if EXACT_ID_RE.search(question) or any("exact" in value for value in values):
        buckets.append("exact")
    if values & NEGATIVE_BUCKETS or any(value in " ".join(values) for value in NEGATIVE_BUCKETS):
        buckets.append("negative")
    if FOLLOWUP_RE.search(question) or any("followup" in value or "follow_up" in value for value in values):
        buckets.append("followup")
    return list(dict.fromkeys(buckets))


def _pii_flags(row: dict[str, str]) -> list[str]:
    text = " ".join(str(row.get(key) or "") for key in ("question", "expected_answer", "expected_answers"))
    flags: list[str] = []
    for marker, name in (("[email]", "email_redacted"), ("[phone]", "phone_redacted"), ("[secret]", "secret_redacted")):
        if marker in text:
            flags.append(name)
    return flags


def build_review_queue(
    *,
    db_path: str,
    kb_id: int,
    limit: int,
    chat_limit: int,
    ticket_limit: int,
    email_limit: int,
    min_question_chars: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    candidates = seed.build_candidates(
        db_path=db_path,
        kb_id=kb_id,
        default_kb_id=kb_id,
        limit=limit,
        chat_limit=chat_limit,
        ticket_limit=ticket_limit,
        email_limit=email_limit,
        min_question_chars=min_question_chars,
        include_chat_answers=True,
        active=False,
    )
    queue: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    pii_counts: Counter[str] = Counter()
    seen_questions: set[str] = set()
    duplicate_count = 0
    for candidate in candidates:
        normalized = _normalized_question(candidate["question"])
        if normalized in seen_questions:
            duplicate_count += 1
            continue
        seen_questions.add(normalized)
        source_type, source_record_id = _provenance(candidate)
        language = _language_hint(candidate["question"])
        buckets = _coverage_buckets(candidate)
        pii = _pii_flags(candidate)
        row = {
            "candidate_id": _candidate_id(candidate),
            **candidate,
            "active": "false",
            "source_type": source_type,
            "source_record_id": source_record_id,
            "language_hint": language,
            "coverage_buckets": ",".join(buckets),
            "pii_flags": ",".join(pii),
            "review_decision": "pending",
            "reviewer_id": "",
            "reviewed_at": "",
            "source_verified": "false",
            "review_notes": "",
        }
        queue.append(row)
        source_counts[source_type] += 1
        language_counts[language] += 1
        bucket_counts.update(buckets)
        pii_counts.update(pii)
    report = {
        "kb_id": kb_id,
        "rows": len(queue),
        "active_rows": 0,
        "duplicate_rows_removed": duplicate_count,
        "source_counts": dict(source_counts),
        "language_counts": dict(language_counts),
        "coverage_bucket_counts": dict(bucket_counts),
        "pii_redaction_counts": dict(pii_counts),
        "review_decision_counts": {"pending": len(queue)},
    }
    return queue, report


def _valid_reviewed_at(value: str) -> bool:
    try:
        datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return True
    except (TypeError, ValueError):
        return False


def _source_ids(row: dict[str, str]) -> list[int]:
    raw = _split_values(row.get("expected_source_file_ids"))
    if not raw and str(row.get("expected_source_file_id") or "").strip():
        raw = [str(row["expected_source_file_id"]).strip()]
    parsed: list[int] = []
    for item in raw:
        value = int(item)
        if value < 1:
            raise ValueError("source IDs must be positive")
        if value not in parsed:
            parsed.append(value)
    return parsed


def _is_negative(row: dict[str, str]) -> bool:
    values = {item.casefold() for item in _split_values(row.get("coverage_buckets"))}
    values.update(item.casefold() for item in _split_values(row.get("expected_categories")))
    values.update(item.casefold() for item in _split_values(row.get("tags")))
    return bool(values & NEGATIVE_BUCKETS)


def _validate_sources(conn: sqlite3.Connection, *, kb_id: int, source_ids: list[int]) -> list[str]:
    reasons: list[str] = []
    for source_id in source_ids:
        row = conn.execute(
            """
            SELECT uf.id, uf.status, kf.status AS kb_status, kf.lifecycle_status
            FROM uploaded_files uf
            LEFT JOIN kb_files kf ON kf.file_id = uf.id AND kf.kb_id = ?
            WHERE uf.id = ?
            """,
            (kb_id, source_id),
        ).fetchone()
        if row is None:
            reasons.append("source_not_found")
            continue
        if row[2] is None:
            reasons.append("source_not_attached_to_kb")
        if str(row[1] or "") != "ingested" or str(row[2] or "") != "ingested":
            reasons.append("source_not_ingested")
        if str(row[3] or "") not in {"reviewed", "published"}:
            reasons.append("source_not_reviewed_or_published")
    return list(dict.fromkeys(reasons))


def finalize_review_queue(
    rows: list[dict[str, str]],
    *,
    db_path: str,
    minimum_reviewed: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    output_rows: list[dict[str, str]] = []
    reason_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    seen_questions: set[str] = set()
    for row in rows:
        decision = str(row.get("review_decision") or "pending").strip().casefold()
        decision_counts[decision] += 1
        if decision not in APPROVED_DECISIONS:
            continue
        reasons: list[str] = []
        question = str(row.get("question") or "").strip()
        answer = str(row.get("expected_answer") or "").strip()
        normalized = _normalized_question(question)
        if not question:
            reasons.append("missing_question")
        if not answer or answer.upper().startswith("TODO:"):
            reasons.append("missing_reviewed_answer")
        if not str(row.get("reviewer_id") or "").strip():
            reasons.append("missing_reviewer_id")
        if not _valid_reviewed_at(str(row.get("reviewed_at") or "")):
            reasons.append("invalid_reviewed_at")
        if normalized in seen_questions:
            reasons.append("duplicate_question")
        if seed.EMAIL_RE.search(question + " " + answer):
            reasons.append("raw_email_detected")
        if seed.PHONE_RE.search(question + " " + answer):
            reasons.append("raw_phone_detected")
        if seed.TOKEN_RE.search(question + " " + answer):
            reasons.append("raw_secret_detected")
        try:
            source_ids = _source_ids(row)
        except (TypeError, ValueError):
            source_ids = []
            reasons.append("invalid_source_id")
        if not _is_negative(row):
            if not source_ids:
                reasons.append("missing_source")
            elif str(row.get("source_verified") or "").strip().casefold() not in {"1", "true", "yes"}:
                reasons.append("source_not_verified_by_reviewer")
            else:
                try:
                    kb_id = int(str(row.get("kb_id") or "0"))
                except ValueError:
                    kb_id = 0
                    reasons.append("invalid_kb_id")
                if kb_id > 0:
                    reasons.extend(_validate_sources(conn, kb_id=kb_id, source_ids=source_ids))
        if reasons:
            reason_counts.update(list(dict.fromkeys(reasons)))
            continue
        seen_questions.add(normalized)
        buckets = _split_values(row.get("coverage_buckets")) or _coverage_buckets(row)
        bucket_counts.update(buckets)
        tags = _split_values(row.get("tags"))
        tags.extend(["reviewed", "source_verified"] if not _is_negative(row) else ["reviewed", "negative"])
        item = {key: str(row.get(key) or "").strip() for key in seed.FIELDNAMES}
        item["tags"] = ",".join(dict.fromkeys(tags))
        item["active"] = "true"
        output_rows.append(item)
    conn.close()
    required_buckets = {"vi", "en", "exact", "citation", "negative", "followup"}
    missing_buckets = sorted(bucket for bucket in required_buckets if bucket_counts.get(bucket, 0) == 0)
    report = {
        "input_rows": len(rows),
        "approved_decision_rows": sum(decision_counts[name] for name in APPROVED_DECISIONS),
        "finalized_rows": len(output_rows),
        "blocked_approved_rows": sum(reason_counts.values()) > 0,
        "decision_counts": dict(decision_counts),
        "blocked_reason_counts": dict(reason_counts),
        "coverage_bucket_counts": dict(bucket_counts),
        "missing_required_buckets": missing_buckets,
        "minimum_reviewed_rows": minimum_reviewed,
        "ready_for_pilot_baseline": len(output_rows) >= minimum_reviewed and not missing_buckets,
    }
    return output_rows, report


def main() -> int:
    args = _parse_args()
    if args.command == "build":
        rows, report = build_review_queue(
            db_path=args.db,
            kb_id=args.kb_id,
            limit=max(1, args.limit),
            chat_limit=max(0, args.chat_limit),
            ticket_limit=max(0, args.ticket_limit),
            email_limit=max(0, args.email_limit),
            min_question_chars=max(1, args.min_question_chars),
        )
        _write_csv(args.output, rows, REVIEW_FIELDS)
        _write_json(args.report_output, report)
        print(json.dumps({"output": str(Path(args.output).resolve()), **report}, ensure_ascii=False))
        return 0
    rows, report = finalize_review_queue(
        _read_csv(args.input),
        db_path=args.db,
        minimum_reviewed=max(1, args.minimum_reviewed),
    )
    _write_csv(args.output, rows, seed.FIELDNAMES)
    _write_json(args.report_output, report)
    print(json.dumps({"output": str(Path(args.output).resolve()), **report}, ensure_ascii=False))
    return 0 if not report["blocked_approved_rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
