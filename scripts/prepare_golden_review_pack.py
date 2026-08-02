"""Prepare a golden dataset review/upload pack from candidate CSV rows.

This script does not invent expected answers. Rows are activated only when they
are explicitly marked as reviewed/approved, unless an operator opts into a smoke
mode. The default output is therefore safe to upload as inactive evidence for a
knowledge owner to finish.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
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

REVIEWED_TAGS = {"reviewed", "approved", "golden_reviewed", "source_verified"}
TODO_PREFIX = "TODO:"
BROKEN_PLACEHOLDERS = ("[phone]", "[secret]", "[email]")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reviewed golden dataset CSV for upload")
    parser.add_argument("--input", required=True, help="Candidate CSV path")
    parser.add_argument("--output", required=True, help="Reviewed/upload CSV path")
    parser.add_argument("--report-output", default="", help="Optional JSON readiness report")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to include")
    parser.add_argument(
        "--activate-reviewed-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only activate rows with a reviewed/approved tag",
    )
    parser.add_argument(
        "--activate-evidence-backed",
        action="store_true",
        help="Smoke mode: activate rows with answer + source + clean chunk IDs even without reviewed tag",
    )
    parser.add_argument("--min-active", type=int, default=80, help="Target active row count for promotion gate readiness")
    return parser.parse_args()


def _read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_rows(path: str, rows: list[dict[str, str]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in FIELDNAMES} for row in rows])


def _tags(row: dict[str, str]) -> list[str]:
    return [item.strip() for item in str(row.get("tags") or "").split(",") if item.strip()]


def _has_reviewed_tag(row: dict[str, str]) -> bool:
    return bool(set(_tags(row)) & REVIEWED_TAGS)


def _answer_is_reviewable(row: dict[str, str]) -> bool:
    answer = str(row.get("expected_answer") or "").strip()
    return bool(answer) and not answer.startswith(TODO_PREFIX)


def _has_source(row: dict[str, str]) -> bool:
    return bool(str(row.get("expected_source_file_id") or "").strip() or str(row.get("expected_source_file_ids") or "").strip())


def _chunk_ids(row: dict[str, str]) -> list[str]:
    raw = str(row.get("expected_chunk_ids") or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in raw.split(",")]
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _chunk_ids_are_clean(row: dict[str, str]) -> bool:
    chunks = _chunk_ids(row)
    if not chunks:
        return False
    return not any(any(marker in chunk for marker in BROKEN_PLACEHOLDERS) for chunk in chunks)


def review_status(row: dict[str, str]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not _answer_is_reviewable(row):
        reasons.append("needs_answer")
    if not _has_source(row):
        reasons.append("needs_source")
    if str(row.get("expected_chunk_ids") or "").strip() and not _chunk_ids_are_clean(row):
        reasons.append("invalid_chunk_id")
    if not _has_reviewed_tag(row):
        reasons.append("needs_reviewed_tag")
    if reasons:
        return "needs_review", reasons
    return "reviewed", []


def prepare_rows(
    rows: list[dict[str, str]],
    *,
    activate_reviewed_only: bool,
    activate_evidence_backed: bool,
    limit: int = 0,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    prepared: list[dict[str, str]] = []
    status_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    active_count = 0
    for row in rows:
        item = {key: str(row.get(key) or "").strip() for key in FIELDNAMES}
        status, reasons = review_status(item)
        evidence_backed = _answer_is_reviewable(item) and _has_source(item) and _chunk_ids_are_clean(item)
        should_activate = False
        if activate_reviewed_only:
            should_activate = status == "reviewed"
        if activate_evidence_backed and evidence_backed:
            should_activate = True
        tags = _tags(item)
        tags.append(f"review_status:{status}")
        tags.extend(f"review_reason:{reason}" for reason in reasons)
        if activate_evidence_backed and evidence_backed and not _has_reviewed_tag(item):
            tags.append("smoke_auto_activated")
        item["tags"] = ",".join(dict.fromkeys(tags))
        item["active"] = "true" if should_activate else "false"
        if should_activate:
            active_count += 1
        status_counter[status] += 1
        for reason in reasons:
            reason_counter[reason] += 1
        prepared.append(item)
        if limit > 0 and len(prepared) >= limit:
            break
    report = {
        "rows": len(prepared),
        "active_rows": active_count,
        "status_counts": dict(status_counter),
        "reason_counts": dict(reason_counter),
        "activate_reviewed_only": activate_reviewed_only,
        "activate_evidence_backed": activate_evidence_backed,
    }
    return prepared, report


def main() -> int:
    args = _parse_args()
    rows, report = prepare_rows(
        _read_rows(args.input),
        activate_reviewed_only=bool(args.activate_reviewed_only),
        activate_evidence_backed=bool(args.activate_evidence_backed),
        limit=max(0, int(args.limit)),
    )
    report["target_active_rows"] = max(1, int(args.min_active))
    report["ready_for_promotion_gate"] = int(report["active_rows"]) >= int(report["target_active_rows"])
    _write_rows(args.output, rows)
    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(Path(args.output).resolve()), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
