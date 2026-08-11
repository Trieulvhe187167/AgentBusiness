"""Audit enterprise-pilot data readiness without printing sensitive values."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts import pilot_dataset_manager as pilot
    from scripts import seed_golden_dataset_from_db as seed
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import pilot_dataset_manager as pilot
    import seed_golden_dataset_from_db as seed


REVIEWED_TAGS = {"reviewed", "approved", "golden_reviewed", "source_verified"}
REQUIRED_BUCKETS = {"vi", "en", "exact", "citation", "negative", "followup"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit enterprise-pilot data readiness")
    parser.add_argument("--db", default="data/metadata.db")
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--review-queue", default="")
    parser.add_argument("--minimum-reviewed", type=int, default=80)
    parser.add_argument("--output", default="data/pilot/pilot_data_readiness.json")
    return parser.parse_args()


def _json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _finding(key: str, status: str, message: str, *, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"key": key, "status": status, "message": message, "evidence": evidence or {}}


def _pii_shape_counts(conn: sqlite3.Connection, table: str, columns: list[str]) -> dict[str, int]:
    available = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
    selected = [column for column in columns if column in available]
    if not selected:
        return {"rows": 0, "email_shape_rows": 0, "phone_shape_rows": 0, "secret_shape_rows": 0}
    query = "SELECT " + ",".join(f'"{column}"' for column in selected) + f' FROM "{table}"'
    counts = Counter()
    for row in conn.execute(query):
        counts["rows"] += 1
        text = " ".join(str(value or "") for value in row)
        counts["email_shape_rows"] += bool(seed.EMAIL_RE.search(text))
        counts["phone_shape_rows"] += bool(seed.PHONE_RE.search(text))
        counts["secret_shape_rows"] += bool(seed.TOKEN_RE.search(text))
    return dict(counts)


def _golden_profile(conn: sqlite3.Connection, kb_id: int) -> dict[str, Any]:
    rows = conn.execute("SELECT * FROM eval_golden_dataset WHERE kb_id = ? ORDER BY id", (kb_id,)).fetchall()
    profile = Counter()
    buckets = Counter()
    seen: set[str] = set()
    for row in rows:
        item = dict(row)
        tags = {str(value).casefold() for value in _json_list(item.get("tags_json"))}
        categories = [str(value) for value in _json_list(item.get("expected_categories_json"))]
        csv_like = {
            "question": str(item.get("question") or ""),
            "expected_answer": str(item.get("expected_answer") or ""),
            "expected_source_file_id": str(item.get("expected_source_file_id") or ""),
            "expected_source_file_ids": ",".join(str(value) for value in _json_list(item.get("expected_source_file_ids_json"))),
            "expected_categories": ",".join(categories),
            "tags": ",".join(tags),
        }
        profile["total"] += 1
        profile["active"] += bool(item.get("active"))
        profile["reviewed"] += bool(tags & REVIEWED_TAGS)
        profile["smoke"] += "smoke_auto_activated" in tags
        profile["todo_answer"] += str(item.get("expected_answer") or "").strip().upper().startswith("TODO:")
        has_source = bool(item.get("expected_source_file_id") or _json_list(item.get("expected_source_file_ids_json")))
        profile["source_backed"] += has_source
        normalized = pilot._normalized_question(str(item.get("question") or ""))
        profile["duplicate_question"] += normalized in seen
        seen.add(normalized)
        bucket_values = pilot._coverage_buckets(csv_like)
        buckets.update(bucket_values)
    return {**dict(profile), "coverage_bucket_counts": dict(buckets)}


def _queue_profile(path: str) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {"available": False, "rows": 0}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    decisions = Counter(str(row.get("review_decision") or "pending").casefold() for row in rows)
    pii = Counter(value for row in rows for value in pilot._split_values(row.get("pii_flags")))
    buckets = Counter(value for row in rows for value in pilot._split_values(row.get("coverage_buckets")))
    return {
        "available": True,
        "rows": len(rows),
        "decision_counts": dict(decisions),
        "pii_redaction_counts": dict(pii),
        "coverage_bucket_counts": dict(buckets),
    }


def audit_readiness(
    *,
    db_path: str,
    kb_id: int,
    review_queue_path: str = "",
    minimum_reviewed: int = 80,
) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    kb_row = conn.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone()
    findings: list[dict[str, Any]] = []
    if kb_row is None:
        conn.close()
        return {
            "status": "blocked",
            "kb_id": kb_id,
            "findings": [_finding("kb_exists", "fail", "Knowledge Base does not exist")],
        }
    kb = dict(kb_row)
    files = dict(
        conn.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN status = 'ingested' THEN 1 ELSE 0 END) AS ingested,
                   SUM(CASE WHEN lifecycle_status = 'reviewed' THEN 1 ELSE 0 END) AS reviewed,
                   SUM(CASE WHEN lifecycle_status = 'published' THEN 1 ELSE 0 END) AS published,
                   SUM(CASE WHEN status = 'failed' OR stale_reason IS NOT NULL THEN 1 ELSE 0 END) AS failed_or_stale
            FROM kb_files WHERE kb_id = ?
            """,
            (kb_id,),
        ).fetchone()
    )
    golden = _golden_profile(conn, kb_id)
    queue = _queue_profile(review_queue_path)
    support = {
        "tickets": conn.execute("SELECT COUNT(*) FROM support_tickets WHERE kb_id = ? OR kb_id IS NULL", (kb_id,)).fetchone()[0],
        "resolved_or_closed_tickets": conn.execute(
            "SELECT COUNT(*) FROM support_tickets WHERE (kb_id = ? OR kb_id IS NULL) AND status IN ('resolved','closed')",
            (kb_id,),
        ).fetchone()[0],
        "inbound_emails": conn.execute("SELECT COUNT(*) FROM support_email_messages WHERE direction = 'inbound'").fetchone()[0],
        "replied_emails": conn.execute("SELECT COUNT(*) FROM support_email_messages WHERE status = 'replied'").fetchone()[0],
        "workflow_runs": conn.execute("SELECT COUNT(*) FROM workflow_runs WHERE kb_id = ?", (kb_id,)).fetchone()[0],
    }
    pii = {
        "chat_logs": _pii_shape_counts(conn, "chat_logs", ["user_message", "answer_text", "user_id"]),
        "support_tickets": _pii_shape_counts(conn, "support_tickets", ["message", "contact", "resolution_summary"]),
        "support_email_messages": _pii_shape_counts(
            conn,
            "support_email_messages",
            ["from_address", "to_addresses_json", "cc_addresses_json", "subject", "body_text", "snippet"],
        ),
    }
    conn.close()

    findings.append(_finding("kb_active", "pass" if kb.get("status") == "active" else "fail", f"KB status is {kb.get('status')}"))
    findings.append(
        _finding(
            "kb_access_scope",
            "pass" if kb.get("access_level") in {"internal", "admin"} else "fail",
            "Pilot KB must not be public",
            evidence={"access_level": kb.get("access_level")},
        )
    )
    findings.append(
        _finding(
            "kb_tenant_scope",
            "pass" if kb.get("tenant_id") or kb.get("org_id") else "fail",
            "Pilot KB needs tenant_id or org_id scope",
            evidence={"tenant_scoped": bool(kb.get("tenant_id")), "org_scoped": bool(kb.get("org_id"))},
        )
    )
    trusted_files = int(files.get("reviewed") or 0) + int(files.get("published") or 0)
    findings.append(
        _finding(
            "trusted_knowledge_sources",
            "pass" if trusted_files > 0 else "fail",
            "At least one reviewed or published source is required",
            evidence={"reviewed": int(files.get("reviewed") or 0), "published": int(files.get("published") or 0)},
        )
    )
    findings.append(
        _finding(
            "knowledge_debt",
            "pass" if int(files.get("failed_or_stale") or 0) == 0 else "warn",
            "Failed or stale sources should be resolved before pilot",
            evidence={"failed_or_stale": int(files.get("failed_or_stale") or 0)},
        )
    )
    findings.append(
        _finding(
            "reviewed_golden_coverage",
            "pass" if int(golden.get("reviewed") or 0) >= minimum_reviewed else "fail",
            f"At least {minimum_reviewed} owner-reviewed golden rows are required",
            evidence={"reviewed": int(golden.get("reviewed") or 0), "active": int(golden.get("active") or 0)},
        )
    )
    missing_buckets = sorted(bucket for bucket in REQUIRED_BUCKETS if golden.get("coverage_bucket_counts", {}).get(bucket, 0) == 0)
    findings.append(
        _finding(
            "golden_bucket_coverage",
            "pass" if not missing_buckets else "fail",
            "Golden data must cover required pilot behaviors",
            evidence={"missing_buckets": missing_buckets},
        )
    )
    findings.append(
        _finding(
            "smoke_rows",
            "pass" if int(golden.get("smoke") or 0) == 0 else "warn",
            "Smoke-auto-activated rows do not count as reviewed production evidence",
            evidence={"smoke_rows": int(golden.get("smoke") or 0)},
        )
    )
    findings.append(
        _finding(
            "resolved_support_outcomes",
            "pass" if support["resolved_or_closed_tickets"] >= 20 else "warn",
            "Resolved outcomes are thin; use KB evidence and collect more pilot outcomes",
            evidence={"resolved_or_closed_tickets": support["resolved_or_closed_tickets"]},
        )
    )
    findings.append(
        _finding(
            "review_queue",
            "pass" if queue.get("available") else "warn",
            "A privacy-safe owner review queue should be generated before activation",
            evidence={"available": bool(queue.get("available")), "rows": int(queue.get("rows") or 0)},
        )
    )
    fail_count = sum(item["status"] == "fail" for item in findings)
    warn_count = sum(item["status"] == "warn" for item in findings)
    return {
        "status": "ready" if fail_count == 0 else "blocked",
        "kb_id": kb_id,
        "kb": {
            "key": kb.get("key"),
            "name": kb.get("name"),
            "status": kb.get("status"),
            "access_level": kb.get("access_level"),
            "tenant_scoped": bool(kb.get("tenant_id")),
            "org_scoped": bool(kb.get("org_id")),
            "kb_version": kb.get("kb_version"),
        },
        "knowledge_files": {key: int(value or 0) for key, value in files.items()},
        "golden_dataset": golden,
        "review_queue": queue,
        "support_operations": support,
        "source_pii_shape_counts": pii,
        "summary": {"fail": fail_count, "warn": warn_count, "pass": len(findings) - fail_count - warn_count},
        "findings": findings,
    }


def main() -> int:
    args = _parse_args()
    report = audit_readiness(
        db_path=args.db,
        kb_id=args.kb_id,
        review_queue_path=args.review_queue,
        minimum_reviewed=max(1, args.minimum_reviewed),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "status": report["status"], **report.get("summary", {})}, ensure_ascii=False))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
