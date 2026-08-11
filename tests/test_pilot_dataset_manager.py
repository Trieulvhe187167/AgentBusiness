from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from scripts import pilot_data_readiness as readiness
from scripts import pilot_dataset_manager as pilot


def _create_pilot_db(path: Path, *, production_scope: bool = True) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE knowledge_bases (
            id INTEGER PRIMARY KEY, key TEXT, name TEXT, description TEXT,
            status TEXT, is_default INTEGER, kb_version TEXT, created_at TEXT,
            updated_at TEXT, access_level TEXT, tenant_id TEXT, org_id TEXT
        );
        CREATE TABLE uploaded_files (
            id INTEGER PRIMARY KEY, filename TEXT, original_name TEXT,
            status TEXT
        );
        CREATE TABLE kb_files (
            id INTEGER PRIMARY KEY, kb_id INTEGER, file_id INTEGER, status TEXT,
            lifecycle_status TEXT, stale_reason TEXT
        );
        CREATE TABLE chat_logs (
            id INTEGER PRIMARY KEY, kb_id INTEGER, user_message TEXT,
            answer_text TEXT, citations_json TEXT, mode TEXT, top_score REAL,
            created_at TEXT, user_id TEXT
        );
        CREATE TABLE support_tickets (
            id INTEGER PRIMARY KEY, ticket_code TEXT, kb_id INTEGER,
            issue_type TEXT, intent TEXT, priority TEXT, status TEXT,
            message TEXT, contact TEXT, resolution_summary TEXT, created_at TEXT
        );
        CREATE TABLE support_email_messages (
            id INTEGER PRIMARY KEY, subject TEXT, snippet TEXT, body_text TEXT,
            ticket_code TEXT, direction TEXT, status TEXT, received_at TEXT,
            created_at TEXT, from_address TEXT, to_addresses_json TEXT,
            cc_addresses_json TEXT
        );
        CREATE TABLE eval_golden_dataset (
            id INTEGER PRIMARY KEY, kb_id INTEGER, question TEXT,
            expected_answer TEXT, expected_source_file_id INTEGER,
            expected_keywords_json TEXT, tags_json TEXT, active INTEGER,
            expected_source_file_ids_json TEXT, expected_categories_json TEXT
        );
        CREATE TABLE workflow_runs (id INTEGER PRIMARY KEY, kb_id INTEGER);
        """
    )
    access_level = "internal" if production_scope else "public"
    tenant_id = "tenant-1" if production_scope else None
    conn.execute(
        "INSERT INTO knowledge_bases VALUES (1,'pilot','Pilot Support','', 'active',0,'v1','now','now',?,?,?)",
        (access_level, tenant_id, None),
    )
    conn.execute("INSERT INTO uploaded_files VALUES (11,'policy.pdf','policy.pdf','ingested')")
    conn.execute("INSERT INTO kb_files VALUES (1,1,11,'ingested','published',NULL)")
    conn.execute(
        "INSERT INTO chat_logs VALUES (1,1,?,?,?,?,?,?,?)",
        (
            "Please contact user@example.com or +84 901 234 567 about order ORD-42",
            "The approved policy applies.",
            '[{"file_id":11,"chunk_id":"chunk-1","filename":"policy.pdf"}]',
            "answer",
            0.9,
            "2026-08-01T00:00:00Z",
            "user-1",
        ),
    )
    conn.execute(
        "INSERT INTO support_tickets VALUES (1,'T-1',1,'policy','policy_question','P2','closed','Question','user@example.com','Resolved','now')"
    )
    conn.execute(
        "INSERT INTO support_email_messages VALUES (1,'Policy','Call +84 901 234 567','',NULL,'inbound','new','now','now','user@example.com','[]','[]')"
    )
    conn.commit()
    conn.close()


def test_build_review_queue_redacts_pii_and_stays_inactive(tmp_path: Path):
    db_path = tmp_path / "pilot.db"
    _create_pilot_db(db_path)

    rows, report = pilot.build_review_queue(
        db_path=str(db_path),
        kb_id=1,
        limit=10,
        chat_limit=10,
        ticket_limit=0,
        email_limit=0,
        min_question_chars=8,
    )

    assert len(rows) == 1
    assert rows[0]["active"] == "false"
    assert rows[0]["source_type"] == "chat_log"
    assert rows[0]["source_record_id"] == "1"
    assert "[email]" in rows[0]["question"]
    assert "[phone]" in rows[0]["question"]
    assert "email_redacted" in rows[0]["pii_flags"]
    assert "phone_redacted" in rows[0]["pii_flags"]
    assert report["active_rows"] == 0


def test_finalize_requires_owner_review_and_verified_source(tmp_path: Path):
    db_path = tmp_path / "pilot.db"
    _create_pilot_db(db_path)
    row = {
        key: "" for key in pilot.REVIEW_FIELDS
    }
    row.update(
        {
            "candidate_id": "pilot-1",
            "kb_id": "1",
            "question": "What does policy POL-42 require?",
            "expected_answer": "The approved policy requires manager review.",
            "expected_source_file_id": "11",
            "expected_source_file_ids": "11",
            "expected_chunk_ids": "chunk-1",
            "expected_categories": "policy",
            "tags": "candidate",
            "coverage_buckets": "en,exact,citation",
            "review_decision": "approved",
            "reviewer_id": "knowledge-owner-1",
            "reviewed_at": "2026-08-02T10:00:00+07:00",
            "source_verified": "true",
        }
    )

    output, report = pilot.finalize_review_queue([row], db_path=str(db_path), minimum_reviewed=1)

    assert len(output) == 1
    assert output[0]["active"] == "true"
    assert "reviewed" in output[0]["tags"]
    assert report["blocked_reason_counts"] == {}


def test_finalize_allows_reviewed_negative_without_source(tmp_path: Path):
    db_path = tmp_path / "pilot.db"
    _create_pilot_db(db_path)
    row = {key: "" for key in pilot.REVIEW_FIELDS}
    row.update(
        {
            "candidate_id": "pilot-negative",
            "kb_id": "1",
            "question": "What if no approved policy exists?",
            "expected_answer": "Say the answer is unavailable and escalate.",
            "expected_categories": "negative",
            "coverage_buckets": "en,negative",
            "review_decision": "approved",
            "reviewer_id": "knowledge-owner-1",
            "reviewed_at": "2026-08-02T10:00:00+07:00",
        }
    )

    output, report = pilot.finalize_review_queue([row], db_path=str(db_path), minimum_reviewed=1)

    assert len(output) == 1
    assert "negative" in output[0]["tags"]
    assert "missing_source" not in report["blocked_reason_counts"]


def test_finalize_blocks_unreviewed_source_and_raw_pii(tmp_path: Path):
    db_path = tmp_path / "pilot.db"
    _create_pilot_db(db_path)
    row = {key: "" for key in pilot.REVIEW_FIELDS}
    row.update(
        {
            "candidate_id": "pilot-blocked",
            "kb_id": "1",
            "question": "Email user@example.com about policy",
            "expected_answer": "Approved answer",
            "expected_source_file_id": "11",
            "review_decision": "approved",
            "reviewed_at": "2026-08-02T10:00:00+07:00",
            "source_verified": "false",
        }
    )

    output, report = pilot.finalize_review_queue([row], db_path=str(db_path), minimum_reviewed=1)

    assert output == []
    assert report["blocked_reason_counts"]["missing_reviewer_id"] == 1
    assert report["blocked_reason_counts"]["raw_email_detected"] == 1
    assert report["blocked_reason_counts"]["source_not_verified_by_reviewer"] == 1


def test_readiness_audit_reports_scope_and_review_gaps_without_raw_text(tmp_path: Path):
    db_path = tmp_path / "pilot.db"
    _create_pilot_db(db_path, production_scope=False)

    report = readiness.audit_readiness(db_path=str(db_path), kb_id=1, minimum_reviewed=80)

    assert report["status"] == "blocked"
    by_key = {item["key"]: item for item in report["findings"]}
    assert by_key["kb_access_scope"]["status"] == "fail"
    assert by_key["kb_tenant_scope"]["status"] == "fail"
    assert by_key["reviewed_golden_coverage"]["status"] == "fail"
    assert report["source_pii_shape_counts"]["support_email_messages"]["email_shape_rows"] == 1
    assert "user@example.com" not in str(report)


def test_review_queue_csv_schema_is_stable(tmp_path: Path):
    output = tmp_path / "queue.csv"
    row = {key: "" for key in pilot.REVIEW_FIELDS}
    pilot._write_csv(str(output), [row], pilot.REVIEW_FIELDS)

    with output.open(encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))

    assert list(parsed[0]) == pilot.REVIEW_FIELDS
