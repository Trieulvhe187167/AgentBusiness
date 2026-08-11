from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path

from scripts import generate_golden_dataset_template as template
from scripts import prepare_golden_review_pack as review_pack
from scripts import run_baseline_eval
from scripts import seed_golden_dataset_from_db as seed


def test_golden_dataset_template_has_production_sized_starter_rows():
    rows = template._rows(kb_id=7)

    assert len(rows) == 96
    assert {row["kb_id"] for row in rows} == {"7"}
    assert all(row["question"] for row in rows)
    assert all(row["expected_answer"].startswith("TODO:") for row in rows)
    assert all(row["active"] == "false" for row in rows)
    assert {"policy_vi", "policy_en", "exact_id", "citation", "negative", "followup"} <= {
        row["expected_categories"] for row in rows
    }


def test_golden_dataset_template_columns_match_upload_contract():
    rows = template._rows(kb_id=1)
    fieldnames = [
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
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows[:2])

    parsed = list(csv.DictReader(io.StringIO(out.getvalue())))
    assert parsed[0]["question"]
    assert parsed[0]["expected_keywords"] == "TODO"
    assert parsed[0]["tags"].startswith("starter,")


def _create_seed_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE uploaded_files (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL
        );
        CREATE TABLE chat_logs (
            id INTEGER PRIMARY KEY,
            kb_id INTEGER,
            user_message TEXT NOT NULL,
            answer_text TEXT,
            citations_json TEXT,
            mode TEXT,
            top_score REAL,
            created_at TEXT
        );
        CREATE TABLE support_tickets (
            id INTEGER PRIMARY KEY,
            ticket_code TEXT,
            kb_id INTEGER,
            issue_type TEXT,
            intent TEXT,
            priority TEXT,
            status TEXT,
            message TEXT,
            resolution_summary TEXT,
            created_at TEXT
        );
        CREATE TABLE support_email_messages (
            id INTEGER PRIMARY KEY,
            subject TEXT,
            snippet TEXT,
            body_text TEXT,
            ticket_code TEXT,
            direction TEXT,
            received_at TEXT,
            created_at TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO uploaded_files (id, filename, original_name) VALUES (11, 'policy.pdf', 'policy.pdf')"
    )
    conn.execute(
        """
        INSERT INTO chat_logs (
            id, kb_id, user_message, answer_text, citations_json, mode, top_score, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            7,
            "When can a customer request a refund?",
            "Refunds are available within seven days with policy evidence.",
            json.dumps([{"filename": "policy.pdf", "chunk_id": "chunk-abc", "score": 0.91}]),
            "answer",
            0.91,
            "2026-06-01T00:00:00Z",
        ),
    )
    conn.execute(
        """
        INSERT INTO support_tickets (
            id, ticket_code, kb_id, issue_type, intent, priority, status,
            message, resolution_summary, created_at
        ) VALUES (2, 'TCK-2', NULL, 'refund', 'refund_request', 'P2', 'resolved',
                  'Customer asks whether refund needs approval.', 'Use the refund policy and approval queue.', '2026-06-02T00:00:00Z')
        """
    )
    conn.execute(
        """
        INSERT INTO support_email_messages (
            id, subject, snippet, body_text, ticket_code, direction, received_at, created_at
        ) VALUES (3, 'Refund timing', 'Please reply to user@example.com about refund timing.',
                  '', 'TCK-2', 'inbound', '2026-06-03T00:00:00Z', '2026-06-03T00:00:00Z')
        """
    )
    conn.commit()
    conn.close()


def test_seed_golden_dataset_from_db_builds_review_ready_candidates(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    _create_seed_db(db_path)

    rows = seed.build_candidates(
        db_path=str(db_path),
        kb_id=None,
        default_kb_id=1,
        limit=10,
        chat_limit=10,
        ticket_limit=10,
        email_limit=10,
        min_question_chars=8,
        include_chat_answers=True,
        active=False,
    )

    assert len(rows) == 3
    chat_row = rows[0]
    assert chat_row["kb_id"] == "7"
    assert chat_row["expected_source_file_id"] == "11"
    assert chat_row["expected_chunk_ids"] == "chunk-abc"
    assert "from_chat_log" in chat_row["tags"]
    assert all(row["active"] == "false" for row in rows)
    assert "[email]" in rows[-1]["question"]


def test_seed_golden_dataset_keeps_chunk_ids_unsanitized(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    _create_seed_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE chat_logs SET citations_json = ? WHERE id = 1",
        (json.dumps([{"filename": "policy.pdf", "chunk_id": "12345678901234567890abcdef"}]),),
    )
    conn.commit()
    conn.close()

    rows = seed.build_candidates(
        db_path=str(db_path),
        kb_id=None,
        default_kb_id=1,
        limit=10,
        chat_limit=10,
        ticket_limit=0,
        email_limit=0,
        min_question_chars=8,
        include_chat_answers=True,
        active=False,
    )

    assert rows[0]["expected_chunk_ids"] == "12345678901234567890abcdef"
    assert "[phone]" not in rows[0]["expected_chunk_ids"]


def test_seed_golden_dataset_write_csv_matches_upload_contract(tmp_path: Path):
    output = tmp_path / "golden.csv"
    rows = [
        {
            "kb_id": "1",
            "question": "What is the SLA?",
            "expected_answer": "TODO: replace with approved source",
            "expected_answers": "",
            "expected_source_file_id": "",
            "expected_source_file_ids": "",
            "expected_chunk_ids": "",
            "expected_categories": "support",
            "expected_keywords": "sla",
            "tags": "candidate,needs_source_review",
            "active": "false",
        }
    ]

    seed.write_csv(rows, str(output))

    parsed = list(csv.DictReader(io.StringIO(output.read_text(encoding="utf-8"))))
    assert list(parsed[0].keys()) == seed.FIELDNAMES
    assert parsed[0]["active"] == "false"


def test_prepare_golden_review_pack_only_activates_reviewed_rows():
    rows = [
        {
            "kb_id": "1",
            "question": "What is the SLA?",
            "expected_answer": "The SLA is four hours.",
            "expected_answers": "",
            "expected_source_file_id": "11",
            "expected_source_file_ids": "11",
            "expected_chunk_ids": "chunk-abc",
            "expected_categories": "support",
            "expected_keywords": "sla",
            "tags": "candidate,reviewed",
            "active": "false",
        },
        {
            "kb_id": "1",
            "question": "What is pricing?",
            "expected_answer": "Pricing is in the quote.",
            "expected_answers": "",
            "expected_source_file_id": "12",
            "expected_source_file_ids": "12",
            "expected_chunk_ids": "chunk-def",
            "expected_categories": "sales",
            "expected_keywords": "pricing",
            "tags": "candidate",
            "active": "false",
        },
        {
            "kb_id": "1",
            "question": "Missing source?",
            "expected_answer": "TODO: replace with approved source.",
            "expected_answers": "",
            "expected_source_file_id": "",
            "expected_source_file_ids": "",
            "expected_chunk_ids": "",
            "expected_categories": "support",
            "expected_keywords": "source",
            "tags": "candidate",
            "active": "false",
        },
    ]

    prepared, report = review_pack.prepare_rows(
        rows,
        activate_reviewed_only=True,
        activate_evidence_backed=False,
    )

    assert [row["active"] for row in prepared] == ["true", "false", "false"]
    assert report["active_rows"] == 1
    assert "review_status:reviewed" in prepared[0]["tags"]
    assert "review_reason:needs_reviewed_tag" in prepared[1]["tags"]
    assert "review_reason:needs_answer" in prepared[2]["tags"]
    assert "review_reason:needs_source" in prepared[2]["tags"]


def test_prepare_golden_review_pack_smoke_mode_can_activate_evidence_backed_rows():
    rows = [
        {
            "kb_id": "1",
            "question": "What is the SLA?",
            "expected_answer": "The SLA is four hours.",
            "expected_answers": "",
            "expected_source_file_id": "11",
            "expected_source_file_ids": "11",
            "expected_chunk_ids": "chunk-abc",
            "expected_categories": "support",
            "expected_keywords": "sla",
            "tags": "candidate",
            "active": "false",
        }
    ]

    prepared, report = review_pack.prepare_rows(
        rows,
        activate_reviewed_only=True,
        activate_evidence_backed=True,
    )

    assert prepared[0]["active"] == "true"
    assert "smoke_auto_activated" in prepared[0]["tags"]
    assert report["active_rows"] == 1


def test_prepare_golden_review_pack_allows_reviewed_negative_without_source():
    rows = [
        {
            "kb_id": "1",
            "question": "What should the agent do when the policy is unavailable?",
            "expected_answer": "State that the policy was not found and escalate to support.",
            "expected_answers": "",
            "expected_source_file_id": "",
            "expected_source_file_ids": "",
            "expected_chunk_ids": "",
            "expected_categories": "negative",
            "expected_keywords": "policy,unavailable",
            "tags": "reviewed,negative",
            "active": "false",
        }
    ]

    prepared, report = review_pack.prepare_rows(
        rows,
        activate_reviewed_only=True,
        activate_evidence_backed=False,
    )

    assert prepared[0]["active"] == "true"
    assert "review_reason:needs_source" not in prepared[0]["tags"]
    assert report["active_rows"] == 1


def test_prepare_golden_review_pack_does_not_smoke_activate_unreviewed_negative_without_source():
    rows = [
        {
            "kb_id": "1",
            "question": "What should the agent do when the policy is unavailable?",
            "expected_answer": "State that the policy was not found and escalate to support.",
            "expected_answers": "",
            "expected_source_file_id": "",
            "expected_source_file_ids": "",
            "expected_chunk_ids": "",
            "expected_categories": "negative",
            "expected_keywords": "policy,unavailable",
            "tags": "negative",
            "active": "false",
        }
    ]

    prepared, report = review_pack.prepare_rows(
        rows,
        activate_reviewed_only=False,
        activate_evidence_backed=True,
    )

    assert prepared[0]["active"] == "false"
    assert "review_reason:needs_source" in prepared[0]["tags"]
    assert "review_reason:needs_reviewed_tag" in prepared[0]["tags"]
    assert report["active_rows"] == 0


def test_golden_coverage_requires_reviewed_representative_buckets():
    items = []
    categories = ["policy_vi", "policy_en", "exact_id", "citation", "negative", "followup"]
    for idx in range(80):
        items.append(
            {
                "active": True,
                "tags": ["reviewed"],
                "expected_categories": [categories[idx % len(categories)]],
                "expected_source_file_id": 10 if idx % 6 == 3 else None,
            }
        )

    coverage = run_baseline_eval._golden_coverage(items, target=80)

    assert coverage["representative_ready"] is True
    assert coverage["reviewed_active_items"] == 80
    assert coverage["missing_required_buckets"] == []


def test_golden_coverage_rejects_smoke_only_rows():
    items = [
        {
            "active": True,
            "tags": ["smoke_auto_activated"],
            "expected_categories": ["policy_vi"],
            "expected_source_file_id": 10,
        }
        for _ in range(80)
    ]

    coverage = run_baseline_eval._golden_coverage(items, target=80)

    assert coverage["representative_ready"] is False
    assert coverage["reviewed_active_items"] == 0
    assert coverage["smoke_active_items"] == 80
