from __future__ import annotations

import csv
import io

from scripts import generate_golden_dataset_template as template


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
