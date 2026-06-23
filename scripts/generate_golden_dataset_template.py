"""Generate a 96-row golden dataset starter CSV for RAG baseline work.

The rows are intentionally domain-neutral prompts. Replace expected answers,
source IDs, categories, and keywords with facts from the target KB before using
the file as a quality gate.
"""

from __future__ import annotations

import argparse
import csv
import sys


CATEGORIES: list[tuple[str, list[str]]] = [
    (
        "policy_vi",
        [
            "Chinh sach doi tra ap dung trong bao lau?",
            "Khi nao khach hang duoc hoan tien?",
            "Dieu kien bao hanh san pham la gi?",
            "Can giay to nao de yeu cau ho tro?",
            "Chinh sach huy don hang nhu the nao?",
            "Phi van chuyen duoc tinh ra sao?",
            "Thoi gian xu ly yeu cau ho tro la bao lau?",
            "Truong hop nao can chuyen len quan ly?",
        ],
    ),
    (
        "policy_en",
        [
            "What is the return window?",
            "When is a customer eligible for a refund?",
            "What documents are required for support?",
            "How does cancellation work?",
            "How is shipping cost calculated?",
            "What is the warranty policy?",
            "What is the expected support response time?",
            "When should a case be escalated?",
        ],
    ),
    (
        "exact_id",
        [
            "What does policy code POLICY-001 cover?",
            "What is the handling rule for ORDER-EXAMPLE-001?",
            "What does SKU-EXAMPLE-001 mean?",
            "Which document mentions FORM-001?",
            "What is the SLA for priority P1?",
            "What is the SLA for priority P2?",
            "Which team owns billing_invoice cases?",
            "Which team owns account_access cases?",
        ],
    ),
    (
        "citation",
        [
            "Which source states the return policy?",
            "Which source states refund exceptions?",
            "Which source states warranty exclusions?",
            "Which source states support contact rules?",
            "Which source states escalation criteria?",
            "Which source states service availability?",
            "Which source states document retention?",
            "Which source states approval requirements?",
        ],
    ),
    (
        "negative",
        [
            "What is the policy for a product not in the KB?",
            "What is the private phone number of the CEO?",
            "What discount will be offered next quarter?",
            "Can you approve a refund without policy evidence?",
            "What is the answer if no source supports it?",
            "What are the credentials for the admin system?",
            "Can you delete a source file for me?",
            "What unsupported claim appears in the documents?",
        ],
    ),
    (
        "followup",
        [
            "What about the same policy for enterprise customers?",
            "Does that apply if the order is already shipped?",
            "Can you summarize the exception?",
            "Which source supports that answer?",
            "What should support do next?",
            "Is there a deadline for that request?",
            "Does this require human approval?",
            "What information should the customer provide?",
        ],
    ),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a golden dataset starter CSV")
    parser.add_argument("--kb-id", type=int, default=1, help="Default KB id for generated rows")
    parser.add_argument("--output", default="", help="Output CSV path. Defaults to stdout")
    return parser.parse_args()


def _rows(kb_id: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for category, questions in CATEGORIES:
        for idx, question in enumerate(questions, start=1):
            rows.append(
                {
                    "kb_id": str(kb_id),
                    "question": question,
                    "expected_answer": "TODO: replace with grounded reference answer from the KB",
                    "expected_answers": "",
                    "expected_source_file_id": "",
                    "expected_source_file_ids": "",
                    "expected_chunk_ids": "",
                    "expected_categories": category,
                    "expected_keywords": "TODO",
                    "tags": f"starter,{category}",
                    "active": "false",
                }
            )
    # Duplicate the categories with no-diacritic/short-form variants to reach a
    # production-sized starter set without inventing domain facts.
    extras: list[dict[str, str]] = []
    for row in rows:
        if len(extras) >= len(rows):
            break
        copy = dict(row)
        copy["question"] = f"Short variant: {row['question']}"
        copy["tags"] = f"{row['tags']},variant"
        extras.append(copy)
    return rows + extras


def main() -> int:
    args = _parse_args()
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
    rows = _rows(args.kb_id)
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
