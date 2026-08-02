from __future__ import annotations

import json
from pathlib import Path

from scripts import compare_rag_eval_reports as compare


def _write_report(path: Path, *, run_id: int, gate_status: str, metrics: dict[str, float]) -> None:
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "run_id": run_id,
                    "gate_status": gate_status,
                    "avg_score": metrics.get("avg_score"),
                    "metrics": metrics,
                }
            }
        ),
        encoding="utf-8",
    )


def test_compare_reports_marks_promotable_candidate(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(
        baseline,
        run_id=1,
        gate_status="baseline_missing",
        metrics={"avg_score": 80, "mrr": 0.5, "recall_at_k": 0.7, "citation_accuracy": 0.8, "latency_p95_ms": 1000},
    )
    _write_report(
        candidate,
        run_id=2,
        gate_status="passed",
        metrics={"avg_score": 82, "mrr": 0.53, "recall_at_k": 0.72, "citation_accuracy": 0.81, "latency_p95_ms": 1100},
    )

    result = compare.build_comparison(
        baseline_path=str(baseline),
        candidates=[("cross_encoder", str(candidate))],
        latency_budget_ratio=1.25,
        min_mrr_improvement=0.05,
    )

    row = result["candidates"][0]
    assert row["decision"] == "promote_candidate"
    assert row["notes"] == ["within configured benchmark budget"]
    assert any(metric["metric"] == "mrr" and round(metric["relative_delta"], 2) == 0.06 for metric in row["metrics"])


def test_compare_reports_blocks_failed_or_slow_candidate(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(
        baseline,
        run_id=1,
        gate_status="baseline_missing",
        metrics={"mrr": 0.5, "recall_at_k": 0.7, "citation_accuracy": 0.8, "latency_p95_ms": 1000},
    )
    _write_report(
        candidate,
        run_id=3,
        gate_status="failed",
        metrics={"mrr": 0.48, "recall_at_k": 0.66, "citation_accuracy": 0.76, "latency_p95_ms": 1600},
    )

    result = compare.build_comparison(
        baseline_path=str(baseline),
        candidates=[("corrective", str(candidate))],
        latency_budget_ratio=1.25,
        min_mrr_improvement=0.05,
    )

    row = result["candidates"][0]
    assert row["decision"] == "do_not_promote"
    assert "eval gate failed" in row["notes"]
    assert "p95 latency exceeded 1.25x budget" in row["notes"]
    assert "recall_at_k dropped more than 0.03" in row["notes"]


def test_render_markdown_includes_decision_table(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_report(baseline, run_id=1, gate_status="baseline_missing", metrics={"mrr": 0.5})
    _write_report(candidate, run_id=2, gate_status="passed", metrics={"mrr": 0.55})

    result = compare.build_comparison(
        baseline_path=str(baseline),
        candidates=[("candidate", str(candidate))],
        latency_budget_ratio=1.25,
        min_mrr_improvement=0.05,
    )
    markdown = compare.render_markdown(result)

    assert "# RAG Benchmark Promotion Summary" in markdown
    assert "| candidate | 2 | passed | promote_candidate |" in markdown
