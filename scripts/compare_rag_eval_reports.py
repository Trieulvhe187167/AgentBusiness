"""Compare RAG golden-eval reports and produce a promotion summary.

The input reports are produced by `scripts/run_baseline_eval.py --report-output`.
This script is intentionally file-based so operators can compare benchmark runs
without connecting to the app or mutating promotion state.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CORE_METRICS = [
    "avg_score",
    "recall_at_k",
    "mrr",
    "citation_accuracy",
    "answer_similarity",
    "latency_p50_ms",
    "latency_p95_ms",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RAG eval report JSON files")
    parser.add_argument("--baseline", required=True, help="Baseline report JSON path")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate report path, or name=path. Repeat for multiple candidates.",
    )
    parser.add_argument("--output", default="", help="Optional output path")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument(
        "--latency-budget-ratio",
        type=float,
        default=1.25,
        help="Warn when candidate p95 latency exceeds baseline by this ratio",
    )
    parser.add_argument(
        "--min-mrr-improvement",
        type=float,
        default=0.05,
        help="Relative MRR improvement expected for neural reranker promotion notes",
    )
    return parser.parse_args()


def _load_report(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid report payload: {path}")
    return payload


def _candidate_arg(value: str) -> tuple[str, str]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), path.strip()
    path = Path(value)
    return path.stem.removeprefix("eval_"), value


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = dict(report.get("summary") or {})
    metrics = dict(summary.get("metrics") or {})
    if "avg_score" not in metrics and summary.get("avg_score") is not None:
        metrics["avg_score"] = summary.get("avg_score")
    summary["metrics"] = metrics
    return summary


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_rows(baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names = list(dict.fromkeys([*CORE_METRICS, *baseline_metrics.keys(), *candidate_metrics.keys()]))
    for metric in metric_names:
        base_value = _float(baseline_metrics.get(metric))
        candidate_value = _float(candidate_metrics.get(metric))
        if base_value is None and candidate_value is None:
            continue
        delta = None if base_value is None or candidate_value is None else candidate_value - base_value
        relative_delta = None
        if delta is not None and base_value not in {None, 0.0}:
            relative_delta = delta / abs(base_value)
        rows.append(
            {
                "metric": metric,
                "baseline": base_value,
                "candidate": candidate_value,
                "delta": delta,
                "relative_delta": relative_delta,
            }
        )
    return rows


def _decision(
    *,
    gate_status: str,
    rows: list[dict[str, Any]],
    latency_budget_ratio: float,
    min_mrr_improvement: float,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    if gate_status == "failed":
        notes.append("eval gate failed")

    metric_by_name = {row["metric"]: row for row in rows}
    latency = metric_by_name.get("latency_p95_ms")
    if latency and latency.get("baseline") and latency.get("candidate"):
        if float(latency["candidate"]) > float(latency["baseline"]) * latency_budget_ratio:
            notes.append(f"p95 latency exceeded {latency_budget_ratio:.2f}x budget")

    mrr = metric_by_name.get("mrr")
    if mrr and mrr.get("baseline") and mrr.get("candidate"):
        rel = mrr.get("relative_delta")
        if rel is not None and rel < min_mrr_improvement:
            notes.append(f"MRR improvement below {min_mrr_improvement:.0%}")

    citation = metric_by_name.get("citation_accuracy")
    if citation and citation.get("delta") is not None and float(citation["delta"]) < -0.03:
        notes.append("citation accuracy dropped more than 0.03")

    recall = metric_by_name.get("recall_at_k")
    if recall and recall.get("delta") is not None and float(recall["delta"]) < -0.03:
        notes.append("recall_at_k dropped more than 0.03")

    if gate_status in {"passed", "not_compared", "baseline_missing"} and not notes:
        return "promote_candidate", ["within configured benchmark budget"]
    if gate_status == "passed" and notes:
        return "needs_owner_review", notes
    return "do_not_promote", notes or [f"gate status is {gate_status or 'unknown'}"]


def build_comparison(
    *,
    baseline_path: str,
    candidates: list[tuple[str, str]],
    latency_budget_ratio: float,
    min_mrr_improvement: float,
) -> dict[str, Any]:
    baseline_report = _load_report(baseline_path)
    baseline_summary = _summary(baseline_report)
    baseline_metrics = baseline_summary["metrics"]
    rows = []
    for name, path in candidates:
        candidate_report = _load_report(path)
        candidate_summary = _summary(candidate_report)
        metric_rows = _metric_rows(baseline_metrics, candidate_summary["metrics"])
        decision, notes = _decision(
            gate_status=str(candidate_summary.get("gate_status") or "unknown"),
            rows=metric_rows,
            latency_budget_ratio=latency_budget_ratio,
            min_mrr_improvement=min_mrr_improvement,
        )
        rows.append(
            {
                "name": name,
                "path": path,
                "run_id": candidate_summary.get("run_id"),
                "baseline_run_id": candidate_summary.get("baseline_run_id") or baseline_summary.get("run_id"),
                "gate_status": candidate_summary.get("gate_status"),
                "decision": decision,
                "notes": notes,
                "metrics": metric_rows,
            }
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline": {
            "path": baseline_path,
            "run_id": baseline_summary.get("run_id"),
            "gate_status": baseline_summary.get("gate_status"),
            "metrics": baseline_metrics,
        },
        "candidates": rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# RAG Benchmark Promotion Summary",
        "",
        f"Generated: {comparison['generated_at']}",
        "",
        f"Baseline report: `{comparison['baseline']['path']}`",
        f"Baseline run ID: `{comparison['baseline'].get('run_id') or '-'}`",
        "",
        "| Candidate | Run | Gate | Decision | Notes |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for candidate in comparison["candidates"]:
        lines.append(
            "| {name} | {run} | {gate} | {decision} | {notes} |".format(
                name=candidate["name"],
                run=candidate.get("run_id") or "-",
                gate=candidate.get("gate_status") or "-",
                decision=candidate["decision"],
                notes="; ".join(candidate["notes"]),
            )
        )
    lines.extend(["", "## Metric Deltas", ""])
    for candidate in comparison["candidates"]:
        lines.extend(
            [
                f"### {candidate['name']}",
                "",
                "| Metric | Baseline | Candidate | Delta | Relative Delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in candidate["metrics"]:
            rel = row.get("relative_delta")
            lines.append(
                f"| {row['metric']} | {_fmt(row.get('baseline'))} | {_fmt(row.get('candidate'))} | "
                f"{_fmt(row.get('delta'))} | {_fmt(None if rel is None else rel * 100)}% |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _parse_args()
    if not args.candidate:
        raise SystemExit("At least one --candidate is required")
    comparison = build_comparison(
        baseline_path=args.baseline,
        candidates=[_candidate_arg(value) for value in args.candidate],
        latency_budget_ratio=max(1.0, float(args.latency_budget_ratio)),
        min_mrr_improvement=max(0.0, float(args.min_mrr_improvement)),
    )
    text = (
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(comparison)
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(json.dumps({"output": str(path.resolve()), "format": args.format}, ensure_ascii=False))
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
