from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import rag_profile_manager as rpm


def test_profile_env_text_contains_required_qdrant_hybrid_settings():
    text = rpm.profile_env_text("qdrant_hybrid_qwen3_reranker")

    assert "RAG_VECTOR_BACKEND=qdrant" in text
    assert "RAG_QDRANT_HYBRID_ENABLED=true" in text
    assert "RAG_RERANKER_PROVIDER=cross_encoder" in text
    assert "RAG_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B" in text
    assert "RAG_RETRIEVAL_SOURCE_DIVERSIFICATION_ENABLED=true" in text


def _report(run_id: int, *, gate_status: str = "passed", representative_ready: bool = True) -> dict:
    coverage = {
        "representative_ready": representative_ready,
        "reviewed_active_items": 80 if representative_ready else 20,
        "target_active_reviewed_items": 80,
        "missing_required_buckets": [] if representative_ready else ["negative", "followup"],
        "bucket_counts": {
            "vi": 12,
            "en": 12,
            "exact": 8,
            "citation": 20,
            "negative": 5 if representative_ready else 0,
            "followup": 5 if representative_ready else 0,
        },
    }
    return {
        "readiness": {"coverage": coverage},
        "summary": {"run_id": run_id, "gate_status": gate_status, "metrics": {"recall_at_k": 0.8}},
    }


def test_n_plus_one_profiles_do_not_require_reindex():
    cross_encoder = rpm.get_profile("local_cross_encoder")
    corrective = rpm.get_profile("local_corrective_rag")
    combined = rpm.get_profile("local_cross_encoder_corrective")

    assert cross_encoder["requires_reindex"] is False
    assert cross_encoder["env"]["RAG_RERANKER_PROVIDER"] == "cross_encoder"
    assert corrective["requires_reindex"] is False
    assert corrective["env"]["RAG_CORRECTIVE_RAG_ENABLED"] == "true"
    assert combined["requires_reindex"] is False
    assert combined["env"]["RAG_RERANKER_PROVIDER"] == "cross_encoder"
    assert combined["env"]["RAG_CORRECTIVE_RAG_ENABLED"] == "true"


def test_promote_rejects_failed_gate_without_override(tmp_path: Path):
    report = tmp_path / "candidate.json"
    report.write_text(
        json.dumps({"summary": {"run_id": 22, "gate_status": "failed", "metrics": {"recall_at_k": 0.2}}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Candidate gate failed"):
        rpm.promote_profile(
            state_path=str(tmp_path / "state.json"),
            profile_name="qdrant_hybrid_bge_m3",
            candidate_report=str(report),
            baseline_run_id=11,
            candidate_run_id=None,
            reason="benchmark failed",
            allow_failed_gate=False,
        )


def test_promote_and_rollback_update_state(tmp_path: Path):
    state = tmp_path / "state.json"
    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text(json.dumps(_report(1)), encoding="utf-8")
    candidate_report = tmp_path / "candidate.json"
    candidate_report.write_text(json.dumps(_report(2)), encoding="utf-8")

    first = rpm.promote_profile(
        state_path=str(state),
        profile_name="current_local",
        candidate_report=str(baseline_report),
        baseline_run_id=None,
        candidate_run_id=None,
        reason="initial baseline",
        allow_failed_gate=False,
    )
    second = rpm.promote_profile(
        state_path=str(state),
        profile_name="qdrant_hybrid_bge_m3",
        candidate_report=str(candidate_report),
        baseline_run_id=1,
        candidate_run_id=None,
        reason="golden eval passed",
        allow_failed_gate=False,
    )

    assert first["profile"] == "current_local"
    assert second["previous_profile"] == "current_local"
    loaded = rpm.load_state(str(state))
    assert loaded["active_profile"] == "qdrant_hybrid_bge_m3"
    assert loaded["previous_profile"] == "current_local"
    assert len(loaded["history"]) == 2

    rollback = rpm.rollback_profile(state_path=str(state), reason="latency budget exceeded")

    assert rollback["to_profile"] == "current_local"
    loaded = rpm.load_state(str(state))
    assert loaded["active_profile"] == "current_local"
    assert loaded["previous_profile"] == "qdrant_hybrid_bge_m3"
    assert len(loaded["history"]) == 3


def test_promote_rejects_index_profile_without_representative_golden(tmp_path: Path):
    report = tmp_path / "candidate.json"
    report.write_text(json.dumps(_report(2, representative_ready=False)), encoding="utf-8")

    with pytest.raises(SystemExit, match="Representative golden coverage is required"):
        rpm.promote_profile(
            state_path=str(tmp_path / "state.json"),
            profile_name="qdrant_hybrid_bge_m3",
            candidate_report=str(report),
            baseline_run_id=1,
            candidate_run_id=None,
            reason="coverage too small",
            allow_failed_gate=False,
        )


def test_promote_allows_local_reranker_without_representative_golden(tmp_path: Path):
    report = tmp_path / "candidate.json"
    report.write_text(json.dumps(_report(2, representative_ready=False)), encoding="utf-8")

    event = rpm.promote_profile(
        state_path=str(tmp_path / "state.json"),
        profile_name="local_cross_encoder",
        candidate_report=str(report),
        baseline_run_id=1,
        candidate_run_id=None,
        reason="local reranker smoke benchmark",
        allow_failed_gate=False,
    )

    assert event["profile"] == "local_cross_encoder"
    assert event["changes_index_contract"] is False


def test_benchmark_plan_adds_baseline_comparison_after_current_profile():
    plan = rpm.benchmark_plan("http://127.0.0.1:8080", kb_id=7, limit=25, baseline_run_id=99)

    assert plan["steps"][0]["profile"] == "current_local"
    assert "--baseline-run-id" not in plan["steps"][0]["command"]
    assert plan["steps"][1]["profile"] == "local_cross_encoder"
    assert "--baseline-run-id 99" in plan["steps"][1]["command"]
    assert plan["steps"][1]["requires_reindex"] is False
    assert any(step["profile"] == "qdrant_dense_bge_m3" and step["requires_reindex"] for step in plan["steps"])
