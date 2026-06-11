from __future__ import annotations

import json

from fastapi.testclient import TestClient

import app.main as main
import app.evaluations as evaluations
from app.database import execute_sync, fetch_one_sync, utcnow_iso
from tests.conftest import admin_headers, auth_headers, configure_test_env, insert_file


def _insert_eval_chat(
    request_id: str,
    *,
    mode: str = "answer",
    top_score: float | None = 0.82,
    answer_text: str = "This is a grounded answer with enough detail.",
    citations_json: str = '[{"source":"kb"}]',
    rating: str | None = None,
) -> int:
    now = utcnow_iso()
    chat_id = int(
        execute_sync(
            """
            INSERT INTO chat_logs (
                session_id, request_id, user_id, roles_json, channel,
                tenant_id, org_id, kb_id, kb_key, user_message, merged_query,
                mode, top_score, answer_text, citations_json, latency_ms,
                llm_provider, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"session-{request_id}",
                request_id,
                "user-1",
                json.dumps(["employee"]),
                "web",
                "tenant-a",
                "org-a",
                1,
                "default",
                f"Question {request_id}",
                f"Question {request_id}",
                mode,
                top_score,
                answer_text,
                citations_json,
                100,
                "none",
                now,
            ),
        )
        or 0
    )
    if rating:
        execute_sync(
            """
            INSERT INTO chat_feedback (
                chat_log_id, request_id, rating, reason_code, comment,
                created_by_user_id, roles_json, channel, tenant_id, org_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                request_id,
                rating,
                "not_helpful" if rating == "down" else "good_answer",
                None,
                "user-1",
                json.dumps(["employee"]),
                "web",
                "tenant-a",
                "org-a",
                now,
                now,
            ),
        )
    return chat_id


def test_admin_can_create_list_and_get_agent_eval_run(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _insert_eval_chat("eval-pass", rating="up")
    _insert_eval_chat(
        "eval-fail",
        mode="fallback",
        top_score=0.1,
        answer_text="",
        citations_json="[]",
        rating="down",
    )

    with TestClient(main.app) as client:
        created = client.post(
            "/api/admin/evaluations/runs",
            headers=admin_headers(),
            json={"name": "Smoke eval", "days": 7, "limit": 10, "kb_id": 1},
        )
        listed = client.get("/api/admin/evaluations/runs?limit=10", headers=admin_headers())
        detail = client.get(f"/api/admin/evaluations/runs/{created.json()['id']}", headers=admin_headers())

    assert created.status_code == 200, created.text
    payload = created.json()
    assert payload["name"] == "Smoke eval"
    assert payload["sample_size"] == 2
    assert payload["pass_count"] == 1
    assert payload["fail_count"] == 1
    assert payload["results"][0]["verdict"] == "fail"
    assert listed.status_code == 200, listed.text
    assert listed.json()["total"] == 1
    assert detail.status_code == 200, detail.text
    assert detail.json()["config"]["scorer"] == "rule_based_v1"


def test_agent_eval_requires_analytics_role(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/admin/evaluations/runs",
            headers=auth_headers(user_id="user-1", roles=["employee"]),
            json={"days": 7},
        )

    assert response.status_code == 403, response.text


def test_admin_can_manage_golden_dataset_and_run_regression_eval(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    source_file_id = insert_file("policy.csv")
    chat_id = _insert_eval_chat("golden-eval-chat")

    def fake_collect(question, *, kb_id, auth):
        return {
            "request_id": "golden-eval-chat",
            "chat_log_id": chat_id,
            "answer_text": "Shipping is free for standard orders.",
            "citations": [{"filename": "policy.csv", "content_preview": "shipping free"}],
            "retrieved": [{"source_id": str(source_file_id), "similarity": 0.91}],
            "mode": "answer",
            "top_score": 0.91,
            "kb_key": "default",
            "latency_ms": 12,
        }

    monkeypatch.setattr(evaluations, "_collect_rag_answer", fake_collect)

    with TestClient(main.app) as client:
        created_item = client.post(
            "/api/admin/evaluations/golden-dataset",
            headers=admin_headers(),
            json={
                "kb_id": 1,
                "question": "What is the shipping policy?",
                "expected_answer": "Shipping is free.",
                "expected_source_file_id": source_file_id,
                "expected_keywords": ["shipping", "free"],
                "tags": ["smoke"],
            },
        )
        listed = client.get("/api/admin/evaluations/golden-dataset?kb_id=1", headers=admin_headers())
        run = client.post(
            "/api/admin/evaluations/runs",
            headers=admin_headers(),
            json={"name": "Golden smoke", "source": "golden_dataset", "kb_id": 1, "limit": 10},
        )

    assert created_item.status_code == 200, created_item.text
    assert listed.status_code == 200, listed.text
    assert listed.json()["total"] == 1
    assert run.status_code == 200, run.text
    payload = run.json()
    assert payload["source"] == "golden_dataset"
    assert payload["sample_size"] == 1
    assert payload["pass_count"] == 1
    snapshot = payload["config"]["rag_config_snapshot"]
    assert snapshot["snapshot_version"] == "phase0_baseline_v1"
    assert snapshot["embedding"]["effective_model_id"] == "models/missing-model"
    assert snapshot["vector_store"]["configured_backend"] == "numpy"
    assert snapshot["retrieval"]["top_k"] == 10
    assert snapshot["retrieval"]["reranker_provider"] == "bm25_lite"
    assert payload["metrics"]["latency_p50_ms"] == 12.0
    assert payload["metrics"]["latency_p95_ms"] == 12.0
    assert payload["metrics"]["latency_avg_ms"] == 12.0
    assert payload["metrics"]["retrieved_count_avg"] == 1.0
    result = payload["results"][0]
    assert result["golden_item_id"] == created_item.json()["id"]
    assert result["answer_similarity"] > 0
    assert result["recall_at_k"] == 1.0
    assert result["citation_accuracy"] == 1.0


def test_golden_eval_quality_drop_creates_notification(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    source_file_id = insert_file("policy.csv")
    chat_id = _insert_eval_chat("golden-alert-chat")
    execute_sync(
        """
        INSERT INTO eval_golden_dataset (
            kb_id, question, expected_answer, expected_source_file_id,
            expected_keywords_json, tags_json, active,
            created_by_user_id, created_at, updated_at
        ) VALUES (1, 'Shipping?', 'Shipping is free.', ?, '["shipping","free"]', '[]', 1, 'admin-1', ?, ?)
        """,
        (source_file_id, utcnow_iso(), utcnow_iso()),
    )
    answers = iter(
        [
            "Shipping is free for standard orders.",
            "I could not find relevant information.",
        ]
    )

    def fake_collect(question, *, kb_id, auth):
        answer = next(answers)
        return {
            "request_id": "golden-alert-chat",
            "chat_log_id": chat_id,
            "answer_text": answer,
            "citations": [{"filename": "policy.csv", "content_preview": answer}],
            "retrieved": [{"source_id": str(source_file_id), "similarity": 0.9}] if "free" in answer else [],
            "mode": "answer" if "free" in answer else "fallback",
            "top_score": 0.9 if "free" in answer else 0.0,
            "kb_key": "default",
            "latency_ms": 10,
        }

    monkeypatch.setattr(evaluations, "_collect_rag_answer", fake_collect)

    with TestClient(main.app) as client:
        first = client.post(
            "/api/admin/evaluations/runs",
            headers=admin_headers(),
            json={"source": "golden_dataset", "kb_id": 1, "limit": 10, "alert_drop_threshold": 10},
        )
        second = client.post(
            "/api/admin/evaluations/runs",
            headers=admin_headers(),
            json={"source": "golden_dataset", "kb_id": 1, "limit": 10, "alert_drop_threshold": 10},
        )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    notification = fetch_one_sync(
        "SELECT event_type, entity_id FROM notifications WHERE event_type = 'evaluation.quality_drop'"
    )
    assert notification is not None
    assert notification["entity_id"] == str(second.json()["id"])


def test_golden_eval_tracks_extended_retrieval_metrics_and_answer_variants(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    source_file_id = insert_file("delivery-policy.csv")
    chat_id = _insert_eval_chat("golden-extended-chat")

    def fake_collect(question, *, kb_id, auth):
        return {
            "request_id": "golden-extended-chat",
            "chat_log_id": chat_id,
            "answer_text": "Delivery has no charge.",
            "citations": [
                {
                    "filename": "delivery-policy.csv",
                    "chunk_id": "chunk-delivery-policy",
                    "content_preview": "Delivery has no charge.",
                }
            ],
            "retrieved": [
                {"source_id": "999", "chunk_id": "chunk-other", "category": "other", "similarity": 0.95},
                {
                    "source_id": str(source_file_id),
                    "chunk_id": "chunk-delivery-policy",
                    "category": "shipping",
                    "similarity": 0.90,
                },
            ],
            "mode": "answer",
            "top_score": 0.95,
            "kb_key": "default",
            "latency_ms": 11,
        }

    monkeypatch.setattr(evaluations, "_collect_rag_answer", fake_collect)

    with TestClient(main.app) as client:
        item = client.post(
            "/api/admin/evaluations/golden-dataset",
            headers=admin_headers(),
            json={
                "kb_id": 1,
                "question": "How much is delivery?",
                "expected_answer": "Shipping is free.",
                "expected_answers": ["Delivery has no charge."],
                "expected_source_file_ids": [source_file_id],
                "expected_chunk_ids": ["chunk-delivery-policy"],
                "expected_categories": ["shipping"],
                "expected_keywords": ["delivery", "charge"],
            },
        )
        run = client.post(
            "/api/admin/evaluations/runs",
            headers=admin_headers(),
            json={"source": "golden_dataset", "kb_id": 1, "limit": 10},
        )

    assert item.status_code == 200, item.text
    assert item.json()["expected_answers"] == ["Delivery has no charge."]
    assert item.json()["expected_source_file_ids"] == [source_file_id]
    assert run.status_code == 200, run.text
    payload = run.json()
    assert payload["gate_status"] == "baseline_missing"
    result = payload["results"][0]
    assert result["answer_similarity"] == 1.0
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] == 0.5
    assert result["matched_source_rank"] == 2
    assert result["citation_accuracy"] == 1.0
    assert result["chunk_match"] == 1.0
    assert result["category_match"] == 1.0
    assert result["retrieved"][1]["rank"] == 2
    assert payload["metrics"]["mrr"] == 0.5


def test_golden_eval_can_use_optional_llm_judge(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    source_file_id = insert_file("judge-policy.csv")
    chat_id = _insert_eval_chat("golden-judge-chat")

    def fake_collect(question, *, kb_id, auth):
        return {
            "request_id": "golden-judge-chat",
            "chat_log_id": chat_id,
            "answer_text": "Shipping is free.",
            "citations": [{"filename": "judge-policy.csv", "content_preview": "shipping free"}],
            "retrieved": [{"source_id": str(source_file_id), "similarity": 0.9}],
            "mode": "answer",
            "top_score": 0.9,
            "kb_key": "default",
            "latency_ms": 10,
        }

    def fake_judge(**kwargs):
        return {
            "judge_provider": "openai",
            "judge_model": "gpt-test",
            "judge_score": 0.5,
            "judge_verdict": "warn",
            "judge_metrics": {
                "correctness": 0.6,
                "groundedness": 0.7,
                "completeness": 0.4,
                "citation_support": 0.8,
                "hallucination_risk": 0.1,
            },
            "judge_reason": "Mostly correct but incomplete.",
            "judge_latency_ms": 123,
            "judge_error": None,
        }

    monkeypatch.setattr(evaluations, "_collect_rag_answer", fake_collect)
    monkeypatch.setattr(evaluations, "evaluate_golden_answer", fake_judge)

    with TestClient(main.app) as client:
        client.post(
            "/api/admin/evaluations/golden-dataset",
            headers=admin_headers(),
            json={
                "kb_id": 1,
                "question": "What is the shipping policy?",
                "expected_answer": "Shipping is free.",
                "expected_source_file_id": source_file_id,
                "expected_keywords": ["shipping", "free"],
            },
        )
        run = client.post(
            "/api/admin/evaluations/runs",
            headers=admin_headers(),
            json={
                "source": "golden_dataset",
                "kb_id": 1,
                "limit": 10,
                "llm_judge": True,
                "llm_judge_weight": 0.4,
            },
        )

    assert run.status_code == 200, run.text
    payload = run.json()
    assert payload["config"]["llm_judge_enabled"] is True
    assert payload["config"]["llm_judge_weight"] == 0.4
    assert payload["metrics"]["judge_score"] == 0.5
    assert payload["metrics"]["judge_groundedness"] == 0.7
    result = payload["results"][0]
    assert result["score"] == 80.0
    assert result["judge_provider"] == "openai"
    assert result["judge_model"] == "gpt-test"
    assert result["judge_score"] == 0.5
    assert result["judge_verdict"] == "warn"
    assert result["judge_metrics"]["hallucination_risk"] == 0.1
    assert result["judge_reason"] == "Mostly correct but incomplete."


def test_golden_eval_gate_fails_when_metrics_regress(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    source_file_id = insert_file("gate-policy.csv")
    chat_id = _insert_eval_chat("golden-gate-chat")
    execute_sync(
        """
        INSERT INTO eval_golden_dataset (
            kb_id, question, expected_answer, expected_source_file_id,
            expected_answers_json, expected_source_file_ids_json,
            expected_chunk_ids_json, expected_categories_json,
            expected_keywords_json, tags_json, active,
            created_by_user_id, created_at, updated_at
        ) VALUES (1, 'Shipping?', 'Shipping is free.', ?, '[]', ?, '[]', '[]', '["shipping","free"]', '[]', 1, 'admin-1', ?, ?)
        """,
        (source_file_id, json.dumps([source_file_id]), utcnow_iso(), utcnow_iso()),
    )
    answers = iter(
        [
            {
                "answer_text": "Shipping is free.",
                "citations": [{"filename": "gate-policy.csv", "content_preview": "shipping free"}],
                "retrieved": [{"source_id": str(source_file_id), "similarity": 0.9}],
                "mode": "answer",
                "top_score": 0.9,
            },
            {
                "answer_text": "I could not find relevant information.",
                "citations": [],
                "retrieved": [],
                "mode": "fallback",
                "top_score": 0.0,
            },
        ]
    )

    def fake_collect(question, *, kb_id, auth):
        result = next(answers)
        return {
            "request_id": "golden-gate-chat",
            "chat_log_id": chat_id,
            "kb_key": "default",
            "latency_ms": 10,
            **result,
        }

    monkeypatch.setattr(evaluations, "_collect_rag_answer", fake_collect)

    with TestClient(main.app) as client:
        first = client.post(
            "/api/admin/evaluations/runs",
            headers=admin_headers(),
            json={"source": "golden_dataset", "kb_id": 1, "limit": 10, "max_metric_drop": 0.1},
        )
        second = client.post(
            "/api/admin/evaluations/runs",
            headers=admin_headers(),
            json={"source": "golden_dataset", "kb_id": 1, "limit": 10, "max_metric_drop": 0.1},
        )

    assert first.status_code == 200, first.text
    assert first.json()["gate_status"] == "baseline_missing"
    assert second.status_code == 200, second.text
    payload = second.json()
    assert payload["baseline_run_id"] == first.json()["id"]
    assert payload["gate_status"] == "failed"
    regressions = {item["metric"] for item in payload["comparison"]["regressions"]}
    assert "answer_similarity" in regressions
    assert "recall_at_k" in regressions
    notification = fetch_one_sync(
        "SELECT event_type, entity_id FROM notifications WHERE event_type = 'evaluation.gate_failed'"
    )
    assert notification is not None
    assert notification["entity_id"] == str(payload["id"])
