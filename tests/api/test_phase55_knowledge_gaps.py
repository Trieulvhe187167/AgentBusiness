from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main
from app.background_jobs import enqueue_background_job, run_due_background_jobs_once
from app.config import settings
from app.database import execute_sync, fetch_all_sync
from app.knowledge_gaps import create_knowledge_gap_report, record_knowledge_gap
from app.models import AuthContext, RequestContext
from tests.conftest import admin_headers, auth_headers, configure_test_env, run


def test_knowledge_gap_clusters_group_repeated_fallbacks_and_alert(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "knowledge_gap_alert_repeat_count", 2)

    context = {
        "kb_id": 1,
        "kb_key": "default",
        "auth": {"tenant_id": "tenant-a", "org_id": "org-a"},
    }
    record_knowledge_gap(
        chat_log_id=None,
        query="Đổi trả qua app được không?",
        mode="fallback",
        top_score=0.12,
        session_id="session-gap-1",
        context=context,
    )
    record_knowledge_gap(
        chat_log_id=None,
        query="doi tra qua app duoc khong",
        mode="fallback",
        top_score=0.08,
        session_id="session-gap-2",
        context=context,
    )

    with TestClient(main.app) as client:
        response = client.get("/api/admin/knowledge-gaps?days=7&kb_id=1", headers=admin_headers())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 1
    item = payload["items"][0]
    assert item["count"] == 2
    assert item["kb_id"] == 1
    assert item["suggested_action"] == "create_faq_entry"
    assert item["min_score"] == 0.08
    assert "doi tra qua app" in " ".join(item["sample_queries"]).lower()

    notifications = fetch_all_sync(
        "SELECT event_type, entity_type, payload_json FROM notifications WHERE event_type = 'knowledge_gap.repeated'"
    )
    assert len(notifications) == 1
    assert notifications[0]["entity_type"] == "knowledge_gap_cluster"


def test_knowledge_gap_clusters_semantic_paraphrases(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "knowledge_gap_semantic_clustering_enabled", True)
    monkeypatch.setattr(settings, "knowledge_gap_semantic_similarity_threshold", 0.75)

    def fake_embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "return" in lowered or "send items back" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors

    monkeypatch.setattr("app.knowledge_gaps.embed_texts", fake_embed_texts)
    context = {"kb_id": 1, "kb_key": "default", "auth": {}}
    record_knowledge_gap(
        chat_log_id=None,
        query="return policy in mobile app",
        mode="fallback",
        top_score=0.10,
        session_id="session-semantic-1",
        context=context,
    )
    record_knowledge_gap(
        chat_log_id=None,
        query="how do I send items back using the application",
        mode="fallback",
        top_score=0.11,
        session_id="session-semantic-2",
        context=context,
    )
    record_knowledge_gap(
        chat_log_id=None,
        query="international shipping options",
        mode="fallback",
        top_score=0.12,
        session_id="session-semantic-3",
        context=context,
    )

    with TestClient(main.app) as client:
        response = client.get("/api/admin/knowledge-gaps?days=7&kb_id=1", headers=admin_headers())

    assert response.status_code == 200, response.text
    items = response.json()["items"]
    counts = sorted(item["count"] for item in items)
    assert counts == [1, 2]
    return_cluster = next(item for item in items if item["count"] == 2)
    assert "return" in " ".join(return_cluster["sample_queries"]).lower()
    assert "send items back" in " ".join(return_cluster["sample_queries"]).lower()


def test_knowledge_gap_endpoint_updates_cluster_status(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    context = {"kb_id": 1, "kb_key": "default", "auth": {}}
    gap_id = record_knowledge_gap(
        chat_log_id=None,
        query="Giao hàng quốc tế?",
        mode="fallback",
        top_score=0.05,
        session_id="session-gap-status",
        context=context,
    )
    assert gap_id

    with TestClient(main.app) as client:
        listed = client.get("/api/admin/knowledge-gaps?days=7&kb_id=1", headers=admin_headers())
        cluster_key = listed.json()["items"][0]["cluster_key"]
        updated = client.patch(
            f"/api/admin/knowledge-gaps/{cluster_key}?kb_id=1",
            headers=admin_headers(),
            json={
                "status": "triaged",
                "owner_user_id": "knowledge-owner",
                "priority": "P1",
                "due_date": "2026-06-22",
                "status_reason": "Needs source owner review.",
            },
        )
        open_after = client.get("/api/admin/knowledge-gaps?days=7&kb_id=1", headers=admin_headers())
        triaged_after = client.get(
            "/api/admin/knowledge-gaps?days=7&kb_id=1&status=triaged",
            headers=admin_headers(),
        )

    assert updated.status_code == 200, updated.text
    assert open_after.json()["total"] == 0
    assert triaged_after.json()["total"] == 1
    item = triaged_after.json()["items"][0]
    assert item["status"] == "triaged"
    assert item["owner_user_id"] == "knowledge-owner"
    assert item["priority"] == "P1"
    assert item["due_date"] == "2026-06-22"
    assert item["status_reason"] == "Needs source owner review."


def test_knowledge_gap_suggest_faq_creates_pending_action(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    context = {"kb_id": 1, "kb_key": "default", "auth": {}}
    record_knowledge_gap(
        chat_log_id=None,
        query="Có đổi trả qua app không?",
        mode="fallback",
        top_score=0.05,
        session_id="session-gap-faq",
        context=context,
    )

    with TestClient(main.app) as client:
        listed = client.get("/api/admin/knowledge-gaps?days=7&kb_id=1", headers=admin_headers())
        cluster_key = listed.json()["items"][0]["cluster_key"]
        created = client.post(
            f"/api/admin/knowledge-gaps/{cluster_key}/suggest-faq?kb_id=1",
            headers=admin_headers(),
        )
        duplicate = client.post(
            f"/api/admin/knowledge-gaps/{cluster_key}/suggest-faq?kb_id=1",
            headers=admin_headers(),
        )
        suggested = client.get(
            "/api/admin/knowledge-gaps?days=7&kb_id=1&status=suggested",
            headers=admin_headers(),
        )

    assert created.status_code == 200, created.text
    payload = created.json()
    action = payload["pending_action"]
    assert payload["created"] is True
    assert action["action_type"] == "create_faq_entry"
    assert action["risk_level"] == "medium"
    assert action["kb_id"] == 1
    assert action["payload"]["cluster_key"] == cluster_key
    assert action["payload"]["question"] == "Có đổi trả qua app không?"
    assert "answer_template" in action["payload"]
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json()["created"] is False
    assert duplicate.json()["pending_action"]["id"] == action["id"]
    assert suggested.json()["total"] == 1


def test_negative_feedback_adds_knowledge_review_queue_item(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    chat_id = execute_sync(
        """
        INSERT INTO chat_logs (
            request_id, session_id, user_id, roles_json, channel, kb_id, kb_key,
            user_message, mode, top_score, answer_text, citations_json, created_at
        ) VALUES (
            'req-feedback-gap', 'session-feedback-gap', 'employee-1', '["employee"]',
            'chat', 1, 'default', 'Can I return through the mobile app?',
            'answer', 0.82, 'Use the normal return policy.', '[]', datetime('now')
        )
        """
    )

    with TestClient(main.app) as client:
        feedback = client.post(
            "/api/feedback/chat",
            headers=auth_headers(user_id="employee-1", roles=["employee"], channel="chat"),
            json={
                "chat_log_id": chat_id,
                "rating": "down",
                "reason_code": "wrong_source",
                "comment": "The cited policy does not mention the mobile app.",
            },
        )
        gaps = client.get(
            "/api/admin/knowledge-gaps?days=7&kb_id=1&status=new",
            headers=admin_headers(),
        )

    assert feedback.status_code == 200, feedback.text
    assert gaps.status_code == 200, gaps.text
    assert gaps.json()["total"] == 1
    item = gaps.json()["items"][0]
    assert item["representative_query"] == "Can I return through the mobile app?"
    assert item["priority"] == "P1"
    assert item["status"] == "new"
    assert "wrong_source" in item["status_reason"]


def test_quality_debt_endpoint_counts_active_gaps_and_stale_docs(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    context = {"kb_id": 1, "kb_key": "default", "auth": {}}
    record_knowledge_gap(
        chat_log_id=None,
        query="Missing warranty policy?",
        mode="fallback",
        top_score=0.05,
        session_id="session-quality-debt",
        context={**context, "priority": "P0", "due_date": "2020-01-01"},
    )
    file_id = execute_sync(
        """
        INSERT INTO uploaded_files (filename, original_name, file_type, file_size, file_hash, status, created_at)
        VALUES ('stale.csv', 'stale.csv', '.csv', 10, 'hash-stale', 'failed', datetime('now'))
        """
    )
    execute_sync(
        """
        INSERT INTO kb_files (kb_id, file_id, status, chunk_count, stale_reason, stale_detected_at, attached_at)
        VALUES (1, ?, 'failed', 0, 'File is not currently ingested.', datetime('now'), datetime('now'))
        """,
        (file_id,),
    )

    with TestClient(main.app) as client:
        response = client.get(
            "/api/admin/knowledge-gaps/quality-debt?days=30&kb_id=1",
            headers=admin_headers(),
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["active_gap_count"] == 1
    assert payload["overdue_gap_count"] == 1
    assert payload["stale_document_count"] == 1
    assert payload["failed_ingest_count"] == 1
    assert payload["zero_chunk_count"] == 1


def test_knowledge_gap_endpoint_requires_analytics_role(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        response = client.get(
            "/api/admin/knowledge-gaps?days=7",
            headers=auth_headers(user_id="user-1", roles=["customer"]),
        )

    assert response.status_code == 403, response.text


def test_knowledge_gap_report_creates_notification_summary(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    context = {
        "kb_id": 1,
        "kb_key": "default",
        "auth": {"tenant_id": "tenant-a", "org_id": "org-a"},
    }
    record_knowledge_gap(
        chat_log_id=None,
        query="Giao hàng quốc tế?",
        mode="fallback",
        top_score=0.05,
        session_id="session-report-1",
        context=context,
    )
    record_knowledge_gap(
        chat_log_id=None,
        query="Đổi trả qua app?",
        mode="fallback",
        top_score=0.07,
        session_id="session-report-2",
        context=context,
    )

    report = create_knowledge_gap_report(
        days=7,
        kb_id=1,
        status="open",
        limit=20,
        context=RequestContext(
            request_id="test-gap-report",
            kb_id=1,
            kb_key="default",
            auth=AuthContext(user_id="admin-1", roles=["admin"], channel="admin"),
        ),
    )

    assert report["event_count"] == 2
    assert report["cluster_count"] == 2
    assert report["notification"]["event_type"] == "knowledge_gap.weekly_report"
    assert report["notification"]["payload"]["event_count"] == 2
    assert len(report["notification"]["payload"]["top_clusters"]) == 2


def test_knowledge_gap_report_background_job_creates_notification(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    context = {
        "kb_id": 1,
        "kb_key": "default",
        "auth": {"tenant_id": "tenant-a", "org_id": "org-a"},
    }
    record_knowledge_gap(
        chat_log_id=None,
        query="Bảo hành ngoài Việt Nam?",
        mode="fallback",
        top_score=0.05,
        session_id="session-report-job",
        context=context,
    )
    enqueue_background_job(
        job_type="knowledge_gap_report",
        payload={"days": 7, "kb_id": 1, "status": "open", "limit": 20},
        context=RequestContext(
            request_id="gap-report-job",
            kb_id=1,
            kb_key="default",
            auth=AuthContext(user_id="admin-1", roles=["admin"], channel="admin"),
        ),
    )

    assert run(run_due_background_jobs_once()) is True
    notifications = fetch_all_sync(
        "SELECT event_type, payload_json FROM notifications WHERE event_type = 'knowledge_gap.weekly_report'"
    )
    assert len(notifications) == 1
    assert '"event_count": 1' in notifications[0]["payload_json"]
