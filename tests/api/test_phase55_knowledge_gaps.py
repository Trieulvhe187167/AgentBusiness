from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main
from app.background_jobs import enqueue_background_job, run_due_background_jobs_once
from app.config import settings
from app.database import fetch_all_sync
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
            json={"status": "resolved"},
        )
        open_after = client.get("/api/admin/knowledge-gaps?days=7&kb_id=1", headers=admin_headers())
        resolved_after = client.get(
            "/api/admin/knowledge-gaps?days=7&kb_id=1&status=resolved",
            headers=admin_headers(),
        )

    assert updated.status_code == 200, updated.text
    assert open_after.json()["total"] == 0
    assert resolved_after.json()["total"] == 1
    assert resolved_after.json()["items"][0]["status"] == "resolved"


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
