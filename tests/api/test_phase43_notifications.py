from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from tests.conftest import admin_headers, configure_test_env


def test_admin_notification_center_create_list_and_mark_read(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        created = client.post(
            "/api/admin/notifications",
            headers=admin_headers(),
            json={
                "event_type": "notification.test",
                "severity": "warning",
                "title": "Test notification",
                "message": "A test event",
                "entity_type": "system",
                "entity_id": "test",
                "payload": {"ok": True},
            },
        )
        listed = client.get("/api/admin/notifications?status=unread", headers=admin_headers())

    assert created.status_code == 200, created.text
    assert created.json()["status"] == "unread"
    assert listed.status_code == 200, listed.text
    payload = listed.json()
    assert payload["unread_count"] == 1
    assert payload["items"][0]["event_type"] == "notification.test"

    notification_id = created.json()["id"]
    with TestClient(main.app) as client:
        marked = client.post(f"/api/admin/notifications/{notification_id}/read", headers=admin_headers())
        unread = client.get("/api/admin/notifications?status=unread", headers=admin_headers())

    assert marked.status_code == 200, marked.text
    assert marked.json()["status"] == "read"
    assert unread.status_code == 200, unread.text
    assert unread.json()["unread_count"] == 0


def test_pending_action_creation_emits_notification(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        action = client.post(
            "/api/admin/pending-actions",
            headers=admin_headers(),
            json={
                "action_type": "support_case_review",
                "risk_level": "high",
                "title": "Review support case",
                "summary": "Needs approval",
                "payload": {"ticket_id": 1},
            },
        )
        notifications = client.get("/api/admin/notifications?status=unread", headers=admin_headers())

    assert action.status_code == 200, action.text
    assert notifications.status_code == 200, notifications.text
    items = notifications.json()["items"]
    assert any(item["event_type"] == "pending_action.created" for item in items)


def test_notification_creates_pending_webhook_delivery_when_enabled(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "notification_webhook_enabled", True)
    monkeypatch.setattr(settings, "notification_webhook_url", "https://example.test/agent-webhook")

    with TestClient(main.app) as client:
        created = client.post(
            "/api/admin/notifications",
            headers=admin_headers(),
            json={
                "event_type": "notification.webhook_test",
                "severity": "info",
                "title": "Webhook test",
                "message": "Should enqueue webhook delivery",
            },
        )
        deliveries = client.get("/api/admin/webhook-deliveries", headers=admin_headers())

    assert created.status_code == 200, created.text
    assert deliveries.status_code == 200, deliveries.text
    items = deliveries.json()["items"]
    assert len(items) == 1
    assert items[0]["notification_id"] == created.json()["id"]
    assert items[0]["event_type"] == "notification.webhook_test"
    assert items[0]["status"] == "pending"


def test_webhook_subscription_filters_events_and_test_delivery(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "notification_webhook_enabled", True)
    monkeypatch.setattr(settings, "notification_webhook_url", "")

    with TestClient(main.app) as client:
        created = client.post(
            "/api/admin/webhook-subscriptions",
            headers=admin_headers(),
            json={
                "name": "Support events",
                "endpoint_url": "https://example.test/support-webhook",
                "secret": "secret-1",
                "event_types": ["support.*", "notification.webhook_test"],
                "enabled": True,
            },
        )
        listed = client.get("/api/admin/webhook-subscriptions", headers=admin_headers())
        ignored_notification = client.post(
            "/api/admin/notifications",
            headers=admin_headers(),
            json={"event_type": "background_job.failed", "severity": "warning", "title": "Ignored"},
        )
        matched_notification = client.post(
            "/api/admin/notifications",
            headers=admin_headers(),
            json={"event_type": "support.employee_replied", "severity": "warning", "title": "Matched"},
        )
        tested = client.post(f"/api/admin/webhook-subscriptions/{created.json()['id']}/test", headers=admin_headers())
        deliveries = client.get("/api/admin/webhook-deliveries", headers=admin_headers())

    assert created.status_code == 200, created.text
    subscription = created.json()
    assert subscription["has_secret"] is True
    assert subscription["event_types"] == ["notification.webhook_test", "support.*"]
    assert listed.status_code == 200, listed.text
    assert listed.json()["total"] == 1
    assert ignored_notification.status_code == 200, ignored_notification.text
    assert matched_notification.status_code == 200, matched_notification.text
    assert tested.status_code == 200, tested.text

    delivery_items = deliveries.json()["items"]
    delivery_notification_ids = {item["notification_id"] for item in delivery_items}
    assert ignored_notification.json()["id"] not in delivery_notification_ids
    assert matched_notification.json()["id"] in delivery_notification_ids
    assert tested.json()["notification"]["id"] in delivery_notification_ids
    assert all(item["subscription_id"] == subscription["id"] for item in delivery_items)
