from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main
from tests.conftest import admin_headers, auth_headers, configure_test_env


def test_support_role_can_access_support_but_not_drive(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        support = client.get(
            "/api/admin/support-tickets",
            headers=auth_headers(user_id="support-1", roles=["support_agent"], channel="admin"),
        )
        drive = client.get(
            "/api/admin/google-drive/sources",
            headers=auth_headers(user_id="support-1", roles=["support_agent"], channel="admin"),
        )

    assert support.status_code == 200, support.text
    assert drive.status_code == 403, drive.text
    assert "integration_admin" in drive.json()["detail"]


def test_integration_role_can_access_drive_but_not_support(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        drive = client.get(
            "/api/admin/google-drive/sources",
            headers=auth_headers(user_id="ops-1", roles=["integration_admin"], channel="admin"),
        )
        support = client.get(
            "/api/admin/support-tickets",
            headers=auth_headers(user_id="ops-1", roles=["integration_admin"], channel="admin"),
        )

    assert drive.status_code == 200, drive.text
    assert support.status_code == 403, support.text
    assert "support_agent" in support.json()["detail"]


def test_analyst_and_auditor_scopes_are_separate(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        analytics = client.get(
            "/api/admin/analytics",
            headers=auth_headers(user_id="analyst-1", roles=["analyst"], channel="admin"),
        )
        audit_denied = client.get(
            "/api/admin/auth-audit-logs",
            headers=auth_headers(user_id="analyst-1", roles=["analyst"], channel="admin"),
        )
        audit = client.get(
            "/api/admin/auth-audit-logs",
            headers=auth_headers(user_id="auditor-1", roles=["auditor"], channel="admin"),
        )
        analytics_denied = client.get(
            "/api/admin/analytics",
            headers=auth_headers(user_id="auditor-1", roles=["auditor"], channel="admin"),
        )

    assert analytics.status_code == 200, analytics.text
    assert audit_denied.status_code == 403, audit_denied.text
    assert audit.status_code == 200, audit.text
    assert analytics_denied.status_code == 403, analytics_denied.text


def test_approver_can_access_pending_actions_but_not_knowledge_stats(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        pending = client.get(
            "/api/admin/pending-actions",
            headers=auth_headers(user_id="approver-1", roles=["approver"], channel="admin"),
        )
        kb_stats = client.get(
            "/api/kb/stats",
            headers=auth_headers(user_id="approver-1", roles=["approver"], channel="admin"),
        )

    assert pending.status_code == 200, pending.text
    assert kb_stats.status_code == 403, kb_stats.text
    assert "content_manager" in kb_stats.json()["detail"]


def test_admin_still_has_all_role_scoped_access(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    endpoints = [
        "/api/admin/support-tickets",
        "/api/admin/google-drive/sources",
        "/api/admin/analytics",
        "/api/admin/auth-audit-logs",
        "/api/admin/pending-actions",
        "/api/kb/stats",
    ]

    with TestClient(main.app) as client:
        responses = [client.get(endpoint, headers=admin_headers()) for endpoint in endpoints]

    assert [response.status_code for response in responses] == [200, 200, 200, 200, 200, 200]
