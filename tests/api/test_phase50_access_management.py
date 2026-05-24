from __future__ import annotations

from app.config import settings
from tests.conftest import admin_headers, auth_headers


def test_access_management_observes_and_lists_users(isolated_client):
    response = isolated_client.get("/api/me", headers=auth_headers(user_id="employee-1", roles=["employee"], channel="portal"))
    assert response.status_code == 200, response.text

    listed = isolated_client.get("/api/admin/access/users", headers=admin_headers())
    assert listed.status_code == 200, listed.text
    payload = listed.json()
    user_ids = {item["user_id"] for item in payload["items"]}
    assert "employee-1" in user_ids
    assert "admin-1" in user_ids


def test_access_management_deactivated_user_is_blocked(isolated_client):
    create = isolated_client.post(
        "/api/admin/access/users",
        headers=admin_headers(user_id="admin-owner"),
        json={
            "user_id": "admin-blocked",
            "roles": ["admin"],
            "is_active": False,
        },
    )
    assert create.status_code == 200, create.text
    assert create.json()["is_active"] is False

    blocked = isolated_client.get("/api/admin/readiness", headers=admin_headers(user_id="admin-blocked"))
    assert blocked.status_code == 403, blocked.text
    assert blocked.json()["detail"] == "User is inactive"


def test_access_management_fallback_roles_can_authorize_user_without_header_roles(isolated_client, monkeypatch):
    monkeypatch.setattr(settings, "access_management_role_mode", "fallback")
    create = isolated_client.post(
        "/api/admin/access/users",
        headers=admin_headers(user_id="admin-owner"),
        json={
            "user_id": "directory-admin",
            "roles": ["admin"],
            "is_active": True,
        },
    )
    assert create.status_code == 200, create.text

    response = isolated_client.get(
        "/api/admin/readiness",
        headers=auth_headers(user_id="directory-admin", channel="admin"),
    )
    assert response.status_code == 200, response.text


def test_production_readiness_reports_dev_auth_and_llm_warning(isolated_client, monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "dev")
    monkeypatch.setattr(settings, "llm_provider", "none")

    response = isolated_client.get("/api/admin/readiness", headers=admin_headers())
    assert response.status_code == 200, response.text
    payload = response.json()
    checks = {item["key"]: item for item in payload["checks"]}

    assert payload["ready_for_production"] is False
    assert checks["auth_secured"]["status"] == "fail"
    assert checks["llm_configured"]["status"] == "warn"


def test_health_includes_readiness_issues(isolated_client, monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "dev")
    monkeypatch.setattr(settings, "llm_provider", "none")

    response = isolated_client.get("/health")
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["ready_for_chat"] is True
    assert payload["ready_for_production"] is False
    assert any(issue["component"] == "auth_secured" for issue in payload["issues"])
