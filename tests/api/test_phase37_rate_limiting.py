from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from tests.conftest import admin_headers, auth_headers, configure_test_env


def test_rate_limit_blocks_after_policy_quota(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "rate_limit_admin_requests_per_window", 2)

    with TestClient(main.app) as client:
        first = client.get("/api/admin/mcp/status", headers=admin_headers())
        second = client.get("/api/admin/mcp/status", headers=admin_headers())
        blocked = client.get("/api/admin/mcp/status", headers=admin_headers())

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert blocked.status_code == 429, blocked.text
    assert blocked.json()["detail"] == "Rate limit exceeded"
    assert blocked.json()["policy"] == "admin"
    assert blocked.headers["X-RateLimit-Limit"] == "2"
    assert blocked.headers["X-RateLimit-Remaining"] == "0"
    assert int(blocked.headers["Retry-After"]) >= 1
    assert blocked.headers["X-Request-ID"]


def test_rate_limit_uses_separate_user_buckets(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "rate_limit_admin_requests_per_window", 1)

    with TestClient(main.app) as client:
        admin_one = client.get("/api/admin/mcp/status", headers=admin_headers(user_id="admin-1"))
        blocked_admin_one = client.get("/api/admin/mcp/status", headers=admin_headers(user_id="admin-1"))
        admin_two = client.get("/api/admin/mcp/status", headers=admin_headers(user_id="admin-2"))

    assert admin_one.status_code == 200, admin_one.text
    assert blocked_admin_one.status_code == 429, blocked_admin_one.text
    assert admin_two.status_code == 200, admin_two.text


def test_mcp_has_its_own_rate_limit_policy(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "rate_limit_mcp_requests_per_window", 1)
    payload = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}

    with TestClient(main.app) as client:
        first = client.post("/mcp", json=payload, headers=admin_headers())
        blocked = client.post("/mcp", json=payload, headers=admin_headers())

    assert first.status_code == 200, first.text
    assert blocked.status_code == 429, blocked.text
    assert blocked.json()["policy"] == "mcp"
    assert blocked.headers["X-RateLimit-Policy"] == "mcp"


def test_chat_rate_limit_applies_before_body_validation(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "rate_limit_chat_requests_per_window", 1)

    with TestClient(main.app) as client:
        invalid_body = client.post("/api/chat", json={}, headers=auth_headers(user_id="user-1", channel="web"))
        blocked = client.post("/api/chat", json={}, headers=auth_headers(user_id="user-1", channel="web"))

    assert invalid_body.status_code == 422, invalid_body.text
    assert blocked.status_code == 429, blocked.text
    assert blocked.json()["policy"] == "chat"


def test_exempt_paths_do_not_consume_quota(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "rate_limit_default_requests_per_window", 1)

    with TestClient(main.app) as client:
        first = client.get("/health")
        second = client.get("/health")

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert "X-RateLimit-Limit" not in first.headers
