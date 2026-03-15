from __future__ import annotations

from fastapi.testclient import TestClient

from app.mock_business_api import app


def test_mock_business_api_serves_default_contracts():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["ok"] is True

        order = client.get("/orders/status", params={"order_code": "DH12345", "user_id": "user-1"})
        assert order.status_code == 200, order.text
        assert order.json()["status"] == "dang_giao"

        recent = client.get("/orders/recent", params={"user_id": "user-1", "limit": 5})
        assert recent.status_code == 200, recent.text
        assert recent.json()["orders"]
        assert recent.json()["orders"][0]["order_code"] == "DH12345"

        alliance = client.get("/alliances/online", params={"alliance_id": "LM01", "server_id": "S1"})
        assert alliance.status_code == 200, alliance.text
        assert alliance.json()["online_count"] == 128


def test_mock_business_api_supports_admin_upserts():
    with TestClient(app) as client:
        upsert_order = client.post(
            "/admin/orders/upsert",
            json={
                "order_code": "DH77777",
                "user_id": "user-7",
                "status": "cho_xu_ly",
                "last_update": "2026-03-15T15:00:00+07:00",
                "tracking_code": None,
                "carrier": None,
            },
        )
        assert upsert_order.status_code == 200, upsert_order.text

        order = client.get("/orders/status", params={"order_code": "DH77777", "user_id": "user-7"})
        assert order.status_code == 200, order.text
        assert order.json()["status"] == "cho_xu_ly"

        upsert_alliance = client.post(
            "/admin/alliances/upsert",
            json={
                "alliance_id": "LM77",
                "server_id": "S7",
                "online_count": 77,
                "observed_at": "2026-03-15T15:05:00+07:00",
            },
        )
        assert upsert_alliance.status_code == 200, upsert_alliance.text

        alliance = client.get("/alliances/online", params={"alliance_id": "LM77", "server_id": "S7"})
        assert alliance.status_code == 200, alliance.text
        assert alliance.json()["online_count"] == 77
