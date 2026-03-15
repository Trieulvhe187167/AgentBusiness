from __future__ import annotations

import app.database as database
from fastapi.testclient import TestClient

from tests.conftest import isolated_client


def _seed_order(
    order_code: str,
    *,
    user_id: str,
    status: str,
    last_update: str = "2026-03-15T01:02:03+00:00",
    tracking_code: str | None = "TRACK-001",
    carrier: str | None = "GHN",
    source: str = "snapshot",
):
    now = database.utcnow_iso()
    database.execute_sync(
        """
        INSERT INTO order_status_cache (
            order_code, user_id, status, last_update, tracking_code, carrier,
            source, raw_json, cached_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)
        """,
        (order_code, user_id, status, last_update, tracking_code, carrier, source, now, now),
    )


def _seed_online_count(
    alliance_id: str,
    *,
    online_count: int,
    server_id: str | None = None,
    observed_at: str = "2026-03-15T02:03:04+00:00",
    source: str = "snapshot",
):
    now = database.utcnow_iso()
    database.execute_sync(
        """
        INSERT INTO game_online_cache (
            alliance_id, server_id, server_scope, online_count, observed_at,
            source, raw_json, cached_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, '{}', ?, ?)
        """,
        (alliance_id, server_id, server_id or "", online_count, observed_at, source, now, now),
    )


def test_chat_agent_suggests_recent_orders_for_signed_in_user(isolated_client: TestClient):
    _seed_order("DH12345", user_id="user-1", status="dang_giao")
    _seed_order("DH12346", user_id="user-1", status="cho_xac_nhan")

    chat = isolated_client.post(
        "/api/chat",
        json={
            "session_id": "phase19-recent-orders",
            "message": "Don hang cua toi toi dau roi?",
            "lang": "vi",
            "user_id": "user-1",
        },
    )

    assert chat.status_code == 200, chat.text
    assert '"route": "tool"' in chat.text
    assert '"tool_name": "find_recent_orders"' in chat.text
    assert '"status": "success"' in chat.text
    assert "DH12345" in chat.text


def test_chat_agent_clarifies_missing_order_code_for_anonymous_user(isolated_client: TestClient):
    chat = isolated_client.post(
        "/api/chat",
        json={
            "session_id": "phase19-order-clarify",
            "message": "Don hang cua toi toi dau roi?",
            "lang": "vi",
        },
    )

    assert chat.status_code == 200, chat.text
    assert '"route": "clarify"' in chat.text
    assert "event: tool_call" not in chat.text
    lowered = chat.text.lower()
    assert "mã đơn" in lowered or "ma don" in lowered


def test_chat_agent_returns_order_status_for_explicit_code(isolated_client: TestClient):
    _seed_order("DH99999", user_id="user-1", status="dang_giao", carrier="VNPOST")

    chat = isolated_client.post(
        "/api/chat",
        json={
            "session_id": "phase19-order-status",
            "message": "Kiem tra don DH99999 giup toi",
            "lang": "vi",
            "user_id": "user-1",
        },
    )

    assert chat.status_code == 200, chat.text
    assert '"tool_name": "get_order_status"' in chat.text
    assert '"status": "success"' in chat.text
    assert "DH99999" in chat.text
    assert "VNPOST" in chat.text


def test_chat_agent_blocks_access_to_other_users_order(isolated_client: TestClient):
    _seed_order("DH00077", user_id="user-2", status="da_giao")

    chat = isolated_client.post(
        "/api/chat",
        json={
            "session_id": "phase19-order-denied",
            "message": "Kiem tra don DH00077",
            "lang": "vi",
            "user_id": "user-1",
        },
    )

    assert chat.status_code == 200, chat.text
    assert '"tool_name": "get_order_status"' in chat.text
    assert '"status": "failed"' in chat.text
    assert "quyen" in chat.text.lower() or "permission" in chat.text.lower()


def test_chat_agent_returns_online_member_count(isolated_client: TestClient):
    _seed_online_count("LM01", online_count=128)

    chat = isolated_client.post(
        "/api/chat",
        json={
            "session_id": "phase19-online-count",
            "message": "Lien minh LM01 co bao nhieu nguoi online?",
            "lang": "vi",
        },
    )

    assert chat.status_code == 200, chat.text
    assert '"tool_name": "get_online_member_count"' in chat.text
    assert '"status": "success"' in chat.text
    assert "128" in chat.text
