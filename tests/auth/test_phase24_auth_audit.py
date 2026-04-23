from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.database as database
import app.main as main
from app.models import AuthContext, KnowledgeBaseCreate, RequestContext
from app.tools import build_default_tool_registry
from app.tools.registry import ToolAuthorizationError
from tests.conftest import auth_headers, configure_test_env, run
from app.kb import create_knowledge_base


def test_admin_route_denial_writes_auth_audit(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    with TestClient(main.app) as client:
        response = client.get(
            "/api/kbs",
            headers=auth_headers(user_id="customer-001", roles=["customer"], channel="web"),
        )
        assert response.status_code == 403

    row = database.fetch_one_sync(
        """
        SELECT resource_type, resource_id, action, decision, reason, user_id
        FROM auth_audit_logs
        ORDER BY id DESC
        LIMIT 1
        """
    )
    assert row == {
        "resource_type": "route",
        "resource_id": "/api/kbs",
        "action": "admin_access",
        "decision": "deny",
        "reason": "admin_role_required",
        "user_id": "customer-001",
    }


def test_chat_visible_kbs_logs_allow_and_deny_decisions(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    run(create_knowledge_base(KnowledgeBaseCreate(name="Public One", key="public-one", access_level="public")))
    run(create_knowledge_base(KnowledgeBaseCreate(name="Internal One", key="internal-one", access_level="internal")))

    with TestClient(main.app) as client:
        response = client.get(
            "/api/chat/kbs",
            headers=auth_headers(user_id="customer-001", roles=["customer"], channel="web"),
        )
        response.raise_for_status()

    rows = database.fetch_all_sync(
        """
        SELECT resource_type, decision, reason
        FROM auth_audit_logs
        WHERE resource_type = 'knowledge_base'
        ORDER BY id DESC
        """
    )
    reasons = {(row["decision"], row["reason"]) for row in rows}
    assert ("allow", "access_level=public") in reasons
    assert ("deny", "internal_access_required") in reasons


def test_tool_authorization_denial_writes_auth_audit(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    registry = build_default_tool_registry()

    with pytest.raises(ToolAuthorizationError):
        run(
            registry.execute(
                "list_kbs",
                {},
                request_context=RequestContext(
                    request_id="req-admin-web-channel",
                    auth=AuthContext(user_id="admin-1", roles=["admin"], channel="web"),
                ),
            )
        )

    row = database.fetch_one_sync(
        """
        SELECT resource_type, resource_id, action, decision, reason, request_id
        FROM auth_audit_logs
        WHERE resource_type = 'tool'
        ORDER BY id DESC
        LIMIT 1
        """
    )
    assert row == {
        "resource_type": "tool",
        "resource_id": "list_kbs",
        "action": "execute",
        "decision": "deny",
        "reason": "allowed_channels=admin",
        "request_id": "req-admin-web-channel",
    }
