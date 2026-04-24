from __future__ import annotations

import pytest

import app.database as database
from app.models import AuthContext, RequestContext
from app.tools import build_default_tool_registry
from app.tools.registry import ToolAuthorizationError, ToolValidationError
from tests.conftest import add_vector, attach_file, configure_test_env, create_kb, fetch_default_kb, insert_file, mark_ingested, run


def _seed_two_kbs():
    default_kb = fetch_default_kb()
    archive_kb = create_kb("Archive KB", "archive")

    default_file = insert_file("default.csv")
    archive_file = insert_file("archive.csv")
    attach_file(default_kb.id, default_file)
    attach_file(archive_kb.id, archive_file)
    mark_ingested(default_kb.id, default_file)
    mark_ingested(archive_kb.id, archive_file)

    add_vector(
        default_kb.id,
        default_file,
        "Default KB shipping answer",
        filename="default.csv",
        kb_version=default_kb.kb_version,
        chunk_id="chunk-default",
    )
    add_vector(
        archive_kb.id,
        archive_file,
        "Archive KB refund answer",
        filename="archive.csv",
        kb_version=archive_kb.kb_version,
        chunk_id="chunk-archive",
    )
    return default_kb, archive_kb


def test_registry_defines_phase2_tools(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    registry = build_default_tool_registry()

    definitions = {item.name: item for item in registry.list_definitions()}
    assert set(definitions) == {
        "search_kb",
        "create_support_ticket",
        "get_order_status",
        "find_recent_orders",
        "get_online_member_count",
        "list_kbs",
        "get_kb_stats",
        "list_google_drive_sources",
        "create_google_drive_source",
        "sync_google_drive_source",
        "get_google_drive_sync_status",
        "delete_google_drive_source",
        "list_support_emails",
        "read_email_thread",
        "create_ticket_from_email",
        "send_email_reply",
    }
    assert definitions["search_kb"].auth_policy["allow_anonymous"] is True
    assert definitions["search_kb"].idempotent is True
    assert definitions["create_support_ticket"].idempotent is False
    assert definitions["get_order_status"].auth_policy["require_user_id"] is True
    assert definitions["find_recent_orders"].auth_policy["require_user_id"] is True
    assert definitions["get_online_member_count"].auth_policy["allow_anonymous"] is True
    assert definitions["list_kbs"].auth_policy["required_roles"] == ["admin"]
    assert definitions["list_kbs"].auth_policy["allowed_channels"] == ["admin"]
    assert definitions["list_kbs"].auth_policy["risk_level"] == "high"
    assert definitions["list_google_drive_sources"].auth_policy["required_roles"] == ["admin"]
    assert definitions["sync_google_drive_source"].idempotent is False
    assert definitions["delete_google_drive_source"].idempotent is False
    assert definitions["list_support_emails"].auth_policy["required_roles"] == ["admin"]
    assert definitions["send_email_reply"].auth_policy["risk_level"] == "critical"
    assert definitions["get_order_status"].auth_policy["requires_tenant_match"] is True
    assert definitions["get_kb_stats"].auth_policy["required_roles"] == ["admin"]


def test_search_kb_tool_returns_scoped_hits_and_logs_audit(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    default_kb, _ = _seed_two_kbs()
    registry = build_default_tool_registry()
    monkeypatch.setattr("app.rag.expand_query", lambda query: [query])
    monkeypatch.setattr("app.rag.embed_query", lambda query: [1.0, 0.0])
    monkeypatch.setattr("app.rag.rerank", lambda query, items: items)

    result = run(
        registry.execute(
            "search_kb",
            {"query": "shipping", "kb_id": default_kb.id, "top_k": 5},
            request_context=RequestContext(request_id="req-search-tool"),
        )
    )

    assert result.output["kb_id"] == default_kb.id
    assert result.output["kb_key"] == "default"
    assert result.output["hits"]
    assert result.output["hits"][0]["filename"] == "default.csv"

    audit_row = database.fetch_one_sync(
        "SELECT tool_name, tool_status, request_id FROM tool_audit_logs WHERE tool_call_id = ?",
        (result.tool_call_id,),
    )
    assert audit_row == {
        "tool_name": "search_kb",
        "tool_status": "success",
        "request_id": "req-search-tool",
    }


def test_admin_tools_require_admin_role_and_return_data(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    default_kb, _ = _seed_two_kbs()
    registry = build_default_tool_registry()

    with pytest.raises(ToolAuthorizationError):
        run(
            registry.execute(
                "list_kbs",
                {},
                request_context=RequestContext(request_id="req-no-admin"),
            )
        )

    denied_row = database.fetch_one_sync(
        "SELECT tool_status FROM tool_audit_logs WHERE request_id = ? ORDER BY id DESC LIMIT 1",
        ("req-no-admin",),
    )
    assert denied_row == {"tool_status": "permission_denied"}

    admin_context = RequestContext(
        request_id="req-admin-tools",
        kb_id=default_kb.id,
        auth=AuthContext(user_id="admin-1", roles=["admin"], channel="admin"),
    )

    kb_list = run(registry.execute("list_kbs", {}, request_context=admin_context))
    kb_stats = run(registry.execute("get_kb_stats", {}, request_context=admin_context))

    assert kb_list.output["total"] >= 2
    assert any(item["key"] == "default" for item in kb_list.output["items"])
    assert kb_stats.output["kb_id"] == default_kb.id
    assert kb_stats.output["total_vectors"] >= 1


def test_admin_tools_require_admin_channel(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    registry = build_default_tool_registry()

    wrong_channel_context = RequestContext(
        request_id="req-admin-web-channel",
        auth=AuthContext(user_id="admin-1", roles=["admin"], channel="web"),
    )

    with pytest.raises(ToolAuthorizationError):
        run(registry.execute("list_kbs", {}, request_context=wrong_channel_context))


def test_order_tools_reject_tenant_scope_mismatch(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    registry = build_default_tool_registry()

    async def _fake_get_order_status(order_code: str, *, user_id: str | None = None):
        return {
            "order_code": order_code,
            "user_id": user_id,
            "tenant_id": "tenant-b",
            "org_id": "org-b",
            "status": "shipping",
            "source": "api",
        }

    monkeypatch.setattr("app.tools.business_tools.get_order_status", _fake_get_order_status)

    tenant_a_context = RequestContext(
        request_id="req-order-tenant-a",
        auth=AuthContext(user_id="cust-1", roles=["customer"], channel="web", tenant_id="tenant-a", org_id="org-a"),
    )

    with pytest.raises(ToolAuthorizationError):
        run(
            registry.execute(
                "get_order_status",
                {"order_code": "ORD-1"},
                request_context=tenant_a_context,
            )
        )


def test_create_support_ticket_persists_and_validates_contact(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    registry = build_default_tool_registry()

    with pytest.raises(ToolValidationError):
        run(
            registry.execute(
                "create_support_ticket",
                {"issue_type": "payment", "message": "Card declined again"},
                request_context=RequestContext(request_id="req-ticket-invalid"),
            )
        )

    invalid_row = database.fetch_one_sync(
        "SELECT tool_status FROM tool_audit_logs WHERE request_id = ? ORDER BY id DESC LIMIT 1",
        ("req-ticket-invalid",),
    )
    assert invalid_row == {"tool_status": "validation_error"}

    result = run(
        registry.execute(
            "create_support_ticket",
            {
                "issue_type": "payment",
                "message": "Card declined again",
                "contact": "user@example.com",
            },
            request_context=RequestContext(
                request_id="req-ticket-valid",
                kb_id=1,
                kb_key="default",
                auth=AuthContext(channel="chat", tenant_id="tenant-a", org_id="org-a"),
            ),
        )
    )

    ticket_row = database.fetch_one_sync(
        """
        SELECT ticket_code, issue_type, contact, channel, tenant_id, org_id, kb_id, kb_key
        FROM support_tickets
        WHERE ticket_code = ?
        """,
        (result.output["ticket_code"],),
    )
    assert ticket_row == {
        "ticket_code": result.output["ticket_code"],
        "issue_type": "payment",
        "contact": "user@example.com",
        "channel": "chat",
        "tenant_id": "tenant-a",
        "org_id": "org-a",
        "kb_id": 1,
        "kb_key": "default",
    }
