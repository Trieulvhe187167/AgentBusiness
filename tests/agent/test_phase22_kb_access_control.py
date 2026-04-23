from __future__ import annotations

import pytest
from fastapi import HTTPException

import app.rag as rag
from app.kb import create_knowledge_base
from app.models import AuthContext, KnowledgeBaseCreate, RequestContext
from app.tools import build_default_tool_registry
from app.tools.registry import ToolExecutionError
from tests.conftest import add_vector, admin_headers, attach_file, auth_headers, configure_test_env, insert_file, mark_ingested, run


def _create_kb(name: str, key: str, *, access_level: str, tenant_id: str | None = None, org_id: str | None = None):
    return run(
        create_knowledge_base(
            KnowledgeBaseCreate(
                name=name,
                key=key,
                access_level=access_level,
                tenant_id=tenant_id,
                org_id=org_id,
            )
        )
    )


def _seed_kb(access_level: str, *, tenant_id: str | None = None, org_id: str | None = None):
    key = f"{access_level}-{tenant_id or 'global'}-{org_id or 'global'}-kb"
    kb = _create_kb(
        f"{access_level.title()} KB",
        key,
        access_level=access_level,
        tenant_id=tenant_id,
        org_id=org_id,
    )
    file_id = insert_file(
        f"{access_level}.csv",
        access_level=access_level,
        tenant_id=tenant_id,
        org_id=org_id,
    )
    attach_file(kb.id, file_id)
    mark_ingested(kb.id, file_id)
    add_vector(
        kb.id,
        file_id,
        f"{access_level.title()} KB shipping answer",
        filename=f"{access_level}.csv",
        kb_version=kb.kb_version,
        chunk_id=f"chunk-{access_level}-{tenant_id or 'global'}-{org_id or 'global'}",
        access_level=access_level,
        tenant_id=tenant_id,
        org_id=org_id,
    )
    return kb


def _stub_retrieval(monkeypatch):
    monkeypatch.setattr("app.rag.expand_query", lambda query: [query])
    monkeypatch.setattr("app.rag.embed_query", lambda query: [1.0, 0.0])
    monkeypatch.setattr("app.rag.rerank", lambda query, items: items)


def test_search_kb_tool_enforces_internal_and_admin_access_levels(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _stub_retrieval(monkeypatch)
    registry = build_default_tool_registry()

    internal_kb = _seed_kb("internal")
    admin_kb = _seed_kb("admin")

    employee_context = RequestContext(
        request_id="req-kb-employee",
        auth=AuthContext(user_id="emp-1", roles=["employee"]),
    )
    admin_context = RequestContext(
        request_id="req-kb-admin",
        auth=AuthContext(user_id="admin-1", roles=["admin"]),
    )
    customer_context = RequestContext(
        request_id="req-kb-customer",
        auth=AuthContext(user_id="cust-1", roles=["customer"]),
    )

    employee_result = run(
        registry.execute(
            "search_kb",
            {"query": "shipping", "kb_id": internal_kb.id, "top_k": 5},
            request_context=employee_context,
        )
    )
    assert employee_result.output["kb_id"] == internal_kb.id
    assert employee_result.output["hits"][0]["filename"] == "internal.csv"

    with pytest.raises(ToolExecutionError):
        run(
            registry.execute(
                "search_kb",
                {"query": "shipping", "kb_id": internal_kb.id, "top_k": 5},
                request_context=customer_context,
            )
        )

    with pytest.raises(ToolExecutionError):
        run(
            registry.execute(
                "search_kb",
                {"query": "shipping", "kb_id": admin_kb.id, "top_k": 5},
                request_context=employee_context,
            )
        )

    admin_result = run(
        registry.execute(
            "search_kb",
            {"query": "shipping", "kb_id": admin_kb.id, "top_k": 5},
            request_context=admin_context,
        )
    )
    assert admin_result.output["kb_id"] == admin_kb.id
    assert admin_result.output["hits"][0]["filename"] == "admin.csv"



def test_rag_retrieve_enforces_kb_access_by_role(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _stub_retrieval(monkeypatch)

    public_kb = _seed_kb("public")
    internal_kb = _seed_kb("internal")

    public_results = rag.retrieve("shipping", top_k=5, kb_id=public_kb.id, auth_context={})
    assert public_results
    assert public_results[0]["filename"] == "public.csv"

    with pytest.raises(HTTPException) as denied:
        rag.retrieve("shipping", top_k=5, kb_id=internal_kb.id, auth_context={})
    assert denied.value.status_code == 403

    internal_results = rag.retrieve(
        "shipping",
        top_k=5,
        kb_id=internal_kb.id,
        auth_context={"user_id": "emp-2", "roles": ["internal"]},
    )
    assert internal_results
    assert internal_results[0]["filename"] == "internal.csv"


def test_chat_visible_kbs_endpoint_filters_by_role(isolated_client):
    _create_kb("Public One", "public-one", access_level="public")
    _create_kb("Public Two", "public-two", access_level="public")
    _create_kb("Internal One", "internal-one", access_level="internal")

    customer_response = isolated_client.get(
        "/api/chat/kbs",
        headers=auth_headers(user_id="customer-001", roles=["customer"], channel="web"),
    )
    customer_response.raise_for_status()
    customer_keys = [item["key"] for item in customer_response.json()]
    assert "default" in customer_keys
    assert "public-one" in customer_keys
    assert "public-two" in customer_keys
    assert "internal-one" not in customer_keys

    employee_response = isolated_client.get(
        "/api/chat/kbs",
        headers=auth_headers(user_id="employee-001", roles=["employee"], channel="web"),
    )
    employee_response.raise_for_status()
    employee_keys = [item["key"] for item in employee_response.json()]
    assert "internal-one" in employee_keys

    admin_only_response = isolated_client.get(
        "/api/kbs",
        headers=auth_headers(user_id="customer-001", roles=["customer"], channel="web"),
    )
    assert admin_only_response.status_code == 403

    admin_response = isolated_client.get("/api/kbs", headers=admin_headers())
    admin_response.raise_for_status()
    admin_keys = [item["key"] for item in admin_response.json()]
    assert "public-one" in admin_keys
    assert "internal-one" in admin_keys


def test_chat_visible_kbs_and_rag_retrieve_enforce_tenant_org_scope(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    _stub_retrieval(monkeypatch)

    tenant_a_kb = _seed_kb("public", tenant_id="tenant-a", org_id="org-a")
    _seed_kb("public", tenant_id="tenant-b", org_id="org-b")

    tenant_a_results = rag.retrieve(
        "shipping",
        top_k=5,
        kb_id=tenant_a_kb.id,
        auth_context={"user_id": "user-a", "roles": ["customer"], "tenant_id": "tenant-a", "org_id": "org-a"},
    )
    assert tenant_a_results
    assert tenant_a_results[0]["tenant_id"] == "tenant-a"
    assert tenant_a_results[0]["org_id"] == "org-a"

    with pytest.raises(HTTPException) as tenant_denied:
        rag.retrieve(
            "shipping",
            top_k=5,
            kb_id=tenant_a_kb.id,
            auth_context={"user_id": "user-b", "roles": ["customer"], "tenant_id": "tenant-b", "org_id": "org-b"},
        )
    assert tenant_denied.value.status_code == 403


def test_rag_retrieve_passes_scope_filters_to_vector_lookup(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    kb = _create_kb("Tenant KB", "tenant-kb", access_level="internal", tenant_id="tenant-a", org_id="org-a")
    monkeypatch.setattr("app.rag.expand_query", lambda query: [query])
    monkeypatch.setattr("app.rag.embed_query", lambda query: [1.0, 0.0])
    monkeypatch.setattr("app.rag.rerank", lambda query, items: items)

    captured: list[dict] = []

    def _fake_retrieve_single(query, top_k, where, cache_scope):
        captured.append({"query": query, "top_k": top_k, "where": dict(where), "cache_scope": cache_scope})
        return [
            {
                "chunk_id": "chunk-tenant",
                "text": "tenant answer",
                "similarity": 0.99,
                "filename": "tenant.csv",
                "access_level": "internal",
                "tenant_id": "tenant-a",
                "org_id": "org-a",
            }
        ]

    monkeypatch.setattr("app.rag._retrieve_single", _fake_retrieve_single)

    results = rag.retrieve(
        "shipping",
        top_k=5,
        kb_id=kb.id,
        auth_context={"user_id": "emp-a", "roles": ["employee"], "tenant_id": "tenant-a", "org_id": "org-a"},
    )

    assert results
    assert captured
    assert captured[0]["where"] == {
        "kb_id": kb.id,
        "access_level": "internal",
        "tenant_id": "tenant-a",
        "org_id": "org-a",
    }
