from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import app.rag as rag
from tests.conftest import admin_headers, poll_jobs


def _prepare_ingested_kb(client: TestClient) -> int:
    admin = admin_headers()
    sample_path = Path("kb_sample.csv")
    with sample_path.open("rb") as handle:
        upload = client.post(
            "/api/upload",
            files={"file": (sample_path.name, handle, "text/csv")},
            headers=admin,
        )
    upload.raise_for_status()

    kb = client.get("/api/kbs/default", headers=admin)
    kb.raise_for_status()
    kb_id = kb.json()["id"]

    ingest = client.post(f"/api/kbs/{kb_id}/ingest", headers=admin)
    ingest.raise_for_status()
    jobs = ingest.json().get("jobs") or []
    if jobs:
        poll_jobs(client, jobs)
    return kb_id


def test_admin_and_debug_endpoints_are_kb_scoped(isolated_client: TestClient):
    admin = admin_headers()
    kb_id = _prepare_ingested_kb(isolated_client)

    events = list(
        rag.rag_stream(
            query="Phí giao hàng là bao nhiêu?",
            session_id="phase10-admin-debug",
            lang="vi",
            kb_id=kb_id,
        )
    )
    assert any(event["event"] == "done" for event in events)

    system = isolated_client.get("/api/system", params={"kb_id": kb_id}, headers=admin)
    assert system.status_code == 200, system.text
    system_payload = system.json()
    assert system_payload["scope"]["type"] == "kb"
    assert system_payload["scope"]["kb_id"] == kb_id
    assert system_payload["source_count"] >= 1

    similarity = isolated_client.get(
        "/api/debug/similarity",
        params={"kb_id": kb_id, "query": "phí giao hàng"},
        headers=admin,
    )
    assert similarity.status_code == 200, similarity.text
    similarity_payload = similarity.json()
    assert similarity_payload["kb"]["id"] == kb_id
    assert isinstance(similarity_payload["results"], list)

    retrieval = isolated_client.get(
        "/api/debug/retrieval",
        params={"kb_id": kb_id, "query": "phí giao hàng", "top_k": 5},
        headers=admin,
    )
    assert retrieval.status_code == 200, retrieval.text
    retrieval_payload = retrieval.json()
    assert retrieval_payload["kb"]["id"] == kb_id
    assert isinstance(retrieval_payload["results"], list)
    assert retrieval_payload["results"]

    chat_logs = isolated_client.get("/api/admin/chat-logs", params={"limit": 10}, headers=admin)
    assert chat_logs.status_code == 200, chat_logs.text
    logs_payload = chat_logs.json()
    assert logs_payload
    assert any(item["session_id"] == f"phase10-admin-debug::kb:{kb_id}" for item in logs_payload)

    cache_stats = isolated_client.get("/api/cache/stats", headers=admin)
    assert cache_stats.status_code == 200, cache_stats.text
    assert cache_stats.json()["total_entries"] >= 1

    cache_clear = isolated_client.post("/api/cache/clear", headers=admin)
    assert cache_clear.status_code == 200, cache_clear.text
    assert cache_clear.json()["message"] == "Cache cleared"
