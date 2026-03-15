from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import isolated_client, poll_jobs


def test_upload_ingest_and_chat_smoke(isolated_client: TestClient):
    sample_path = Path("kb_sample.csv")
    with sample_path.open("rb") as handle:
        upload = isolated_client.post(
            "/api/upload",
            files={"file": (sample_path.name, handle, "text/csv")},
        )
    assert upload.status_code == 200, upload.text
    uploaded = upload.json()["file"]
    assert uploaded["original_name"] == "kb_sample.csv"
    assert uploaded["status"] == "uploaded"

    default_kb = isolated_client.get("/api/kbs/default")
    assert default_kb.status_code == 200, default_kb.text
    kb = default_kb.json()

    kb_files = isolated_client.get(f"/api/kbs/{kb['id']}/files")
    assert kb_files.status_code == 200, kb_files.text
    attached_names = {item["original_name"] for item in kb_files.json()}
    assert "kb_sample.csv" in attached_names

    ingest = isolated_client.post(f"/api/kbs/{kb['id']}/ingest")
    assert ingest.status_code == 200, ingest.text
    jobs = ingest.json().get("jobs") or []
    assert jobs, ingest.json()
    poll_jobs(isolated_client, jobs)

    stats = isolated_client.get("/api/kb/stats", params={"kb_id": kb["id"]})
    assert stats.status_code == 200, stats.text
    stats_payload = stats.json()
    assert stats_payload["scope"] == "kb"
    assert stats_payload["ingested_files"] >= 1
    assert "kb_sample.csv" in stats_payload["sources"]

    chat = isolated_client.post(
        "/api/chat",
        json={
            "session_id": "phase9-smoke",
            "message": "Phí giao hàng là bao nhiêu?",
            "lang": "vi",
            "kb_id": kb["id"],
        },
    )
    assert chat.status_code == 200, chat.text
    assert "event: start" in chat.text
    assert '"kb_key": "default"' in chat.text
    assert "event: done" in chat.text


def test_ingest_endpoint_reports_no_stale_files_after_smoke_run(isolated_client: TestClient):
    sample_path = Path("kb_sample.csv")
    with sample_path.open("rb") as handle:
        isolated_client.post(
            "/api/upload",
            files={"file": (sample_path.name, handle, "text/csv")},
        ).raise_for_status()

    kb = isolated_client.get("/api/kbs/default")
    kb.raise_for_status()
    kb_id = kb.json()["id"]

    first_ingest = isolated_client.post(f"/api/kbs/{kb_id}/ingest")
    first_ingest.raise_for_status()
    jobs = first_ingest.json().get("jobs") or []
    assert jobs
    poll_jobs(isolated_client, jobs)

    second_ingest = isolated_client.post(f"/api/kbs/{kb_id}/ingest")
    second_ingest.raise_for_status()
    payload = second_ingest.json()
    assert payload["jobs"] == []
    assert payload["message"] == "No stale files to ingest for this Knowledge Base"
