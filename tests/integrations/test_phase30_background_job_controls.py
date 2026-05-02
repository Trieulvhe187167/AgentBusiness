from __future__ import annotations

import app.database as database
from app.background_jobs import (
    cancel_background_job,
    claim_next_background_job,
    enqueue_background_job,
    get_background_job,
    recover_stale_background_jobs,
    retry_background_job,
    run_due_background_jobs_once,
)
from app.models import AuthContext, RequestContext
from tests.conftest import admin_headers, configure_test_env, run


def _admin_context() -> RequestContext:
    return RequestContext(
        request_id="req-background-controls",
        auth=AuthContext(user_id="admin-1", roles=["admin"], channel="admin"),
    )


def test_background_job_retries_failed_transient_job(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    job = enqueue_background_job(
        job_type="unsupported_transient_job",
        payload={},
        context=_admin_context(),
        max_attempts=2,
    )

    assert run(run_due_background_jobs_once()) is True

    retrying = get_background_job(job["job_id"])
    assert retrying["status"] == "retrying"
    assert retrying["attempts"] == 1
    assert retrying["retry_after"]
    assert retrying["error_message"]

    cancelled = cancel_background_job(job["job_id"], reason="stop retry")
    assert cancelled["status"] == "cancelled"
    retried = retry_background_job(job["job_id"])
    assert retried["status"] == "queued"
    assert retried["attempts"] == 0
    assert retried["retry_after"] is None
    assert retried["error_message"] is None


def test_background_job_cancel_queued_job(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    job = enqueue_background_job(
        job_type="kb_ingest",
        payload={"kb_id": 1},
        context=_admin_context(),
    )

    cancelled = cancel_background_job(job["job_id"], reason="wrong source")

    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_requested_at"]
    assert cancelled["cancelled_at"]
    assert cancelled["cancel_reason"] == "wrong source"
    assert run(run_due_background_jobs_once()) is False


def test_background_job_cancel_and_retry_routes(isolated_client):
    admin = admin_headers()
    job = enqueue_background_job(
        job_type="kb_ingest",
        payload={"kb_id": 1},
        context=_admin_context(),
    )
    job_id = job["job_id"]
    database.execute_sync(
        """
        UPDATE background_jobs
        SET status = 'retrying', retry_after = '2999-01-01T00:00:00+00:00'
        WHERE job_id = ?
        """,
        (job_id,),
    )

    cancel = isolated_client.post(
        f"/api/admin/background-jobs/{job_id}/cancel",
        headers=admin,
        json={"reason": "manual test"},
    )
    assert cancel.status_code == 200, cancel.text
    assert cancel.json()["status"] == "cancelled"

    retry = isolated_client.post(
        f"/api/admin/background-jobs/{job_id}/retry",
        headers=admin,
    )
    assert retry.status_code == 200, retry.text
    assert retry.json()["status"] == "queued"


def test_stale_running_job_is_recovered_for_retry(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    job = enqueue_background_job(
        job_type="unsupported_transient_job",
        payload={},
        context=_admin_context(),
        max_attempts=2,
    )

    claimed = claim_next_background_job(worker_id="worker-a")
    assert claimed is not None
    assert claimed["job_id"] == job["job_id"]
    assert claimed["status"] == "running"
    assert claimed["worker_id"] == "worker-a"
    assert claimed["heartbeat_at"]

    database.execute_sync(
        """
        UPDATE background_jobs
        SET heartbeat_at = '2000-01-01T00:00:00+00:00'
        WHERE job_id = ?
        """,
        (job["job_id"],),
    )
    recovered = recover_stale_background_jobs(stale_seconds=1)
    assert recovered == 1

    retrying = get_background_job(job["job_id"])
    assert retrying["status"] == "retrying"
    assert retrying["worker_id"] is None
    assert retrying["retry_after"]

    reclaimed = claim_next_background_job(worker_id="worker-b")
    assert reclaimed is not None
    assert reclaimed["job_id"] == job["job_id"]
    assert reclaimed["worker_id"] == "worker-b"


def test_stale_cancelling_job_becomes_cancelled(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    job = enqueue_background_job(
        job_type="kb_ingest",
        payload={"kb_id": 1},
        context=_admin_context(),
    )
    claimed = claim_next_background_job(worker_id="worker-a")
    assert claimed is not None
    cancel_background_job(job["job_id"], reason="stop running job")
    database.execute_sync(
        """
        UPDATE background_jobs
        SET heartbeat_at = '2000-01-01T00:00:00+00:00'
        WHERE job_id = ?
        """,
        (job["job_id"],),
    )

    recovered = recover_stale_background_jobs(stale_seconds=1)
    assert recovered == 1
    cancelled = get_background_job(job["job_id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["worker_id"] is None
    assert cancelled["cancelled_at"]
