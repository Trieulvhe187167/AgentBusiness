from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def _poll_jobs(client: httpx.Client, base_url: str, jobs: list[dict], out):
    pending = {job["job_id"] for job in jobs}
    while pending:
        for job_id in list(pending):
            response = client.get(f"{base_url}/api/jobs/{job_id}", timeout=60.0)
            response.raise_for_status()
            payload = response.json()
            out.write(f"Job {job_id}: {json.dumps(payload, ensure_ascii=False)}\n")
            if payload["status"] in {"done", "failed"}:
                pending.remove(job_id)
                if payload["status"] == "failed":
                    raise RuntimeError(f"Job {job_id} failed: {payload.get('error_message')}")
        if pending:
            time.sleep(2)


def _read_sse_lines(client: httpx.Client, base_url: str, payload: dict, out):
    with client.stream("POST", f"{base_url}/api/chat", json=payload, timeout=60.0) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                out.write(f"{line}\n")


def main():
    parser = argparse.ArgumentParser(description="Run a live smoke flow against the local RAG API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="API base URL")
    parser.add_argument("--kb-id", type=int, default=None, help="Optional KB id to ingest/chat against")
    parser.add_argument("--kb-key", default=None, help="Optional KB key to resolve first")
    parser.add_argument("--message", default="Phí giao hàng là bao nhiêu?", help="Chat question")
    parser.add_argument("--lang", default="vi", help="Language hint")
    parser.add_argument("--output", default="chat_output.txt", help="Output log file")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    output_path = Path(args.output)

    with httpx.Client() as client, output_path.open("w", encoding="utf-8") as out:
        kb = None
        if args.kb_id is not None:
            response = client.get(f"{base_url}/api/kbs/{args.kb_id}", timeout=30.0)
            response.raise_for_status()
            kb = response.json()
        elif args.kb_key:
            response = client.get(f"{base_url}/api/kb/stats", params={"kb_key": args.kb_key}, timeout=30.0)
            response.raise_for_status()
            stats = response.json()
            kb = {
                "id": stats["kb_id"],
                "key": stats["kb_key"],
                "name": stats["kb_name"],
            }
        else:
            response = client.get(f"{base_url}/api/kbs/default", timeout=30.0)
            response.raise_for_status()
            kb = response.json()

        out.write(f"Using KB: {json.dumps(kb, ensure_ascii=False)}\n")

        ingest_endpoint = f"{base_url}/api/kbs/{kb['id']}/ingest"
        response = client.post(ingest_endpoint, timeout=60.0)
        response.raise_for_status()
        ingest_payload = response.json()
        out.write(f"Ingest response: {json.dumps(ingest_payload, ensure_ascii=False)}\n")

        jobs = ingest_payload.get("jobs") or []
        if jobs:
            _poll_jobs(client, base_url, jobs, out)

        stats_response = client.get(f"{base_url}/api/kb/stats", params={"kb_id": kb["id"]}, timeout=30.0)
        stats_response.raise_for_status()
        out.write(f"KB stats: {json.dumps(stats_response.json(), ensure_ascii=False)}\n")

        out.write("\nTesting chat...\n")
        chat_payload = {
            "session_id": "test_session_1",
            "message": args.message,
            "lang": args.lang,
            "kb_id": kb["id"],
        }
        _read_sse_lines(client, base_url, chat_payload, out)


if __name__ == "__main__":
    main()
