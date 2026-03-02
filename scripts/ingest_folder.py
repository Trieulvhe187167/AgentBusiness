"""Batch upload + ingest utility for local/product deployments."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a folder and trigger ingest")
    parser.add_argument("--input", required=True, help="Folder containing files")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="API base URL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base_url.rstrip("/")
    folder = Path(args.input)

    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Input folder does not exist: {folder}")

    files = [p for p in folder.iterdir() if p.is_file()]
    if not files:
        raise SystemExit("No files found")

    uploaded = 0
    with httpx.Client(timeout=60) as client:
        for file_path in files:
            with file_path.open("rb") as fh:
                response = client.post(
                    f"{base}/api/upload",
                    files={"file": (file_path.name, fh)},
                )
            if response.status_code >= 400:
                print(f"[skip] {file_path.name}: {response.text[:200]}")
                continue

            uploaded += 1
            print(f"[ok] uploaded {file_path.name}")

        ingest = client.post(f"{base}/api/ingest/all")
        if ingest.status_code >= 400:
            raise SystemExit(f"Ingest failed: {ingest.text}")

    print(f"Uploaded {uploaded}/{len(files)} files and queued ingest jobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
