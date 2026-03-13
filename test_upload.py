from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description="Upload a file to the local RAG API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="API base URL")
    parser.add_argument("--file", default="kb_sample.csv", help="Path to the file to upload")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")

    with file_path.open("rb") as handle:
        files = {"file": (file_path.name, handle, "text/csv")}
        response = httpx.post(f"{args.base_url.rstrip('/')}/api/upload", files=files, timeout=60.0)

    print("Status:", response.status_code)
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)

    response.raise_for_status()


if __name__ == "__main__":
    main()
