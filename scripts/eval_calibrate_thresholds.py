"""Simple threshold calibration helper for answer/clarify/fallback modes."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate threshold_good/threshold_low")
    parser.add_argument("--queries", required=True, help="Text file with one query per line")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="API base URL")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query_file = Path(args.queries)
    if not query_file.exists():
        raise SystemExit(f"File not found: {query_file}")

    queries = [line.strip() for line in query_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not queries:
        raise SystemExit("No queries found")

    scores = []
    base = args.base_url.rstrip("/")

    with httpx.Client(timeout=30) as client:
        for query in queries:
            response = client.get(
                f"{base}/api/debug/similarity",
                params={"query": query, "top_k": args.top_k},
            )
            if response.status_code >= 400:
                print(f"[error] {query}: {response.text[:160]}")
                continue

            payload = response.json()
            score = float(payload.get("top_score", 0.0))
            scores.append(score)
            print(f"{score:0.4f}\t{payload.get('predicted_mode')}\t{query}")

    if not scores:
        raise SystemExit("No valid score returned")

    print("\nSummary")
    print(f"count={len(scores)}")
    print(f"min={min(scores):0.4f}")
    print(f"max={max(scores):0.4f}")
    print(f"mean={statistics.mean(scores):0.4f}")
    print(f"median={statistics.median(scores):0.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
