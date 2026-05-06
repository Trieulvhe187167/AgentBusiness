"""
In-process fixed-window rate limiting for API and MCP endpoints.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from fastapi import Request

from app.config import settings


@dataclass(slots=True)
class RateLimitDecision:
    allowed: bool
    policy: str
    limit: int
    remaining: int
    reset_after_seconds: int
    retry_after_seconds: int = 0


@dataclass(slots=True)
class _Bucket:
    count: int
    reset_at: float


class RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, _Bucket] = {}

    def reset(self) -> None:
        self._buckets.clear()

    def check(self, request: Request) -> RateLimitDecision | None:
        policy = self._policy_for_request(request)
        if policy is None:
            return None

        policy_name, limit = policy
        limit = max(1, int(limit))
        window_seconds = max(1, int(settings.rate_limit_window_seconds))
        now = time.monotonic()
        bucket_key = self._bucket_key(request, policy_name)
        bucket = self._buckets.get(bucket_key)

        if bucket is None or bucket.reset_at <= now:
            bucket = _Bucket(count=0, reset_at=now + window_seconds)
            self._buckets[bucket_key] = bucket
            self._cleanup(now)

        reset_after = max(1, int(bucket.reset_at - now))
        if bucket.count >= limit:
            return RateLimitDecision(
                allowed=False,
                policy=policy_name,
                limit=limit,
                remaining=0,
                reset_after_seconds=reset_after,
                retry_after_seconds=reset_after,
            )

        bucket.count += 1
        return RateLimitDecision(
            allowed=True,
            policy=policy_name,
            limit=limit,
            remaining=max(0, limit - bucket.count),
            reset_after_seconds=reset_after,
        )

    def _policy_for_request(self, request: Request) -> tuple[str, int] | None:
        if not settings.rate_limit_enabled:
            return None

        path = request.url.path
        method = request.method.upper()
        if self._is_exempt(path):
            return None

        if path == "/api/chat":
            return ("chat", settings.rate_limit_chat_requests_per_window)
        if path == "/mcp":
            return ("mcp", settings.rate_limit_mcp_requests_per_window)
        if path in {"/api/upload", "/api/admin/upload"}:
            return ("upload", settings.rate_limit_upload_requests_per_window)
        if self._is_sync_or_ingest_mutation(path, method):
            return ("sync", settings.rate_limit_sync_requests_per_window)
        if path.startswith("/api/admin") or self._is_kb_or_file_mutation(path, method):
            return ("admin", settings.rate_limit_admin_requests_per_window)
        return ("default", settings.rate_limit_default_requests_per_window)

    def _is_exempt(self, path: str) -> bool:
        if path.startswith("/static/"):
            return True
        configured = settings.rate_limit_exempt_paths_set
        return path in configured

    @staticmethod
    def _is_sync_or_ingest_mutation(path: str, method: str) -> bool:
        if method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return False
        sync_markers = (
            "/sync",
            "/ingest",
            "/reindex",
            "/sync-schedules",
            "/background-jobs",
        )
        return any(marker in path for marker in sync_markers)

    @staticmethod
    def _is_kb_or_file_mutation(path: str, method: str) -> bool:
        if method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return False
        return path.startswith("/api/kbs") or path.startswith("/api/files")

    def _bucket_key(self, request: Request, policy_name: str) -> str:
        identity = self._identity_for_request(request)
        return f"{policy_name}:{identity}"

    def _identity_for_request(self, request: Request) -> str:
        user_id = self._first_header(
            request,
            "X-User-Id",
            settings.gateway_user_id_header,
        )
        tenant_id = self._first_header(
            request,
            "X-Tenant-Id",
            settings.gateway_tenant_id_header,
        )
        org_id = self._first_header(
            request,
            "X-Org-Id",
            settings.gateway_org_id_header,
        )
        if user_id:
            scope = ":".join(part for part in [tenant_id, org_id, user_id] if part)
            return f"user:{scope or user_id}"

        auth = (request.headers.get("Authorization") or "").strip()
        if auth:
            digest = hashlib.sha256(auth.encode("utf-8")).hexdigest()[:20]
            return f"auth:{digest}"

        forwarded_for = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        ip = forwarded_for or (request.client.host if request.client else "unknown")
        return f"ip:{ip}"

    @staticmethod
    def _first_header(request: Request, *names: str) -> str | None:
        for name in names:
            value = (request.headers.get(name) or "").strip()
            if value:
                return value
        return None

    def _cleanup(self, now: float) -> None:
        if len(self._buckets) <= settings.rate_limit_max_buckets:
            return
        expired = [key for key, bucket in self._buckets.items() if bucket.reset_at <= now]
        for key in expired:
            self._buckets.pop(key, None)


def rate_limit_headers(decision: RateLimitDecision | None) -> dict[str, str]:
    if decision is None:
        return {}
    headers = {
        "X-RateLimit-Policy": decision.policy,
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
        "X-RateLimit-Reset": str(decision.reset_after_seconds),
    }
    if not decision.allowed:
        headers["Retry-After"] = str(decision.retry_after_seconds)
    return headers


def rate_limit_error_payload(decision: RateLimitDecision) -> dict[str, Any]:
    return {
        "detail": "Rate limit exceeded",
        "policy": decision.policy,
        "limit": decision.limit,
        "retry_after_seconds": decision.retry_after_seconds,
    }


rate_limiter = RateLimiter()
