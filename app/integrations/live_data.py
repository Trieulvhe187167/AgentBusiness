"""
External API adapters with SQLite-backed snapshot/cache fallback.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import settings
from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.tools.registry import ToolExecutionError


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_fresh(cached_at: str | None) -> bool:
    parsed = _parse_iso_timestamp(cached_at)
    if parsed is None:
        return False
    age_seconds = (datetime.now(timezone.utc) - parsed).total_seconds()
    return age_seconds <= settings.integration_cache_ttl_seconds


def _auth_headers(api_key: str) -> dict[str, str]:
    if not api_key.strip():
        return {}
    return {"Authorization": f"Bearer {api_key.strip()}"}


def _extract_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), (dict, list)):
            return payload["data"]
        if isinstance(payload.get("result"), (dict, list)):
            return payload["result"]
    return payload


def _normalize_order_payload(payload: dict[str, Any]) -> dict[str, Any]:
    order_code = str(payload.get("order_code") or payload.get("code") or "").strip()
    if not order_code:
        raise ToolExecutionError("Order payload is missing order_code")
    return {
        "order_code": order_code,
        "user_id": str(payload.get("user_id") or "").strip() or None,
        "tenant_id": str(payload.get("tenant_id") or "").strip() or None,
        "org_id": str(payload.get("org_id") or "").strip() or None,
        "status": str(payload.get("status") or "").strip() or "unknown",
        "last_update": str(payload.get("last_update") or payload.get("updated_at") or "").strip() or None,
        "tracking_code": str(payload.get("tracking_code") or "").strip() or None,
        "carrier": str(payload.get("carrier") or "").strip() or None,
    }


def _normalize_online_payload(payload: dict[str, Any], *, alliance_id: str, server_id: str | None) -> dict[str, Any]:
    online_count = payload.get("online_count", payload.get("count"))
    if online_count is None:
        raise ToolExecutionError("Online payload is missing online_count")
    return {
        "alliance_id": alliance_id,
        "server_id": server_id,
        "online_count": int(online_count),
        "observed_at": str(payload.get("observed_at") or payload.get("updated_at") or utcnow_iso()),
    }


def _upsert_order_cache(payload: dict[str, Any], *, source: str, raw_payload: dict[str, Any] | None = None) -> None:
    now = utcnow_iso()
    execute_sync(
        """
        INSERT INTO order_status_cache (
            order_code, user_id, status, last_update, tracking_code, carrier,
            tenant_id, org_id, source, raw_json, cached_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(order_code) DO UPDATE SET
            user_id=excluded.user_id,
            status=excluded.status,
            last_update=excluded.last_update,
            tracking_code=excluded.tracking_code,
            carrier=excluded.carrier,
            tenant_id=excluded.tenant_id,
            org_id=excluded.org_id,
            source=excluded.source,
            raw_json=excluded.raw_json,
            cached_at=excluded.cached_at,
            updated_at=excluded.updated_at
        """,
        (
            payload["order_code"],
            payload.get("user_id"),
            payload["status"],
            payload.get("last_update"),
            payload.get("tracking_code"),
            payload.get("carrier"),
            payload.get("tenant_id"),
            payload.get("org_id"),
            source,
            json.dumps(raw_payload or payload, ensure_ascii=False),
            now,
            now,
        ),
    )


def _upsert_game_cache(payload: dict[str, Any], *, source: str, raw_payload: dict[str, Any] | None = None) -> None:
    now = utcnow_iso()
    server_scope = str(payload.get("server_id") or "").strip()
    execute_sync(
        """
        INSERT INTO game_online_cache (
            alliance_id, server_id, server_scope, online_count, observed_at,
            source, raw_json, cached_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(alliance_id, server_scope) DO UPDATE SET
            server_id=excluded.server_id,
            online_count=excluded.online_count,
            observed_at=excluded.observed_at,
            source=excluded.source,
            raw_json=excluded.raw_json,
            cached_at=excluded.cached_at,
            updated_at=excluded.updated_at
        """,
        (
            payload["alliance_id"],
            payload.get("server_id"),
            server_scope,
            int(payload["online_count"]),
            payload["observed_at"],
            source,
            json.dumps(raw_payload or payload, ensure_ascii=False),
            now,
            now,
        ),
    )


def _serialize_order_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_code": row["order_code"],
        "user_id": row.get("user_id"),
        "tenant_id": row.get("tenant_id"),
        "org_id": row.get("org_id"),
        "status": row["status"],
        "last_update": row.get("last_update"),
        "tracking_code": row.get("tracking_code"),
        "carrier": row.get("carrier"),
        "source": row.get("source") or "snapshot",
    }


def _serialize_online_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "alliance_id": row["alliance_id"],
        "server_id": row.get("server_id"),
        "online_count": int(row.get("online_count") or 0),
        "observed_at": row["observed_at"],
        "source": row.get("source") or "snapshot",
    }


async def _http_get_json(base_url: str, path: str, *, api_key: str, params: dict[str, Any]) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    timeout = settings.integration_http_timeout_seconds
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, params=params, headers=_auth_headers(api_key))
        response.raise_for_status()
        return response.json()


async def get_order_status(order_code: str, *, user_id: str | None = None) -> dict[str, Any]:
    row = fetch_one_sync(
        """
        SELECT order_code, user_id, tenant_id, org_id, status, last_update, tracking_code, carrier, source, cached_at
        FROM order_status_cache
        WHERE order_code = ?
        """,
        (order_code,),
    )
    if row and (_is_fresh(row.get("cached_at")) or not settings.order_api_base_url.strip()):
        return _serialize_order_row(dict(row))

    if settings.order_api_base_url.strip():
        payload = _extract_payload(
            await _http_get_json(
                settings.order_api_base_url,
                settings.order_api_status_path,
                api_key=settings.order_api_key,
                params={"order_code": order_code, "user_id": user_id},
            )
        )
        if not isinstance(payload, dict):
            raise ToolExecutionError("Order status API returned an invalid payload")
        normalized = _normalize_order_payload(payload)
        _upsert_order_cache(normalized, source="api", raw_payload=payload)
        return {**normalized, "source": "api"}

    if row:
        return _serialize_order_row(dict(row))

    raise ToolExecutionError("Order not found and no order API is configured")


async def find_recent_orders(user_id: str, *, limit: int = 5) -> dict[str, Any]:
    rows = fetch_all_sync(
        """
        SELECT order_code, user_id, tenant_id, org_id, status, last_update, tracking_code, carrier, source, cached_at
        FROM order_status_cache
        WHERE user_id = ?
        ORDER BY COALESCE(last_update, cached_at) DESC, id DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    snapshot_orders = [_serialize_order_row(dict(row)) for row in rows]
    snapshot_fresh = bool(rows) and all(_is_fresh(row.get("cached_at")) for row in rows)

    if rows and (snapshot_fresh or not settings.order_api_base_url.strip()):
        return {
            "user_id": user_id,
            "total": len(rows),
            "orders": snapshot_orders,
            "source": "snapshot",
        }

    if settings.order_api_base_url.strip():
        try:
            payload = _extract_payload(
                await _http_get_json(
                    settings.order_api_base_url,
                    settings.order_api_recent_path,
                    api_key=settings.order_api_key,
                    params={"user_id": user_id, "limit": limit},
                )
            )
            if isinstance(payload, dict):
                items = payload.get("orders") or payload.get("items") or []
            elif isinstance(payload, list):
                items = payload
            else:
                raise ToolExecutionError("Recent orders API returned an invalid payload")

            normalized_orders: list[dict[str, Any]] = []
            for item in items[:limit]:
                if not isinstance(item, dict):
                    continue
                normalized = _normalize_order_payload(item)
                normalized["user_id"] = normalized.get("user_id") or user_id
                _upsert_order_cache(normalized, source="api", raw_payload=item)
                normalized_orders.append({**normalized, "source": "api"})

            return {
                "user_id": user_id,
                "total": len(normalized_orders),
                "orders": normalized_orders,
                "source": "api",
            }
        except Exception:
            if rows:
                return {
                    "user_id": user_id,
                    "total": len(rows),
                    "orders": snapshot_orders,
                    "source": "snapshot",
                }
            raise

    return {
        "user_id": user_id,
        "total": 0,
        "orders": [],
        "source": "snapshot",
    }


async def get_online_member_count(alliance_id: str, *, server_id: str | None = None) -> dict[str, Any]:
    row = fetch_one_sync(
        """
        SELECT alliance_id, server_id, online_count, observed_at, source, cached_at
        FROM game_online_cache
        WHERE alliance_id = ? AND server_scope = ?
        """,
        (alliance_id, str(server_id or "")),
    )
    if row and (_is_fresh(row.get("cached_at")) or not settings.game_api_base_url.strip()):
        return _serialize_online_row(dict(row))

    if settings.game_api_base_url.strip():
        payload = _extract_payload(
            await _http_get_json(
                settings.game_api_base_url,
                settings.game_api_online_path,
                api_key=settings.game_api_key,
                params={"alliance_id": alliance_id, "server_id": server_id},
            )
        )
        if not isinstance(payload, dict):
            raise ToolExecutionError("Game online API returned an invalid payload")
        normalized = _normalize_online_payload(payload, alliance_id=alliance_id, server_id=server_id)
        _upsert_game_cache(normalized, source="api", raw_payload=payload)
        return {**normalized, "source": "api"}

    if row:
        return _serialize_online_row(dict(row))

    raise ToolExecutionError("Alliance online count not found and no game API is configured")
