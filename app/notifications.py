"""
Notification center and outbound webhook delivery.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.config import settings
from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.models import AuthContext, RequestContext


class NotificationItem(BaseModel):
    id: int
    event_type: str
    severity: str
    status: str
    title: str
    message: str
    entity_type: str | None = None
    entity_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_by_user_id: str | None = None
    read_by_user_id: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    kb_id: int | None = None
    kb_key: str | None = None
    created_at: str
    read_at: str | None = None


class ListNotificationsOutput(BaseModel):
    total: int
    unread_count: int
    items: list[NotificationItem]


class WebhookDeliveryItem(BaseModel):
    id: int
    notification_id: int
    subscription_id: int | None = None
    event_type: str
    endpoint_url: str
    status: str
    attempts: int
    response_status: int | None = None
    response_body: str | None = None
    error_message: str | None = None
    last_attempt_at: str | None = None
    next_retry_at: str | None = None
    created_at: str
    updated_at: str


class ListWebhookDeliveriesOutput(BaseModel):
    total: int
    items: list[WebhookDeliveryItem]


class WebhookSubscriptionItem(BaseModel):
    id: int
    name: str
    endpoint_url: str
    event_types: list[str] = Field(default_factory=list)
    enabled: bool = True
    has_secret: bool = False
    created_by_user_id: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    last_test_at: str | None = None
    created_at: str
    updated_at: str


class ListWebhookSubscriptionsOutput(BaseModel):
    total: int
    items: list[WebhookSubscriptionItem]


class UpsertWebhookSubscriptionInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=160)
    endpoint_url: str = Field(..., min_length=8, max_length=1000)
    secret: str | None = Field(default=None, max_length=500)
    event_types: list[str] = Field(default_factory=list)
    enabled: bool = True


class CreateNotificationInput(BaseModel):
    event_type: str = Field(..., min_length=1, max_length=120)
    severity: str = Field(default="info", min_length=1, max_length=40)
    title: str = Field(..., min_length=1, max_length=240)
    message: str = Field(default="", max_length=2000)
    entity_type: str | None = Field(default=None, max_length=80)
    entity_id: str | None = Field(default=None, max_length=160)
    payload: dict[str, Any] = Field(default_factory=dict)


def _parse_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _json_list_dumps(value: list[str]) -> str:
    cleaned = sorted({str(item).strip() for item in value if str(item).strip()})
    return json.dumps(cleaned, ensure_ascii=False)


def _event_matches_subscription(event_type: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    event = event_type.strip()
    for pattern in patterns:
        item = pattern.strip()
        if not item or item == "*":
            return True
        if item.endswith("*") and event.startswith(item[:-1]):
            return True
        if item == event:
            return True
    return False


def _normalize_severity(value: str) -> str:
    severity = str(value or "info").strip().lower()
    return severity if severity in {"info", "success", "warning", "critical"} else "info"


def _serialize_notification(row: dict[str, Any]) -> NotificationItem:
    return NotificationItem(
        id=int(row["id"]),
        event_type=row["event_type"],
        severity=row["severity"],
        status=row["status"],
        title=row["title"],
        message=row.get("message") or "",
        entity_type=row.get("entity_type"),
        entity_id=row.get("entity_id"),
        payload=_parse_json(row.get("payload_json")),
        created_by_user_id=row.get("created_by_user_id"),
        read_by_user_id=row.get("read_by_user_id"),
        tenant_id=row.get("tenant_id"),
        org_id=row.get("org_id"),
        kb_id=row.get("kb_id"),
        kb_key=row.get("kb_key"),
        created_at=row["created_at"],
        read_at=row.get("read_at"),
    )


def _serialize_delivery(row: dict[str, Any]) -> WebhookDeliveryItem:
    return WebhookDeliveryItem(
        id=int(row["id"]),
        notification_id=int(row["notification_id"]),
        subscription_id=row.get("subscription_id"),
        event_type=row["event_type"],
        endpoint_url=row["endpoint_url"],
        status=row["status"],
        attempts=int(row.get("attempts") or 0),
        response_status=row.get("response_status"),
        response_body=row.get("response_body"),
        error_message=row.get("error_message"),
        last_attempt_at=row.get("last_attempt_at"),
        next_retry_at=row.get("next_retry_at"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _serialize_subscription(row: dict[str, Any]) -> WebhookSubscriptionItem:
    return WebhookSubscriptionItem(
        id=int(row["id"]),
        name=row["name"],
        endpoint_url=row["endpoint_url"],
        event_types=_parse_json_list(row.get("event_types_json")),
        enabled=bool(row.get("enabled")),
        has_secret=bool(row.get("secret")),
        created_by_user_id=row.get("created_by_user_id"),
        tenant_id=row.get("tenant_id"),
        org_id=row.get("org_id"),
        last_test_at=row.get("last_test_at"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _notification_payload(item: NotificationItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "event_type": item.event_type,
        "severity": item.severity,
        "title": item.title,
        "message": item.message,
        "entity_type": item.entity_type,
        "entity_id": item.entity_id,
        "payload": item.payload,
        "tenant_id": item.tenant_id,
        "org_id": item.org_id,
        "kb_id": item.kb_id,
        "kb_key": item.kb_key,
        "created_at": item.created_at,
    }


def get_notification(notification_id: int) -> NotificationItem:
    row = fetch_one_sync("SELECT * FROM notifications WHERE id = ?", (notification_id,))
    if not row:
        raise ValueError("Notification not found")
    return _serialize_notification(row)


def list_notifications(*, status: str | None = None, severity: str | None = None, limit: int = 50) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if severity:
        clauses.append("severity = ?")
        params.append(_normalize_severity(severity))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = fetch_all_sync(
        f"""
        SELECT * FROM notifications
        {where}
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (*params, max(1, min(limit, 200))),
    )
    unread = fetch_one_sync("SELECT COUNT(*) AS count FROM notifications WHERE status = 'unread'")
    items = [_serialize_notification(row).model_dump() for row in rows]
    return {"total": len(items), "unread_count": int((unread or {}).get("count") or 0), "items": items}


def list_webhook_deliveries(*, status: str | None = None, limit: int = 50) -> dict[str, Any]:
    if status:
        rows = fetch_all_sync(
            """
            SELECT * FROM webhook_deliveries
            WHERE status = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (status, max(1, min(limit, 200))),
        )
    else:
        rows = fetch_all_sync(
            """
            SELECT * FROM webhook_deliveries
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        )
    items = [_serialize_delivery(row).model_dump() for row in rows]
    return {"total": len(items), "items": items}


def list_webhook_subscriptions(*, include_disabled: bool = True, limit: int = 100) -> dict[str, Any]:
    if include_disabled:
        rows = fetch_all_sync(
            """
            SELECT * FROM webhook_subscriptions
            ORDER BY enabled DESC, updated_at DESC, id DESC
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        )
    else:
        rows = fetch_all_sync(
            """
            SELECT * FROM webhook_subscriptions
            WHERE enabled = 1
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        )
    items = [_serialize_subscription(row).model_dump() for row in rows]
    return {"total": len(items), "items": items}


def get_webhook_subscription(subscription_id: int) -> WebhookSubscriptionItem:
    row = fetch_one_sync("SELECT * FROM webhook_subscriptions WHERE id = ?", (subscription_id,))
    if not row:
        raise ValueError("Webhook subscription not found")
    return _serialize_subscription(row)


def create_webhook_subscription(payload: UpsertWebhookSubscriptionInput, *, auth: AuthContext) -> dict[str, Any]:
    now = utcnow_iso()
    subscription_id = execute_sync(
        """
        INSERT INTO webhook_subscriptions (
            name, endpoint_url, secret, event_types_json, enabled,
            created_by_user_id, tenant_id, org_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.name.strip(),
            payload.endpoint_url.strip(),
            payload.secret.strip() if payload.secret else None,
            _json_list_dumps(payload.event_types),
            1 if payload.enabled else 0,
            auth.user_id,
            auth.tenant_id,
            auth.org_id,
            now,
            now,
        ),
    )
    return get_webhook_subscription(int(subscription_id or 0)).model_dump()


def update_webhook_subscription(subscription_id: int, payload: UpsertWebhookSubscriptionInput) -> dict[str, Any]:
    existing = fetch_one_sync("SELECT * FROM webhook_subscriptions WHERE id = ?", (subscription_id,))
    if not existing:
        raise ValueError("Webhook subscription not found")
    now = utcnow_iso()
    secret = existing.get("secret") if payload.secret is None else payload.secret.strip()
    execute_sync(
        """
        UPDATE webhook_subscriptions
        SET name = ?,
            endpoint_url = ?,
            secret = ?,
            event_types_json = ?,
            enabled = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            payload.name.strip(),
            payload.endpoint_url.strip(),
            secret,
            _json_list_dumps(payload.event_types),
            1 if payload.enabled else 0,
            now,
            subscription_id,
        ),
    )
    return get_webhook_subscription(subscription_id).model_dump()


def delete_webhook_subscription(subscription_id: int) -> dict[str, Any]:
    existing = get_webhook_subscription(subscription_id)
    execute_sync("DELETE FROM webhook_subscriptions WHERE id = ?", (subscription_id,))
    return {"deleted": True, "id": subscription_id, "name": existing.name}


def _create_webhook_delivery(notification: NotificationItem) -> None:
    if not settings.notification_webhook_enabled:
        return
    now = utcnow_iso()
    payload_json = _json_dumps(_notification_payload(notification))
    subscriptions = fetch_all_sync(
        """
        SELECT id, endpoint_url, event_types_json
        FROM webhook_subscriptions
        WHERE enabled = 1
        ORDER BY id ASC
        """
    )
    for subscription in subscriptions:
        if not _event_matches_subscription(notification.event_type, _parse_json_list(subscription.get("event_types_json"))):
            continue
        execute_sync(
            """
            INSERT INTO webhook_deliveries (
                notification_id, subscription_id, event_type, endpoint_url, status, request_json,
                attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', ?, 0, ?, ?)
            """,
            (
                notification.id,
                subscription["id"],
                notification.event_type,
                subscription["endpoint_url"],
                payload_json,
                now,
                now,
            ),
        )
    if subscriptions or not settings.notification_webhook_url.strip():
        return
    execute_sync(
        """
        INSERT INTO webhook_deliveries (
            notification_id, event_type, endpoint_url, status, request_json,
            attempts, created_at, updated_at
        ) VALUES (?, ?, ?, 'pending', ?, 0, ?, ?)
        """,
        (
            notification.id,
            notification.event_type,
            settings.notification_webhook_url.strip(),
            payload_json,
            now,
            now,
        ),
    )


def create_notification(
    *,
    event_type: str,
    title: str,
    message: str = "",
    severity: str = "info",
    entity_type: str | None = None,
    entity_id: str | int | None = None,
    payload: dict[str, Any] | None = None,
    context: RequestContext | None = None,
    auth: AuthContext | None = None,
) -> dict[str, Any]:
    resolved_auth = auth or (context.auth if context else AuthContext())
    now = utcnow_iso()
    notification_id = execute_sync(
        """
        INSERT INTO notifications (
            event_type, severity, status, title, message, entity_type, entity_id,
            payload_json, created_by_user_id, tenant_id, org_id, kb_id, kb_key,
            created_at
        ) VALUES (?, ?, 'unread', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_type.strip(),
            _normalize_severity(severity),
            title.strip(),
            message.strip(),
            entity_type,
            str(entity_id) if entity_id is not None else None,
            _json_dumps(payload or {}),
            resolved_auth.user_id,
            resolved_auth.tenant_id,
            resolved_auth.org_id,
            context.kb_id if context else None,
            context.kb_key if context else None,
            now,
        ),
    )
    item = get_notification(int(notification_id or 0))
    _create_webhook_delivery(item)
    return item.model_dump()


def mark_notification_read(notification_id: int, *, auth: AuthContext) -> dict[str, Any]:
    now = utcnow_iso()
    execute_sync(
        """
        UPDATE notifications
        SET status = 'read',
            read_by_user_id = ?,
            read_at = ?
        WHERE id = ?
        """,
        (auth.user_id, now, notification_id),
    )
    return get_notification(notification_id).model_dump()


def mark_all_notifications_read(*, auth: AuthContext) -> dict[str, Any]:
    now = utcnow_iso()
    execute_sync(
        """
        UPDATE notifications
        SET status = 'read',
            read_by_user_id = COALESCE(read_by_user_id, ?),
            read_at = COALESCE(read_at, ?)
        WHERE status = 'unread'
        """,
        (auth.user_id, now),
    )
    return list_notifications(status="unread", limit=1)


def _next_retry_after(attempts: int) -> str:
    delay_seconds = min(900, 30 * (2 ** max(0, attempts - 1)))
    return (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat()


def _signature(payload: bytes) -> str | None:
    secret = settings.notification_webhook_secret.strip()
    if not secret:
        return None
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _signature_for_secret(payload: bytes, secret: str) -> str | None:
    if not secret:
        return None
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _delivery_secret(row: dict[str, Any]) -> str:
    if row.get("subscription_id"):
        subscription = fetch_one_sync("SELECT secret FROM webhook_subscriptions WHERE id = ?", (row["subscription_id"],))
        if subscription and subscription.get("secret"):
            return str(subscription["secret"]).strip()
    return settings.notification_webhook_secret.strip()


async def deliver_webhook_delivery(delivery_id: int) -> dict[str, Any]:
    row = fetch_one_sync("SELECT * FROM webhook_deliveries WHERE id = ?", (delivery_id,))
    if not row:
        raise ValueError("Webhook delivery not found")
    delivery = _serialize_delivery(row)
    payload = row.get("request_json") or "{}"
    payload_bytes = payload.encode("utf-8")
    headers = {"Content-Type": "application/json", "X-AgentBusiness-Event": delivery.event_type}
    signature = _signature_for_secret(payload_bytes, _delivery_secret(row))
    if signature:
        headers["X-AgentBusiness-Signature"] = signature

    now = utcnow_iso()
    try:
        async with httpx.AsyncClient(timeout=settings.notification_webhook_timeout_seconds) as client:
            response = await client.post(delivery.endpoint_url, content=payload_bytes, headers=headers)
        status = "sent" if 200 <= response.status_code < 300 else "failed"
        execute_sync(
            """
            UPDATE webhook_deliveries
            SET status = ?,
                attempts = attempts + 1,
                response_status = ?,
                response_body = ?,
                error_message = ?,
                last_attempt_at = ?,
                next_retry_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                status,
                response.status_code,
                response.text[:2000],
                None if status == "sent" else f"HTTP {response.status_code}",
                now,
                None if status == "sent" else _next_retry_after(delivery.attempts + 1),
                now,
                delivery_id,
            ),
        )
    except Exception as err:
        execute_sync(
            """
            UPDATE webhook_deliveries
            SET status = 'failed',
                attempts = attempts + 1,
                error_message = ?,
                last_attempt_at = ?,
                next_retry_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (str(err), now, _next_retry_after(delivery.attempts + 1), now, delivery_id),
        )
    return _serialize_delivery(fetch_one_sync("SELECT * FROM webhook_deliveries WHERE id = ?", (delivery_id,))).model_dump()


async def deliver_due_webhooks_once(*, limit: int = 10) -> int:
    if not settings.notification_webhook_enabled:
        return 0
    now = utcnow_iso()
    rows = fetch_all_sync(
        """
        SELECT id FROM webhook_deliveries
        WHERE status IN ('pending', 'failed')
          AND attempts < ?
          AND (next_retry_at IS NULL OR next_retry_at <= ?)
        ORDER BY created_at ASC, id ASC
        LIMIT ?
        """,
        (max(1, settings.notification_webhook_max_attempts), now, max(1, min(limit, 50))),
    )
    delivered = 0
    for row in rows:
        await deliver_webhook_delivery(int(row["id"]))
        delivered += 1
    return delivered


async def retry_webhook_delivery(delivery_id: int) -> dict[str, Any]:
    execute_sync(
        """
        UPDATE webhook_deliveries
        SET status = 'pending',
            next_retry_at = NULL,
            updated_at = ?
        WHERE id = ?
        """,
        (utcnow_iso(), delivery_id),
    )
    return await deliver_webhook_delivery(delivery_id)


def test_webhook_subscription(subscription_id: int, *, auth: AuthContext) -> dict[str, Any]:
    subscription = get_webhook_subscription(subscription_id)
    notification = create_notification(
        event_type="notification.webhook_test",
        severity="info",
        title=f"Webhook test: {subscription.name}",
        message="Webhook subscription test event from Admin UI.",
        entity_type="webhook_subscription",
        entity_id=subscription_id,
        payload={"subscription_id": subscription_id, "source": "admin_ui"},
        auth=auth,
    )
    now = utcnow_iso()
    execute_sync("UPDATE webhook_subscriptions SET last_test_at = ?, updated_at = ? WHERE id = ?", (now, now, subscription_id))
    rows = fetch_all_sync(
        """
        SELECT * FROM webhook_deliveries
        WHERE notification_id = ? AND subscription_id = ?
        ORDER BY id DESC
        """,
        (notification["id"], subscription_id),
    )
    if not rows:
        now = utcnow_iso()
        execute_sync(
            """
            INSERT INTO webhook_deliveries (
                notification_id, subscription_id, event_type, endpoint_url, status, request_json,
                attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', ?, 0, ?, ?)
            """,
            (
                notification["id"],
                subscription_id,
                notification["event_type"],
                subscription.endpoint_url,
                _json_dumps(_notification_payload(get_notification(notification["id"]))),
                now,
                now,
            ),
        )
        rows = fetch_all_sync(
            """
            SELECT * FROM webhook_deliveries
            WHERE notification_id = ? AND subscription_id = ?
            ORDER BY id DESC
            """,
            (notification["id"], subscription_id),
        )
    return {
        "notification": notification,
        "deliveries": [_serialize_delivery(row).model_dump() for row in rows],
    }
