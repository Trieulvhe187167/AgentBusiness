"""
Business tools backed by external APIs with SQLite snapshot fallback.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.integrations.live_data import find_recent_orders, get_online_member_count, get_order_status
from app.models import RequestContext
from app.tools.registry import ToolAuthPolicy, ToolAuthorizationError, ToolSpec


class GetOrderStatusInput(BaseModel):
    order_code: str = Field(..., min_length=3, max_length=80)
    user_id: str | None = Field(default=None, min_length=1, max_length=120)


class GetOrderStatusOutput(BaseModel):
    order_code: str
    user_id: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    status: str
    last_update: str | None = None
    tracking_code: str | None = None
    carrier: str | None = None
    source: str


class RecentOrderItem(BaseModel):
    order_code: str
    user_id: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    status: str
    last_update: str | None = None
    tracking_code: str | None = None
    carrier: str | None = None
    source: str


class FindRecentOrdersInput(BaseModel):
    user_id: str | None = Field(default=None, min_length=1, max_length=120)
    limit: int = Field(default=5, ge=1, le=10)


class FindRecentOrdersOutput(BaseModel):
    user_id: str
    total: int
    orders: list[RecentOrderItem]
    source: str


class GetOnlineMemberCountInput(BaseModel):
    alliance_id: str = Field(..., min_length=1, max_length=80)
    server_id: str | None = Field(default=None, max_length=80)


class GetOnlineMemberCountOutput(BaseModel):
    alliance_id: str
    server_id: str | None = None
    online_count: int
    observed_at: str
    source: str


def _resolve_order_user_id(payload_user_id: str | None, context: RequestContext) -> str:
    requested_user_id = payload_user_id or context.auth.user_id
    if not requested_user_id:
        raise ToolAuthorizationError("Order lookup requires user_id")

    is_admin = "admin" in set(context.auth.roles)
    if payload_user_id and payload_user_id != context.auth.user_id and not is_admin:
        raise ToolAuthorizationError("You cannot access another user's orders")
    return requested_user_id


def _ensure_order_scope_match(payload: dict, context: RequestContext) -> None:
    if "admin" in set(context.auth.roles):
        return

    result_tenant = str(payload.get("tenant_id") or "").strip()
    result_org = str(payload.get("org_id") or "").strip()
    auth_tenant = str(context.auth.tenant_id or "").strip()
    auth_org = str(context.auth.org_id or "").strip()

    if auth_tenant and result_tenant and auth_tenant != result_tenant:
        raise ToolAuthorizationError("You cannot access another tenant's orders")
    if auth_org and result_org and auth_org != result_org:
        raise ToolAuthorizationError("You cannot access another org's orders")


async def _get_order_status_tool(payload: GetOrderStatusInput, context: RequestContext) -> dict:
    resolved_user_id = _resolve_order_user_id(payload.user_id, context)
    result = await get_order_status(payload.order_code.strip(), user_id=resolved_user_id)
    row_user_id = result.get("user_id")
    if row_user_id and row_user_id != resolved_user_id and "admin" not in set(context.auth.roles):
        raise ToolAuthorizationError("You cannot access another user's orders")
    _ensure_order_scope_match(result, context)
    return result


async def _find_recent_orders_tool(payload: FindRecentOrdersInput, context: RequestContext) -> dict:
    resolved_user_id = _resolve_order_user_id(payload.user_id, context)
    result = await find_recent_orders(resolved_user_id, limit=payload.limit)
    for item in result.get("orders", []):
        if isinstance(item, dict):
            _ensure_order_scope_match(item, context)
    return result


async def _get_online_member_count_tool(payload: GetOnlineMemberCountInput, _: RequestContext) -> dict:
    return await get_online_member_count(payload.alliance_id.strip(), server_id=payload.server_id)


def build_get_order_status_tool() -> ToolSpec:
    return ToolSpec(
        name="get_order_status",
        description="Get the live status of a user's order by order code.",
        input_model=GetOrderStatusInput,
        output_model=GetOrderStatusOutput,
        auth_policy=ToolAuthPolicy(
            require_user_id=True,
            allowed_channels=["web", "chat", "admin"],
            requires_tenant_match=True,
            risk_level="medium",
            scope="order",
        ),
        timeout_seconds=15,
        idempotent=True,
        handler=_get_order_status_tool,
        summarize_result=lambda payload: f"order {payload.get('order_code', '')} is {payload.get('status', 'unknown')}",
    )


def build_find_recent_orders_tool() -> ToolSpec:
    return ToolSpec(
        name="find_recent_orders",
        description="Find recent orders for the authenticated user when the order code is unknown.",
        input_model=FindRecentOrdersInput,
        output_model=FindRecentOrdersOutput,
        auth_policy=ToolAuthPolicy(
            require_user_id=True,
            allowed_channels=["web", "chat", "admin"],
            requires_tenant_match=True,
            risk_level="medium",
            scope="order",
        ),
        timeout_seconds=15,
        idempotent=True,
        handler=_find_recent_orders_tool,
        summarize_result=lambda payload: f"found {payload.get('total', 0)} recent order(s)",
    )


def build_get_online_member_count_tool() -> ToolSpec:
    return ToolSpec(
        name="get_online_member_count",
        description="Get the current online member/player count for a game alliance or group.",
        input_model=GetOnlineMemberCountInput,
        output_model=GetOnlineMemberCountOutput,
        auth_policy=ToolAuthPolicy(
            allow_anonymous=True,
            allowed_channels=["web", "chat", "admin"],
            risk_level="low",
            scope="game",
        ),
        timeout_seconds=15,
        idempotent=True,
        handler=_get_online_member_count_tool,
        summarize_result=lambda payload: (
            f"alliance {payload.get('alliance_id', '')} has {payload.get('online_count', 0)} online member(s)"
        ),
    )
