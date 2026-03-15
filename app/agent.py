"""
Agent orchestration for manual JSON tool routing.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
import uuid
from typing import Any, AsyncGenerator, Literal

from pydantic import BaseModel, Field

import app.rag as rag
from app.config import settings
from app.lang import detect_language
from app.llm_client import active_provider_name, complete_chat, generate_stream, is_llm_ready
from app.models import RequestContext
from app.session_memory import load_slots, merge_slots
from app.tools import tool_registry
from app.tools.registry import ToolAuthorizationError, ToolExecutionError, ToolValidationError

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_ORDER_CODE_RE = re.compile(r"\b(?:DH|ORD|ORDER)[A-Z0-9-]{3,}\b", re.IGNORECASE)
_ALLIANCE_ID_PATTERNS = (
    re.compile(r"(?:alliance|lien minh)\s*(?:id)?\s*[:#-]?\s*([a-z0-9_-]{2,40})", re.IGNORECASE),
    re.compile(r"(?:group|clan)\s*(?:id)?\s*[:#-]?\s*([a-z0-9_-]{2,40})", re.IGNORECASE),
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_GREETING_KEYWORDS = {
    "hi",
    "hello",
    "hey",
    "xin chao",
    "chao",
}
_LIST_KB_KEYWORDS = (
    "list kb",
    "list kbs",
    "list knowledge base",
    "list knowledge bases",
    "danh sach kb",
    "liet ke kb",
)
_KB_STATS_KEYWORDS = (
    "kb stats",
    "thong ke kb",
    "thong tin kb",
    "so lieu kb",
    "vector cua kb",
)
_TICKET_KEYWORDS = (
    "tao ticket",
    "mo ticket",
    "support ticket",
    "create ticket",
    "phieu ho tro",
    "ticket ho tro",
)
_TICKET_REFERENCE_KEYWORDS = (
    "ma ticket",
    "ticket code",
    "ticket vua tao",
    "ticket moi tao",
    "ticket gan nhat",
    "recent ticket",
    "latest ticket",
)
_ORDER_STATUS_KEYWORDS = (
    "don hang",
    "ma don",
    "order status",
    "track order",
    "shipping status",
    "trang thai don",
    "tien do don",
    "recent orders",
    "don gan day",
)
_ONLINE_COUNT_KEYWORDS = (
    "bao nhieu nguoi online",
    "bao nhieu thanh vien online",
    "how many online",
    "online member count",
    "active players",
    "nguoi choi dang hoat dong",
    "thanh vien dang online",
)
_KB_STATS_FOLLOWUP_KEYWORDS = (
    "bao nhieu vector",
    "how many vectors",
    "bao nhieu file",
    "how many files",
    "ingested files",
    "vector count",
    "thong ke kb do",
    "kb do",
)
_CANCEL_KEYWORDS = (
    "cancel",
    "khong can",
    "khong tao nua",
    "thoi",
    "bo qua",
    "huy",
)

_ROUTER_SYSTEM_PROMPT = """
You are an orchestration router.
Choose exactly one route: rag, tool, clarify, or fallback.

Return JSON only with this shape:
{
  "route": "rag" | "tool" | "clarify" | "fallback",
  "tool_name": string | null,
  "arguments": object,
  "message": string | null,
  "reason": string
}

Rules:
- Use route="rag" for document or knowledge-base questions.
- Use route="tool" for backend actions or admin data lookups.
- Use route="clarify" when required arguments are missing.
- Use route="fallback" for greeting, small talk, or out-of-scope requests.
- Never invent tool names.
- If route="tool", provide only one tool name.
""".strip()

_NATIVE_ROUTER_SYSTEM_PROMPT = """
You are an orchestration router.
Use the provided tools when one of them is the best way to handle the request.

Rules:
- Use search_kb for document or knowledge-base questions.
- Use business/admin tools only when the user clearly asks for them.
- If required arguments are missing, do not call a tool. Return compact JSON:
  {"route":"clarify","message":"...","reason":"..."}
- For greeting, small talk, or out-of-scope requests, do not call a tool. Return compact JSON:
  {"route":"fallback","message":"...","reason":"..."}
- If no tool is needed and the request is still best handled by the normal RAG answer flow, return compact JSON:
  {"route":"rag","message":null,"reason":"..."}
- Call at most one tool.
""".strip()


class AgentDecision(BaseModel):
    route: Literal["rag", "tool", "clarify", "fallback", "memory"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
    reason: str = ""


def _normalize_query(text: str) -> str:
    return " ".join(text.strip().split())


def _ascii_hint(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(stripped.split())


def _extract_contact(text: str) -> str | None:
    email = _EMAIL_RE.search(text)
    if email:
        return email.group(0)
    phone = _PHONE_RE.search(text)
    if phone:
        return " ".join(phone.group(0).split())
    return None


def _extract_order_code(text: str) -> str | None:
    match = _ORDER_CODE_RE.search(text.upper())
    if not match:
        return None
    return match.group(0).upper()


def _extract_alliance_id(text: str) -> str | None:
    lowered = _ascii_hint(text)
    for pattern in _ALLIANCE_ID_PATTERNS:
        match = pattern.search(lowered)
        if match:
            return match.group(1).upper()
    return None


def _infer_issue_type(text: str) -> str:
    lowered = _ascii_hint(text)
    if any(token in lowered for token in ("thanh toan", "payment", "card", "the ")):
        return "payment"
    if any(token in lowered for token in ("giao hang", "shipping", "ship", "van chuyen")):
        return "shipping"
    if any(token in lowered for token in ("refund", "hoan tien", "tra hang")):
        return "refund"
    if any(token in lowered for token in ("tai khoan", "account", "dang nhap", "login")):
        return "account"
    if any(token in lowered for token in ("bug", "loi", "technical", "ky thuat")):
        return "technical"
    return "other"


def _fallback_message(lang: str) -> str:
    if lang == "vi":
        return (
            "Mình có thể hỗ trợ tra cứu KB, tạo ticket hỗ trợ, hoặc xem thông tin KB "
            "nếu bạn có quyền phù hợp."
        )
    return "I can help search KB content, create support tickets, or inspect KB information when you have permission."


def _ticket_contact_clarify(lang: str) -> str:
    if lang == "vi":
        return "Mình cần email hoặc số điện thoại liên hệ để tạo ticket hỗ trợ cho bạn."
    return "I need an email address or phone number to create a support ticket for you."


def _permission_message(tool_name: str, lang: str) -> str:
    if lang == "vi":
        return f"Bạn chưa có quyền dùng tool `{tool_name}`."
    return f"You do not have permission to use tool `{tool_name}`."


def _tool_error_message(tool_name: str, lang: str) -> str:
    if lang == "vi":
        return f"Mình chưa thể thực hiện tool `{tool_name}` lúc này."
    return f"I could not complete tool `{tool_name}` right now."


def _clarify_message(lang: str) -> str:
    if lang == "vi":
        return "Bạn có thể nói rõ hơn yêu cầu của mình không?"
    return "Could you clarify your request?"


def _compose_tool_answer(tool_name: str, payload: dict[str, Any], lang: str) -> str:
    if tool_name == "create_support_ticket":
        if lang == "vi":
            return (
                f"Mình đã tạo ticket {payload['ticket_code']} cho vấn đề {payload['issue_type']}. "
                f"Trạng thái hiện tại là {payload['status']}."
            )
        return (
            f"I created ticket {payload['ticket_code']} for {payload['issue_type']}. "
            f"The current status is {payload['status']}."
        )

    if tool_name == "list_kbs":
        items = payload.get("items") or []
        names = ", ".join(item.get("name") or item.get("key") or "-" for item in items[:5])
        if lang == "vi":
            return f"Hiện có {payload.get('total', 0)} KB. Một số KB: {names}."
        return f"There are {payload.get('total', 0)} KBs. Some of them: {names}."

    if tool_name == "get_kb_stats":
        kb_name = payload.get("kb_name") or payload.get("kb_key")
        if lang == "vi":
            return (
                f"KB {kb_name} hiện có {payload.get('total_files', 0)} file, "
                f"{payload.get('ingested_files', 0)} file đã ingest và {payload.get('total_vectors', 0)} vectors."
            )
        return (
            f"KB {kb_name} currently has {payload.get('total_files', 0)} files, "
            f"{payload.get('ingested_files', 0)} ingested files, and {payload.get('total_vectors', 0)} vectors."
        )

    if tool_name == "search_kb":
        hits = payload.get("hits") or []
        if not hits:
            return "Mình chưa tìm thấy kết quả phù hợp trong KB." if lang == "vi" else "I could not find a relevant result in the KB."
        top_hit = hits[0]
        if lang == "vi":
            return f"Kết quả gần nhất đến từ {top_hit.get('filename')}: {top_hit.get('preview')}"
        return f"The closest result is from {top_hit.get('filename')}: {top_hit.get('preview')}"

    return json.dumps(payload, ensure_ascii=False)


def _tool_result_summary(tool_name: str, payload: dict[str, Any]) -> str:
    if tool_name == "create_support_ticket":
        return f"created {payload.get('ticket_code')}"
    if tool_name == "list_kbs":
        return f"{payload.get('total', 0)} KB(s)"
    if tool_name == "get_kb_stats":
        return f"{payload.get('total_vectors', 0)} vector(s)"
    if tool_name == "search_kb":
        return f"{payload.get('total_hits', 0)} hit(s)"
    return "completed"


def _fallback_message(lang: str) -> str:
    if lang == "vi":
        return (
            "Mình có thể hỗ trợ tra cứu KB, tạo ticket, tra trạng thái đơn hàng, "
            "gợi ý đơn gần đây, hoặc xem số người online trong game khi backend đã kết nối."
        )
    return (
        "I can help search KB content, create support tickets, check order status, suggest recent orders, "
        "or look up online member counts when your backend data is connected."
    )


def _ticket_contact_clarify(lang: str) -> str:
    if lang == "vi":
        return "Mình cần email hoặc số điện thoại liên hệ để tạo ticket hỗ trợ cho bạn."
    return "I need an email address or phone number to create a support ticket for you."


def _order_code_clarify(lang: str, *, logged_in: bool) -> str:
    if lang == "vi":
        if logged_in:
            return "Bạn chưa gửi mã đơn. Mình có thể gợi ý các đơn gần đây trong tài khoản của bạn."
        return "Bạn cho mình mã đơn hàng nhé. Nếu có đăng nhập, mình có thể gợi ý đơn gần đây."
    if logged_in:
        return "You have not provided an order code. I can suggest your recent orders from the signed-in account."
    return "Please share your order code. If the user is signed in, I can also suggest recent orders."


def _alliance_clarify(lang: str) -> str:
    if lang == "vi":
        return "Bạn cho mình alliance/liên minh ID để mình kiểm tra số người đang online."
    return "Please share the alliance ID so I can check the online member count."


def _compose_tool_answer(tool_name: str, payload: dict[str, Any], lang: str) -> str:
    if tool_name == "create_support_ticket":
        if lang == "vi":
            return (
                f"Mình đã tạo ticket {payload['ticket_code']} cho vấn đề {payload['issue_type']}. "
                f"Trạng thái hiện tại là {payload['status']}."
            )
        return (
            f"I created ticket {payload['ticket_code']} for {payload['issue_type']}. "
            f"The current status is {payload['status']}."
        )

    if tool_name == "list_kbs":
        items = payload.get("items") or []
        names = ", ".join(item.get("name") or item.get("key") or "-" for item in items[:5])
        if lang == "vi":
            return f"Hiện có {payload.get('total', 0)} KB. Một số KB: {names}."
        return f"There are {payload.get('total', 0)} KBs. Some of them: {names}."

    if tool_name == "get_kb_stats":
        kb_name = payload.get("kb_name") or payload.get("kb_key")
        if lang == "vi":
            return (
                f"KB {kb_name} hiện có {payload.get('total_files', 0)} file, "
                f"{payload.get('ingested_files', 0)} file đã ingest và {payload.get('total_vectors', 0)} vectors."
            )
        return (
            f"KB {kb_name} currently has {payload.get('total_files', 0)} files, "
            f"{payload.get('ingested_files', 0)} ingested files, and {payload.get('total_vectors', 0)} vectors."
        )

    if tool_name == "search_kb":
        hits = payload.get("hits") or []
        if not hits:
            return "Mình chưa tìm thấy kết quả phù hợp trong KB." if lang == "vi" else "I could not find a relevant result in the KB."
        top_hit = hits[0]
        if lang == "vi":
            return f"Kết quả gần nhất đến từ {top_hit.get('filename')}: {top_hit.get('preview')}"
        return f"The closest result is from {top_hit.get('filename')}: {top_hit.get('preview')}"

    if tool_name == "get_order_status":
        if lang == "vi":
            return (
                f"Đơn {payload.get('order_code')} hiện đang {payload.get('status')}. "
                f"Cập nhật gần nhất: {payload.get('last_update') or 'chưa có'}. "
                f"Đơn vị vận chuyển: {payload.get('carrier') or 'chưa có'}."
            )
        return (
            f"Order {payload.get('order_code')} is currently {payload.get('status')}. "
            f"Last update: {payload.get('last_update') or 'n/a'}. Carrier: {payload.get('carrier') or 'n/a'}."
        )

    if tool_name == "find_recent_orders":
        orders = payload.get("orders") or []
        if not orders:
            return (
                "Mình chưa tìm thấy đơn gần đây nào trong tài khoản này."
                if lang == "vi"
                else "I could not find any recent orders for this account."
            )
        preview = ", ".join(f"{item.get('order_code')} ({item.get('status')})" for item in orders[:3])
        if lang == "vi":
            return f"Mình tìm thấy {payload.get('total', 0)} đơn gần đây: {preview}. Bạn muốn kiểm tra đơn nào?"
        return f"I found {payload.get('total', 0)} recent orders: {preview}. Which one do you want to inspect?"

    if tool_name == "get_online_member_count":
        if lang == "vi":
            return (
                f"Liên minh {payload.get('alliance_id')} hiện có {payload.get('online_count')} người đang online. "
                f"Mốc quan sát: {payload.get('observed_at')}."
            )
        return (
            f"Alliance {payload.get('alliance_id')} currently has {payload.get('online_count')} members online. "
            f"Observed at {payload.get('observed_at')}."
        )

    return json.dumps(payload, ensure_ascii=False)


def _tool_result_summary(tool_name: str, payload: dict[str, Any]) -> str:
    if tool_name == "create_support_ticket":
        return f"created {payload.get('ticket_code')}"
    if tool_name == "list_kbs":
        return f"{payload.get('total', 0)} KB(s)"
    if tool_name == "get_kb_stats":
        return f"{payload.get('total_vectors', 0)} vector(s)"
    if tool_name == "search_kb":
        return f"{payload.get('total_hits', 0)} hit(s)"
    if tool_name == "get_order_status":
        return f"{payload.get('order_code')} => {payload.get('status')}"
    if tool_name == "find_recent_orders":
        return f"{payload.get('total', 0)} recent order(s)"
    if tool_name == "get_online_member_count":
        return f"{payload.get('online_count', 0)} online"
    return "completed"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _load_session_slots(request_context: RequestContext) -> dict[str, Any]:
    return load_slots(request_context.session_id)


def _ticket_memory_message(slots: dict[str, Any], lang: str) -> str | None:
    ticket_code = str(slots.get("last_ticket_code") or "").strip()
    if not ticket_code:
        return None

    issue_type = str(slots.get("last_issue_type") or "other").strip() or "other"
    if lang == "vi":
        return f"Ticket gần nhất là {ticket_code} cho vấn đề {issue_type}."
    return f"The most recent ticket is {ticket_code} for {issue_type}."


def _memory_decision(query: str, request_context: RequestContext, lang: str) -> AgentDecision | None:
    slots = _load_session_slots(request_context)
    if not slots:
        return None

    lowered = _ascii_hint(query).strip(" .,!?:;")

    if slots.get("pending_tool_name") == "create_support_ticket":
        if any(keyword in lowered for keyword in _CANCEL_KEYWORDS):
            merge_slots(
                request_context.session_id,
                {
                    "pending_tool_name": None,
                    "pending_tool_arguments": None,
                    "pending_subject_type": None,
                },
            )
        else:
            contact = _extract_contact(query)
            if contact or request_context.auth.user_id:
                arguments = dict(slots.get("pending_tool_arguments") or {})
                if contact:
                    arguments["contact"] = contact
                return AgentDecision(
                    route="tool",
                    tool_name="create_support_ticket",
                    arguments=arguments,
                    reason="resume_pending_tool_from_slots",
                )

    if any(keyword in lowered for keyword in _TICKET_REFERENCE_KEYWORDS):
        message = _ticket_memory_message(slots, lang)
        if message:
            return AgentDecision(
                route="memory",
                message=message,
                reason="reply_from_slot_memory",
            )

    if any(keyword in lowered for keyword in _KB_STATS_FOLLOWUP_KEYWORDS):
        kb_id = slots.get("last_kb_id")
        kb_key = slots.get("last_kb_key")
        if kb_id or kb_key:
            return AgentDecision(
                route="tool",
                tool_name="get_kb_stats",
                arguments={
                    "kb_id": kb_id,
                    "kb_key": kb_key,
                },
                reason="reuse_last_kb_slots",
            )

    return None


def _slot_updates_for_decision(decision: AgentDecision) -> dict[str, Any]:
    if decision.route == "clarify" and decision.tool_name == "create_support_ticket" and decision.arguments:
        return {
            "pending_tool_name": "create_support_ticket",
            "pending_tool_arguments": decision.arguments,
            "pending_subject_type": "support_ticket",
        }
    return {}


def _slot_updates_for_tool(
    tool_name: str,
    payload: dict[str, Any],
    *,
    arguments: dict[str, Any],
    request_context: RequestContext,
) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "last_tool": tool_name,
        "pending_tool_name": None,
        "pending_tool_arguments": None,
        "pending_subject_type": None,
    }

    if tool_name == "create_support_ticket":
        updates.update(
            {
                "subject_type": "support_ticket",
                "last_ticket_code": payload.get("ticket_code"),
                "last_issue_type": payload.get("issue_type"),
                "last_contact": payload.get("contact") or arguments.get("contact"),
            }
        )
    elif tool_name == "get_kb_stats":
        updates.update(
            {
                "subject_type": "kb",
                "last_kb_id": payload.get("kb_id") or request_context.kb_id,
                "last_kb_key": payload.get("kb_key") or request_context.kb_key,
                "last_kb_name": payload.get("kb_name"),
                "last_kb_stats": {
                    "total_files": payload.get("total_files"),
                    "ingested_files": payload.get("ingested_files"),
                    "total_vectors": payload.get("total_vectors"),
                },
            }
        )
    elif tool_name == "search_kb":
        hits = payload.get("hits") or []
        top_hit = hits[0] if hits else {}
        updates.update(
            {
                "subject_type": "kb_search",
                "last_kb_id": payload.get("kb_id") or request_context.kb_id,
                "last_kb_key": payload.get("kb_key") or request_context.kb_key,
                "last_kb_name": payload.get("kb_name"),
                "last_search_query": payload.get("query"),
                "last_search_filename": top_hit.get("filename"),
                "last_search_preview": top_hit.get("preview"),
            }
        )
    elif tool_name == "get_order_status":
        updates.update(
            {
                "subject_type": "order",
                "last_order_code": payload.get("order_code"),
                "last_order_status": payload.get("status"),
                "last_tracking_code": payload.get("tracking_code"),
                "last_order_user_id": payload.get("user_id") or request_context.auth.user_id,
            }
        )
    elif tool_name == "find_recent_orders":
        orders = payload.get("orders") or []
        updates.update(
            {
                "subject_type": "recent_orders",
                "last_recent_order_codes": [item.get("order_code") for item in orders[:5] if item.get("order_code")],
                "last_order_code": (orders[0] or {}).get("order_code") if orders else None,
                "last_order_user_id": payload.get("user_id") or request_context.auth.user_id,
            }
        )
    elif tool_name == "get_online_member_count":
        updates.update(
            {
                "subject_type": "game_online",
                "last_alliance_id": payload.get("alliance_id"),
                "last_server_id": payload.get("server_id"),
                "last_online_count": payload.get("online_count"),
            }
        )
    elif tool_name == "list_kbs":
        updates.update(
            {
                "subject_type": "kb_list",
                "last_kb_count": payload.get("total"),
            }
        )

    return updates


def _decision_with_hydrated_arguments(
    decision: AgentDecision,
    *,
    query: str,
    request_context: RequestContext,
) -> AgentDecision:
    arguments = dict(decision.arguments or {})
    tool_name = decision.tool_name

    if tool_name == "search_kb":
        arguments.setdefault("query", query)
        if request_context.kb_id is not None:
            arguments.setdefault("kb_id", request_context.kb_id)
        if request_context.kb_key:
            arguments.setdefault("kb_key", request_context.kb_key)
    elif tool_name == "create_support_ticket":
        arguments.setdefault("issue_type", _infer_issue_type(query))
        arguments.setdefault("message", query)
        contact = _extract_contact(query)
        if contact:
            arguments.setdefault("contact", contact)
    elif tool_name == "get_order_status":
        order_code = _extract_order_code(query)
        if order_code:
            arguments.setdefault("order_code", order_code)
        if request_context.auth.user_id:
            arguments.setdefault("user_id", request_context.auth.user_id)
    elif tool_name == "find_recent_orders":
        if request_context.auth.user_id:
            arguments.setdefault("user_id", request_context.auth.user_id)
    elif tool_name == "get_online_member_count":
        alliance_id = _extract_alliance_id(query)
        if alliance_id:
            arguments.setdefault("alliance_id", alliance_id)
    elif tool_name == "get_kb_stats":
        if request_context.kb_id is not None:
            arguments.setdefault("kb_id", request_context.kb_id)
        if request_context.kb_key:
            arguments.setdefault("kb_key", request_context.kb_key)

    return decision.model_copy(update={"arguments": arguments})


def _heuristic_route(query: str, request_context: RequestContext, lang: str) -> AgentDecision:
    lowered = _ascii_hint(query).strip(" .,!?:;")
    auth = request_context.auth
    order_code = _extract_order_code(query)
    alliance_id = _extract_alliance_id(query)

    if lowered in _GREETING_KEYWORDS:
        return AgentDecision(route="fallback", message=_fallback_message(lang), reason="greeting_or_small_talk")

    if any(keyword in lowered for keyword in _LIST_KB_KEYWORDS):
        return AgentDecision(route="tool", tool_name="list_kbs", reason="admin_list_kbs_intent")

    if any(keyword in lowered for keyword in _KB_STATS_KEYWORDS):
        return AgentDecision(
            route="tool",
            tool_name="get_kb_stats",
            arguments={},
            reason="admin_kb_stats_intent",
        )

    if any(keyword in lowered for keyword in _TICKET_KEYWORDS):
        contact = _extract_contact(query)
        if not auth.user_id and not contact:
            return AgentDecision(
                route="clarify",
                tool_name="create_support_ticket",
                arguments={
                    "issue_type": _infer_issue_type(query),
                    "message": query,
                },
                message=_ticket_contact_clarify(lang),
                reason="missing_ticket_contact",
            )
        return AgentDecision(
            route="tool",
            tool_name="create_support_ticket",
            arguments={},
            reason="support_ticket_intent",
        )

    if order_code:
        return AgentDecision(
            route="tool",
            tool_name="get_order_status",
            arguments={"order_code": order_code},
            reason="explicit_order_code",
        )

    if any(keyword in lowered for keyword in _ORDER_STATUS_KEYWORDS):
        if order_code:
            return AgentDecision(
                route="tool",
                tool_name="get_order_status",
                arguments={"order_code": order_code},
                reason="order_status_intent",
            )
        if auth.user_id:
            return AgentDecision(
                route="tool",
                tool_name="find_recent_orders",
                arguments={"user_id": auth.user_id},
                reason="recent_orders_intent",
            )
        return AgentDecision(
            route="clarify",
            tool_name="get_order_status",
            arguments={},
            message=_order_code_clarify(lang, logged_in=False),
            reason="missing_order_code",
        )

    if any(keyword in lowered for keyword in _ONLINE_COUNT_KEYWORDS):
        if not alliance_id:
            return AgentDecision(
                route="clarify",
                tool_name="get_online_member_count",
                arguments={},
                message=_alliance_clarify(lang),
                reason="missing_alliance_id",
            )
        return AgentDecision(
            route="tool",
            tool_name="get_online_member_count",
            arguments={"alliance_id": alliance_id},
            reason="game_online_intent",
        )

    return AgentDecision(route="rag", reason="default_rag_route")


def _llm_route(query: str, request_context: RequestContext, lang: str) -> AgentDecision | None:
    if not is_llm_ready():
        return None

    tool_summaries = [item.model_dump() for item in tool_registry.list_definitions()]
    prompt = json.dumps(
        {
            "query": query,
            "lang": lang,
            "request_context": request_context.model_dump(),
            "tools": tool_summaries,
        },
        ensure_ascii=False,
    )
    try:
        raw_text = "".join(generate_stream(prompt, system_prompt=_ROUTER_SYSTEM_PROMPT)).strip()
    except Exception:
        logger.exception("LLM router failed")
        return None

    payload = _extract_json_object(raw_text)
    if not payload:
        logger.warning("LLM router returned non-JSON payload: %s", raw_text[:240])
        return None

    try:
        decision = AgentDecision.model_validate(payload)
    except Exception:
        logger.warning("LLM router returned invalid decision payload: %s", payload)
        return None

    valid_tools = {item.name for item in tool_registry.list_definitions()}
    if decision.route == "tool" and decision.tool_name not in valid_tools:
        logger.warning("LLM router suggested unknown tool: %s", decision.tool_name)
        return None

    return _decision_with_hydrated_arguments(decision, query=query, request_context=request_context)


def _native_tool_route(query: str, request_context: RequestContext, lang: str) -> AgentDecision | None:
    if not is_llm_ready():
        return None

    tools = tool_registry.list_openai_tools()
    prompt = json.dumps(
        {
            "query": query,
            "lang": lang,
            "request_context": request_context.model_dump(),
        },
        ensure_ascii=False,
    )
    try:
        result = complete_chat(
            prompt,
            system_prompt=_NATIVE_ROUTER_SYSTEM_PROMPT,
            tools=tools,
            tool_choice=settings.normalized_agent_tool_choice_mode,
        )
    except Exception:
        logger.exception("Native tool router failed")
        return None

    if result.tool_calls:
        tool_call = result.tool_calls[0]
        if tool_call.name == "search_kb":
            return _decision_with_hydrated_arguments(
                AgentDecision(
                    route="rag",
                    tool_name="search_kb",
                    arguments=tool_call.arguments,
                    reason="native_tool_search_kb",
                ),
                query=query,
                request_context=request_context,
            )

        valid_tools = {item.name for item in tool_registry.list_definitions()}
        if tool_call.name in valid_tools:
            return _decision_with_hydrated_arguments(
                AgentDecision(
                    route="tool",
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    reason="native_tool_call",
                ),
                query=query,
                request_context=request_context,
            )
        return None

    payload = _extract_json_object(result.text)
    if not payload:
        return None

    try:
        decision = AgentDecision.model_validate(payload)
    except Exception:
        logger.warning("Native tool router returned invalid JSON decision: %s", payload)
        return None
    return _decision_with_hydrated_arguments(decision, query=query, request_context=request_context)


def decide_route(
    query: str,
    *,
    request_context: RequestContext | dict[str, Any] | None = None,
    lang: str | None = None,
) -> AgentDecision:
    context = request_context if isinstance(request_context, RequestContext) else RequestContext.model_validate(
        request_context or {"request_id": uuid.uuid4().hex[:8]}
    )
    resolved_lang = detect_language(query, explicit_lang=lang)
    memory = _memory_decision(query, context, resolved_lang)
    if memory is not None:
        return _decision_with_hydrated_arguments(memory, query=query, request_context=context)

    if settings.agent_native_tool_ready:
        native = _native_tool_route(query, context, resolved_lang)
        if native is not None:
            return native

    heuristic = _decision_with_hydrated_arguments(
        _heuristic_route(query, context, resolved_lang),
        query=query,
        request_context=context,
    )
    if heuristic.route != "rag":
        return heuristic

    llm_decision = _llm_route(query, context, resolved_lang)
    return llm_decision or heuristic


def _emit_start(
    *,
    query: str,
    mode: str,
    route: str,
    request_context: RequestContext,
    session_id: str,
    lang: str,
    tool_name: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "event": "start",
        "data": {
            "query": query,
            "mode": mode,
            "route": route,
            "request_id": request_context.request_id,
            "session_id": session_id,
            "llm_provider": active_provider_name(),
            "lang": lang,
            "kb_id": request_context.kb_id,
            "kb_key": request_context.kb_key,
            "tool_name": tool_name,
            "reason": reason,
        },
    }


def _log_agent_chat(
    *,
    session_id: str,
    query: str,
    mode: str,
    answer_text: str,
    llm_provider: str,
    request_context: RequestContext,
    latency_ms: int,
) -> None:
    rag._log_chat(
        session_id=session_id,
        user_message=query,
        merged_query=query,
        mode=mode,
        top_score=0.0,
        answer_text=answer_text,
        citations=[],
        latency_ms=latency_ms,
        llm_provider=llm_provider,
        request_context=request_context,
    )


def _resolved_scoped_session_id(session_id: str, request_context: RequestContext) -> str:
    if request_context.kb_id:
        return rag._scoped_session_id(session_id, request_context.kb_id)
    return session_id


async def agent_stream(
    query: str,
    session_id: str | None = None,
    lang: str | None = None,
    kb_id: int | None = None,
    kb_key: str | None = None,
    request_context: RequestContext | dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    start_time = time.perf_counter()
    normalized_query = _normalize_query(query)
    resolved_lang = detect_language(normalized_query, explicit_lang=lang)
    sid = session_id or uuid.uuid4().hex
    context = request_context if isinstance(request_context, RequestContext) else RequestContext.model_validate(
        request_context
        or {
            "request_id": uuid.uuid4().hex[:8],
            "session_id": sid,
            "kb_id": kb_id,
            "kb_key": kb_key,
        }
    )
    if not context.session_id:
        context.session_id = sid
    if kb_id is not None:
        context.kb_id = kb_id
    if kb_key:
        context.kb_key = kb_key

    decision = decide_route(normalized_query, request_context=context, lang=resolved_lang)
    yield {
        "event": "route",
        "data": {
            "route": decision.route,
            "tool_name": decision.tool_name,
            "arguments": decision.arguments,
            "request_id": context.request_id,
            "reason": decision.reason,
        },
    }

    if decision.route == "rag":
        rag_query = str(decision.arguments.get("query") or normalized_query)
        if decision.arguments.get("kb_id") and not context.kb_id:
            context.kb_id = int(decision.arguments["kb_id"])
        if decision.arguments.get("kb_key") and not context.kb_key:
            context.kb_key = str(decision.arguments["kb_key"])
        for event in rag.rag_stream(
            query=rag_query,
            session_id=sid,
            lang=resolved_lang,
            kb_id=context.kb_id,
            kb_key=context.kb_key,
            request_context=context,
        ):
            if event["event"] == "start":
                event["data"]["route"] = "rag"
                event["data"]["reason"] = decision.reason
            yield event
        return

    llm_provider = active_provider_name()

    if decision.route == "clarify":
        pending_updates = _slot_updates_for_decision(decision)
        if pending_updates:
            merge_slots(context.session_id, pending_updates)
        answer_text = decision.message or _clarify_message(resolved_lang)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        yield _emit_start(
            query=normalized_query,
            mode="clarify",
            route="clarify",
            request_context=context,
            session_id=sid,
            lang=resolved_lang,
            reason=decision.reason,
        )
        yield {"event": "token", "data": {"text": answer_text}}
        yield {"event": "citations", "data": {"items": []}}
        _log_agent_chat(
            session_id=_resolved_scoped_session_id(sid, context),
            query=normalized_query,
            mode="clarify",
            answer_text=answer_text,
            llm_provider=llm_provider,
            request_context=context,
            latency_ms=latency_ms,
        )
        yield {"event": "done", "data": {"ok": True, "latency_ms": latency_ms}}
        return

    if decision.route == "memory":
        answer_text = decision.message or _clarify_message(resolved_lang)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        yield _emit_start(
            query=normalized_query,
            mode="memory",
            route="memory",
            request_context=context,
            session_id=sid,
            lang=resolved_lang,
            reason=decision.reason,
        )
        yield {"event": "token", "data": {"text": answer_text}}
        yield {"event": "citations", "data": {"items": []}}
        _log_agent_chat(
            session_id=_resolved_scoped_session_id(sid, context),
            query=normalized_query,
            mode="memory",
            answer_text=answer_text,
            llm_provider=llm_provider,
            request_context=context,
            latency_ms=latency_ms,
        )
        yield {"event": "done", "data": {"ok": True, "latency_ms": latency_ms}}
        return

    if decision.route == "fallback":
        answer_text = decision.message or _fallback_message(resolved_lang)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        yield _emit_start(
            query=normalized_query,
            mode="fallback",
            route="fallback",
            request_context=context,
            session_id=sid,
            lang=resolved_lang,
            reason=decision.reason,
        )
        yield {"event": "token", "data": {"text": answer_text}}
        yield {"event": "citations", "data": {"items": []}}
        _log_agent_chat(
            session_id=_resolved_scoped_session_id(sid, context),
            query=normalized_query,
            mode="fallback",
            answer_text=answer_text,
            llm_provider=llm_provider,
            request_context=context,
            latency_ms=latency_ms,
        )
        yield {"event": "done", "data": {"ok": True, "latency_ms": latency_ms}}
        return

    tool_name = decision.tool_name or ""
    yield _emit_start(
        query=normalized_query,
        mode="tool",
        route="tool",
        request_context=context,
        session_id=sid,
        lang=resolved_lang,
        tool_name=tool_name,
        reason=decision.reason,
    )
    yield {"event": "tool_call", "data": {"tool_name": tool_name, "arguments": decision.arguments}}

    done_ok = True
    tool_result_payload: dict[str, Any]
    answer_text: str
    chat_mode: str

    try:
        result = await tool_registry.execute(tool_name, decision.arguments, request_context=context)
    except ToolAuthorizationError:
        done_ok = False
        answer_text = _permission_message(tool_name, resolved_lang)
        chat_mode = "tool_error"
        tool_result_payload = {
            "tool_name": tool_name,
            "status": "failed",
            "summary": "permission_denied",
        }
    except ToolValidationError as err:
        answer_text = decision.message or (str(err) if str(err) else _ticket_contact_clarify(resolved_lang))
        chat_mode = "clarify"
        tool_result_payload = {
            "tool_name": tool_name,
            "status": "clarify",
            "summary": str(err) or "validation_error",
        }
    except (ToolExecutionError, KeyError):
        done_ok = False
        answer_text = _tool_error_message(tool_name, resolved_lang)
        chat_mode = "tool_error"
        tool_result_payload = {
            "tool_name": tool_name,
            "status": "failed",
            "summary": "execution_error",
        }
    else:
        if not context.kb_id and result.output.get("kb_id"):
            context.kb_id = int(result.output["kb_id"])
        if not context.kb_key and result.output.get("kb_key"):
            context.kb_key = str(result.output["kb_key"])
        merge_slots(
            context.session_id,
            _slot_updates_for_tool(
                tool_name,
                result.output,
                arguments=decision.arguments,
                request_context=context,
            ),
        )
        answer_text = _compose_tool_answer(tool_name, result.output, resolved_lang)
        chat_mode = "tool"
        tool_result_payload = {
            "tool_name": tool_name,
            "tool_call_id": result.tool_call_id,
            "status": "success",
            "summary": _tool_result_summary(tool_name, result.output),
        }

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    yield {"event": "tool_result", "data": tool_result_payload}
    yield {"event": "token", "data": {"text": answer_text}}
    yield {"event": "citations", "data": {"items": []}}
    _log_agent_chat(
        session_id=_resolved_scoped_session_id(sid, context),
        query=normalized_query,
        mode=chat_mode,
        answer_text=answer_text,
        llm_provider=llm_provider,
        request_context=context,
        latency_ms=latency_ms,
    )
    yield {"event": "done", "data": {"ok": done_ok, "latency_ms": latency_ms}}
