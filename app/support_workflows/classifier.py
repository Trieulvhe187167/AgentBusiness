from __future__ import annotations

import re

from app.support_workflows.schemas import CaseClassification

_ORDER_CODE_RE = re.compile(r"\b(?:DH|ORD|ORDER)[A-Z0-9-]{3,}\b", re.IGNORECASE)


def classify_text(text: str, *, issue_type: str | None = None) -> CaseClassification:
    normalized = " ".join((text or "").split())
    lower = normalized.lower()
    entities: dict[str, str] = {}
    order_match = _ORDER_CODE_RE.search(normalized.upper())
    if order_match:
        entities["order_code"] = order_match.group(0).upper()

    intent = "unknown"
    confidence = 0.45
    risk_level = "low"
    requires_auth = False
    reason = "No strong rule matched."

    if issue_type in {"refund", "payment"} or any(token in lower for token in ["refund", "hoàn tiền", "hoan tien"]):
        intent = "refund_request"
        confidence = 0.86
        risk_level = "high"
        requires_auth = True
        reason = "Refund/payment keyword detected."
    elif any(token in lower for token in ["cancel", "huỷ đơn", "hủy đơn", "cancel order"]):
        intent = "cancel_order"
        confidence = 0.84
        risk_level = "high"
        requires_auth = True
        reason = "Order cancellation keyword detected."
    elif entities.get("order_code") or any(token in lower for token in ["đơn hàng", "don hang", "order", "tracking", "giao hàng", "delivery"]):
        intent = "order_status"
        confidence = 0.82 if entities.get("order_code") else 0.68
        risk_level = "low"
        requires_auth = True
        reason = "Order or delivery signal detected."
    elif any(token in lower for token in ["login", "password", "mật khẩu", "dang nhap", "đăng nhập", "account"]):
        intent = "account_access"
        confidence = 0.78
        risk_level = "medium"
        requires_auth = True
        reason = "Account access keyword detected."
    elif any(token in lower for token in ["bug", "lỗi", "loi", "crash", "không chạy", "khong chay"]):
        intent = "technical_issue"
        confidence = 0.76
        risk_level = "medium"
        reason = "Technical issue keyword detected."
    elif any(token in lower for token in ["invoice", "billing", "hoá đơn", "hóa đơn", "thanh toán"]):
        intent = "billing_invoice"
        confidence = 0.76
        risk_level = "medium"
        requires_auth = True
        reason = "Billing keyword detected."
    elif any(token in lower for token in ["human", "agent", "nhân viên", "nguoi that", "người thật"]):
        intent = "human_request"
        confidence = 0.8
        risk_level = "medium"
        reason = "Human handoff keyword detected."
    elif issue_type and issue_type != "other":
        intent = "complaint" if issue_type in {"shipping", "technical"} else "product_question"
        confidence = 0.62
        risk_level = "medium" if intent == "complaint" else "low"
        reason = "Issue type provided by ticket."

    sentiment = "neutral"
    if any(token in lower for token in ["angry", "bực", "tệ", "bad", "khiếu nại", "complaint", "quá lâu"]):
        sentiment = "angry"
    elif any(token in lower for token in ["urgent", "gấp", "ngay", "asap"]):
        sentiment = "frustrated"

    return CaseClassification(
        intent=intent,  # type: ignore[arg-type]
        confidence=confidence,
        entities=entities,
        customer_sentiment=sentiment,
        requires_auth=requires_auth,
        risk_level=risk_level,
        reason=reason,
    )
