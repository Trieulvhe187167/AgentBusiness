from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.support_workflows.schemas import CaseClassification, PriorityAssessment


def _due(hours: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def assign_priority(classification: CaseClassification) -> PriorityAssessment:
    intent = classification.intent
    sentiment = classification.customer_sentiment

    if intent in {"refund_request", "cancel_order"}:
        return PriorityAssessment(
            priority="P1",
            sla_due_at=_due(4),
            assigned_team="billing_ops" if intent == "refund_request" else "order_ops",
            reason=f"High-risk {intent} requires human approval.",
        )

    if sentiment in {"angry", "frustrated"} or intent in {"complaint", "technical_issue", "account_access"}:
        return PriorityAssessment(
            priority="P2",
            sla_due_at=_due(12),
            assigned_team="support_ops",
            reason="Customer sentiment or issue type requires timely support follow-up.",
        )

    if intent == "order_status":
        return PriorityAssessment(
            priority="P2",
            sla_due_at=_due(24),
            assigned_team="order_ops",
            reason="Order status requests should be answered within one business day.",
        )

    return PriorityAssessment(
        priority="P3",
        sla_due_at=_due(48),
        assigned_team="general_support",
        reason="Low-risk informational support case.",
    )
