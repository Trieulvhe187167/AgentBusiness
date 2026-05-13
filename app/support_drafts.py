"""AI-assisted draft replies for support cases."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings
from app.database import fetch_all_sync, fetch_one_sync
from app.lang import detect_language
from app.llm_client import active_provider_name, complete_chat, is_llm_ready
from app.models import RequestContext
from app.rag import _build_citations, retrieve


class SupportDraftReplyInput(BaseModel):
    tone: str = Field(default="professional", max_length=80)
    instruction: str | None = Field(default=None, max_length=1000)


class SupportDraftReplyOutput(BaseModel):
    ticket_id: int
    ticket_code: str
    draft_reply: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    provider: str
    used_llm: bool
    retrieval_query: str
    context_summary: dict[str, Any] = Field(default_factory=dict)


_SYSTEM_PROMPT = """You draft customer support replies for an internal support team.
Rules:
- Write only the reply body, no markdown title.
- Be concise, accurate, and professional.
- Use the same language as the customer's message.
- Use KB/context facts only. If facts are missing, say support will verify and follow up.
- Do not promise refunds, policy exceptions, deletion, or irreversible actions.
- Do not say the reply was written by AI.
- Include short citation markers like [1], [2] only when using KB facts from the supplied citations.
"""


def _parse_json(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _compact(value: str | None, *, limit: int = 1400) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _ticket(ticket_id: int) -> dict[str, Any]:
    row = fetch_one_sync("SELECT * FROM support_tickets WHERE id = ?", (ticket_id,))
    if not row:
        raise ValueError("Support ticket not found")
    return row


def _notes(ticket_id: int) -> list[dict[str, Any]]:
    return fetch_all_sync(
        """
        SELECT note_type, visibility, body, created_by_user_id, created_at
        FROM support_ticket_notes
        WHERE ticket_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 8
        """,
        (ticket_id,),
    )


def _email_thread(ticket_code: str) -> list[dict[str, Any]]:
    return fetch_all_sync(
        """
        SELECT from_address, subject, snippet, body_text, received_at, direction
        FROM support_email_messages
        WHERE ticket_code = ?
        ORDER BY received_at ASC, id ASC
        LIMIT 8
        """,
        (ticket_code,),
    )


def _retrieval_query(ticket: dict[str, Any], notes: list[dict[str, Any]]) -> str:
    parts = [
        ticket.get("message") or "",
        ticket.get("issue_type") or "",
        ticket.get("intent") or "",
        ticket.get("resolution_summary") or "",
    ]
    for note in notes[:3]:
        parts.append(note.get("body") or "")
    return _compact(" ".join(part for part in parts if part), limit=900)


def _citation_context(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, item in enumerate(results[: settings.max_answer_chunks], start=1):
        source = item.get("filename") or "KB"
        if item.get("sheet_name"):
            source += f"/{item['sheet_name']}"
        if item.get("row_num") is not None:
            source += f" row {item['row_num']}"
        if item.get("page_num") is not None:
            source += f" p.{item['page_num']}"
        lines.append(f"[{index}] {source}: {_compact(item.get('text') or item.get('content_preview') or '', limit=700)}")
    return "\n".join(lines)


def _prompt(
    *,
    ticket: dict[str, Any],
    notes: list[dict[str, Any]],
    email_thread: list[dict[str, Any]],
    citations_context: str,
    tone: str,
    instruction: str | None,
) -> str:
    classification = _parse_json(ticket.get("classification_json"), {})
    action_plan = _parse_json(ticket.get("action_plan_json"), {})
    escalation = _parse_json(ticket.get("escalation_package_json"), {})
    note_lines = [
        f"- {row.get('note_type')}/{row.get('visibility')} by {row.get('created_by_user_id') or '-'}: {_compact(row.get('body'), limit=260)}"
        for row in notes
    ]
    email_lines = [
        f"- {row.get('direction') or 'email'} {row.get('received_at') or '-'} {row.get('from_address') or '-'}: "
        f"{_compact(row.get('body_text') or row.get('snippet') or row.get('subject'), limit=260)}"
        for row in email_thread
    ]
    return f"""
Tone: {tone or 'professional'}
Extra support instruction: {instruction or '-'}

Ticket:
- Code: {ticket.get('ticket_code')}
- Status: {ticket.get('workflow_status') or ticket.get('status')}
- Issue type: {ticket.get('issue_type')}
- Intent: {ticket.get('intent') or classification.get('intent') or '-'}
- Priority: {ticket.get('priority') or '-'}
- Risk: {ticket.get('risk_level') or classification.get('risk_level') or '-'}
- Customer/contact: {ticket.get('created_by_user_id') or '-'} / {ticket.get('contact') or '-'}
- Customer message: {_compact(ticket.get('message'), limit=900)}

Workflow context:
- Classification: {_compact(json.dumps(classification, ensure_ascii=False), limit=900)}
- Action plan: {_compact(json.dumps(action_plan, ensure_ascii=False), limit=900)}
- Escalation package: {_compact(json.dumps(escalation, ensure_ascii=False), limit=900)}

Recent notes:
{chr(10).join(note_lines) if note_lines else '-'}

Email thread:
{chr(10).join(email_lines) if email_lines else '-'}

KB citations:
{citations_context or '-'}

Draft the support reply now.
""".strip()


def _fallback_draft(ticket: dict[str, Any], citations: list[dict[str, Any]], lang: str) -> str:
    if lang == "en":
        if citations:
            return (
                "Thank you for contacting support. I checked the available knowledge base information "
                "and will use it to verify your request. Based on the current context, we will review "
                "the details and follow up with an accurate answer shortly."
            )
        return (
            "Thank you for contacting support. We have received your request and will verify the details "
            "before replying with accurate information."
        )
    if citations:
        return (
            "Cảm ơn bạn đã liên hệ bộ phận hỗ trợ. Mình đã kiểm tra thông tin hiện có trong hệ thống/KB "
            "và sẽ dùng các dữ liệu này để xác minh yêu cầu của bạn. Bộ phận hỗ trợ sẽ phản hồi lại với "
            "thông tin chính xác trong thời gian sớm nhất."
        )
    return (
        "Cảm ơn bạn đã liên hệ bộ phận hỗ trợ. Mình đã tiếp nhận yêu cầu của bạn và sẽ kiểm tra thêm "
        "thông tin trước khi phản hồi chính xác."
    )


def generate_support_draft_reply(
    ticket_id: int,
    payload: SupportDraftReplyInput,
    *,
    context: RequestContext,
) -> dict[str, Any]:
    ticket = _ticket(ticket_id)
    notes = _notes(ticket_id)
    email_thread = _email_thread(ticket["ticket_code"])
    query = _retrieval_query(ticket, notes)
    lang = detect_language(ticket.get("message") or query)

    results: list[dict[str, Any]] = []
    try:
        results = retrieve(
            query or ticket.get("message") or ticket["ticket_code"],
            top_k=max(settings.max_citations, 3),
            kb_id=ticket.get("kb_id") or context.kb_id,
            kb_key=ticket.get("kb_key") or context.kb_key,
            auth_context=context.auth.model_dump(),
        )
    except Exception:
        results = []

    citations = _build_citations(results) if results else []
    citations_context = _citation_context(results)
    draft = ""
    used_llm = False
    provider = active_provider_name()

    if is_llm_ready():
        try:
            result = complete_chat(
                _prompt(
                    ticket=ticket,
                    notes=notes,
                    email_thread=email_thread,
                    citations_context=citations_context,
                    tone=payload.tone,
                    instruction=payload.instruction,
                ),
                system_prompt=_SYSTEM_PROMPT,
                timeout_seconds=min(settings.llm_timeout_seconds, 45),
                max_tokens=min(settings.llm_max_tokens, 500),
            )
            draft = result.text.strip()
            provider = result.provider
            used_llm = bool(draft)
        except Exception:
            draft = ""

    if not draft:
        draft = _fallback_draft(ticket, citations, lang)
        provider = provider or "none"

    return SupportDraftReplyOutput(
        ticket_id=int(ticket["id"]),
        ticket_code=ticket["ticket_code"],
        draft_reply=draft,
        citations=citations,
        provider=provider,
        used_llm=used_llm,
        retrieval_query=query,
        context_summary={
            "note_count": len(notes),
            "email_message_count": len(email_thread),
            "citation_count": len(citations),
            "status": ticket.get("workflow_status") or ticket.get("status"),
            "intent": ticket.get("intent"),
            "priority": ticket.get("priority"),
        },
    ).model_dump()
