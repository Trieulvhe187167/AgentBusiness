"""Optional LLM-as-judge scoring for golden dataset evaluation."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.config import settings
from app.llm_client import active_provider_name, complete_chat, provider_available
from app.observability import content_attrs, trace_span

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a retrieval-augmented business assistant.
Grade the answer only against the question, expected answer, retrieved context, and citations.
Return one JSON object only. Do not include markdown.
Scores must be numbers from 0.0 to 1.0.
hallucination_risk means unsupported or contradictory content risk, where 0.0 is no risk and 1.0 is severe risk.
verdict must be pass, warn, or fail."""

JUDGE_USER_TEMPLATE = """Evaluate this RAG answer.

Question:
{question}

Expected answer:
{expected_answer}

Accepted answer variants:
{expected_answers}

Actual answer:
{actual_answer}

Retrieved context preview:
{retrieved}

Citations:
{citations}

Return JSON with exactly these keys:
correctness, groundedness, completeness, citation_support, hallucination_risk, verdict, reason"""


def _preview_json(value: Any, *, limit: int = 6000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text[:limit]


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty judge response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("judge response is not a JSON object")
    return parsed


def _score(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return round(max(0.0, min(numeric, 1.0)), 4)


def _normalize_verdict(value: Any, judge_score: float) -> str:
    verdict = str(value or "").strip().lower()
    if verdict in {"pass", "warn", "fail"}:
        return verdict
    if judge_score >= 0.75:
        return "pass"
    if judge_score >= 0.50:
        return "warn"
    return "fail"


def judge_enabled_for_run(explicit: bool | None) -> bool:
    return bool(settings.eval_llm_judge_enabled if explicit is None else explicit)


def resolved_judge_provider() -> str:
    return settings.normalized_eval_llm_judge_provider or active_provider_name()


def resolved_judge_model(provider: str) -> str | None:
    configured = settings.eval_llm_judge_model.strip()
    if configured:
        return configured
    if provider == "openai":
        return settings.openai_model
    if provider == "openai_compatible":
        return settings.llm_model
    if provider == "gemini":
        return settings.gemini_model
    if provider == "ollama":
        return settings.ollama_model
    if provider == "llama_cpp":
        return settings.llm_model_path
    return settings.effective_chat_model or None


def judge_weight_for_run(explicit: float | None) -> float:
    if explicit is not None:
        return max(0.0, min(float(explicit), 1.0))
    return settings.effective_eval_llm_judge_weight


def evaluate_golden_answer(
    *,
    question: str,
    expected_answer: str,
    expected_answers: list[str],
    actual_answer: str,
    retrieved: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> dict[str, Any]:
    provider = resolved_judge_provider()
    model = resolved_judge_model(provider)
    if provider in {"none", "extractive"} or not provider_available(provider):
        return {
            "judge_provider": provider,
            "judge_model": model,
            "judge_metrics": {},
            "judge_error": "LLM judge is enabled but no LLM provider is ready.",
        }

    prompt = JUDGE_USER_TEMPLATE.format(
        question=question,
        expected_answer=expected_answer,
        expected_answers=_preview_json(expected_answers),
        actual_answer=actual_answer,
        retrieved=_preview_json(retrieved),
        citations=_preview_json(citations),
    )
    started = time.perf_counter()
    try:
        with trace_span(
            "eval.llm_judge",
            {
                "eval.judge.provider": provider,
                "eval.judge.model": model or "",
                **content_attrs("eval.judge.question", question),
            },
        ) as span:
            result = complete_chat(
                prompt,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                provider=provider,
                model=model,
                response_format={"type": "json_object"},
                timeout_seconds=settings.effective_eval_llm_judge_timeout_seconds,
                max_tokens=settings.effective_eval_llm_judge_max_tokens,
            )
            payload = _extract_json_object(result.text)
            metrics = {
                "correctness": _score(payload.get("correctness")),
                "groundedness": _score(payload.get("groundedness")),
                "completeness": _score(payload.get("completeness")),
                "citation_support": _score(payload.get("citation_support")),
                "hallucination_risk": _score(payload.get("hallucination_risk")),
            }
            judge_score = round(
                max(
                    0.0,
                    (
                        metrics["correctness"] * 0.40
                        + metrics["groundedness"] * 0.25
                        + metrics["completeness"] * 0.20
                        + metrics["citation_support"] * 0.15
                    )
                    * (1.0 - metrics["hallucination_risk"] * 0.35),
                ),
                4,
            )
            verdict = _normalize_verdict(payload.get("verdict"), judge_score)
            latency_ms = int((time.perf_counter() - started) * 1000)
            span.set_attribute("eval.judge.score", judge_score)
            span.set_attribute("eval.judge.verdict", verdict)
            span.set_attribute("eval.judge.latency_ms", latency_ms)
            return {
                "judge_provider": result.provider or provider,
                "judge_model": result.model or model,
                "judge_score": judge_score,
                "judge_verdict": verdict,
                "judge_metrics": metrics,
                "judge_reason": str(payload.get("reason") or "").strip()[:1000],
                "judge_latency_ms": latency_ms,
                "judge_error": None,
            }
    except Exception as err:
        logger.warning("LLM judge failed: %s", err)
        return {
            "judge_provider": provider,
            "judge_model": model,
            "judge_metrics": {},
            "judge_latency_ms": int((time.perf_counter() - started) * 1000),
            "judge_error": f"{err.__class__.__name__}: {str(err)[:240]}",
        }
