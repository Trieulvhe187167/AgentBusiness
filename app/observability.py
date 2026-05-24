"""
Optional OpenTelemetry tracing.

The app must remain runnable without OpenTelemetry packages installed. When
RAG_OTEL_ENABLED=false or dependencies are missing, all helpers degrade to no-op.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

from app.config import settings

logger = logging.getLogger(__name__)

_configured = False
_enabled = False
_provider: Any = None


class NoopSpan:
    def set_attribute(self, _: str, __: Any) -> None:
        return None

    def set_attributes(self, attributes: Mapping[str, Any] | None) -> None:
        for key, value in (attributes or {}).items():
            self.set_attribute(key, value)

    def add_event(self, _: str, __: Mapping[str, Any] | None = None) -> None:
        return None

    def record_exception(self, _: BaseException) -> None:
        return None


def _safe_attrs(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in (attributes or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            safe[str(key)] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(item, (str, bool, int, float)) for item in value):
            safe[str(key)] = list(value)
        else:
            safe[str(key)] = str(value)[:500]
    return safe


def content_attrs(prefix: str, text: str | None, *, limit: int = 500) -> dict[str, Any]:
    if not settings.otel_trace_content or not text:
        return {}
    return {f"{prefix}.content": str(text)[: max(1, limit)]}


def gen_ai_attrs(
    *,
    operation: str,
    system: str | None = None,
    model: str | None = None,
    request_model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": operation,
    }
    if system:
        attrs["gen_ai.system"] = system
    if model:
        attrs["gen_ai.response.model"] = model
    if request_model:
        attrs["gen_ai.request.model"] = request_model
    if max_tokens is not None:
        attrs["gen_ai.request.max_tokens"] = max_tokens
    if temperature is not None:
        attrs["gen_ai.request.temperature"] = temperature
    if top_p is not None:
        attrs["gen_ai.request.top_p"] = top_p
    return attrs


def tool_trace_attrs(
    *,
    name: str,
    call_id: str | None = None,
    status: str | None = None,
    risk_level: str | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": name,
        "tool.name": name,
    }
    if call_id:
        attrs["tool.call_id"] = call_id
    if status:
        attrs["tool.status"] = status
    if risk_level:
        attrs["tool.risk_level"] = risk_level
    if scope:
        attrs["tool.scope"] = scope
    return attrs


def retrieval_trace_attrs(
    *,
    query: str | None = None,
    top_k: int | None = None,
    kb_id: int | None = None,
    kb_key: str | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "retrieve",
        "rag.operation": "retrieve",
    }
    if top_k is not None:
        attrs["rag.top_k"] = top_k
    if kb_id is not None:
        attrs["rag.kb_id"] = kb_id
    if kb_key:
        attrs["rag.kb_key"] = kb_key
    attrs.update(content_attrs("rag.query", query, limit=240))
    return attrs


def workflow_trace_attrs(
    *,
    workflow_type: str,
    run_id: int | None = None,
    step: str | None = None,
    step_type: str | None = None,
    status: str | None = None,
    entity_type: str | None = None,
    entity_id: str | int | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "workflow.type": workflow_type,
    }
    if run_id is not None:
        attrs["workflow.run_id"] = run_id
    if step:
        attrs["workflow.step"] = step
    if step_type:
        attrs["workflow.step_type"] = step_type
    if status:
        attrs["workflow.status"] = status
    if entity_type:
        attrs["workflow.entity_type"] = entity_type
    if entity_id is not None:
        attrs["workflow.entity_id"] = str(entity_id)
    return attrs


def _parse_headers(raw: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in str(raw or "").split(","):
        name, sep, value = item.partition("=")
        if sep and name.strip() and value.strip():
            headers[name.strip()] = value.strip()
    return headers


def configure_tracing() -> bool:
    global _configured, _enabled, _provider
    if _configured:
        return _enabled
    _configured = True
    _enabled = False

    if not settings.otel_enabled:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except Exception as err:
        logger.warning("OpenTelemetry tracing requested but dependencies are unavailable: %s", err)
        return False

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": settings.mcp_server_version,
            "deployment.environment": settings.otel_environment,
        }
    )
    provider = TracerProvider(resource=resource)
    exporter_name = settings.otel_exporter.strip().lower()
    if exporter_name == "console":
        exporter = ConsoleSpanExporter()
    else:
        exporter_kwargs: dict[str, Any] = {}
        endpoint = settings.otel_exporter_otlp_endpoint.strip()
        if endpoint:
            exporter_kwargs["endpoint"] = endpoint
        headers = _parse_headers(settings.otel_exporter_otlp_headers)
        if headers:
            exporter_kwargs["headers"] = headers
        exporter = OTLPSpanExporter(**exporter_kwargs)

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _provider = provider
    _enabled = True
    logger.info("OpenTelemetry tracing enabled: exporter=%s service=%s", exporter_name or "otlp", settings.otel_service_name)
    return True


def shutdown_tracing() -> None:
    provider = _provider
    if provider is not None:
        try:
            provider.shutdown()
        except Exception:
            logger.debug("OpenTelemetry provider shutdown failed", exc_info=True)


def tracing_enabled() -> bool:
    return bool(_enabled)


def tracing_status() -> dict[str, Any]:
    return {
        "enabled": bool(settings.otel_enabled),
        "active": bool(_enabled),
        "service_name": settings.otel_service_name,
        "environment": settings.otel_environment,
        "exporter": settings.otel_exporter,
        "endpoint": settings.otel_exporter_otlp_endpoint or None,
        "trace_content": settings.otel_trace_content,
    }


@contextmanager
def trace_span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
    *,
    carrier: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    if not _enabled:
        yield NoopSpan()
        return

    try:
        from opentelemetry import propagate, trace
        from opentelemetry.trace import Status, StatusCode

        tracer = trace.get_tracer("agent-business")
        context = propagate.extract(dict(carrier or {})) if carrier else None
    except Exception as err:
        if settings.otel_fail_closed:
            raise
        logger.debug("OpenTelemetry span failed and was ignored: %s", err, exc_info=True)
        yield NoopSpan()
        return

    with tracer.start_as_current_span(name, context=context, attributes=_safe_attrs(attributes)) as span:
        try:
            yield span
        except Exception as err:
            span.record_exception(err)
            span.set_status(Status(StatusCode.ERROR, str(err)[:256]))
            raise
