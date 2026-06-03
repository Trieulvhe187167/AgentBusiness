from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import app.llm_client as llm_client
from tests.conftest import configure_test_env


class _StreamingResponse:
    def __init__(self, lines: list[str]):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self.lines)


def test_openai_responses_stream_uses_sse_prompt_cache_and_usage_callback(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(llm_client.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(llm_client.settings, "openai_prompt_cache_key", "support-rag-v1")
    monkeypatch.setattr(llm_client.settings, "openai_prompt_cache_retention", "24h")

    captured: dict[str, Any] = {}
    usage: dict[str, int] = {}

    def fake_stream(method, url, **kwargs):
        captured.update({"method": method, "url": url, **kwargs})
        return _StreamingResponse(
            [
                "event: response.output_text.delta",
                f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': 'Hello '})}",
                f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': 'world'})}",
                f"data: {json.dumps({'type': 'response.completed', 'response': {'usage': {'input_tokens': 1200, 'output_tokens': 5, 'total_tokens': 1205, 'input_tokens_details': {'cached_tokens': 1024}}}})}",
                "data: [DONE]",
            ]
        )

    monkeypatch.setattr(llm_client.httpx, "stream", fake_stream)

    text = "".join(llm_client._openai_stream("hello", on_completed=usage.update))

    assert text == "Hello world"
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/responses")
    assert captured["json"]["stream"] is True
    assert captured["json"]["prompt_cache_key"] == "support-rag-v1"
    assert captured["json"]["prompt_cache_retention"] == "24h"
    assert usage == {
        "input_tokens": 1200,
        "output_tokens": 5,
        "total_tokens": 1205,
        "cached_tokens": 1024,
    }


def test_generate_stream_records_openai_usage_on_trace_span(tmp_path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(llm_client.settings, "openai_prompt_cache_key", "support-rag-v1")
    monkeypatch.setattr(llm_client.settings, "openai_prompt_cache_retention", "in-memory")

    attributes: dict[str, Any] = {}

    class _Span:
        def set_attribute(self, key, value):
            attributes[key] = value

    @contextmanager
    def fake_trace_span(*args, **kwargs):
        yield _Span()

    def fake_openai_stream(*args, on_completed=None, **kwargs):
        assert on_completed is not None
        on_completed(
            {
                "input_tokens": 800,
                "output_tokens": 10,
                "total_tokens": 810,
                "cached_tokens": 0,
            }
        )
        yield "ok"

    monkeypatch.setattr(llm_client, "trace_span", fake_trace_span)
    monkeypatch.setattr(llm_client, "_openai_stream", fake_openai_stream)

    assert "".join(llm_client.generate_stream("hello", provider="openai")) == "ok"
    assert attributes["gen_ai.request.prompt_cache_key_configured"] is True
    assert attributes["gen_ai.request.prompt_cache_retention"] == "in-memory"
    assert attributes["gen_ai.usage.input_tokens"] == 800
    assert attributes["gen_ai.usage.output_tokens"] == 10
    assert attributes["gen_ai.usage.total_tokens"] == 810
    assert attributes["gen_ai.usage.cached_tokens"] == 0
    assert llm_client.get_last_generation_usage() == {
        "input_tokens": 800,
        "output_tokens": 10,
        "total_tokens": 810,
        "cached_tokens": 0,
    }
