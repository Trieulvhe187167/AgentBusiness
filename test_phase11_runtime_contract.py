from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.main as main
from app.config import settings
from tests.conftest import configure_test_env, run


def test_system_info_reports_phase0_runtime_contract(monkeypatch: pytest.MonkeyPatch, tmp_path):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(settings, "llm_model", "Qwen/Qwen3-4B-Instruct-2507")

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(llm_loaded=False, vector_store_ready=True, embeddings_loaded=False)
        )
    )

    system = run(main.system_info(request, kb_id=None, kb_key=None))

    assert system["agent_runtime"] == {
        "serving_stack": "vllm",
        "tool_protocol": "manual_json",
        "native_tool_calling": False,
        "tool_choice_mode": "auto",
        "native_tool_status": "disabled",
        "native_tool_ready": False,
        "native_tool_reason": "Native tool calling is disabled; the router will keep using the manual JSON path.",
        "native_tool_warning": None,
        "tool_parser": None,
        "target_model": "Qwen/Qwen3-4B-Instruct-2507",
    }
    assert system["llm_provider_config"] == "openai_compatible"
    assert system["llm_model"] == "Qwen/Qwen3-4B-Instruct-2507"
