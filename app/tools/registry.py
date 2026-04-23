"""
Tool registry, validation, auth policy, and audited execution.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.authorization import AuthorizationDeniedError, authorize_tool_access
from app.models import RequestContext
from app.tool_audit import log_tool_call


class ToolValidationError(ValueError):
    pass


class ToolAuthorizationError(PermissionError):
    pass


class ToolExecutionError(RuntimeError):
    pass


class ToolAuthPolicy(BaseModel):
    allow_anonymous: bool = False
    require_user_id: bool = False
    required_roles: list[str] = Field(default_factory=list)
    allowed_channels: list[str] = Field(default_factory=list)
    requires_tenant_match: bool = False
    risk_level: str = "low"
    scope: str = "general"


class ToolDefinitionSummary(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    auth_policy: dict[str, Any]
    timeout_seconds: int
    idempotent: bool

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    auth_policy: ToolAuthPolicy
    timeout_seconds: int
    idempotent: bool
    handler: Callable[[BaseModel, RequestContext], Any]
    summarize_result: Callable[[dict[str, Any]], str] | None = None

    def summary(self) -> ToolDefinitionSummary:
        return ToolDefinitionSummary(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
            output_schema=self.output_model.model_json_schema(),
            auth_policy=self.auth_policy.model_dump(),
            timeout_seconds=self.timeout_seconds,
            idempotent=self.idempotent,
        )


class ToolExecutionResult(BaseModel):
    tool_name: str
    tool_call_id: str
    output: dict[str, Any]
    latency_ms: int


@dataclass(slots=True)
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"Unknown tool: {name}")
        return spec

    def list_definitions(self) -> list[ToolDefinitionSummary]:
        return [self._tools[name].summary() for name in sorted(self._tools)]

    def list_openai_tools(self) -> list[dict[str, Any]]:
        return [item.to_openai_tool() for item in self.list_definitions()]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        *,
        request_context: RequestContext | dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        spec = self.get(name)
        context = request_context if isinstance(request_context, RequestContext) else RequestContext.model_validate(
            request_context or {"request_id": "tool-exec"}
        )
        raw_args = arguments or {}
        tool_call_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()

        try:
            self._authorize(spec, context, raw_args)
            validated_input = spec.input_model.model_validate(raw_args)
        except ToolAuthorizationError as err:
            self._log_failure(spec, tool_call_id, context, raw_args, "permission_denied", err, started)
            raise
        except Exception as err:  # pydantic validation errors land here
            wrapped = ToolValidationError(str(err))
            self._log_failure(spec, tool_call_id, context, raw_args, "validation_error", wrapped, started)
            raise wrapped from err

        try:
            payload = await asyncio.wait_for(
                self._invoke(spec, validated_input, context),
                timeout=spec.timeout_seconds,
            )
        except asyncio.TimeoutError as err:
            wrapped = ToolExecutionError(f"Tool '{name}' timed out after {spec.timeout_seconds}s")
            self._log_failure(spec, tool_call_id, context, raw_args, "timeout", wrapped, started)
            raise wrapped from err
        except Exception as err:
            if isinstance(err, ToolValidationError):
                status = "validation_error"
                wrapped = err
            elif isinstance(err, ToolAuthorizationError):
                status = "permission_denied"
                wrapped = err
            else:
                status = "error"
                wrapped = err if isinstance(err, ToolExecutionError) else ToolExecutionError(str(err))
            self._log_failure(spec, tool_call_id, context, raw_args, status, wrapped, started)
            raise wrapped from err

        latency_ms = int((time.perf_counter() - started) * 1000)
        summary = spec.summarize_result(payload) if spec.summarize_result else self._default_summary(spec.name, payload)
        log_tool_call(
            name,
            tool_call_id=tool_call_id,
            request_context=context,
            args=raw_args,
            tool_status="success",
            result_summary=summary,
            latency_ms=latency_ms,
        )
        return ToolExecutionResult(
            tool_name=name,
            tool_call_id=tool_call_id,
            output=payload,
            latency_ms=latency_ms,
        )

    def _authorize(self, spec: ToolSpec, context: RequestContext, raw_args: dict[str, Any]) -> None:
        try:
            authorize_tool_access(spec.name, spec.auth_policy, context=context, arguments=raw_args)
        except AuthorizationDeniedError as err:
            raise ToolAuthorizationError(str(err)) from err

    async def _invoke(self, spec: ToolSpec, validated_input: BaseModel, context: RequestContext) -> dict[str, Any]:
        result = spec.handler(validated_input, context)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, BaseModel):
            validated_output = spec.output_model.model_validate(result.model_dump())
        else:
            validated_output = spec.output_model.model_validate(result)
        return validated_output.model_dump()

    def _log_failure(
        self,
        spec: ToolSpec,
        tool_call_id: str,
        context: RequestContext,
        raw_args: dict[str, Any],
        status: str,
        err: Exception,
        started: float,
    ) -> None:
        log_tool_call(
            spec.name,
            tool_call_id=tool_call_id,
            request_context=context,
            args=raw_args,
            tool_status=status,
            result_summary=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error_message=str(err),
        )

    @staticmethod
    def _default_summary(tool_name: str, payload: dict[str, Any]) -> str:
        preview = json.dumps(payload, ensure_ascii=False)[:220]
        return f"{tool_name} completed: {preview}"
