"""
Pydantic schemas for API requests and responses.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


_ROLE_TOKEN_RE = re.compile(r"[^a-z0-9:_-]+")
_ACCESS_LEVEL_ALLOWED = {"public", "internal", "admin"}


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def _normalize_roles(raw_roles: object) -> list[str]:
    if raw_roles is None:
        return []

    if isinstance(raw_roles, str):
        values = [part.strip() for part in raw_roles.split(",")]
    elif isinstance(raw_roles, (list, tuple, set)):
        values = [str(part).strip() for part in raw_roles]
    else:
        values = [str(raw_roles).strip()]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        role = _ROLE_TOKEN_RE.sub("_", value.lower()).strip("_")
        if not role or role in seen:
            continue
        seen.add(role)
        normalized.append(role)
    return normalized


def _normalize_access_level(value: object) -> str:
    normalized = (_normalize_optional_text(str(value) if value is not None else None) or "public").lower()
    if normalized not in _ACCESS_LEVEL_ALLOWED:
        raise ValueError(f"Invalid access_level. Allowed: {sorted(_ACCESS_LEVEL_ALLOWED)}")
    return normalized


class AuthContext(BaseModel):
    user_id: str | None = Field(default=None, min_length=1, max_length=120)
    roles: list[str] = Field(default_factory=list)
    channel: str = Field(default="web", min_length=1, max_length=40)
    tenant_id: str | None = Field(default=None, min_length=1, max_length=120)
    org_id: str | None = Field(default=None, min_length=1, max_length=120)

    @field_validator("user_id", "tenant_id", "org_id", mode="before")
    @classmethod
    def _normalize_identity_fields(cls, value):
        return _normalize_optional_text(value)

    @field_validator("channel", mode="before")
    @classmethod
    def _normalize_channel(cls, value):
        return (_normalize_optional_text(value) or "web").lower()

    @field_validator("roles", mode="before")
    @classmethod
    def _normalize_role_list(cls, value):
        return _normalize_roles(value)


class RequestContext(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=80)
    session_id: str | None = None
    kb_id: int | None = Field(default=None, ge=1)
    kb_key: str | None = Field(default=None, min_length=1, max_length=80)
    auth: AuthContext = Field(default_factory=AuthContext)
    runtime_controls: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "session_id", "kb_key", mode="before")
    @classmethod
    def _normalize_request_fields(cls, value):
        return _normalize_optional_text(value)


class KnowledgeBaseCreate(BaseModel):
    key: str = Field(..., min_length=1, max_length=80)
    name: str = Field(..., min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1000)
    is_default: bool = False
    access_level: str = Field(default="public", min_length=1, max_length=40)
    tenant_id: str | None = Field(default=None, min_length=1, max_length=120)
    org_id: str | None = Field(default=None, min_length=1, max_length=120)

    @field_validator("access_level", mode="before")
    @classmethod
    def _normalize_kb_access_level(cls, value):
        return _normalize_access_level(value)

    @field_validator("tenant_id", "org_id", mode="before")
    @classmethod
    def _normalize_kb_scope_fields(cls, value):
        return _normalize_optional_text(value)


class KnowledgeBaseUpdate(BaseModel):
    key: str | None = Field(default=None, min_length=1, max_length=80)
    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1000)
    status: str | None = Field(default=None, max_length=40)
    is_default: bool | None = None
    access_level: str | None = Field(default=None, min_length=1, max_length=40)
    tenant_id: str | None = Field(default=None, min_length=1, max_length=120)
    org_id: str | None = Field(default=None, min_length=1, max_length=120)

    @field_validator("access_level", mode="before")
    @classmethod
    def _normalize_optional_kb_access_level(cls, value):
        if value is None:
            return None
        return _normalize_access_level(value)

    @field_validator("tenant_id", "org_id", mode="before")
    @classmethod
    def _normalize_optional_kb_scope_fields(cls, value):
        return _normalize_optional_text(value)


class KnowledgeBaseSummary(BaseModel):
    id: int
    key: str
    name: str
    description: str | None = None
    status: str
    access_level: str = "public"
    tenant_id: str | None = None
    org_id: str | None = None
    is_default: bool
    kb_version: str
    file_count: int = 0
    ingested_file_count: int = 0
    created_at: str
    updated_at: str


class KnowledgeBaseDeleteResponse(BaseModel):
    message: str
    id: int
    key: str


class KBFileSummary(BaseModel):
    kb_id: int
    file_id: int
    mapping_id: int
    filename: str
    original_name: str
    file_type: str
    file_size: int
    file_hash: str
    upload_status: str
    access_level: str = "public"
    tenant_id: str | None = None
    org_id: str | None = None
    owner_user_id: str | None = None
    kb_status: str
    chunk_count: int = 0
    ingest_signature: str | None = None
    last_job_id: str | None = None
    lifecycle_status: str = "draft"
    reviewed_by_user_id: str | None = None
    reviewed_at: str | None = None
    published_at: str | None = None
    archived_at: str | None = None
    quality_score: float | None = None
    stale_reason: str | None = None
    stale_detected_at: str | None = None
    attached_at: str
    last_ingest_at: str | None = None
    created_at: str


class FileInfo(BaseModel):
    id: int
    filename: str
    original_name: str
    file_type: str
    file_size: int
    file_hash: str
    status: str
    access_level: str = "public"
    tenant_id: str | None = None
    org_id: str | None = None
    owner_user_id: str | None = None
    parser_type: str | None = None
    pages_or_rows: int | None = None
    ingested_at: str | None = None
    error_message: str | None = None
    created_at: str


class FileVersionItem(BaseModel):
    id: int
    file_id: int
    version_number: int
    file_hash: str
    file_size: int
    filename: str
    original_name: str
    file_type: str
    parser_type: str | None = None
    pages_or_rows: int | None = None
    chunk_count: int | None = None
    ingest_signature: str | None = None
    has_snapshot: bool = False
    change_summary: str | None = None
    created_by_user_id: str | None = None
    created_at: str
    is_current: bool = False
    is_active: bool = False


class ListFileVersionsOutput(BaseModel):
    file_id: int
    current_version: int | None = None
    versions: list[FileVersionItem]


class DiffFileVersionsOutput(BaseModel):
    file_id: int
    from_version: FileVersionItem
    to_version: FileVersionItem
    changed: bool
    additions: int
    deletions: int
    from_line_count: int
    to_line_count: int
    diff_lines: list[str] = Field(default_factory=list)
    truncated: bool = False


class RollbackFileVersionInput(BaseModel):
    reingest: bool = True
    kb_id: int | None = None
    reason: str | None = Field(default=None, max_length=500)


class RollbackFileVersionOutput(BaseModel):
    message: str
    file: FileInfo
    restored_from: FileVersionItem
    restored_as: FileVersionItem
    changed: bool
    jobs: list[dict[str, Any]] = Field(default_factory=list)


class UploadResponse(BaseModel):
    message: str
    file: FileInfo


class IngestJobResponse(BaseModel):
    job_id: str
    file_id: int
    kb_id: int | None = None
    status: str


class JobStatus(BaseModel):
    job_id: str
    file_id: int
    kb_id: int | None = None
    status: str
    progress: float = 0.0
    error_message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=3000)
    session_id: str | None = None
    conversation_id: str | None = None  # backward compatibility
    lang: str | None = Field(default=None, description="Optional language hint: vi or en")
    kb_id: int | None = Field(default=None, ge=1)
    kb_key: str | None = Field(default=None, min_length=1, max_length=80)
    request_id: str | None = Field(default=None, min_length=1, max_length=80)
    user_id: str | None = Field(default=None, min_length=1, max_length=120)
    roles: list[str] = Field(default_factory=list)
    channel: str = Field(default="web", min_length=1, max_length=40)
    tenant_id: str | None = Field(default=None, min_length=1, max_length=120)
    org_id: str | None = Field(default=None, min_length=1, max_length=120)
    disable_reranker: bool = False
    disable_corrective_rag: bool = False

    @field_validator("session_id", "conversation_id", "request_id", "user_id", "tenant_id", "org_id", "kb_key", mode="before")
    @classmethod
    def _normalize_optional_request_fields(cls, value):
        return _normalize_optional_text(value)

    @field_validator("channel", mode="before")
    @classmethod
    def _normalize_chat_channel(cls, value):
        return (_normalize_optional_text(value) or "web").lower()

    @field_validator("roles", mode="before")
    @classmethod
    def _normalize_chat_roles(cls, value):
        return _normalize_roles(value)

    @property
    def resolved_session_id(self) -> str | None:
        return self.session_id or self.conversation_id

    @property
    def auth_context(self) -> AuthContext:
        return AuthContext(
            user_id=self.user_id,
            roles=self.roles,
            channel=self.channel,
            tenant_id=self.tenant_id,
            org_id=self.org_id,
        )

    def build_request_context(self, request_id: str) -> RequestContext:
        return RequestContext(
            request_id=request_id,
            session_id=self.resolved_session_id,
            kb_id=self.kb_id,
            kb_key=self.kb_key,
            auth=self.auth_context,
            runtime_controls={
                "disable_reranker": self.disable_reranker,
                "disable_corrective_rag": self.disable_corrective_rag,
            },
        )


class Citation(BaseModel):
    filename: str
    file_type: str
    page_num: int | None = None
    sheet_name: str | None = None
    row_range: str | None = None
    content_preview: str
    chunk_id: str
    score: float | None = None


class KBStats(BaseModel):
    total_files: int
    ingested_files: int
    total_chunks: int
    total_vectors: int
    sources: list[str]
    scope: str = "global"
    kb_id: int | None = None
    kb_key: str | None = None
    kb_name: str | None = None
    kb_version: str | None = None
    is_default: bool | None = None


class KBSource(BaseModel):
    source_id: int
    filename: str
    file_type: str
    chunk_count: int
    ingested_at: str | None = None


class CacheStats(BaseModel):
    total_entries: int
    size_mb: float
    semantic_retrieval_scopes: int = 0
    semantic_retrieval_entries: int = 0
    semantic_response_scopes: int = 0
    semantic_response_entries: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    llm_loaded: bool = False
    embeddings_loaded: bool = False
    embeddings_backend: str = "hashing"   # "sentence-transformers" | "hashing"
    embeddings_ready: bool = False        # True only after warm-up completes
    vector_store_ready: bool = False
    ready_for_chat: bool = False
    ready_for_production: bool = False
    setup_complete: bool = False
    issues: list[dict[str, str | None]] = Field(default_factory=list)
    timestamp: str


class DocumentSummary(BaseModel):
    doc_id: int
    file_name: str
    status: str
    chunks: int
    kb_version: str | None = None
    created_at: str
    ingested_at: str | None = None


class ChatLogItem(BaseModel):
    id: int
    session_id: str
    request_id: str | None = None
    user_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    channel: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    kb_id: int | None = None
    kb_key: str | None = None
    mode: str
    top_score: float | None = None
    latency_ms: int | None = None
    llm_provider: str | None = None
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0
    llm_cached_tokens: int = 0
    user_message: str
    answer_text: str
    created_at: str
    feedback_up: int = 0
    feedback_down: int = 0


class SubmitChatFeedbackInput(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=80)
    chat_log_id: int | None = Field(default=None, ge=1)
    rating: str = Field(..., min_length=1, max_length=10)
    reason_code: str | None = Field(default=None, max_length=80)
    comment: str | None = Field(default=None, max_length=1000)

    @field_validator("request_id", "reason_code", "comment", mode="before")
    @classmethod
    def _normalize_feedback_text(cls, value):
        return _normalize_optional_text(value)

    @field_validator("rating", mode="before")
    @classmethod
    def _normalize_rating(cls, value):
        normalized = (_normalize_optional_text(value) or "").lower()
        if normalized not in {"up", "down"}:
            raise ValueError("rating must be 'up' or 'down'")
        return normalized

    @model_validator(mode="after")
    def _require_target(self):
        if self.chat_log_id is None and not self.request_id:
            raise ValueError("request_id or chat_log_id is required")
        return self


class ChatFeedbackItem(BaseModel):
    id: int
    chat_log_id: int
    request_id: str | None = None
    rating: str
    reason_code: str | None = None
    comment: str | None = None
    created_by_user_id: str
    roles: list[str] = Field(default_factory=list)
    channel: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    chat_session_id: str | None = None
    kb_id: int | None = None
    kb_key: str | None = None
    chat_user_message: str | None = None
    chat_answer_text: str | None = None
    created_at: str
    updated_at: str


class ListChatFeedbackOutput(BaseModel):
    total: int
    items: list[ChatFeedbackItem]


class FeedbackSummaryGroup(BaseModel):
    kb_id: int | None = None
    kb_key: str | None = None
    total: int
    up: int
    down: int
    positive_rate: float | None = None


class FeedbackSummaryOutput(BaseModel):
    total: int
    up: int
    down: int
    positive_rate: float | None = None
    by_kb: list[FeedbackSummaryGroup] = Field(default_factory=list)


class AnalyticsSummary(BaseModel):
    chat_count: int = 0
    unique_users: int = 0
    avg_latency_ms: float | None = None
    fallback_count: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0
    llm_cached_tokens: int = 0
    llm_cached_input_rate: float | None = None
    feedback_total: int = 0
    feedback_up: int = 0
    feedback_down: int = 0
    positive_rate: float | None = None
    tool_calls: int = 0
    tool_error_rate: float | None = None
    background_jobs_total: int = 0
    background_jobs_failed: int = 0
    pending_actions_open: int = 0
    support_tickets_open: int = 0
    support_tickets_escalated: int = 0
    sla_overdue: int = 0
    uploaded_files: int = 0
    ingested_files: int = 0


class AnalyticsTimeBucket(BaseModel):
    bucket: str
    chats: int = 0
    feedback_up: int = 0
    feedback_down: int = 0
    tool_calls: int = 0
    job_failures: int = 0
    support_tickets: int = 0


class AnalyticsBreakdownItem(BaseModel):
    key: str
    count: int
    label: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AnalyticsDashboardOutput(BaseModel):
    period_days: int
    kb_id: int | None = None
    summary: AnalyticsSummary
    timeseries: list[AnalyticsTimeBucket] = Field(default_factory=list)
    chat_modes: list[AnalyticsBreakdownItem] = Field(default_factory=list)
    kb_usage: list[AnalyticsBreakdownItem] = Field(default_factory=list)
    top_tools: list[AnalyticsBreakdownItem] = Field(default_factory=list)
    job_status: list[AnalyticsBreakdownItem] = Field(default_factory=list)
    support_status: list[AnalyticsBreakdownItem] = Field(default_factory=list)
    support_intents: list[AnalyticsBreakdownItem] = Field(default_factory=list)
    pending_status: list[AnalyticsBreakdownItem] = Field(default_factory=list)


class AiOpsReplayInput(BaseModel):
    mode: str = Field(default="retrieval_only", max_length=40)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_replay_mode(cls, value):
        normalized = (_normalize_optional_text(value) or "retrieval_only").lower()
        if normalized != "retrieval_only":
            raise ValueError("mode must be 'retrieval_only'")
        return normalized


class KnowledgeGapClusterItem(BaseModel):
    cluster_key: str
    representative_query: str
    count: int
    kb_id: int | None = None
    kb_key: str | None = None
    mode: str | None = None
    min_score: float | None = None
    avg_score: float | None = None
    last_seen_at: str
    first_seen_at: str
    status: str = "new"
    owner_user_id: str | None = None
    priority: str = "P2"
    due_date: str | None = None
    overdue: bool = False
    status_reason: str | None = None
    suggested_action: str | None = None
    sample_queries: list[str] = Field(default_factory=list)


class ListKnowledgeGapClustersOutput(BaseModel):
    total: int
    period_days: int
    kb_id: int | None = None
    items: list[KnowledgeGapClusterItem] = Field(default_factory=list)


class UpdateKnowledgeGapStatusInput(BaseModel):
    status: str | None = Field(default=None, min_length=1, max_length=40)
    owner_user_id: str | None = Field(default=None, max_length=120)
    priority: str | None = Field(default=None, max_length=10)
    due_date: str | None = Field(default=None, max_length=40)
    status_reason: str | None = Field(default=None, max_length=500)

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_gap_status(cls, value):
        if value is None:
            return None
        normalized = (_normalize_optional_text(value) or "").lower()
        aliases = {"open": "new", "suggested": "patch_pending", "resolved": "fixed"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"new", "triaged", "source_needed", "patch_pending", "fixed", "ignored"}:
            raise ValueError("status must be one of: new, triaged, source_needed, patch_pending, fixed, ignored")
        return normalized

    @field_validator("owner_user_id", "due_date", "status_reason", mode="before")
    @classmethod
    def _normalize_optional_review_text(cls, value):
        return _normalize_optional_text(value)

    @field_validator("priority", mode="before")
    @classmethod
    def _normalize_gap_priority(cls, value):
        if value is None:
            return None
        normalized = (_normalize_optional_text(value) or "").upper()
        if normalized not in {"P0", "P1", "P2", "P3"}:
            raise ValueError("priority must be one of: P0, P1, P2, P3")
        return normalized


class CreateAgentEvalRunInput(BaseModel):
    name: str | None = Field(default=None, max_length=160)
    days: int = Field(default=7, ge=1, le=90)
    kb_id: int | None = Field(default=None, ge=1)
    limit: int = Field(default=50, ge=1, le=500)
    min_pass_score: int = Field(default=75, ge=0, le=100)
    min_warn_score: int = Field(default=50, ge=0, le=100)
    source: str = Field(default="chat_logs", max_length=40)
    alert_drop_threshold: float = Field(default=10.0, ge=0.0, le=100.0)
    baseline_run_id: int | None = Field(default=None, ge=1)
    max_metric_drop: float = Field(default=0.05, ge=0.0, le=1.0)
    llm_judge: bool | None = None
    llm_judge_weight: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_eval_name(cls, value):
        return _normalize_optional_text(value)

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_eval_source(cls, value):
        normalized = (_normalize_optional_text(value) or "chat_logs").lower()
        if normalized not in {"chat_logs", "golden_dataset"}:
            raise ValueError("source must be 'chat_logs' or 'golden_dataset'")
        return normalized

    @model_validator(mode="after")
    def _validate_thresholds(self):
        if self.min_warn_score > self.min_pass_score:
            raise ValueError("min_warn_score must be <= min_pass_score")
        return self


class AgentEvalCheck(BaseModel):
    name: str
    status: str
    impact: int = 0
    message: str


class AgentEvalResultItem(BaseModel):
    id: int
    run_id: int
    chat_log_id: int | None = None
    golden_item_id: int | None = None
    request_id: str | None = None
    kb_id: int | None = None
    kb_key: str | None = None
    mode: str | None = None
    top_score: float | None = None
    feedback_rating: str | None = None
    expected_answer: str | None = None
    answer_similarity: float | None = None
    recall_at_k: float | None = None
    citation_accuracy: float | None = None
    mrr: float | None = None
    source_match: float | None = None
    chunk_match: float | None = None
    category_match: float | None = None
    matched_source_rank: int | None = None
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_score: float | None = None
    judge_verdict: str | None = None
    judge_metrics: dict[str, Any] = Field(default_factory=dict)
    judge_reason: str | None = None
    judge_latency_ms: int | None = None
    judge_error: str | None = None
    latency_ms: int | None = None
    verdict: str
    score: float
    checks: list[AgentEvalCheck] = Field(default_factory=list)
    reason: str | None = None
    user_message: str
    answer_text: str
    created_at: str


class AgentEvalRunItem(BaseModel):
    id: int
    name: str
    status: str
    source: str = "chat_logs"
    kb_id: int | None = None
    kb_key: str | None = None
    period_days: int
    sample_size: int
    pass_count: int
    warn_count: int
    fail_count: int
    avg_score: float | None = None
    baseline_run_id: int | None = None
    metrics: dict[str, float | None] = Field(default_factory=dict)
    comparison: dict[str, Any] = Field(default_factory=dict)
    gate_status: str = "not_compared"
    created_by_user_id: str | None = None
    created_at: str
    completed_at: str | None = None


class AgentEvalRunDetail(AgentEvalRunItem):
    config: dict[str, Any] = Field(default_factory=dict)
    results: list[AgentEvalResultItem] = Field(default_factory=list)


class ListAgentEvalRunsOutput(BaseModel):
    total: int
    items: list[AgentEvalRunItem]


class GoldenDatasetItem(BaseModel):
    id: int
    kb_id: int
    question: str
    expected_answer: str
    expected_answers: list[str] = Field(default_factory=list)
    expected_source_file_id: int | None = None
    expected_source_file_ids: list[int] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_categories: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    active: bool = True
    created_by_user_id: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    created_at: str
    updated_at: str


class CreateGoldenDatasetItemInput(BaseModel):
    kb_id: int = Field(..., ge=1)
    question: str = Field(..., min_length=1, max_length=3000)
    expected_answer: str = Field(..., min_length=1, max_length=8000)
    expected_answers: list[str] = Field(default_factory=list)
    expected_source_file_id: int | None = Field(default=None, ge=1)
    expected_source_file_ids: list[int] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_categories: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    active: bool = True

    @field_validator("question", "expected_answer", mode="before")
    @classmethod
    def _normalize_required_text(cls, value):
        normalized = _normalize_optional_text(value)
        if not normalized:
            raise ValueError("field is required")
        return normalized

    @field_validator("expected_answers", "expected_chunk_ids", "expected_categories", "expected_keywords", "tags", mode="before")
    @classmethod
    def _normalize_text_list(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = [part.strip() for part in value.split(",")]
        elif isinstance(value, list):
            raw_items = value
        else:
            raise ValueError("must be a list or comma-separated string")
        items = []
        for item in raw_items:
            normalized = _normalize_optional_text(str(item))
            if normalized and normalized not in items:
                items.append(normalized)
        return items[:50]

    @field_validator("expected_source_file_ids", mode="before")
    @classmethod
    def _normalize_int_list(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = [part.strip() for part in value.split(",")]
        elif isinstance(value, list):
            raw_items = value
        else:
            raise ValueError("must be a list or comma-separated string")
        items: list[int] = []
        for item in raw_items:
            if item in (None, ""):
                continue
            parsed = int(item)
            if parsed < 1:
                raise ValueError("source file IDs must be positive integers")
            if parsed not in items:
                items.append(parsed)
        return items[:50]


class ListGoldenDatasetOutput(BaseModel):
    total: int
    items: list[GoldenDatasetItem]


class GoldenDatasetUploadOutput(BaseModel):
    created: int
    items: list[GoldenDatasetItem] = Field(default_factory=list)


class ToolAuditLogItem(BaseModel):
    id: int
    tool_call_id: str
    request_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    channel: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    kb_id: int | None = None
    kb_key: str | None = None
    tool_name: str
    tool_status: str
    args_json: str | None = None
    result_summary: str | None = None
    latency_ms: int | None = None
    error_message: str | None = None
    created_at: str


class AuthAuditLogItem(BaseModel):
    id: int
    request_id: str | None = None
    user_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    channel: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    resource_type: str
    resource_id: str | None = None
    action: str
    decision: str
    reason: str | None = None
    created_at: str


class CurrentUserProfile(BaseModel):
    authenticated: bool
    auth_mode: str
    debug_auth_inputs_enabled: bool
    access_management_enabled: bool = True
    access_role_mode: str = "fallback"
    user_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    channel: str = "web"
    tenant_id: str | None = None
    org_id: str | None = None


class SystemRuntime(BaseModel):
    scope: dict[str, str | int | bool | None]
    agent_runtime: dict[str, str | bool | None]
    observability: dict[str, str | bool | None]
    llm_capabilities: dict[str, Any] = Field(default_factory=dict)
    vector_backend: str
    retrieval_mode: str | None = None
    qdrant_hybrid_enabled: bool = False
    reranker_provider: str = "bm25_lite"
    reranker_model: str | None = None
    reranker_top_n: int | None = None
    llm_provider_active: str
    llm_provider_config: str
    answer_mode_config: str
    corrective_rag: dict[str, Any] = Field(default_factory=dict)
    runtime_budget: dict[str, Any] = Field(default_factory=dict)
    top_k: int
    threshold_good: float
    threshold_low: float
    min_similarity_threshold: float
    embedding_model: str
    embedding_source: str
    embedding_provider: str = "sentence_transformers"
    embedding_model_fingerprint: str | None = None
    embedding_dimension: int | None = None
    total_files: int
    ingested_files: int
    source_count: int
    total_vectors: int
    cache_entries: int
    cache_size_mb: float
    collection_name: str | None = None
    embedding_backend: str
    vector_backend_config: str
    vector_backend_active: str
    effective_threshold_good: float
    effective_threshold_low: float
    effective_min_similarity_threshold: float
    chunk_size: int
    chunk_overlap: int
    llm_model: str
    llm_loaded: bool
    vector_store_ready: bool
    embeddings_loaded: bool
    embeddings_ready: bool
    timestamp: str
