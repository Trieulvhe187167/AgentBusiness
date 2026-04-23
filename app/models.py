"""
Pydantic schemas for API requests and responses.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


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


class HealthResponse(BaseModel):
    status: str = "ok"
    llm_loaded: bool = False
    embeddings_loaded: bool = False
    embeddings_backend: str = "hashing"   # "sentence-transformers" | "hashing"
    embeddings_ready: bool = False        # True only after warm-up completes
    vector_store_ready: bool = False
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
    user_message: str
    answer_text: str
    created_at: str


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
    user_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    channel: str = "web"
    tenant_id: str | None = None
    org_id: str | None = None


class SystemRuntime(BaseModel):
    scope: dict[str, str | int | bool | None]
    agent_runtime: dict[str, str | bool | None]
    vector_backend: str
    llm_provider_active: str
    llm_provider_config: str
    answer_mode_config: str
    top_k: int
    threshold_good: float
    threshold_low: float
    min_similarity_threshold: float
    embedding_model: str
    embedding_source: str
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
