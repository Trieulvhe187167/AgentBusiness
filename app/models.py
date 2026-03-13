"""
Pydantic schemas for API requests and responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    key: str = Field(..., min_length=1, max_length=80)
    name: str = Field(..., min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1000)
    is_default: bool = False


class KnowledgeBaseUpdate(BaseModel):
    key: str | None = Field(default=None, min_length=1, max_length=80)
    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1000)
    status: str | None = Field(default=None, max_length=40)
    is_default: bool | None = None


class KnowledgeBaseSummary(BaseModel):
    id: int
    key: str
    name: str
    description: str | None = None
    status: str
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

    @property
    def resolved_session_id(self) -> str | None:
        return self.session_id or self.conversation_id


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
    mode: str
    top_score: float | None = None
    latency_ms: int | None = None
    llm_provider: str | None = None
    user_message: str
    answer_text: str
    created_at: str


class SystemRuntime(BaseModel):
    scope: dict[str, str | int | bool | None]
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
