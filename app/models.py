"""
Pydantic schemas for API requests and responses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    status: str


class JobStatus(BaseModel):
    job_id: str
    file_id: int
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
    total_vectors: int
    cache_entries: int
    cache_size_mb: float
