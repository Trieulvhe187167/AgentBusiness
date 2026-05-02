"""
Application configuration with environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Paths
    # MVP required: yes
    # ------------------------------------------------------------------
    data_dir: Path = BASE_DIR / "data"
    raw_upload_dir: Path = BASE_DIR / "data" / "raw"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    vectordb_dir: Path = BASE_DIR / "data" / "vectordb"
    chroma_dir: Path = BASE_DIR / "data" / "vectordb" / "chroma"
    cache_dir: Path = BASE_DIR / "data" / "cache"
    sqlite_path: Path = BASE_DIR / "data" / "metadata.db"
    models_dir: Path = BASE_DIR / "models"

    # ------------------------------------------------------------------
    # Upload
    # MVP required: yes
    # ------------------------------------------------------------------
    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = [
        ".xlsx", ".xls", ".csv", ".pdf", ".html", ".htm",
        ".txt", ".md", ".docx", ".json", ".jsonl", ".ndjson", ".tsv", ".xml",
    ]

    # ------------------------------------------------------------------
    # Parsing
    # MVP required: yes
    # ------------------------------------------------------------------
    # Custom column overrides for CSV/Excel (empty = auto-detect)
    kb_text_columns: list[str] = []
    kb_meta_columns: list[str] = []

    # ------------------------------------------------------------------
    # Chunking
    # MVP required: yes
    # ------------------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 120

    # ------------------------------------------------------------------
    # Vector store
    # MVP required: yes
    # Advanced options: Chroma HTTP / persistent server
    # ------------------------------------------------------------------
    vector_backend: str = "numpy"  # chroma | numpy
    top_k: int = 10
    chroma_collection_name: str = "kb_chunks"
    chroma_http_url: str = ""
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"

    # Retrieval thresholds
    threshold_good: float = 0.60
    threshold_low: float = 0.40
    min_similarity_threshold: float = 0.30

    # Hashing fallback thresholds
    hashing_threshold_good: float = 0.32
    hashing_threshold_low: float = 0.18
    hashing_min_similarity_threshold: float = 0.12

    # ------------------------------------------------------------------
    # Answer presentation
    # MVP required: yes
    # ------------------------------------------------------------------
    max_citations: int = 3
    max_extractive_chunks: int = 2
    max_answer_chunks: int = 5
    debug_show_retrieval: bool = False

    # ------------------------------------------------------------------
    # Embeddings
    # MVP required: yes
    # ------------------------------------------------------------------
    # Multilingual default to support both VI and EN retrieval.
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = 64
    embedding_model_path: str = ""
    # Optional manual prefixes. Leave empty to auto-detect by model family.
    # E5 usually needs: query="query: " and passage="passage: "
    # BGE usually needs query instruction and empty passage prefix.
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""

    # ------------------------------------------------------------------
    # LLM
    # MVP required: no
    # Advanced modes: OpenAI-compatible, OpenAI, Gemini, Ollama, llama.cpp
    # ------------------------------------------------------------------
    llm_provider: str = "none"  # auto|openai|gemini|ollama|llama_cpp|openai_compatible|none
    answer_mode: str = "auto"   # auto|extractive|generative

    # ------------------------------------------------------------------
    # Agent / tool runtime
    # MVP required: no
    # Phase advanced rollout only
    # ------------------------------------------------------------------
    agent_serving_stack: str = "vllm"      # vllm|qwen_agent|sglang|ollama|llama_cpp|openai|gemini|custom
    agent_tool_protocol: str = "manual_json"  # manual_json|openai_tools
    agent_native_tool_calling: bool = False
    agent_tool_choice_mode: str = "auto"  # auto|required|none
    agent_tool_parser: str = ""
    agent_enable_llm_router: bool = True
    agent_brain_mode: str = "hybrid"  # hybrid|ai_first
    agent_followup_reaction_llm_timeout_seconds: int = 8
    agent_followup_reaction_llm_max_tokens: int = 96

    # OpenAI-compatible / vLLM path
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    llm_timeout_seconds: int = 120

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_timeout_seconds: int = 60

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout_seconds: int = 60

    # Ollama
    ollama_base_url: str = ""
    ollama_url: str = ""  # backward compatibility (old env: RAG_OLLAMA_URL)
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout_seconds: int = 120

    # llama.cpp
    llm_model_path: str = ""
    llm_n_ctx: int = 2048
    llm_n_threads: int = 0
    llm_temperature: float = 0.25
    llm_top_p: float = 0.9
    llm_max_tokens: int = 768
    llm_repeat_penalty: float = 1.1
    llm_n_batch: int = 512

    # ------------------------------------------------------------------
    # Session / logs
    # MVP required: chat logs yes, slot memory optional
    # ------------------------------------------------------------------
    session_ttl_minutes: int = 120
    chat_log_limit_default: int = 50
    conversation_memory_turn_limit: int = 5

    # ------------------------------------------------------------------
    # Authentication / authorization
    # MVP required: no
    # Production rollout only
    # ------------------------------------------------------------------
    auth_mode: str = "dev"  # dev|jwt|gateway
    allow_header_auth_in_dev: bool = True
    jwt_issuer: str = ""
    jwt_audience: str = ""
    jwt_jwks_url: str = ""
    jwt_shared_secret: str = ""
    jwt_jwks_cache_ttl_seconds: int = 300
    gateway_shared_secret: str = ""
    gateway_user_id_header: str = "X-Auth-User-Id"
    gateway_roles_header: str = "X-Auth-Roles"
    gateway_channel_header: str = "X-Auth-Channel"
    gateway_tenant_id_header: str = "X-Auth-Tenant-Id"
    gateway_org_id_header: str = "X-Auth-Org-Id"
    gateway_secret_header: str = "X-Auth-Gateway-Secret"

    # ------------------------------------------------------------------
    # External business integrations
    # MVP required: no
    # Phase advanced rollout only
    # ------------------------------------------------------------------
    integration_cache_ttl_seconds: int = 120
    integration_http_timeout_seconds: int = 15

    order_api_base_url: str = ""
    order_api_key: str = ""
    order_api_status_path: str = "/orders/status"
    order_api_recent_path: str = "/orders/recent"

    game_api_base_url: str = ""
    game_api_key: str = ""
    game_api_online_path: str = "/alliances/online"

    # ------------------------------------------------------------------
    # Google Drive sync
    # MVP required: no
    # Admin-only Knowledge Base sync source
    # ------------------------------------------------------------------
    google_drive_enabled: bool = False
    google_drive_service_account_file: str = ""
    google_drive_delegated_subject: str = ""
    google_drive_timeout_seconds: int = 30
    google_drive_export_google_doc_as: str = "docx"
    google_drive_export_google_sheet_as: str = "xlsx"
    google_drive_export_google_slide_as: str = "pdf"
    google_drive_sync_batch_size: int = 50

    # ------------------------------------------------------------------
    # Support email action tools
    # MVP required: no
    # Gmail/Outlook compatible mailbox adapter via IMAP + SMTP
    # ------------------------------------------------------------------
    email_integration_enabled: bool = False
    email_provider: str = "imap_smtp"
    email_imap_host: str = ""
    email_imap_port: int = 993
    email_imap_username: str = ""
    email_imap_password: str = ""
    email_imap_mailbox: str = "INBOX"
    email_imap_use_ssl: bool = True
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_smtp_username: str = ""
    email_smtp_password: str = ""
    email_smtp_use_tls: bool = True
    email_smtp_use_ssl: bool = False
    email_from_address: str = ""
    email_support_address: str = ""
    email_lookback_days: int = 7
    email_fetch_limit: int = 20

    # ------------------------------------------------------------------
    # Cache
    # MVP required: optional but enabled by default
    # ------------------------------------------------------------------
    cache_ttl_seconds: int = 3600
    cache_max_size_mb: int = 500

    # ------------------------------------------------------------------
    # Background worker
    # API defaults to in-process worker for local/dev. Production can set
    # this false on the API container and run `python -m app.worker`.
    # ------------------------------------------------------------------
    background_worker_enabled: bool = True
    background_worker_poll_interval_seconds: float = 0.5
    background_worker_heartbeat_interval_seconds: float = 5.0
    background_worker_stale_seconds: int = 60
    scheduled_sync_enabled: bool = True
    scheduled_sync_poll_interval_seconds: float = 10.0

    # Logging
    log_level: str = "INFO"

    def ensure_dirs(self):
        for directory in [
            self.data_dir,
            self.raw_upload_dir,
            self.processed_dir,
            self.vectordb_dir,
            self.chroma_dir,
            self.cache_dir,
            self.models_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def effective_threads(self) -> int:
        if self.llm_n_threads > 0:
            return self.llm_n_threads
        return max(1, (os.cpu_count() or 4) - 1)

    @property
    def effective_embedding_source(self) -> str:
        if self.embedding_model_path:
            path = Path(self.embedding_model_path)
            if not path.is_absolute():
                path = BASE_DIR / path
            return str(path)
        return self.embedding_model

    @property
    def effective_embedding_model_id(self) -> str:
        return self.embedding_model_path or self.embedding_model

    @property
    def effective_ollama_base_url(self) -> str:
        return self.ollama_base_url or self.ollama_url

    @property
    def normalized_vector_backend(self) -> str:
        backend = self.vector_backend.strip().lower()
        return backend if backend in {"chroma", "numpy"} else "chroma"

    @property
    def normalized_answer_mode(self) -> str:
        mode = self.answer_mode.strip().lower()
        return mode if mode in {"auto", "extractive", "generative"} else "auto"

    @property
    def normalized_llm_provider(self) -> str:
        provider = self.llm_provider.strip().lower()
        valid = {"auto", "openai", "gemini", "ollama", "llama_cpp", "openai_compatible", "none"}
        return provider if provider in valid else "auto"

    @property
    def normalized_agent_serving_stack(self) -> str:
        stack = self.agent_serving_stack.strip().lower()
        valid = {"vllm", "qwen_agent", "sglang", "ollama", "llama_cpp", "openai", "gemini", "custom"}
        return stack if stack in valid else "vllm"

    @property
    def normalized_agent_tool_protocol(self) -> str:
        protocol = self.agent_tool_protocol.strip().lower()
        valid = {"manual_json", "openai_tools"}
        return protocol if protocol in valid else "manual_json"

    @property
    def normalized_auth_mode(self) -> str:
        mode = self.auth_mode.strip().lower()
        return mode if mode in {"dev", "jwt", "gateway"} else "dev"

    @property
    def normalized_agent_tool_choice_mode(self) -> str:
        mode = self.agent_tool_choice_mode.strip().lower()
        valid = {"auto", "required", "none"}
        return mode if mode in valid else "auto"

    @property
    def normalized_agent_brain_mode(self) -> str:
        mode = self.agent_brain_mode.strip().lower()
        return mode if mode in {"hybrid", "ai_first"} else "hybrid"

    @property
    def agent_native_tool_status(self) -> str:
        if self.normalized_agent_tool_protocol != "openai_tools" or not self.agent_native_tool_calling:
            return "disabled"
        if self.normalized_llm_provider not in {"openai", "openai_compatible"}:
            return "misconfigured"
        if self.normalized_llm_provider == "openai_compatible" and not self.llm_base_url.strip():
            return "misconfigured"
        if self.normalized_llm_provider == "openai" and not self.openai_api_key.strip():
            return "misconfigured"
        if not self.effective_chat_model.strip():
            return "misconfigured"
        return "ready"

    @property
    def agent_native_tool_ready(self) -> bool:
        return self.agent_native_tool_status == "ready"

    @property
    def agent_native_tool_reason(self) -> str:
        status = self.agent_native_tool_status
        if status == "disabled":
            return "Native tool calling is disabled; the router will keep using the manual JSON path."
        if self.normalized_llm_provider not in {"openai", "openai_compatible"}:
            return "Native tool calling requires an OpenAI-compatible chat completions provider."
        if self.normalized_llm_provider == "openai_compatible" and not self.llm_base_url.strip():
            return "RAG_LLM_BASE_URL must be set for openai_compatible native tool calling."
        if self.normalized_llm_provider == "openai" and not self.openai_api_key.strip():
            return "RAG_OPENAI_API_KEY must be set for native OpenAI tool calling."
        if not self.effective_chat_model.strip():
            return "A target chat model must be configured before enabling native tool calling."
        return "Runtime is ready for native tool calling."

    @property
    def agent_native_tool_warning(self) -> str | None:
        if not self.agent_native_tool_ready:
            return None
        if self.normalized_agent_serving_stack in {"vllm", "sglang"} and not self.agent_tool_parser.strip():
            return "Set RAG_AGENT_TOOL_PARSER if your serving stack requires an explicit parser for auto tool choice."
        return None

    def validate_runtime_settings(self) -> None:
        if self.normalized_auth_mode != "gateway":
            return

        secret = self.gateway_shared_secret.strip()
        weak_placeholders = {
            "change-me",
            "changeme",
            "your-long-random-secret",
            "replace-me",
        }
        if not secret:
            raise ValueError("RAG_GATEWAY_SHARED_SECRET is required when RAG_AUTH_MODE=gateway")
        if secret.lower() in weak_placeholders or len(secret) < 16:
            raise ValueError("RAG_GATEWAY_SHARED_SECRET must be a non-placeholder secret with at least 16 characters")

    @property
    def effective_chat_model(self) -> str:
        provider = self.normalized_llm_provider
        if provider == "openai":
            return self.openai_model
        if provider == "gemini":
            return self.gemini_model
        if provider == "ollama":
            return self.ollama_model
        if provider == "llama_cpp":
            return Path(self.llm_model_path).name if self.llm_model_path else ""
        if provider == "none":
            return ""
        return self.llm_model


settings = Settings()
