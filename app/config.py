"""
Application configuration with environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Paths
    data_dir: Path = BASE_DIR / "data"
    raw_upload_dir: Path = BASE_DIR / "data" / "raw"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    vectordb_dir: Path = BASE_DIR / "data" / "vectordb"
    chroma_dir: Path = BASE_DIR / "data" / "vectordb" / "chroma"
    cache_dir: Path = BASE_DIR / "data" / "cache"
    sqlite_path: Path = BASE_DIR / "data" / "metadata.db"
    models_dir: Path = BASE_DIR / "models"

    # Upload security
    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = [".xlsx", ".xls", ".csv", ".pdf", ".html", ".htm", ".txt", ".md"]

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 120

    # Vector backend
    vector_backend: str = "chroma"  # chroma | numpy
    top_k: int = 10
    chroma_collection_name: str = "kb_chunks"
    chroma_http_url: str = ""
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"

    # 3-mode thresholds
    threshold_good: float = 0.60
    threshold_low: float = 0.40
    min_similarity_threshold: float = 0.30

    # Auto thresholds when hashing embedding fallback is active
    hashing_threshold_good: float = 0.32
    hashing_threshold_low: float = 0.18
    hashing_min_similarity_threshold: float = 0.12

    # Answer presentation
    max_citations: int = 3
    max_extractive_chunks: int = 2
    max_answer_chunks: int = 5
    debug_show_retrieval: bool = False

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    embedding_model_path: str = ""
    # Optional manual prefixes. Leave empty to auto-detect by model family.
    # E5 usually needs: query="query: " and passage="passage: "
    # BGE usually needs query instruction and empty passage prefix.
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""

    # LLM provider switch
    llm_provider: str = "auto"  # auto|openai|gemini|ollama|llama_cpp|none
    answer_mode: str = "auto"   # auto|extractive|generative

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

    # Session and logs
    session_ttl_minutes: int = 120
    chat_log_limit_default: int = 50

    # Cache
    cache_ttl_seconds: int = 3600
    cache_max_size_mb: int = 500

    # Logging
    log_level: str = "INFO"

    class Config:
        env_prefix = "RAG_"
        env_file = ".env"
        env_file_encoding = "utf-8"

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
        valid = {"auto", "openai", "gemini", "ollama", "llama_cpp", "none"}
        return provider if provider in valid else "auto"


settings = Settings()
