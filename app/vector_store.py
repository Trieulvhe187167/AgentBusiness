"""
Pluggable vector store.

Backends:
- Chroma (persistent or HTTP server)
- Numpy (local fallback)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_SCALE_WARNING_THRESHOLD = 50_000


class NumpyVectorStore:
    """Thread-safe vector store backed by numpy files."""

    def __init__(self):
        self._vectors: np.ndarray | None = None
        self._metadatas: list[dict[str, Any]] = []
        self._documents: list[str] = []
        self._ids: list[str] = []
        self._dimension: int | None = None
        self._lock = threading.Lock()

    @property
    def _store_dir(self) -> Path:
        return Path(settings.vectordb_dir) / "numpy"

    @property
    def _vectors_path(self) -> Path:
        return self._store_dir / "vectors.npy"

    @property
    def _meta_path(self) -> Path:
        return self._store_dir / "metadata.json"

    def initialize(self, expected_dim: int | None = None):
        settings.ensure_dirs()
        self._store_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._dimension = expected_dim
            self._load_locked(expected_dim)

    def _load_locked(self, expected_dim: int | None):
        if self._vectors_path.exists() and self._meta_path.exists():
            try:
                self._vectors = np.load(str(self._vectors_path))
                with open(self._meta_path, "r", encoding="utf-8") as file_obj:
                    payload = json.load(file_obj)
                self._metadatas = payload.get("metadatas", [])
                self._documents = payload.get("documents", [])
                self._ids = payload.get("ids", [])

                if expected_dim and self._vectors.shape[1] != expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: stored={self._vectors.shape[1]} expected={expected_dim}"
                    )
                logger.info("Numpy vector store loaded: %s vectors", len(self._ids))
            except Exception as err:
                logger.warning("Failed to load numpy vectors, resetting store: %s", err)
                self._reset_locked()
        else:
            self._reset_locked()
            logger.info("Numpy vector store initialized (empty)")

    def _reset_locked(self):
        self._vectors = None
        self._metadatas = []
        self._documents = []
        self._ids = []

    def _save_locked(self):
        vtmp = str(self._store_dir / "_vectors.tmp.npy")
        mtmp = str(self._store_dir / "_metadata.tmp.json")

        if self._vectors is not None and self._ids:
            np.save(vtmp, self._vectors)
            with open(mtmp, "w", encoding="utf-8") as file_obj:
                json.dump(
                    {
                        "ids": self._ids,
                        "documents": self._documents,
                        "metadatas": self._metadatas,
                    },
                    file_obj,
                    ensure_ascii=False,
                )
            os.replace(vtmp, str(self._vectors_path))
            os.replace(mtmp, str(self._meta_path))
            return

        for path in [str(self._vectors_path), str(self._meta_path), vtmp, mtmp]:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def add_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]):
        if not chunks:
            return

        new_ids = [chunk["chunk_id"] for chunk in chunks]
        new_docs = [chunk["text"] for chunk in chunks]
        new_metas = [self._chunk_metadata(chunk) for chunk in chunks]
        new_vectors = np.array(embeddings, dtype=np.float32)

        with self._lock:
            old_to_keep = [i for i, chunk_id in enumerate(self._ids) if chunk_id not in set(new_ids)]

            if old_to_keep and self._vectors is not None:
                self._vectors = self._vectors[old_to_keep]
                self._metadatas = [self._metadatas[i] for i in old_to_keep]
                self._documents = [self._documents[i] for i in old_to_keep]
                self._ids = [self._ids[i] for i in old_to_keep]
            elif not old_to_keep:
                self._reset_locked()

            if self._vectors is not None and len(self._vectors) > 0:
                self._vectors = np.vstack([self._vectors, new_vectors])
            else:
                self._vectors = new_vectors

            self._ids.extend(new_ids)
            self._documents.extend(new_docs)
            self._metadatas.extend(new_metas)
            self._save_locked()

        total = len(self._ids)
        logger.info("Numpy vector upserted %s chunks (total=%s)", len(new_ids), total)
        if total >= _SCALE_WARNING_THRESHOLD:
            logger.warning(
                "Numpy vector store has %s vectors. Consider Chroma/Qdrant for large datasets.",
                total,
            )

    def _chunk_metadata(self, chunk: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            "chunk_id": chunk["chunk_id"],
            "source_id": str(chunk["source_id"]),
            "filename": chunk["filename"],
            "file_type": chunk["file_type"],
            "kb_version": chunk["kb_version"],
            "content_preview": chunk.get("content_preview", ""),
        }
        for key in (
            "page_num",
            "sheet_name",
            "row_num",
            "category",
            "keywords",
        ):
            value = chunk.get(key)
            if value is not None:
                metadata[key] = value
        return metadata

    def delete_by_source(self, source_id: str):
        with self._lock:
            if not self._ids:
                return

            keep_idx = [i for i, meta in enumerate(self._metadatas) if str(meta.get("source_id")) != str(source_id)]
            if len(keep_idx) == len(self._ids):
                return

            if keep_idx and self._vectors is not None:
                self._vectors = self._vectors[keep_idx]
                self._metadatas = [self._metadatas[i] for i in keep_idx]
                self._documents = [self._documents[i] for i in keep_idx]
                self._ids = [self._ids[i] for i in keep_idx]
            else:
                self._reset_locked()
            self._save_locked()
        logger.info("Deleted vectors for source_id=%s", source_id)

    def query(self, query_embedding: list[float], top_k: int | None = None, where: dict | None = None) -> list[dict[str, Any]]:
        k = top_k or settings.top_k

        with self._lock:
            if self._vectors is None or not self._ids:
                return []

            query_vec = np.array(query_embedding, dtype=np.float32)
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            vec_norms = self._vectors / (np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10)
            similarities = vec_norms @ query_norm

            valid_idx = list(range(len(self._ids)))
            if where:
                valid_idx = [
                    i
                    for i, meta in enumerate(self._metadatas)
                    if all(meta.get(key) == value for key, value in where.items())
                ]
            if not valid_idx:
                return []

            ranked = sorted(((i, similarities[i]) for i in valid_idx), key=lambda x: x[1], reverse=True)
            ranked = ranked[:k]

            out: list[dict[str, Any]] = []
            for idx, sim in ranked:
                out.append(
                    {
                        "chunk_id": self._ids[idx],
                        "text": self._documents[idx],
                        "distance": float(1.0 - sim),
                        "similarity": float(sim),
                        **self._metadatas[idx],
                    }
                )
            return out

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": "numpy",
                "total_vectors": len(self._ids),
                "dimension": int(self._vectors.shape[1]) if self._vectors is not None else 0,
                "collection_name": "numpy-local",
            }

    def get_sources(self) -> list[str]:
        with self._lock:
            return sorted({meta.get("filename", "") for meta in self._metadatas if meta.get("filename")})


class ChromaVectorStore:
    """Chroma vector backend with persistence support."""

    def __init__(self):
        self._client = None
        self._collection = None
        self._dimension: int | None = None
        self._lock = threading.Lock()

    def initialize(self, expected_dim: int | None = None):
        self._dimension = expected_dim
        self._collection = self._connect_collection()

    def _connect_collection(self):
        try:
            import chromadb
        except ImportError as err:
            raise RuntimeError("chromadb is not installed") from err

        if settings.chroma_http_url:
            parsed = urlparse(settings.chroma_http_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            ssl = parsed.scheme == "https"
            client = chromadb.HttpClient(host=host, port=port, ssl=ssl)
            logger.info("Using Chroma HTTP backend at %s", settings.chroma_http_url)
        else:
            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(settings.chroma_dir))
            logger.info("Using Chroma persistent backend at %s", settings.chroma_dir)

        self._client = client
        return client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _chunk_metadata(self, chunk: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            "chunk_id": chunk["chunk_id"],
            "source_id": str(chunk["source_id"]),
            "filename": chunk["filename"],
            "file_type": chunk["file_type"],
            "kb_version": chunk["kb_version"],
            "content_preview": chunk.get("content_preview", ""),
        }
        for key in (
            "page_num",
            "sheet_name",
            "row_num",
            "category",
            "keywords",
        ):
            value = chunk.get(key)
            if value is not None:
                metadata[key] = value
        return metadata

    def add_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]):
        if not chunks:
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        docs = [chunk["text"] for chunk in chunks]
        metas = [self._chunk_metadata(chunk) for chunk in chunks]

        with self._lock:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=docs,
                metadatas=metas,
            )
        logger.info("Chroma upserted %s chunks", len(ids))

    def delete_by_source(self, source_id: str):
        with self._lock:
            self._collection.delete(where={"source_id": str(source_id)})
        logger.info("Deleted Chroma vectors for source_id=%s", source_id)

    def query(self, query_embedding: list[float], top_k: int | None = None, where: dict | None = None) -> list[dict[str, Any]]:
        k = top_k or settings.top_k
        with self._lock:
            payload = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        ids = (payload.get("ids") or [[]])[0]
        docs = (payload.get("documents") or [[]])[0]
        metas = (payload.get("metadatas") or [[]])[0]
        dists = (payload.get("distances") or [[]])[0]

        out: list[dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            distance = float(dists[idx]) if idx < len(dists) else 1.0
            similarity = max(-1.0, min(1.0, 1.0 - distance))
            metadata = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
            out.append(
                {
                    "chunk_id": chunk_id,
                    "text": docs[idx] if idx < len(docs) else "",
                    "distance": distance,
                    "similarity": similarity,
                    **metadata,
                }
            )
        return out

    def get_stats(self) -> dict[str, Any]:
        count = self._collection.count() if self._collection else 0
        return {
            "backend": "chroma",
            "total_vectors": int(count),
            "dimension": self._dimension or 0,
            "collection_name": settings.chroma_collection_name,
        }

    def get_sources(self) -> list[str]:
        if not self._collection:
            return []
        count = self._collection.count()
        if count == 0:
            return []

        payload = self._collection.get(include=["metadatas"], limit=count)
        metadatas = payload.get("metadatas") or []
        return sorted(
            {
                meta.get("filename")
                for meta in metadatas
                if isinstance(meta, dict) and meta.get("filename")
            }
        )


class VectorStoreFacade:
    """Facade that switches backend based on settings."""

    def __init__(self):
        self._backend_name = "numpy"
        self._backend: Any = NumpyVectorStore()

    def initialize(self, expected_dim: int | None = None):
        preferred = settings.normalized_vector_backend

        if preferred == "chroma":
            try:
                backend = ChromaVectorStore()
                backend.initialize(expected_dim=expected_dim)
                self._backend = backend
                self._backend_name = "chroma"
                logger.info("Vector backend active: chroma")
                return
            except Exception as err:
                logger.warning("Chroma init failed, fallback to numpy: %s", err)

        backend = NumpyVectorStore()
        backend.initialize(expected_dim=expected_dim)
        self._backend = backend
        self._backend_name = "numpy"
        logger.info("Vector backend active: numpy")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def add_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]):
        self._backend.add_chunks(chunks, embeddings)

    def delete_by_source(self, source_id: str):
        self._backend.delete_by_source(source_id)

    def query(self, query_embedding: list[float], top_k: int | None = None, where: dict | None = None) -> list[dict[str, Any]]:
        return self._backend.query(query_embedding, top_k=top_k, where=where)

    def get_stats(self) -> dict[str, Any]:
        stats = self._backend.get_stats()
        stats["backend"] = self._backend_name
        return stats

    def get_sources(self) -> list[str]:
        return self._backend.get_sources()

    def get_index_fingerprint(self) -> str:
        stats = self.get_stats()
        return f"{stats.get('backend')}:{stats.get('total_vectors', 0)}"


vector_store = VectorStoreFacade()
