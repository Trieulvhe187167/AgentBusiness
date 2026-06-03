from __future__ import annotations

import sys
from types import SimpleNamespace

import app.vector_store as vector_store_module
from app.config import settings
from app.vector_store import QdrantVectorStore, VectorStoreFacade


class _Value:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Models:
    Distance = SimpleNamespace(COSINE="cosine")
    Modifier = SimpleNamespace(IDF="idf")
    Fusion = SimpleNamespace(RRF="rrf")
    VectorParams = _Value
    SparseVectorParams = _Value
    Document = _Value
    PointStruct = _Value
    Filter = _Value
    FieldCondition = _Value
    MatchValue = _Value
    FilterSelector = _Value
    Prefetch = _Value
    FusionQuery = _Value


class _Client:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.created = []
        self.upserts = []
        self.queries = []
        self.deletes = []
        _Client.instances.append(self)

    def collection_exists(self, collection_name):
        return False

    def create_collection(self, **kwargs):
        self.created.append(kwargs)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-1",
                    score=0.42,
                    payload={"chunk_id": "chunk-1", "text": "Shipping fee", "filename": "faq.csv"},
                )
            ]
        )

    def delete(self, **kwargs):
        self.deletes.append(kwargs)

    def count(self, **kwargs):
        return SimpleNamespace(count=0)

    def scroll(self, **kwargs):
        return [], None

    def get_collection(self, collection_name):
        return SimpleNamespace()


def _install_fake_qdrant(monkeypatch):
    _Client.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "qdrant_client",
        SimpleNamespace(QdrantClient=_Client, models=_Models),
    )


def _chunk():
    return {
        "chunk_id": "chunk-1",
        "kb_id": 1,
        "source_id": "10",
        "file_id": 10,
        "filename": "faq.csv",
        "file_type": ".csv",
        "kb_version": "v1",
        "access_level": "internal",
        "text": "Shipping fee",
    }


def test_qdrant_hybrid_collection_ingest_and_query(monkeypatch):
    _install_fake_qdrant(monkeypatch)
    monkeypatch.setattr(settings, "qdrant_url", "http://127.0.0.1:6333")
    monkeypatch.setattr(settings, "qdrant_hybrid_enabled", True)
    monkeypatch.setattr(settings, "qdrant_hybrid_prefetch_k", 25)

    backend = QdrantVectorStore()
    backend.initialize(expected_dim=2)
    backend.add_chunks([_chunk()], [[1.0, 0.0]])
    results = backend.query(
        [1.0, 0.0],
        top_k=5,
        where={"kb_id": 1, "access_level": "internal"},
        query_text="delivery charge",
    )

    client = _Client.instances[-1]
    created = client.created[0]
    assert created["vectors_config"]["dense"].size == 2
    assert created["sparse_vectors_config"]["sparse"].modifier == "idf"
    point = client.upserts[0]["points"][0]
    assert point.vector["sparse"].model == "Qdrant/bm25"
    query = client.queries[0]
    assert query["query"].fusion == "rrf"
    assert [prefetch.using for prefetch in query["prefetch"]] == ["sparse", "dense"]
    assert [prefetch.limit for prefetch in query["prefetch"]] == [25, 25]
    assert len(query["query_filter"].must) == 2
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["retrieval_mode"] == "hybrid_rrf"


def test_qdrant_dense_query_does_not_require_query_text(monkeypatch):
    _install_fake_qdrant(monkeypatch)
    monkeypatch.setattr(settings, "qdrant_url", "http://127.0.0.1:6333")
    monkeypatch.setattr(settings, "qdrant_hybrid_enabled", False)

    backend = QdrantVectorStore()
    backend.initialize(expected_dim=2)
    backend.query([1.0, 0.0], top_k=3)

    query = _Client.instances[-1].queries[0]
    assert query["query"] == [1.0, 0.0]
    assert query["using"] == "dense"
    assert "prefetch" not in query


def test_facade_falls_back_to_numpy_when_qdrant_is_unavailable(monkeypatch, tmp_path):
    class _UnavailableQdrant:
        def initialize(self, expected_dim=None):
            raise RuntimeError("offline")

    monkeypatch.setattr(settings, "vector_backend", "qdrant")
    monkeypatch.setattr(settings, "vectordb_dir", tmp_path / "vectordb")
    monkeypatch.setattr(vector_store_module, "QdrantVectorStore", _UnavailableQdrant)

    facade = VectorStoreFacade()
    facade.initialize(expected_dim=2)

    assert facade.backend_name == "numpy"
