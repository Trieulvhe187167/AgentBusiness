from __future__ import annotations

import app.embeddings as embeddings
from app.config import settings


def test_embedding_instruction_prefix_and_fingerprint_change(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "sentence_transformers")
    monkeypatch.setattr(settings, "embedding_model", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.setattr(settings, "embedding_model_path", "")
    monkeypatch.setattr(settings, "embedding_query_prefix", "")
    monkeypatch.setattr(settings, "embedding_passage_prefix", "")
    monkeypatch.setattr(settings, "embedding_query_instruction", "Retrieve business support passages.")
    monkeypatch.setattr(settings, "embedding_document_instruction", "Represent an internal KB passage.")

    query_text = embeddings._prepare_texts(["Shipping fee?"], is_query=True)[0]
    doc_text = embeddings._prepare_texts(["Shipping is free."], is_query=False)[0]
    fingerprint = settings.effective_embedding_fingerprint

    assert query_text == "Retrieve business support passages.\nQuery: Shipping fee?"
    assert doc_text == "Represent an internal KB passage.\nDocument: Shipping is free."
    assert "provider:sentence_transformers" in fingerprint
    assert "model:Qwen/Qwen3-Embedding-0.6B" in fingerprint
    assert "query_instruction:Retrieve business support passages." in fingerprint


def test_sentence_transformer_provider_passes_trust_remote_code(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, source, **kwargs):
            captured["source"] = source
            captured["kwargs"] = kwargs

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True):
            captured["texts"] = texts
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(settings, "embedding_provider", "sentence_transformers")
    monkeypatch.setattr(settings, "embedding_model", "custom/model")
    monkeypatch.setattr(settings, "embedding_model_path", "")
    monkeypatch.setattr(settings, "embedding_trust_remote_code", True)
    monkeypatch.setattr(embeddings, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(embeddings, "_model", None)

    assert embeddings.get_dimension() == 3
    vectors = embeddings.embed_texts(["hello"], is_query=True)

    assert captured["source"] == "custom/model"
    assert captured["kwargs"] == {"trust_remote_code": True}
    assert vectors == [[1.0, 0.0, 0.0]]


def test_openai_compatible_embedding_provider_parses_sorted_embeddings(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }

    class FakeClient:
        def __init__(self, timeout, headers):
            captured["timeout"] = timeout
            captured["headers"] = headers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(settings, "embedding_provider", "openai_compatible")
    monkeypatch.setattr(settings, "embedding_base_url", "http://127.0.0.1:8000/v1/")
    monkeypatch.setattr(settings, "embedding_api_key", "secret")
    monkeypatch.setattr(settings, "embedding_model", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.setattr(settings, "embedding_model_path", "")
    monkeypatch.setattr(settings, "embedding_timeout_seconds", 12)
    monkeypatch.setattr(embeddings.httpx, "Client", FakeClient)

    vectors = embeddings.embed_texts(["first", "second"], is_query=False)

    assert captured["url"] == "http://127.0.0.1:8000/v1/embeddings"
    assert captured["json"] == {"model": "Qwen/Qwen3-Embedding-0.6B", "input": ["first", "second"]}
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["timeout"] == 12
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


def test_remote_embedding_dimension_uses_configured_value_without_probe(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "tei")
    monkeypatch.setattr(settings, "embedding_dimension", 1024)
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts, is_query=False: (_ for _ in ()).throw(AssertionError("probe not expected")))

    assert embeddings.get_dimension() == 1024
    assert embeddings.using_hashing_fallback() is False
    assert embeddings.embedding_backend_name() == "tei"
