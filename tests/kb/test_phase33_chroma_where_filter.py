from app.vector_store import ChromaVectorStore


class _FakeChromaCollection:
    def __init__(self):
        self.deleted_where = None
        self.queried_where = None
        self.got_where = None

    def delete(self, where):
        self.deleted_where = where

    def query(self, **kwargs):
        self.queried_where = kwargs.get("where")
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def get(self, where=None, include=None):
        self.got_where = where
        return {"metadatas": []}


def test_chroma_delete_normalizes_multi_field_where():
    backend = ChromaVectorStore()
    collection = _FakeChromaCollection()
    backend._collection = collection

    backend.delete_by_where({"kb_id": 2, "file_id": 30})

    assert collection.deleted_where == {"$and": [{"kb_id": 2}, {"file_id": 30}]}


def test_chroma_query_and_get_normalize_multi_field_where():
    backend = ChromaVectorStore()
    collection = _FakeChromaCollection()
    backend._collection = collection

    backend.query([0.1, 0.2], where={"kb_id": 2, "access_level": "public"})
    backend.count_by_where({"kb_id": 2, "file_id": 30})

    assert collection.queried_where == {"$and": [{"kb_id": 2}, {"access_level": "public"}]}
    assert collection.got_where == {"$and": [{"kb_id": 2}, {"file_id": 30}]}


def test_chroma_keeps_single_field_and_operator_where_unchanged():
    assert ChromaVectorStore._normalize_where({"kb_id": 2}) == {"kb_id": 2}
    assert ChromaVectorStore._normalize_where({"$and": [{"kb_id": 2}, {"file_id": 30}]}) == {
        "$and": [{"kb_id": 2}, {"file_id": 30}]
    }
