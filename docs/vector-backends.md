# Vector backends

## Supported backends

| Backend | Install profile | Best for | Tradeoff |
| --- | --- | --- | --- |
| `numpy` | Core | MVP, easiest local mode, tests | Simple but not ideal for large datasets |
| `chroma` | RAG / full profile | Larger local KBs, more realistic persistence | Extra dependency and more setup |
| `qdrant` | Qdrant profile | Production-like scale, optional dense+sparse hybrid retrieval | Separate opt-in dependency and index rebuild when schema changes |

## Backend contract

The facade exposes:

- `add_chunks`
- `query`
- `delete_by_source`
- `delete_by_where`
- `get_stats`
- `healthcheck`

## Behavior notes

### Numpy fallback

- stores vectors in local `.npy` and metadata JSON files
- easiest to run locally
- warns at larger scale

### Chroma

- supports local persistent mode or external HTTP mode
- preferred when datasets grow beyond the MVP use case

### Qdrant

Install the optional profile:

```powershell
pip install -r requirements-qdrant.txt
```

Use a Qdrant server:

```dotenv
RAG_VECTOR_BACKEND=qdrant
RAG_QDRANT_URL=http://127.0.0.1:6333
RAG_QDRANT_COLLECTION_NAME=kb_chunks
```

For local persisted development, use `RAG_QDRANT_PATH=data/vectordb/qdrant` instead of `RAG_QDRANT_URL`.

Dense retrieval is the default. Hybrid mode is opt-in:

```dotenv
RAG_QDRANT_HYBRID_ENABLED=true
RAG_QDRANT_SPARSE_MODEL=Qdrant/bm25
RAG_QDRANT_HYBRID_PREFETCH_K=30
```

Hybrid mode stores named `dense` and `sparse` vectors, prefetches candidates from both, and fuses them with reciprocal-rank fusion (RRF). The existing application reranker still runs after fusion.

RRF scores are not cosine similarities. Tune `RAG_QDRANT_HYBRID_MIN_SIMILARITY_THRESHOLD`, `RAG_QDRANT_HYBRID_THRESHOLD_LOW`, and `RAG_QDRANT_HYBRID_THRESHOLD_GOOD` against the golden evaluation gate before production rollout.

Switching an existing Qdrant collection between dense-only and hybrid changes its vector schema. Create a new collection name or rebuild the collection before re-ingesting the KB.

References:

- [Qdrant hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [Qdrant hybrid search with reranking](https://qdrant.tech/documentation/tutorials-basics/reranking-hybrid-search/)

## Dimension mismatch

If you see an embedding dimension mismatch:

1. remove the old vector index
2. rebuild the KB
3. verify the embedding model or embedding model path did not change unexpectedly
