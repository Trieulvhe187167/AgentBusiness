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

### Production hybrid profile

Use a collection name that encodes the embedding model and retrieval schema. Do not reuse a dense-only collection after enabling sparse vectors.

```dotenv
RAG_VECTOR_BACKEND=qdrant
RAG_QDRANT_URL=http://127.0.0.1:6333
RAG_QDRANT_COLLECTION_NAME=kb_chunks_qdrant_hybrid_v1
RAG_QDRANT_HYBRID_ENABLED=true
RAG_QDRANT_SPARSE_MODEL=Qdrant/bm25
RAG_QDRANT_HYBRID_PREFETCH_K=50

# RRF scores need their own thresholds, calibrated with golden eval.
RAG_QDRANT_HYBRID_MIN_SIMILARITY_THRESHOLD=0.10
RAG_QDRANT_HYBRID_THRESHOLD_LOW=0.20
RAG_QDRANT_HYBRID_THRESHOLD_GOOD=0.35

# Optional, useful after reranking when one large file dominates top results.
RAG_RETRIEVAL_SOURCE_DIVERSIFICATION_ENABLED=true
RAG_RETRIEVAL_SOURCE_MAX_CHUNKS_PER_SOURCE=2

# Optional neural reranker after hybrid retrieval.
RAG_RERANKER_PROVIDER=cross_encoder
RAG_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
RAG_RERANKER_TOP_N=50
```

### Embedding profiles for Qdrant

Changing the embedding provider, model, dimension, prefix, or instruction changes the index contract. The app includes these values in the embedding fingerprint used for embedding cache, retrieval cache scope, eval snapshots, and ingest signatures. You still need a new Qdrant collection or a full rebuild when the embedding fingerprint changes.

Local SentenceTransformer baseline:

```dotenv
RAG_EMBEDDING_PROVIDER=sentence_transformers
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_EMBEDDING_DIMENSION=0
RAG_EMBEDDING_TRUST_REMOTE_CODE=false
RAG_QDRANT_COLLECTION_NAME=kb_chunks_bge_m3_v1
```

Qwen3 embedding profile:

```dotenv
RAG_EMBEDDING_PROVIDER=sentence_transformers
RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RAG_EMBEDDING_DIMENSION=0
RAG_EMBEDDING_QUERY_INSTRUCTION=Given a business support question in Vietnamese or English, retrieve relevant internal knowledge base passages.
RAG_QDRANT_COLLECTION_NAME=kb_chunks_qwen3_06b_v1
```

Remote OpenAI-compatible embedding server:

```dotenv
RAG_EMBEDDING_PROVIDER=openai_compatible
RAG_EMBEDDING_BASE_URL=http://127.0.0.1:8000/v1
RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RAG_EMBEDDING_DIMENSION=1024
RAG_QDRANT_COLLECTION_NAME=kb_chunks_qwen3_06b_remote_v1
```

Recommended rollout:

1. Run a Phase 0 golden baseline on the current backend.
2. Enable Qdrant dense-only in a new collection and re-ingest.
3. Enable Qdrant hybrid in a new collection and re-ingest.
4. Compare `recall_at_k`, `mrr`, `citation_accuracy`, `latency_p50_ms`, and `latency_p95_ms`.
5. Only then enable source diversification and neural reranking, one at a time.

`/api/debug/retrieval` includes `retrieval_mode`, `qdrant_score`, `qdrant_query_mode`, `qdrant_fusion`, `qdrant_prefetch_k`, reranker scores, and `source_diversified` so threshold calibration can inspect what happened for each result.

References:

- [Qdrant hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [Qdrant hybrid search with reranking](https://qdrant.tech/documentation/tutorials-basics/reranking-hybrid-search/)

## Dimension mismatch

If you see an embedding dimension mismatch:

1. remove the old vector index
2. rebuild the KB
3. verify the embedding model or embedding model path did not change unexpectedly
