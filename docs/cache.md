# Cache Behavior

The app uses DiskCache for local cache storage. Exact-match embedding and retrieval caches remain enabled by default. Semantic retrieval and response caches are feature-flagged and disabled by default.

## Settings

```dotenv
RAG_CACHE_TTL_SECONDS=3600
RAG_CACHE_MAX_SIZE_MB=500
RAG_SEMANTIC_RETRIEVAL_CACHE_ENABLED=false
RAG_RESPONSE_CACHE_ENABLED=false
RAG_SEMANTIC_RESPONSE_CACHE_ENABLED=false
RAG_SEMANTIC_CACHE_THRESHOLD=0.96
RAG_SEMANTIC_CACHE_MAX_ENTRIES_PER_SCOPE=500
```

## Scope Rules

Retrieval cache scope includes:

- KB id and KB version
- vector backend
- embedding model id
- top_k
- vector filters, including access level, tenant id, and org id when present
- auth scope: channel, roles, tenant id, and org id

Response cache scope includes:

- KB id and KB version
- access level, tenant id, and org id
- auth scope
- language
- answer mode
- LLM provider and model
- embedding model id
- vector backend
- top_k and max answer chunks
- system prompt hash

Semantic cache is automatically disabled when hashing embeddings are active, because hashing fallback is not reliable enough for semantic equivalence.

## Response Cache Safety

Response cache is only used for RAG `answer` responses with citations. It is not used for:

- tool routes
- live-data/business integrations
- memory responses
- clarify/fallback responses
- follow-up questions
- sessions with recent conversation memory
- debug retrieval mode

This keeps cached answers from ignoring user-specific conversation context or live backend state.

## Measurement

`/api/cache/stats` includes semantic retrieval and semantic response scope/entry counts. Chat SSE `start` and `done` events include cache metadata such as response hit type, semantic score, cached query, and response store status.
