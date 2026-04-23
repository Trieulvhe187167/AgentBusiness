# Vector backends

## Supported backends

| Backend | Install profile | Best for | Tradeoff |
| --- | --- | --- | --- |
| `numpy` | Core | MVP, easiest local mode, tests | Simple but not ideal for large datasets |
| `chroma` | RAG / full profile | Larger local KBs, more realistic persistence | Extra dependency and more setup |

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

## Dimension mismatch

If you see an embedding dimension mismatch:

1. remove the old vector index
2. rebuild the KB
3. verify the embedding model or embedding model path did not change unexpectedly
