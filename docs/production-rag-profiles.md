# Production RAG profiles

This workflow turns optional RAG features into benchmarked production profiles.
Use it after the golden dataset has enough active reviewed rows.

## Built-in profiles

Profiles are defined in `scripts/rag_profile_manager.py`.

| Profile | Purpose | Reindex required |
| --- | --- | --- |
| `current_local` | Current lightweight baseline; useful as a control run. | No |
| `local_cross_encoder` | Current index with Qwen3 cross-encoder reranking. | No |
| `local_corrective_rag` | Current index with one corrective RAG rewrite attempt. | No |
| `local_cross_encoder_corrective` | Combined Sprint N+1 reranker plus corrective candidate. | No |
| `qdrant_dense_bge_m3` | Qdrant dense retrieval with BGE-M3 embeddings. | Yes |
| `qdrant_hybrid_bge_m3` | Qdrant dense+sparse hybrid with source diversification. | Yes |
| `qdrant_hybrid_qwen3` | Qdrant hybrid with Qwen3 embeddings. | Yes |
| `qdrant_hybrid_qwen3_reranker` | Qdrant hybrid with Qwen3 embeddings and Qwen3 cross-encoder reranking. | Yes |

List profiles:

```powershell
.\.venv\Scripts\python.exe scripts\rag_profile_manager.py list
```

Show one profile as env:

```powershell
.\.venv\Scripts\python.exe scripts\rag_profile_manager.py show qdrant_hybrid_bge_m3 --format env
```

Write an env overlay:

```powershell
.\.venv\Scripts\python.exe scripts\rag_profile_manager.py write-env qdrant_hybrid_bge_m3 `
  --output data\rag_profiles\qdrant_hybrid_bge_m3.env
```

## Benchmark sequence

1. Run `current_local` against the active golden dataset and keep the returned `run_id`.
2. For Sprint N+1, benchmark `local_cross_encoder`, `local_corrective_rag`, and `local_cross_encoder_corrective` first because they do not require reindexing.
3. Apply a candidate env overlay in a local copy of `.env`.
4. Rebuild/reingest the KB only if the profile changes vector backend, embedding model, dimension, or hybrid schema.
5. Restart the app.
6. Run `scripts\run_baseline_eval.py` with `--baseline-run-id <current_run_id>` and `--fail-on-gate`.
7. Promote only when the gate passes or an owner records an explicit override.

Generate a benchmark checklist:

```powershell
.\.venv\Scripts\python.exe scripts\rag_profile_manager.py plan-benchmark `
  --base-url http://127.0.0.1:8080 `
  --kb-id 1 `
  --limit 150 `
  --baseline-run-id 123
```

Summarize completed report files:

```powershell
.\.venv\Scripts\python.exe scripts\compare_rag_eval_reports.py `
  --baseline data\rag_profiles\eval_current_local.json `
  --candidate cross_encoder=data\rag_profiles\eval_local_cross_encoder.json `
  --candidate corrective=data\rag_profiles\eval_local_corrective_rag.json `
  --candidate combined=data\rag_profiles\eval_local_cross_encoder_corrective.json `
  --output data\rag_profiles\sprint_n1_promotion_summary.md
```

## Promote

After a candidate benchmark passes:

```powershell
.\.venv\Scripts\python.exe scripts\rag_profile_manager.py promote qdrant_hybrid_bge_m3 `
  --candidate-report data\rag_profiles\eval_qdrant_hybrid_bge_m3.json `
  --baseline-run-id 123 `
  --reason "Golden eval passed and p95 latency stayed within budget"
```

Promotion records audit state in:

```text
data/rag_profiles/state.json
```

That state is intentionally separate from `.env`. It is the operator decision
log; deployment still applies environment variables through the normal runtime
path.

Profiles that change the embedding model, vector backend, collection schema, or
Qdrant hybrid mode require representative golden coverage before promotion. The
candidate report must include `readiness.coverage.representative_ready=true`.
This means at least 80 reviewed active rows and coverage for `vi`, `en`,
`exact`, `citation`, `negative`, and `followup` buckets. If an owner must
override this guard, use:

```powershell
.\.venv\Scripts\python.exe scripts\rag_profile_manager.py promote qdrant_hybrid_bge_m3 `
  --candidate-report data\rag_profiles\eval_qdrant_hybrid_bge_m3.json `
  --baseline-run-id 123 `
  --reason "Owner override: limited pilot KB, not production traffic" `
  --allow-insufficient-golden
```

## Rollback

If the promoted profile regresses quality, latency, or operations:

```powershell
.\.venv\Scripts\python.exe scripts\rag_profile_manager.py rollback `
  --reason "Citation accuracy regression after production smoke"
```

Rollback switches the recorded active profile back to the previous promoted
profile and appends an audit event. Then redeploy the previous env/collection
that the state points to.

## Suggested gates

Use stricter gates once the golden dataset has 80-150 reviewed active rows:

- `recall_at_k` should not drop more than 0.03-0.05.
- `citation_accuracy` should not drop more than 0.03-0.05.
- `avg_score` should not materially drop.
- `latency_p95_ms` should not increase more than 25% unless accepted by an owner.
- No high-risk tool/live-data route should be evaluated through semantic cache by default.

## Important rollout notes

- Qdrant dense and hybrid collections use different vector schemas; use separate collection names.
- Embedding model, dimension, prefix, instruction, or trust flag changes require a new index/reingest.
- Neural reranking can improve ranking but may exceed local CPU latency budgets; benchmark before promoting.
- Source diversification should be evaluated after hybrid retrieval so it does not hide useful repeated chunks prematurely.
