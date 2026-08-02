# Golden eval and mandatory RAG benchmark

Use this workflow before promoting a retrieval, embedding, reranker, cache, or
prompt change. The goal is to prove that a candidate config improves quality or
stays inside the agreed error budget.

## Sprint N+1: Benchmark & Promotion Gate

Sprint N+1 does not change embeddings or rebuild Qdrant collections by default.
It benchmarks the current index with three no-reindex candidates:

- `current_local`: BM25-lite baseline.
- `local_cross_encoder`: Qwen3 cross-encoder reranker only.
- `local_corrective_rag`: corrective RAG rewrite only.
- `local_cross_encoder_corrective`: reranker plus corrective RAG.

The promotion decision is evidence-based:

1. Build or review the active golden dataset.
2. Run `current_local` and keep the baseline `run_id`.
3. Restart the app with one candidate env overlay.
4. Run the candidate eval with `--baseline-run-id`.
5. Compare reports and promote only if quality and latency stay inside budget.

Generate env overlays:

```powershell
.\.venv\Scripts\python.exe scripts\rag_profile_manager.py write-env current_local `
  --output data\rag_profiles\current_local.env
.\.venv\Scripts\python.exe scripts\rag_profile_manager.py write-env local_cross_encoder `
  --output data\rag_profiles\local_cross_encoder.env
.\.venv\Scripts\python.exe scripts\rag_profile_manager.py write-env local_corrective_rag `
  --output data\rag_profiles\local_corrective_rag.env
.\.venv\Scripts\python.exe scripts\rag_profile_manager.py write-env local_cross_encoder_corrective `
  --output data\rag_profiles\local_cross_encoder_corrective.env
```

## 1. Export review-ready candidates from real local data

This creates a CSV from chat logs, support tickets, and inbound support email.
Rows are inactive by default because a knowledge owner must verify the reference
answer and source before a row becomes part of the quality gate.

```powershell
.\.venv\Scripts\python.exe scripts\seed_golden_dataset_from_db.py `
  --db data\metadata.db `
  --kb-id 1 `
  --limit 150 `
  --output data\golden_candidates.csv
```

The CSV uses the same columns as `/api/admin/evaluations/golden-dataset/upload`:

- `question`
- `expected_answer`
- `expected_source_file_id` and `expected_source_file_ids`
- `expected_chunk_ids`
- `expected_categories`
- `expected_keywords`
- `tags`
- `active`

Review steps:

1. Replace weak or incomplete `expected_answer` values with source-grounded answers.
2. Fill or correct expected source file IDs/chunk IDs where citations did not map cleanly.
3. Keep ambiguous rows inactive until the source of truth is clear.
4. Add a `reviewed` or `approved` tag to rows that a knowledge owner has checked.
5. Set reviewed rows to `active=true`.

Prepare an upload pack after review:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_golden_review_pack.py `
  --input data\golden_candidates.csv `
  --output data\golden_review_pack.csv `
  --report-output data\golden_review_pack.report.json
```

By default, this activates only rows tagged `reviewed`, `approved`,
`golden_reviewed`, or `source_verified`. For smoke testing only, you can activate
rows that already have answer + source + clean chunk IDs:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_golden_review_pack.py `
  --input data\golden_candidates.csv `
  --output data\golden_review_pack_smoke.csv `
  --report-output data\golden_review_pack_smoke.report.json `
  --activate-evidence-backed
```

Do not treat `smoke_auto_activated` rows as promotion-quality golden data until
a human has verified the expected answer and source.

Recommended coverage for the first gate:

- 80-150 active rows.
- Rows used for promotion should have a human-review tag: `reviewed`,
  `approved`, `golden_reviewed`, or `source_verified`.
- `smoke_auto_activated` rows are useful for wiring tests, but do not count as
  representative production coverage.
- Vietnamese with and without diacritics.
- English questions.
- exact policy/order/SKU identifiers.
- citation-heavy questions.
- support workflow and SLA questions.
- negative or unknown questions where the agent should refuse/fallback.
- follow-up or ambiguous questions.

Embedding or vector-index profiles, such as Qdrant, Qwen3, and BGE-M3 profiles,
must pass this representative coverage gate before promotion. `promote` blocks
those profiles unless the candidate report contains `readiness.coverage` with:

- at least 80 reviewed active rows.
- required buckets present: `vi`, `en`, `exact`, `citation`, `negative`, and
  `followup`.

Owner overrides are possible with `--allow-insufficient-golden`, but the reason
is recorded in the profile state history.

## 2. Import reviewed rows

Use the Admin UI golden dataset upload, or call the API through Swagger at
`/docs`. Keep candidate rows inactive until they are reviewed.

Endpoint:

```text
POST /api/admin/evaluations/golden-dataset/upload
```

## 3. Run a baseline on the current config

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_eval.py `
  --base-url http://127.0.0.1:8080 `
  --kb-id 1 `
  --name baseline_current_local `
  --limit 150 `
  --report-output data\rag_profiles\eval_current_local.json
```

For a smoke run while the dataset is still small:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_eval.py `
  --base-url http://127.0.0.1:8080 `
  --kb-id 1 `
  --limit 20 `
  --allow-small `
  --report-output data\eval_smoke.json
```

Record the returned `run_id`. This is the comparison baseline for candidate
RAG configs.

The JSON report also includes trace diagnostics:

- `failed_or_warn_examples`
- `diagnostics.top_latency_examples`
- `diagnostics.retrieval_miss_examples`
- `diagnostics.citation_mismatch_examples`

Each trace includes the query, corrective RAG metadata, expected source IDs,
matched source rank, retrieved candidates, reranker scores, Qdrant/hybrid
metadata when available, and explanation flags. Use this to understand why a
reranker, embedding model, or vector backend won or lost.

## 4. Benchmark Sprint N+1 candidate configs

Apply one env overlay at a time, restart the app, then run the candidate eval.
No reingest is required for these N+1 profiles because they do not change the
embedding model, vector dimension, or vector backend schema.

Cross-encoder reranker candidate:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_eval.py `
  --base-url http://127.0.0.1:8080 `
  --kb-id 1 `
  --name candidate_local_cross_encoder `
  --limit 150 `
  --baseline-run-id 123 `
  --max-metric-drop 0.03 `
  --fail-on-gate `
  --report-output data\rag_profiles\eval_local_cross_encoder.json
```

Corrective RAG candidate:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_eval.py `
  --base-url http://127.0.0.1:8080 `
  --kb-id 1 `
  --name candidate_local_corrective_rag `
  --limit 150 `
  --baseline-run-id 123 `
  --max-metric-drop 0.03 `
  --fail-on-gate `
  --report-output data\rag_profiles\eval_local_corrective_rag.json
```

Combined candidate:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_eval.py `
  --base-url http://127.0.0.1:8080 `
  --kb-id 1 `
  --name candidate_local_cross_encoder_corrective `
  --limit 150 `
  --baseline-run-id 123 `
  --max-metric-drop 0.03 `
  --fail-on-gate `
  --report-output data\rag_profiles\eval_local_cross_encoder_corrective.json
```

Summarize the candidate reports:

```powershell
.\.venv\Scripts\python.exe scripts\compare_rag_eval_reports.py `
  --baseline data\rag_profiles\eval_current_local.json `
  --candidate cross_encoder=data\rag_profiles\eval_local_cross_encoder.json `
  --candidate corrective=data\rag_profiles\eval_local_corrective_rag.json `
  --candidate combined=data\rag_profiles\eval_local_cross_encoder_corrective.json `
  --output data\rag_profiles\sprint_n1_promotion_summary.md
```

## 5. Promote or rollback

Promote only when the comparison gate passes, or when an owner records an
explicit override reason. Practical defaults:

- `avg_score` should not drop materially.
- `recall_at_k` should not drop more than 0.03-0.05.
- `citation_accuracy` should not drop more than 0.03-0.05.
- `latency_p95_ms` should not increase more than 25% unless accepted.

Keep these values with every promotion:

- baseline run ID
- candidate run ID
- config snapshot
- collection/index name
- embedding fingerprint
- reranker provider/model
- rollback target
