# Enterprise pilot data playbook

Use this playbook to prepare evidence for a Support Operations pilot without
committing operational data or personally identifiable information to Git.

## Scope and ownership

Copy `config/pilot/support-operations.example.json` to the deployment secret or
configuration store and fill in the four owners. Do not commit a local copy that
contains real user, tenant, mailbox, or API identifiers.

The first pilot should use one internal or admin-scoped Knowledge Base. Do not
use the cross-project default KB as an enterprise source of truth. Only files in
`reviewed` or `published` lifecycle state should support golden answers.

## Build the privacy-safe review queue

```powershell
.\.venv\Scripts\python.exe scripts\pilot_dataset_manager.py build `
  --db data\metadata.db `
  --kb-id <pilot-kb-id> `
  --limit 150 `
  --output data\pilot\golden_review_queue.csv `
  --report-output data\pilot\golden_review_queue.report.json
```

The queue remains inactive and contains redacted questions, suggested answers,
provenance, coverage hints, PII flags, and owner-review columns. The source DB,
queue, and generated reports are under `data/`, which is ignored by Git.

For every accepted row, the knowledge owner must:

1. Verify or replace `expected_answer` using an approved source.
2. Verify the source file IDs and chunk IDs.
3. Set `source_verified=true` for source-backed cases.
4. Set `review_decision=approved`.
5. Fill `reviewer_id` and an ISO-8601 `reviewed_at` timestamp.
6. Keep ambiguous, duplicated, or policy-conflicting rows pending or rejected.

Negative/no-answer cases may intentionally omit source IDs, but they still need
an approved expected refusal/fallback response and a `negative` coverage tag.

## Finalize approved rows

```powershell
.\.venv\Scripts\python.exe scripts\pilot_dataset_manager.py finalize `
  --input data\pilot\golden_review_queue.csv `
  --db data\metadata.db `
  --output data\pilot\golden_upload.csv `
  --report-output data\pilot\golden_finalize.report.json `
  --minimum-reviewed 80
```

Finalization fails approved rows that have raw email, phone, or secret shapes;
missing reviewer metadata; missing or unverified evidence; sources outside the
selected KB; sources not ingested; or sources not reviewed/published.

## Audit readiness

```powershell
.\.venv\Scripts\python.exe scripts\pilot_data_readiness.py `
  --db data\metadata.db `
  --kb-id <pilot-kb-id> `
  --review-queue data\pilot\golden_review_queue.csv `
  --minimum-reviewed 80 `
  --output data\pilot\pilot_data_readiness.json
```

The auditor emits aggregate counts only. It does not print messages, email
addresses, phone numbers, ticket bodies, or expected answers.

The enterprise pilot remains blocked until:

- the KB is internal/admin scoped and assigned to a tenant or organization;
- at least one trusted source is reviewed or published;
- at least 80 golden rows are explicitly owner-reviewed;
- Vietnamese, English, exact, citation, negative, and follow-up buckets exist;
- raw operational PII stays outside Git and is redacted before review/export.

## Run the baseline

After uploading `data\pilot\golden_upload.csv`, run the existing dry-run first:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_eval.py `
  --base-url http://127.0.0.1:8080 `
  --kb-id <pilot-kb-id> `
  --limit 150 `
  --dry-run
```

Only create the production baseline after the dry-run reports representative
coverage. Record the run ID, config snapshot, KB version, index/collection, and
rollback target.
