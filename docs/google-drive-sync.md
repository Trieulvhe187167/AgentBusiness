# Google Drive -> Knowledge Base Sync

This integration lets admin users attach a Google Drive folder to a Knowledge Base and sync supported files into the existing upload + ingest pipeline.

## What it does

The sync flow:

1. Create a Google Drive source bound to one KB.
2. Scan the configured folder or shared drive folder.
3. Download or export changed files.
4. Upsert them into `uploaded_files`.
5. Attach them to `kb_files`.
6. Queue normal ingest jobs.
7. Reuse the same `uploaded_file_id` when a Drive file changes revision.

## Supported file types

Drive sync reuses the current parser pipeline, so it only imports files that the app already knows how to parse.

Direct file support:

- PDF
- DOCX
- XLSX
- CSV
- TXT
- MD
- JSON

Google native export support:

- Google Docs -> DOCX
- Google Sheets -> XLSX
- Google Slides -> PDF

## Required environment variables

```dotenv
RAG_GOOGLE_DRIVE_ENABLED=true
RAG_GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE=secrets/google-drive-service-account.json
RAG_GOOGLE_DRIVE_TIMEOUT_SECONDS=30
RAG_GOOGLE_DRIVE_EXPORT_GOOGLE_DOC_AS=docx
RAG_GOOGLE_DRIVE_EXPORT_GOOGLE_SHEET_AS=xlsx
RAG_GOOGLE_DRIVE_EXPORT_GOOGLE_SLIDE_AS=pdf
RAG_GOOGLE_DRIVE_SYNC_BATCH_SIZE=50
```

Optional:

```dotenv
RAG_GOOGLE_DRIVE_DELEGATED_SUBJECT=
```

Use `RAG_GOOGLE_DRIVE_DELEGATED_SUBJECT` only when your Google Workspace admin has configured domain-wide delegation for the service account.

## Admin endpoints

- `GET /api/admin/google-drive/sources`
- `POST /api/admin/google-drive/sources`
- `POST /api/admin/google-drive/sources/{source_id}/sync`
- `GET /api/admin/google-drive/sources/{source_id}/status`
- `DELETE /api/admin/google-drive/sources/{source_id}?mode=unlink|purge`

All of these require the admin role.

## Agent tools

- `list_google_drive_sources`
- `create_google_drive_source`
- `sync_google_drive_source`
- `get_google_drive_sync_status`
- `delete_google_drive_source`

These tools are admin-only and `channel=admin` only.

## Delete behavior

Phase 1 uses `delete_policy=detach`.

If a tracked file disappears from Google Drive:

- the Drive mapping is marked `deleted_remote`
- the file is detached from the bound KB
- vectors for that KB/file pair are removed

The raw uploaded file record is kept for auditability.

## Delete modes

When deleting a configured source:

- `mode=unlink`
  - remove only the sync source and its sync history
  - keep already imported files in the KB
- `mode=purge`
  - remove the sync source
  - detach imported files from the bound KB
  - delete vectors for those files in that KB
  - delete the underlying uploaded file only if it is no longer attached to any KB
