"""
Admin tools for Google Drive sync sources.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.drive_sync import (
    create_google_drive_source,
    delete_google_drive_source,
    get_google_drive_sync_status,
    list_google_drive_sources,
    sync_google_drive_source,
)
from app.models import RequestContext
from app.tools.registry import ToolAuthPolicy, ToolSpec


class GoogleDriveSourceItem(BaseModel):
    id: int
    kb_id: int
    name: str
    folder_id: str
    shared_drive_id: str | None = None
    recursive: bool
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    supported_mime_types: list[str] = Field(default_factory=list)
    delete_policy: str
    status: str
    tenant_id: str | None = None
    org_id: str | None = None
    created_by_user_id: str | None = None
    last_sync_at: str | None = None
    created_at: str
    updated_at: str


class ListGoogleDriveSourcesInput(BaseModel):
    pass


class ListGoogleDriveSourcesOutput(BaseModel):
    total: int
    items: list[GoogleDriveSourceItem]


class CreateGoogleDriveSourceInput(BaseModel):
    kb_id: int | None = Field(default=None, ge=1)
    kb_key: str | None = Field(default=None, min_length=1, max_length=80)
    name: str = Field(..., min_length=1, max_length=160)
    folder_id: str = Field(..., min_length=5, max_length=200)
    shared_drive_id: str | None = Field(default=None, max_length=200)
    recursive: bool = True
    include_patterns: list[str] = Field(default_factory=list, max_length=20)
    exclude_patterns: list[str] = Field(default_factory=list, max_length=20)
    supported_mime_types: list[str] = Field(default_factory=list, max_length=20)
    delete_policy: Literal["detach"] = "detach"


class CreateGoogleDriveSourceOutput(GoogleDriveSourceItem):
    pass


class SyncGoogleDriveSourceInput(BaseModel):
    source_id: int = Field(..., ge=1)
    force_full: bool = False


class SyncGoogleDriveSourceOutput(BaseModel):
    source_id: int
    kb_id: int
    run_id: int
    status: str
    scanned_count: int
    changed_count: int
    imported_count: int
    skipped_count: int
    failed_count: int
    queued_job_ids: list[str] = Field(default_factory=list)
    last_sync_at: str | None = None


class GetGoogleDriveSyncStatusInput(BaseModel):
    source_id: int = Field(..., ge=1)


class GoogleDriveSyncRunItem(BaseModel):
    id: int
    status: str
    scanned_count: int
    changed_count: int
    imported_count: int
    skipped_count: int
    failed_count: int
    started_at: str
    finished_at: str | None = None
    error_message: str | None = None


class GetGoogleDriveSyncStatusOutput(GoogleDriveSourceItem):
    last_run: GoogleDriveSyncRunItem | None = None


class DeleteGoogleDriveSourceInput(BaseModel):
    source_id: int = Field(..., ge=1)
    mode: Literal["unlink", "purge"] = "unlink"


class DeleteGoogleDriveSourceOutput(BaseModel):
    source_id: int
    kb_id: int
    name: str
    mode: str
    tracked_file_count: int
    detached_file_count: int
    deleted_file_count: int
    preserved_file_count: int
    message: str


async def _list_google_drive_sources_tool(_: ListGoogleDriveSourcesInput, __: RequestContext) -> dict[str, Any]:
    return list_google_drive_sources()


async def _create_google_drive_source_tool(payload: CreateGoogleDriveSourceInput, context: RequestContext) -> dict[str, Any]:
    return await create_google_drive_source(
        kb_id=payload.kb_id,
        kb_key=payload.kb_key,
        name=payload.name,
        folder_id=payload.folder_id,
        shared_drive_id=payload.shared_drive_id,
        recursive=payload.recursive,
        include_patterns=payload.include_patterns,
        exclude_patterns=payload.exclude_patterns,
        supported_mime_types=payload.supported_mime_types,
        delete_policy=payload.delete_policy,
        auth=context.auth,
    )


async def _sync_google_drive_source_tool(payload: SyncGoogleDriveSourceInput, context: RequestContext) -> dict[str, Any]:
    return await sync_google_drive_source(
        payload.source_id,
        triggered_by_user_id=context.auth.user_id,
        trigger_mode="tool",
        force_full=payload.force_full,
    )


async def _get_google_drive_sync_status_tool(payload: GetGoogleDriveSyncStatusInput, __: RequestContext) -> dict[str, Any]:
    return get_google_drive_sync_status(payload.source_id)


async def _delete_google_drive_source_tool(payload: DeleteGoogleDriveSourceInput, __: RequestContext) -> dict[str, Any]:
    return delete_google_drive_source(payload.source_id, mode=payload.mode)


def _admin_tool_policy() -> ToolAuthPolicy:
    return ToolAuthPolicy(
        required_roles=["admin"],
        allowed_channels=["admin"],
        risk_level="high",
        scope="admin",
    )


def build_list_google_drive_sources_tool() -> ToolSpec:
    return ToolSpec(
        name="list_google_drive_sources",
        description="List configured Google Drive sync sources for Knowledge Bases.",
        input_model=ListGoogleDriveSourcesInput,
        output_model=ListGoogleDriveSourcesOutput,
        auth_policy=_admin_tool_policy(),
        timeout_seconds=15,
        idempotent=True,
        handler=_list_google_drive_sources_tool,
        summarize_result=lambda payload: f"listed {payload.get('total', 0)} Google Drive source(s)",
    )


def build_create_google_drive_source_tool() -> ToolSpec:
    return ToolSpec(
        name="create_google_drive_source",
        description="Create a Google Drive sync source attached to a Knowledge Base.",
        input_model=CreateGoogleDriveSourceInput,
        output_model=CreateGoogleDriveSourceOutput,
        auth_policy=_admin_tool_policy(),
        timeout_seconds=20,
        idempotent=False,
        handler=_create_google_drive_source_tool,
        summarize_result=lambda payload: f"created Google Drive source {payload.get('id')}",
    )


def build_sync_google_drive_source_tool() -> ToolSpec:
    return ToolSpec(
        name="sync_google_drive_source",
        description="Sync one configured Google Drive source into its Knowledge Base and queue ingest jobs.",
        input_model=SyncGoogleDriveSourceInput,
        output_model=SyncGoogleDriveSourceOutput,
        auth_policy=_admin_tool_policy(),
        timeout_seconds=120,
        idempotent=False,
        handler=_sync_google_drive_source_tool,
        summarize_result=lambda payload: (
            f"synced source {payload.get('source_id')} with {payload.get('imported_count', 0)} imported file(s)"
        ),
    )


def build_get_google_drive_sync_status_tool() -> ToolSpec:
    return ToolSpec(
        name="get_google_drive_sync_status",
        description="Return the latest sync status for one Google Drive source.",
        input_model=GetGoogleDriveSyncStatusInput,
        output_model=GetGoogleDriveSyncStatusOutput,
        auth_policy=_admin_tool_policy(),
        timeout_seconds=10,
        idempotent=True,
        handler=_get_google_drive_sync_status_tool,
        summarize_result=lambda payload: f"Google Drive source {payload.get('id')} status fetched",
    )


def build_delete_google_drive_source_tool() -> ToolSpec:
    return ToolSpec(
        name="delete_google_drive_source",
        description="Delete a Google Drive sync source. Use mode=unlink to remove only the sync link, or mode=purge to also purge imported files from the bound KB.",
        input_model=DeleteGoogleDriveSourceInput,
        output_model=DeleteGoogleDriveSourceOutput,
        auth_policy=_admin_tool_policy(),
        timeout_seconds=30,
        idempotent=False,
        handler=_delete_google_drive_source_tool,
        summarize_result=lambda payload: f"deleted source {payload.get('source_id')} with mode {payload.get('mode')}",
    )
