"""
Internal user directory and production readiness helpers.

This layer does not replace JWT/gateway identity. It records users observed by
the app, allows admins to block/deactivate users, and can provide fallback roles
when an upstream identity source does not send role claims.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.database import execute_sync, fetch_all_sync, fetch_one_sync, utcnow_iso
from app.models import AuthContext

_ROLE_TOKEN_REPLACEMENTS = str.maketrans({" ": "_", ".": "_"})


def _normalize_roles(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(part).strip() for part in raw]
    else:
        values = [str(raw).strip()]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        role = value.lower().translate(_ROLE_TOKEN_REPLACEMENTS)
        role = "".join(ch for ch in role if ch.isalnum() or ch in {"_", "-", ":"}).strip("_")
        if not role or role in seen:
            continue
        seen.add(role)
        normalized.append(role)
    return normalized


def _loads_roles(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return _normalize_roles(parsed)


def _dumps_roles(roles: object) -> str:
    return json.dumps(_normalize_roles(roles), ensure_ascii=False)


class AppUserItem(BaseModel):
    id: int
    user_id: str
    display_name: str | None = None
    email: str | None = None
    roles: list[str] = Field(default_factory=list)
    last_roles: list[str] = Field(default_factory=list)
    channel: str | None = None
    tenant_id: str | None = None
    org_id: str | None = None
    source: str = "observed"
    is_active: bool = True
    created_by_user_id: str | None = None
    notes: str | None = None
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    created_at: str
    updated_at: str


class UpsertAppUserInput(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=200)
    display_name: str | None = Field(default=None, max_length=200)
    email: str | None = Field(default=None, max_length=300)
    roles: list[str] = Field(default_factory=list)
    is_active: bool = True
    tenant_id: str | None = Field(default=None, max_length=200)
    org_id: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=1000)

    @field_validator("roles")
    @classmethod
    def normalize_roles(cls, value: list[str]) -> list[str]:
        return _normalize_roles(value)


class UpdateAppUserInput(BaseModel):
    display_name: str | None = Field(default=None, max_length=200)
    email: str | None = Field(default=None, max_length=300)
    roles: list[str] | None = None
    is_active: bool | None = None
    tenant_id: str | None = Field(default=None, max_length=200)
    org_id: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=1000)

    @field_validator("roles")
    @classmethod
    def normalize_roles(cls, value: list[str] | None) -> list[str] | None:
        return _normalize_roles(value) if value is not None else None


class ListAppUsersOutput(BaseModel):
    total: int
    items: list[AppUserItem]


class ReadinessCheckItem(BaseModel):
    key: str
    label: str
    status: Literal["pass", "warn", "fail"]
    message: str
    fix: str | None = None


class ProductionReadinessOutput(BaseModel):
    ready_for_chat: bool
    ready_for_production: bool
    setup_complete: bool
    checks: list[ReadinessCheckItem]


def _serialize_user(row: dict) -> AppUserItem:
    return AppUserItem(
        id=int(row["id"]),
        user_id=str(row["user_id"]),
        display_name=row.get("display_name"),
        email=row.get("email"),
        roles=_loads_roles(row.get("roles_json")),
        last_roles=_loads_roles(row.get("last_roles_json")),
        channel=row.get("channel"),
        tenant_id=row.get("tenant_id"),
        org_id=row.get("org_id"),
        source=row.get("source") or "observed",
        is_active=bool(row.get("is_active")),
        created_by_user_id=row.get("created_by_user_id"),
        notes=row.get("notes"),
        first_seen_at=row.get("first_seen_at"),
        last_seen_at=row.get("last_seen_at"),
        created_at=row.get("created_at") or "",
        updated_at=row.get("updated_at") or "",
    )


def observe_app_user(auth: AuthContext) -> None:
    if not settings.access_management_enabled or not auth.user_id:
        return

    now = utcnow_iso()
    try:
        execute_sync(
            """
            INSERT INTO app_users (
                user_id, roles_json, last_roles_json, channel, tenant_id, org_id,
                source, is_active, first_seen_at, last_seen_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'observed', 1, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                last_roles_json = excluded.last_roles_json,
                channel = COALESCE(excluded.channel, app_users.channel),
                tenant_id = COALESCE(excluded.tenant_id, app_users.tenant_id),
                org_id = COALESCE(excluded.org_id, app_users.org_id),
                last_seen_at = excluded.last_seen_at,
                updated_at = excluded.updated_at
            """,
            (
                auth.user_id,
                _dumps_roles(auth.roles),
                _dumps_roles(auth.roles),
                auth.channel,
                auth.tenant_id,
                auth.org_id,
                now,
                now,
                now,
                now,
            ),
        )
    except sqlite3.OperationalError:
        # During very early startup or old DBs before migration, auth should not
        # make the whole application unusable.
        return


def apply_user_directory_policy(auth: AuthContext) -> AuthContext:
    if not settings.access_management_enabled or not auth.user_id:
        return auth

    try:
        row = fetch_one_sync("SELECT * FROM app_users WHERE user_id = ?", (auth.user_id,))
    except sqlite3.OperationalError:
        return auth
    if not row:
        return auth

    if settings.access_management_enforce_active and not bool(row.get("is_active")):
        raise HTTPException(status_code=403, detail="User is inactive")

    role_mode = settings.normalized_access_management_role_mode
    directory_roles = _loads_roles(row.get("roles_json"))
    if role_mode == "override":
        auth.roles = directory_roles
    elif role_mode == "fallback" and not auth.roles and directory_roles:
        auth.roles = directory_roles
    return auth


def register_auth_context(auth: AuthContext) -> AuthContext:
    observe_app_user(auth)
    return apply_user_directory_policy(auth)


def list_app_users(*, query: str | None = None, status: str = "all", limit: int = 100) -> ListAppUsersOutput:
    clauses: list[str] = []
    params: list[object] = []
    if query and query.strip():
        like = f"%{query.strip()}%"
        clauses.append("(user_id LIKE ? OR email LIKE ? OR display_name LIKE ?)")
        params.extend([like, like, like])
    if status == "active":
        clauses.append("is_active = 1")
    elif status == "inactive":
        clauses.append("is_active = 0")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = fetch_all_sync(
        f"""
        SELECT *
        FROM app_users
        {where}
        ORDER BY COALESCE(last_seen_at, created_at) DESC, id DESC
        LIMIT ?
        """,
        (*params, max(1, min(limit, 500))),
    )
    return ListAppUsersOutput(total=len(rows), items=[_serialize_user(row) for row in rows])


def upsert_app_user(payload: UpsertAppUserInput, *, auth: AuthContext) -> AppUserItem:
    now = utcnow_iso()
    execute_sync(
        """
        INSERT INTO app_users (
            user_id, display_name, email, roles_json, last_roles_json, channel,
            tenant_id, org_id, source, is_active, created_by_user_id, notes,
            first_seen_at, last_seen_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, '[]', NULL, ?, ?, 'admin', ?, ?, ?, NULL, NULL, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            display_name = excluded.display_name,
            email = excluded.email,
            roles_json = excluded.roles_json,
            tenant_id = excluded.tenant_id,
            org_id = excluded.org_id,
            source = 'admin',
            is_active = excluded.is_active,
            notes = excluded.notes,
            updated_at = excluded.updated_at
        """,
        (
            payload.user_id.strip(),
            payload.display_name,
            payload.email,
            _dumps_roles(payload.roles),
            payload.tenant_id,
            payload.org_id,
            1 if payload.is_active else 0,
            auth.user_id,
            payload.notes,
            now,
            now,
        ),
    )
    row = fetch_one_sync("SELECT * FROM app_users WHERE user_id = ?", (payload.user_id.strip(),))
    if not row:
        raise HTTPException(status_code=500, detail="User was not persisted")
    return _serialize_user(row)


def update_app_user(user_id: str, payload: UpdateAppUserInput, *, auth: AuthContext) -> AppUserItem:
    row = fetch_one_sync("SELECT * FROM app_users WHERE user_id = ?", (user_id,))
    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    updates: list[str] = []
    params: list[object] = []
    fields = payload.model_dump(exclude_unset=True)
    for field, value in fields.items():
        if field == "roles":
            updates.append("roles_json = ?")
            params.append(_dumps_roles(value))
        elif field == "is_active":
            updates.append("is_active = ?")
            params.append(1 if value else 0)
        else:
            updates.append(f"{field} = ?")
            params.append(value)
    updates.extend(["source = 'admin'", "updated_at = ?"])
    params.append(utcnow_iso())
    params.append(user_id)
    execute_sync(f"UPDATE app_users SET {', '.join(updates)} WHERE user_id = ?", tuple(params))
    updated = fetch_one_sync("SELECT * FROM app_users WHERE user_id = ?", (user_id,))
    if not updated:
        raise HTTPException(status_code=500, detail="User update failed")
    return _serialize_user(updated)


def build_production_readiness() -> ProductionReadinessOutput:
    checks: list[ReadinessCheckItem] = []

    checks.append(ReadinessCheckItem(
        key="system_running",
        label="System running",
        status="pass",
        message="API is responding.",
    ))

    kb_count = int((fetch_one_sync("SELECT COUNT(*) AS count FROM knowledge_bases") or {}).get("count") or 0)
    checks.append(ReadinessCheckItem(
        key="knowledge_base",
        label="Knowledge base created",
        status="pass" if kb_count > 0 else "fail",
        message=f"{kb_count} knowledge base(s) found.",
        fix=None if kb_count > 0 else "Create the first Knowledge Base in Knowledge Workspace.",
    ))

    ingested_count = int((fetch_one_sync("SELECT COUNT(*) AS count FROM uploaded_files WHERE status = 'ingested'") or {}).get("count") or 0)
    checks.append(ReadinessCheckItem(
        key="ingested_files",
        label="Files uploaded and ingested",
        status="pass" if ingested_count > 0 else "warn",
        message=f"{ingested_count} ingested file(s) found.",
        fix=None if ingested_count > 0 else "Upload source files and run ingest before relying on answers.",
    ))

    llm_ready = settings.normalized_llm_provider != "none" and bool(settings.effective_chat_model)
    checks.append(ReadinessCheckItem(
        key="llm_configured",
        label="LLM configured",
        status="pass" if llm_ready else "warn",
        message=(
            f"LLM provider is {settings.normalized_llm_provider}."
            if llm_ready else
            "LLM provider is 'none'; chat will use extractive answers."
        ),
        fix=None if llm_ready else "Set RAG_LLM_PROVIDER and the matching API/model settings.",
    ))

    auth_mode = settings.normalized_auth_mode
    auth_secure = auth_mode in {"jwt", "gateway"}
    checks.append(ReadinessCheckItem(
        key="auth_secured",
        label="Production auth secured",
        status="pass" if auth_secure else "fail",
        message=f"Auth mode is {auth_mode}.",
        fix=None if auth_secure else "Set RAG_AUTH_MODE=jwt or gateway before exposing the app to a team.",
    ))

    if auth_mode == "gateway":
        secret_ok = len(settings.gateway_shared_secret.strip()) >= 16
        checks.append(ReadinessCheckItem(
            key="gateway_secret",
            label="Gateway shared secret",
            status="pass" if secret_ok else "fail",
            message="Gateway shared secret is configured." if secret_ok else "Gateway shared secret is missing or weak.",
            fix=None if secret_ok else "Set a long random RAG_GATEWAY_SHARED_SECRET.",
        ))

    user_count = int((fetch_one_sync("SELECT COUNT(*) AS count FROM app_users") or {}).get("count") or 0)
    active_user_count = int((fetch_one_sync("SELECT COUNT(*) AS count FROM app_users WHERE is_active = 1") or {}).get("count") or 0)
    checks.append(ReadinessCheckItem(
        key="user_directory",
        label="User directory active",
        status="pass" if active_user_count > 0 else "warn",
        message=f"{active_user_count}/{user_count} active user(s) in directory.",
        fix=None if active_user_count > 0 else "Open Access Management or make an authenticated request to register the first user.",
    ))

    fail_count = sum(1 for item in checks if item.status == "fail")
    warn_count = sum(1 for item in checks if item.status == "warn")
    return ProductionReadinessOutput(
        ready_for_chat=kb_count > 0,
        ready_for_production=fail_count == 0 and warn_count == 0,
        setup_complete=kb_count > 0 and active_user_count > 0 and auth_secure,
        checks=checks,
    )
