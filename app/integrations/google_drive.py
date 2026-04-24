"""
Minimal Google Drive REST adapter for admin KB sync.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from app.config import settings
from app.tools.registry import ToolExecutionError

GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
GOOGLE_SHEET_MIME = "application/vnd.google-apps.spreadsheet"
GOOGLE_SLIDE_MIME = "application/vnd.google-apps.presentation"
GOOGLE_NATIVE_MIMES = {GOOGLE_DOC_MIME, GOOGLE_SHEET_MIME, GOOGLE_SLIDE_MIME}
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"

_TOKEN_CACHE: dict[str, tuple[float, str]] = {}


def _b64url_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _safe_export_filename(name: str, ext: str | None) -> str:
    cleaned = " ".join((name or "drive-file").split()).strip().rstrip(".")
    if not cleaned:
        cleaned = "drive-file"
    normalized_ext = (ext or "").strip().lstrip(".").lower()
    if normalized_ext and not cleaned.lower().endswith(f".{normalized_ext}"):
        return f"{cleaned}.{normalized_ext}"
    return cleaned


def normalize_google_drive_id(raw: str | None) -> str | None:
    value = " ".join(str(raw or "").split()).strip()
    if not value:
        return None

    if "://" not in value:
        return value

    parsed = urlparse(value)
    path_parts = [part for part in parsed.path.split("/") if part]
    for marker in ("folders", "drives", "d"):
        if marker in path_parts:
            marker_index = path_parts.index(marker)
            if marker_index + 1 < len(path_parts):
                return path_parts[marker_index + 1]

    query = parse_qs(parsed.query)
    for key in ("id", "folder", "driveId"):
        candidate = (query.get(key) or [None])[0]
        if candidate:
            return candidate

    return value


class GoogleDriveClient:
    def __init__(self) -> None:
        if not settings.google_drive_enabled:
            raise ToolExecutionError("Google Drive sync is disabled")
        service_account_file = settings.google_drive_service_account_file.strip()
        if not service_account_file:
            raise ToolExecutionError("RAG_GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE is not configured")
        self._service_account_path = Path(service_account_file)
        if not self._service_account_path.is_absolute():
            self._service_account_path = (Path.cwd() / self._service_account_path).resolve()
        self._service_account = self._load_service_account()
        self._timeout = float(settings.google_drive_timeout_seconds or 30)

    def _load_service_account(self) -> dict[str, Any]:
        try:
            return json.loads(self._service_account_path.read_text(encoding="utf-8"))
        except FileNotFoundError as err:
            raise ToolExecutionError(f"Google service account file not found: {self._service_account_path}") from err
        except json.JSONDecodeError as err:
            raise ToolExecutionError("Google service account file is not valid JSON") from err

    def _sign_jwt(self, assertion_header: dict[str, Any], assertion_payload: dict[str, Any]) -> str:
        private_key_pem = str(self._service_account.get("private_key") or "").strip()
        if not private_key_pem:
            raise ToolExecutionError("Google service account file is missing private_key")
        signing_input = f"{_b64url_json(assertion_header)}.{_b64url_json(assertion_payload)}".encode("ascii")
        try:
            private_key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
            signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
        except Exception as err:
            raise ToolExecutionError("Unable to sign Google service account JWT assertion") from err
        return f"{signing_input.decode('ascii')}.{base64.urlsafe_b64encode(signature).decode('ascii').rstrip('=')}"

    async def _access_token(self) -> str:
        cache_key = str(self._service_account_path)
        cached = _TOKEN_CACHE.get(cache_key)
        now = time.time()
        if cached and cached[0] > now:
            return cached[1]

        client_email = str(self._service_account.get("client_email") or "").strip()
        token_uri = str(self._service_account.get("token_uri") or DEFAULT_TOKEN_URI).strip()
        if not client_email:
            raise ToolExecutionError("Google service account file is missing client_email")

        issued_at = int(now)
        payload = {
            "iss": client_email,
            "scope": DRIVE_SCOPE,
            "aud": token_uri,
            "exp": issued_at + 3600,
            "iat": issued_at,
        }
        delegated_subject = settings.google_drive_delegated_subject.strip()
        if delegated_subject:
            payload["sub"] = delegated_subject
        assertion = self._sign_jwt({"alg": "RS256", "typ": "JWT"}, payload)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    token_uri,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                        "assertion": assertion,
                    },
                )
                response.raise_for_status()
                token_payload = response.json()
            except (httpx.HTTPError, ValueError) as err:
                raise ToolExecutionError("Failed to obtain Google Drive access token") from err

        access_token = str(token_payload.get("access_token") or "").strip()
        expires_in = int(token_payload.get("expires_in") or 3600)
        if not access_token:
            raise ToolExecutionError("Google OAuth token response did not include access_token")
        _TOKEN_CACHE[cache_key] = (now + max(60, expires_in - 60), access_token)
        return access_token

    async def _request_json(self, method: str, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        token = await self._access_token()
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.request(method, url, params=params, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as err:
                detail = (err.response.text or "").strip()
                if detail:
                    detail = detail[:300]
                    raise ToolExecutionError(f"Google Drive request failed: {detail}") from err
                raise ToolExecutionError(f"Google Drive request failed: {url}") from err
            except (httpx.HTTPError, ValueError) as err:
                raise ToolExecutionError(f"Google Drive request failed: {url}") from err

    async def _request_bytes(self, method: str, url: str, *, params: dict[str, Any] | None = None) -> bytes:
        token = await self._access_token()
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.request(method, url, params=params, headers=headers)
                response.raise_for_status()
                return response.content
            except httpx.HTTPError as err:
                raise ToolExecutionError(f"Google Drive download failed: {url}") from err

    async def _list_folder_page(
        self,
        folder_id: str,
        *,
        shared_drive_id: str | None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "pageSize": max(1, int(settings.google_drive_sync_batch_size or 50)),
            "fields": "nextPageToken, files(id,name,mimeType,md5Checksum,size,modifiedTime,parents,version,driveId)",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if shared_drive_id:
            params["corpora"] = "drive"
            params["driveId"] = shared_drive_id
        if page_token:
            params["pageToken"] = page_token
        return await self._request_json("GET", "https://www.googleapis.com/drive/v3/files", params=params)

    async def list_files(
        self,
        folder_id: str,
        *,
        shared_drive_id: str | None = None,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        pending_folders = [folder_id]
        seen_folders: set[str] = set()
        collected: list[dict[str, Any]] = []

        while pending_folders:
            current_folder = pending_folders.pop(0)
            if current_folder in seen_folders:
                continue
            seen_folders.add(current_folder)
            next_page_token: str | None = None
            while True:
                page = await self._list_folder_page(
                    current_folder,
                    shared_drive_id=shared_drive_id,
                    page_token=next_page_token,
                )
                for item in page.get("files") or []:
                    if not isinstance(item, dict):
                        continue
                    mime_type = str(item.get("mimeType") or "").strip()
                    if mime_type == GOOGLE_FOLDER_MIME:
                        if recursive:
                            pending_folders.append(str(item.get("id") or ""))
                        continue
                    collected.append(item)
                next_page_token = str(page.get("nextPageToken") or "").strip() or None
                if not next_page_token:
                    break
        return collected

    def export_target_for_mime(self, mime_type: str) -> tuple[str | None, str | None]:
        normalized = (mime_type or "").strip()
        if normalized == GOOGLE_DOC_MIME:
            ext = settings.google_drive_export_google_doc_as.strip().lower() or "docx"
            return ext, {
                "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "pdf": "application/pdf",
                "txt": "text/plain",
            }.get(ext)
        if normalized == GOOGLE_SHEET_MIME:
            ext = settings.google_drive_export_google_sheet_as.strip().lower() or "xlsx"
            return ext, {
                "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "csv": "text/csv",
                "pdf": "application/pdf",
            }.get(ext)
        if normalized == GOOGLE_SLIDE_MIME:
            ext = settings.google_drive_export_google_slide_as.strip().lower() or "pdf"
            return ext, {
                "pdf": "application/pdf",
                "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            }.get(ext)
        return None, None

    async def download_file(self, item: dict[str, Any]) -> tuple[bytes, str, str | None]:
        file_id = str(item.get("id") or "").strip()
        if not file_id:
            raise ToolExecutionError("Google Drive item is missing id")
        name = str(item.get("name") or "drive-file").strip()
        mime_type = str(item.get("mimeType") or "").strip()

        if mime_type in GOOGLE_NATIVE_MIMES:
            ext, export_mime = self.export_target_for_mime(mime_type)
            if not export_mime:
                raise ToolExecutionError(f"Unsupported export format for Google native mime type: {mime_type}")
            content = await self._request_bytes(
                "GET",
                f"https://www.googleapis.com/drive/v3/files/{file_id}/export",
                params={"mimeType": export_mime},
            )
            return content, _safe_export_filename(name, ext), ext

        content = await self._request_bytes(
            "GET",
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            params={"alt": "media"},
        )
        suffix = Path(name).suffix.lstrip(".").lower() or None
        return content, _safe_export_filename(name, suffix), suffix
