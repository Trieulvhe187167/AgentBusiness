from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from tests.conftest import admin_headers, auth_headers, configure_test_env


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _make_hs256_token(secret: str, claims: dict[str, object]) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url(signature)}"


def _gateway_headers(
    secret: str,
    *,
    user_id: str,
    roles: list[str],
    channel: str = "web",
    tenant_id: str | None = None,
    org_id: str | None = None,
) -> dict[str, str]:
    headers = {
        "X-Auth-Gateway-Secret": secret,
        "X-Auth-User-Id": user_id,
        "X-Auth-Roles": ",".join(roles),
        "X-Auth-Channel": channel,
    }
    if tenant_id:
        headers["X-Auth-Tenant-Id"] = tenant_id
    if org_id:
        headers["X-Auth-Org-Id"] = org_id
    return headers


def test_dev_mode_keeps_header_auth_for_local_testing(tmp_path: Path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "auth_mode", "dev")
    monkeypatch.setattr(settings, "allow_header_auth_in_dev", True)

    with TestClient(main.app) as client:
        response = client.get(
            "/api/chat/kbs",
            headers=auth_headers(user_id="customer-001", roles=["customer"], channel="web"),
        )
        response.raise_for_status()
        payload = response.json()
        assert any(item["key"] == "default" for item in payload)


def test_api_me_reflects_effective_auth_profile_in_dev_and_jwt_modes(tmp_path: Path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)

    monkeypatch.setattr(settings, "auth_mode", "dev")
    monkeypatch.setattr(settings, "allow_header_auth_in_dev", True)
    with TestClient(main.app) as client:
        dev_response = client.get(
            "/api/me",
            headers=auth_headers(user_id="employee-001", roles=["employee"], channel="web"),
        )
        dev_response.raise_for_status()
        dev_profile = dev_response.json()
        assert dev_profile["authenticated"] is True
        assert dev_profile["auth_mode"] == "dev"
        assert dev_profile["debug_auth_inputs_enabled"] is True
        assert dev_profile["user_id"] == "employee-001"
        assert dev_profile["roles"] == ["employee"]
        assert dev_profile["channel"] == "web"

    monkeypatch.setattr(settings, "auth_mode", "jwt")
    monkeypatch.setattr(settings, "allow_header_auth_in_dev", False)
    monkeypatch.setattr(settings, "jwt_shared_secret", "phase23-secret")
    monkeypatch.setattr(settings, "jwt_issuer", "https://issuer.example.test")
    monkeypatch.setattr(settings, "jwt_audience", "campusrag-api")

    now = int(time.time())
    token = _make_hs256_token(
        "phase23-secret",
        {
            "sub": "customer-001",
            "roles": ["customer"],
            "iss": "https://issuer.example.test",
            "aud": "campusrag-api",
            "exp": now + 300,
        },
    )

    with TestClient(main.app) as client:
        jwt_response = client.get(
            "/api/me",
            headers={
                **admin_headers(),
                "Authorization": f"Bearer {token}",
            },
        )
        jwt_response.raise_for_status()
        jwt_profile = jwt_response.json()
        assert jwt_profile["authenticated"] is True
        assert jwt_profile["auth_mode"] == "jwt"
        assert jwt_profile["debug_auth_inputs_enabled"] is False
        assert jwt_profile["user_id"] == "customer-001"
        assert jwt_profile["roles"] == ["customer"]


def test_jwt_mode_requires_bearer_and_ignores_admin_header_override(tmp_path: Path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "auth_mode", "jwt")
    monkeypatch.setattr(settings, "allow_header_auth_in_dev", False)
    monkeypatch.setattr(settings, "jwt_shared_secret", "phase23-secret")
    monkeypatch.setattr(settings, "jwt_issuer", "https://issuer.example.test")
    monkeypatch.setattr(settings, "jwt_audience", "campusrag-api")

    now = int(time.time())
    customer_token = _make_hs256_token(
        "phase23-secret",
        {
            "sub": "customer-001",
            "roles": ["customer"],
            "iss": "https://issuer.example.test",
            "aud": "campusrag-api",
            "exp": now + 300,
        },
    )
    admin_token = _make_hs256_token(
        "phase23-secret",
        {
            "sub": "admin-001",
            "roles": ["admin"],
            "iss": "https://issuer.example.test",
            "aud": "campusrag-api",
            "exp": now + 300,
        },
    )

    with TestClient(main.app) as client:
        missing_bearer = client.get("/api/chat/kbs", headers=auth_headers(user_id="customer-001", roles=["customer"]))
        assert missing_bearer.status_code == 401

        header_override = client.get(
            "/api/kbs",
            headers={
                **admin_headers(),
                "Authorization": f"Bearer {customer_token}",
            },
        )
        assert header_override.status_code == 403

        admin_response = client.get(
            "/api/kbs",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        admin_response.raise_for_status()
        assert any(item["key"] == "default" for item in admin_response.json())


def test_gateway_mode_requires_trusted_headers_and_ignores_client_header_override(tmp_path: Path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "auth_mode", "gateway")
    monkeypatch.setattr(settings, "gateway_shared_secret", "phase26-gateway-secret")

    with TestClient(main.app) as client:
        missing_secret = client.get(
            "/api/chat/kbs",
            headers=auth_headers(user_id="customer-001", roles=["customer"], channel="web"),
        )
        assert missing_secret.status_code == 401

        trusted_customer = client.get(
            "/api/chat/kbs",
            headers={
                **auth_headers(user_id="admin-1", roles=["admin"], channel="admin"),
                **_gateway_headers("phase26-gateway-secret", user_id="customer-001", roles=["customer"], channel="web"),
            },
        )
        trusted_customer.raise_for_status()
        payload = trusted_customer.json()
        assert any(item["key"] == "default" for item in payload)

        admin_override_denied = client.get(
            "/api/kbs",
            headers={
                **auth_headers(user_id="admin-1", roles=["admin"], channel="admin"),
                **_gateway_headers("phase26-gateway-secret", user_id="customer-001", roles=["customer"], channel="web"),
            },
        )
        assert admin_override_denied.status_code == 403

        trusted_admin = client.get(
            "/api/kbs",
            headers=_gateway_headers("phase26-gateway-secret", user_id="admin-1", roles=["admin"], channel="admin"),
        )
        trusted_admin.raise_for_status()
        assert any(item["key"] == "default" for item in trusted_admin.json())


def test_gateway_runtime_validation_rejects_missing_or_placeholder_secret(tmp_path: Path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "auth_mode", "gateway")

    monkeypatch.setattr(settings, "gateway_shared_secret", "")
    with pytest.raises(ValueError, match="RAG_GATEWAY_SHARED_SECRET"):
        settings.validate_runtime_settings()

    monkeypatch.setattr(settings, "gateway_shared_secret", "change-me")
    with pytest.raises(ValueError, match="non-placeholder"):
        settings.validate_runtime_settings()

    monkeypatch.setattr(settings, "gateway_shared_secret", "phase26-gateway-secret")
    settings.validate_runtime_settings()


def test_jwt_mode_rejects_auth_fields_in_chat_body(tmp_path: Path, monkeypatch):
    configure_test_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "auth_mode", "jwt")
    monkeypatch.setattr(settings, "allow_header_auth_in_dev", False)
    monkeypatch.setattr(settings, "jwt_shared_secret", "phase23-secret")
    monkeypatch.setattr(settings, "jwt_issuer", "https://issuer.example.test")
    monkeypatch.setattr(settings, "jwt_audience", "campusrag-api")

    now = int(time.time())
    token = _make_hs256_token(
        "phase23-secret",
        {
            "sub": "customer-001",
            "roles": ["customer"],
            "iss": "https://issuer.example.test",
            "aud": "campusrag-api",
            "exp": now + 300,
        },
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "session_id": "jwt-body-auth-denied",
                "message": "hello",
                "roles": ["admin"],
            },
        )
        assert response.status_code == 400
        assert "AUTH_MODE=jwt" in response.text
