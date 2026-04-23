"""
Authentication helpers for HTTP routes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import HTTPException, Request

from app.auth_audit import log_auth_decision
from app.authorization import can_manage_kb
from app.config import settings
from app.models import AuthContext

ADMIN_REQUIRED_DETAIL = "Admin role required"
AUTH_REQUIRED_DETAIL = "Authentication required"
INVALID_TOKEN_DETAIL = "Invalid bearer token"
INVALID_GATEWAY_DETAIL = "Invalid trusted gateway authentication"

_JWKS_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _parse_roles_header(raw: str | None) -> list[str]:
    value = (raw or "").strip()
    if not value:
        return []

    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]

    return [part.strip() for part in value.split(",") if part.strip()]


def _infer_channel(raw_channel: str | None, roles: list[str]) -> str:
    channel = (raw_channel or "").strip().lower()
    if channel:
        return channel
    normalized_roles = {str(role).strip().lower() for role in roles if str(role).strip()}
    if "admin" in normalized_roles:
        return "admin"
    return "web"


def _auth_context_from_headers(request: Request) -> AuthContext:
    roles = _parse_roles_header(request.headers.get("X-Roles"))
    return AuthContext(
        user_id=request.headers.get("X-User-Id"),
        roles=roles,
        channel=_infer_channel(request.headers.get("X-Channel"), roles),
        tenant_id=request.headers.get("X-Tenant-Id"),
        org_id=request.headers.get("X-Org-Id"),
    )


def _trusted_gateway_header(name: str) -> str:
    return name.strip()


def _auth_context_from_trusted_gateway(request: Request) -> AuthContext:
    configured_secret = settings.gateway_shared_secret.strip()
    if not configured_secret:
        raise HTTPException(status_code=401, detail="Trusted gateway shared secret is not configured")

    secret_header = _trusted_gateway_header(settings.gateway_secret_header)
    presented_secret = (request.headers.get(secret_header) or "").strip()
    if not presented_secret or not hmac.compare_digest(presented_secret, configured_secret):
        raise HTTPException(status_code=401, detail=INVALID_GATEWAY_DETAIL)

    user_id_header = _trusted_gateway_header(settings.gateway_user_id_header)
    roles_header = _trusted_gateway_header(settings.gateway_roles_header)
    channel_header = _trusted_gateway_header(settings.gateway_channel_header)
    tenant_id_header = _trusted_gateway_header(settings.gateway_tenant_id_header)
    org_id_header = _trusted_gateway_header(settings.gateway_org_id_header)
    roles = _parse_roles_header(request.headers.get(roles_header))

    return AuthContext(
        user_id=request.headers.get(user_id_header),
        roles=roles,
        channel=_infer_channel(request.headers.get(channel_header), roles),
        tenant_id=request.headers.get(tenant_id_header),
        org_id=request.headers.get(org_id_header),
    )


def _parse_bearer_token(request: Request) -> str | None:
    raw = (request.headers.get("Authorization") or "").strip()
    if not raw:
        return None
    scheme, _, token = raw.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)
    return token.strip()


def _base64url_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, binascii.Error) as err:
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL) from err


def _decode_jwt_segments(token: str) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)

    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

    try:
        header = json.loads(_base64url_decode(header_b64))
        payload = json.loads(_base64url_decode(payload_b64))
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL) from err

    if not isinstance(header, dict) or not isinstance(payload, dict):
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)

    signature = _base64url_decode(signature_b64)
    return header, payload, signing_input, signature


def _hash_algorithm_for_jwt(alg: str):
    mapping = {
        "HS256": hashlib.sha256,
        "HS384": hashlib.sha384,
        "HS512": hashlib.sha512,
        "RS256": hashes.SHA256,
        "RS384": hashes.SHA384,
        "RS512": hashes.SHA512,
    }
    algorithm = mapping.get(alg.upper())
    if algorithm is None:
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)
    return algorithm


def _verify_hmac_signature(alg: str, signing_input: bytes, signature: bytes) -> None:
    secret = settings.jwt_shared_secret.strip()
    if not secret:
        raise HTTPException(status_code=401, detail="JWT shared secret is not configured")

    digest = _hash_algorithm_for_jwt(alg)
    expected = hmac.new(secret.encode("utf-8"), signing_input, digest).digest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)


def _load_jwks_keys() -> list[dict[str, Any]]:
    jwks_url = settings.jwt_jwks_url.strip()
    if not jwks_url:
        raise HTTPException(status_code=401, detail="JWT JWKS URL is not configured")

    now = time.time()
    cached = _JWKS_CACHE.get(jwks_url)
    if cached and cached[0] > now:
        return cached[1]

    try:
        response = httpx.get(jwks_url, timeout=5.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as err:
        raise HTTPException(status_code=401, detail="Unable to load JWKS for token verification") from err

    keys = payload.get("keys")
    if not isinstance(keys, list):
        raise HTTPException(status_code=401, detail="Invalid JWKS payload")

    ttl = max(30, int(settings.jwt_jwks_cache_ttl_seconds or 300))
    _JWKS_CACHE[jwks_url] = (now + ttl, keys)
    return keys


def _match_rsa_jwk(keys: list[dict[str, Any]], kid: str | None) -> dict[str, Any]:
    rsa_keys = [key for key in keys if str(key.get("kty") or "").upper() == "RSA"]
    if not rsa_keys:
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)

    if kid:
        for key in rsa_keys:
            if key.get("kid") == kid:
                return key
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)

    if len(rsa_keys) == 1:
        return rsa_keys[0]

    raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)


def _verify_rsa_signature(alg: str, kid: str | None, signing_input: bytes, signature: bytes) -> None:
    jwk = _match_rsa_jwk(_load_jwks_keys(), kid)
    modulus = jwk.get("n")
    exponent = jwk.get("e")
    if not isinstance(modulus, str) or not isinstance(exponent, str):
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)

    try:
        public_numbers = rsa.RSAPublicNumbers(
            int.from_bytes(_base64url_decode(exponent), "big"),
            int.from_bytes(_base64url_decode(modulus), "big"),
        )
        public_key = public_numbers.public_key()
        public_key.verify(signature, signing_input, padding.PKCS1v15(), _hash_algorithm_for_jwt(alg)())
    except Exception as err:
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL) from err


def _coerce_numeric_claim(payload: dict[str, Any], claim_name: str) -> int | None:
    raw = payload.get(claim_name)
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)


def _validate_registered_claims(payload: dict[str, Any]) -> None:
    now = int(time.time())
    exp = _coerce_numeric_claim(payload, "exp")
    if exp is None or exp <= now:
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)

    nbf = _coerce_numeric_claim(payload, "nbf")
    if nbf is not None and nbf > now:
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)

    issuer = settings.jwt_issuer.strip()
    if issuer and str(payload.get("iss") or "").strip() != issuer:
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)

    audience = settings.jwt_audience.strip()
    if audience:
        aud_claim = payload.get("aud")
        if isinstance(aud_claim, str):
            audiences = [aud_claim]
        elif isinstance(aud_claim, list):
            audiences = [str(item) for item in aud_claim]
        else:
            raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)
        if audience not in audiences:
            raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)


def _extract_roles_from_claims(payload: dict[str, Any]) -> list[str]:
    roles: list[str] = []

    direct_roles = payload.get("roles")
    if isinstance(direct_roles, str):
        roles.extend([direct_roles])
    elif isinstance(direct_roles, list):
        roles.extend([str(item) for item in direct_roles])

    groups = payload.get("groups")
    if isinstance(groups, str):
        roles.extend([groups])
    elif isinstance(groups, list):
        roles.extend([str(item) for item in groups])

    realm_access = payload.get("realm_access")
    if isinstance(realm_access, dict):
        realm_roles = realm_access.get("roles")
        if isinstance(realm_roles, list):
            roles.extend([str(item) for item in realm_roles])

    seen: set[str] = set()
    normalized: list[str] = []
    for value in roles:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


def _auth_context_from_jwt(request: Request, token: str) -> AuthContext:
    header, payload, signing_input, signature = _decode_jwt_segments(token)
    alg = str(header.get("alg") or "").upper()
    kid = str(header.get("kid")).strip() if header.get("kid") is not None else None
    if not alg or alg == "NONE":
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)

    if alg.startswith("HS"):
        _verify_hmac_signature(alg, signing_input, signature)
    elif alg.startswith("RS"):
        _verify_rsa_signature(alg, kid, signing_input, signature)
    else:
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)

    _validate_registered_claims(payload)

    user_id = (
        str(payload.get("sub") or "").strip()
        or str(payload.get("user_id") or "").strip()
        or str(payload.get("preferred_username") or "").strip()
        or str(payload.get("email") or "").strip()
        or None
    )
    if not user_id:
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_DETAIL)

    return AuthContext(
        user_id=user_id,
        roles=_extract_roles_from_claims(payload),
        channel=str(payload.get("channel") or request.headers.get("X-Channel") or "web"),
        tenant_id=payload.get("tenant_id") or payload.get("tid"),
        org_id=payload.get("org_id") or payload.get("org"),
    )


def auth_context_from_request(request: Request) -> AuthContext:
    mode = settings.normalized_auth_mode
    bearer_token = _parse_bearer_token(request)

    if bearer_token:
        return _auth_context_from_jwt(request, bearer_token)

    if mode == "gateway":
        return _auth_context_from_trusted_gateway(request)

    if mode == "jwt":
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_DETAIL)

    if settings.allow_header_auth_in_dev:
        return _auth_context_from_headers(request)

    return AuthContext(channel=request.headers.get("X-Channel") or "web")


def get_request_auth(request: Request) -> AuthContext:
    return auth_context_from_request(request)


def require_admin(request: Request) -> AuthContext:
    auth = auth_context_from_request(request)
    if not can_manage_kb(auth):
        log_auth_decision(
            resource_type="route",
            resource_id=request.url.path,
            action="admin_access",
            decision="deny",
            reason="admin_role_required",
            auth_context=auth,
            request_context={"request_id": getattr(request.state, "request_id", None)},
        )
        raise HTTPException(status_code=403, detail=ADMIN_REQUIRED_DETAIL)
    log_auth_decision(
        resource_type="route",
        resource_id=request.url.path,
        action="admin_access",
        decision="allow",
        reason="admin_role_verified",
        auth_context=auth,
        request_context={"request_id": getattr(request.state, "request_id", None)},
    )
    return auth
