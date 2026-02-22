"""Authentication and request-identity helpers."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Any

from bist_quant.settings.settings import ProductionSettings

try:
    import jwt as pyjwt  # type: ignore
except Exception:  # pragma: no cover - optional dependency in some test envs.
    pyjwt = None


@dataclass(frozen=True)
class AuthContext:
    mode: str
    principal: str | None
    claims: dict[str, Any]


def _get_header(source: Any, name: str) -> str:
    headers = getattr(source, "headers", None)
    if headers is None:
        headers = source
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return ""
    value = getter(name, "")
    if value is None:
        return ""
    return str(value)


def resolve_client_ip(request: Any) -> str:
    # Honor proxy-forwarded chain if available.
    forwarded = _get_header(request, "x-forwarded-for").strip()
    if forwarded:
        head = forwarded.split(",")[0].strip()
        if head:
            return head
    real_ip = _get_header(request, "x-real-ip").strip()
    if real_ip:
        return real_ip
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    if host:
        return str(host)
    return "unknown"


def _api_key_from_request(request: Any) -> str | None:
    key = _get_header(request, "x-api-key")
    if key and key.strip():
        return key.strip()
    auth = _get_header(request, "authorization")
    if auth.lower().startswith("apikey "):
        token = auth[7:].strip()
        return token or None
    return None


def _bearer_token_from_request(request: Any) -> str | None:
    auth = _get_header(request, "authorization")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    return token or None


def _validate_api_key(api_key: str | None, expected_keys: tuple[str, ...]) -> bool:
    if not api_key:
        return False
    for expected in expected_keys:
        if hmac.compare_digest(api_key, expected):
            return True
    return False


def validate_license_key(request: Any, expected_keys: tuple[str, ...]) -> bool:
    if not expected_keys:
        return False
    key = _get_header(request, "x-license-key").strip()
    if not key:
        return False
    for expected in expected_keys:
        if hmac.compare_digest(key, expected):
            return True
    return False


def decode_jwt(token: str, settings: ProductionSettings) -> dict[str, Any] | None:
    if pyjwt is None or not settings.jwt_secret:
        return None
    decode_kwargs: dict[str, Any] = {
        "algorithms": [settings.jwt_algorithm],
        "options": {"verify_signature": True},
    }
    if settings.jwt_audience:
        decode_kwargs["audience"] = settings.jwt_audience
    if settings.jwt_issuer:
        decode_kwargs["issuer"] = settings.jwt_issuer
    try:
        payload = pyjwt.decode(token, settings.jwt_secret, **decode_kwargs)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _decode_jwt(token: str, settings: ProductionSettings) -> dict[str, Any] | None:
    return decode_jwt(token, settings)


def authenticate_request(
    request: Any,
    settings: ProductionSettings,
) -> tuple[bool, str | None, AuthContext | None]:
    if not settings.auth_enabled:
        return True, None, AuthContext(mode="none", principal=None, claims={})

    api_key = _api_key_from_request(request)
    token = _bearer_token_from_request(request)

    api_ok = _validate_api_key(api_key, settings.api_keys)
    jwt_claims = decode_jwt(token, settings) if token else None
    jwt_ok = jwt_claims is not None

    if settings.auth_mode == "api_key":
        if not api_ok:
            return False, "Valid API key is required.", None
        return True, None, AuthContext(mode="api_key", principal="api-key", claims={})

    if settings.auth_mode == "jwt":
        if not jwt_ok:
            return False, "Valid Bearer JWT token is required.", None
        principal = str(jwt_claims.get("sub") or jwt_claims.get("email") or "jwt-user")
        return True, None, AuthContext(mode="jwt", principal=principal, claims=jwt_claims)

    # either
    if api_ok:
        return True, None, AuthContext(mode="api_key", principal="api-key", claims={})
    if jwt_ok and jwt_claims is not None:
        principal = str(jwt_claims.get("sub") or jwt_claims.get("email") or "jwt-user")
        return True, None, AuthContext(mode="jwt", principal=principal, claims=jwt_claims)
    return False, "Provide a valid API key or Bearer JWT token.", None
