"""Runtime configuration helpers for production API behavior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .environment import parse_env_bool, parse_env_int, parse_env_list, parse_env_str


@dataclass(frozen=True)
class ProductionSettings:
    environment: str
    log_level: str
    log_json: bool
    log_file: str | None
    allow_origins: list[str]
    trust_proxy_headers: bool
    enforce_https: bool
    max_request_bytes: int
    request_timeout_seconds: int
    auth_mode: str
    api_keys: tuple[str, ...]
    jwt_secret: str | None
    jwt_algorithm: str
    jwt_audience: str | None
    jwt_issuer: str | None
    rate_limit_per_minute: int
    rate_limit_exempt_paths: tuple[str, ...]
    telemetry_enabled: bool
    telemetry_file: str | None
    license_required: bool
    license_keys: tuple[str, ...]
    tradingview_auth_token: str | None = None
    tradingview_connect_timeout_seconds: int = 10

    @property
    def auth_enabled(self) -> bool:
        return self.auth_mode != "none"

    @property
    def tradingview_auth_config(self) -> dict[str, str]:
        if not self.tradingview_auth_token:
            return {}
        return {"auth_token": self.tradingview_auth_token}

    def is_rate_limit_exempt(self, path: str) -> bool:
        for prefix in self.rate_limit_exempt_paths:
            if path.startswith(prefix):
                return True
        return False

    @classmethod
    def from_env(cls, *, environ: Mapping[str, str] | None = None) -> "ProductionSettings":
        mode = parse_env_str("BIST_API_AUTH_MODE", "none", environ=environ).lower()
        if mode in {"apikey", "api-key"}:
            mode = "api_key"
        if mode not in {"none", "api_key", "jwt", "either"}:
            mode = "none"

        origins = parse_env_list("BIST_API_ALLOW_ORIGINS", environ=environ)
        if not origins:
            origins = parse_env_list("BIST_API_CORS_ORIGINS", environ=environ)
        if not origins:
            origins = ["*"]

        exempt = parse_env_list("BIST_API_RATE_LIMIT_EXEMPT", environ=environ)
        if not exempt:
            exempt = ["/", "/api/health", "/py-api/api/health", "/api/health/live", "/api/health/ready"]

        telemetry_file = parse_env_str("BIST_API_TELEMETRY_FILE", "", environ=environ)
        log_file = parse_env_str("BIST_API_LOG_FILE", "", environ=environ)
        license_required = parse_env_bool(
            "BIST_API_LICENSE_REQUIRED",
            parse_env_bool("BIST_LICENSE_REQUIRED", False, environ=environ),
            environ=environ,
        )
        license_keys = parse_env_list("BIST_API_LICENSE_KEYS", environ=environ)
        if not license_keys:
            license_keys = parse_env_list("BIST_LICENSE_KEYS", environ=environ)

        tradingview_auth_token = parse_env_str("BIST_API_TRADINGVIEW_AUTH_TOKEN", "", environ=environ)
        if not tradingview_auth_token:
            tradingview_auth_token = parse_env_str("BIST_TRADINGVIEW_AUTH_TOKEN", "", environ=environ)
        if not tradingview_auth_token:
            tradingview_auth_token = parse_env_str("TRADINGVIEW_AUTH_TOKEN", "", environ=environ)

        return cls(
            environment=parse_env_str(
                "BIST_APP_ENV",
                parse_env_str(
                    "BIST_API_ENV",
                    parse_env_str("ENVIRONMENT", "development", environ=environ),
                    environ=environ,
                ),
                environ=environ,
            ).lower(),
            log_level=parse_env_str("BIST_API_LOG_LEVEL", "INFO", environ=environ).upper(),
            log_json=parse_env_bool("BIST_API_LOG_JSON", True, environ=environ),
            log_file=log_file or None,
            allow_origins=origins,
            trust_proxy_headers=parse_env_bool("BIST_API_TRUST_PROXY_HEADERS", True, environ=environ),
            enforce_https=parse_env_bool("BIST_API_ENFORCE_HTTPS", False, environ=environ),
            max_request_bytes=parse_env_int(
                "BIST_API_MAX_REQUEST_BYTES",
                2_000_000,
                1_024,
                25_000_000,
                environ=environ,
            ),
            request_timeout_seconds=parse_env_int(
                "BIST_API_REQUEST_TIMEOUT_SECONDS",
                300,
                5,
                1_200,
                environ=environ,
            ),
            auth_mode=mode,
            api_keys=tuple(parse_env_list("BIST_API_KEYS", environ=environ)),
            jwt_secret=parse_env_str("BIST_API_JWT_SECRET", "", environ=environ) or None,
            jwt_algorithm=parse_env_str("BIST_API_JWT_ALGORITHM", "HS256", environ=environ),
            jwt_audience=parse_env_str("BIST_API_JWT_AUDIENCE", "", environ=environ) or None,
            jwt_issuer=parse_env_str("BIST_API_JWT_ISSUER", "", environ=environ) or None,
            rate_limit_per_minute=parse_env_int(
                "BIST_API_RATE_LIMIT_PER_MINUTE",
                90,
                1,
                10_000,
                environ=environ,
            ),
            rate_limit_exempt_paths=tuple(exempt),
            telemetry_enabled=parse_env_bool("BIST_API_TELEMETRY_ENABLED", True, environ=environ),
            telemetry_file=telemetry_file or None,
            license_required=license_required,
            license_keys=tuple(license_keys),
            tradingview_auth_token=tradingview_auth_token or None,
            tradingview_connect_timeout_seconds=parse_env_int(
                "BIST_API_TRADINGVIEW_CONNECT_TIMEOUT_SECONDS",
                10,
                1,
                120,
                environ=environ,
            ),
        )


def load_production_settings(*, environ: Mapping[str, str] | None = None) -> ProductionSettings:
    return ProductionSettings.from_env(environ=environ)
