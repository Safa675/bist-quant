# `settings/` — Environment Configuration

## Purpose

Loads all production settings from environment variables with type-safe parsing, defaults, and backward-compatible alias resolution. The `ProductionSettings` dataclass is the single source of truth for runtime configuration.

## Files

```
settings/
├── settings.py      # ProductionSettings dataclass + load_production_settings()
└── environment.py   # Environment variable parsing primitives
```

---

## `settings.py` — Production Settings

`ProductionSettings` is a `frozen=True` dataclass loaded entirely from environment variables.

### Key Fields

| Field | Env Var | Default | Description |
|---|---|---|---|
| `environment` | `BIST_APP_ENV` / `ENVIRONMENT` | `"development"` | Deployment environment |
| `log_level` | `BIST_LOG_LEVEL` | `"INFO"` | Logging level |
| `json_logs` | `BIST_JSON_LOGS` | `False` | Structured JSON log output |
| `cors_origins` | `BIST_CORS_ORIGINS` | `["*"]` | CORS allowed origins (comma-separated) |
| `auth_mode` | `BIST_AUTH_MODE` | `"none"` | `"none"`, `"api_key"`, `"jwt"`, `"either"` |
| `api_keys` | `BIST_API_KEYS` | `[]` | Comma-separated valid API keys |
| `jwt_secret` | `BIST_JWT_SECRET` | `None` | JWT signing secret |
| `jwt_algorithm` | `BIST_JWT_ALGORITHM` | `"HS256"` | JWT algorithm |
| `rate_limit_enabled` | `BIST_RATE_LIMIT_ENABLED` | `False` | Enable rate limiting |
| `rate_limit_max_requests` | `BIST_RATE_LIMIT_MAX_REQUESTS` | `100` | Requests per window |
| `enforce_https` | `BIST_ENFORCE_HTTPS` | `False` | Redirect HTTP → HTTPS |
| `telemetry_enabled` | `BIST_TELEMETRY_ENABLED` | `False` | Crash telemetry |
| `tradingview_auth_token` | `BIST_TV_AUTH_TOKEN` | `None` | TradingView streaming auth |

### Computed Properties

- `auth_enabled` — `True` when `auth_mode != "none"`.
- `tradingview_auth_config` — Formats auth token into `{"auth_token": "..."}` dict.
- `is_rate_limit_exempt(path)` — Checks `path` prefix against exempt list (e.g. `/health`, `/metrics`).

### Loading

```python
from bist_quant.settings.settings import load_production_settings

settings = load_production_settings()          # reads from os.environ
settings = load_production_settings(environ={"BIST_AUTH_MODE": "api_key", ...})  # inject for testing
```

**Alias resolution in `from_env()`:** Normalizes legacy variable names (e.g. `BIST_APP_ENV` → `ENVIRONMENT`) so both old and new names work.

---

## `environment.py` — Parsing Primitives

| Function | Description |
|---|---|
| `parse_env_str(name, default, environ?)` | Reads string with fallback |
| `parse_env_bool(name, default, environ?)` | Accepts `"1"`, `"true"`, `"yes"`, `"on"`, `"y"` (case-insensitive) |
| `parse_env_int(name, default, minimum?, maximum?, environ?)` | Parses int and clamps to `[minimum, maximum]` |
| `parse_env_list(name, environ?)` | Comma-split into cleaned list |

All functions accept an optional `environ` mapping argument for dependency injection in tests — always use this instead of `os.environ` directly in test code.

---

## Local Rules for Contributors

1. **All settings must come from environment variables** — no hardcoded secrets or credentials anywhere in the codebase.
2. **Use `parse_env_*` helpers** — do not use `os.environ.get()` directly in `settings.py`. The helpers handle empty strings, type coercion, and clamping.
3. **`frozen=True`** — `ProductionSettings` is immutable after construction. Do not add mutable fields.
4. **Add alias resolution in `from_env()`** when renaming an environment variable — both old and new names must work for at least one major version.
5. **`environ` injection** — All `parse_env_*` functions accept an `environ` parameter. Tests must inject a dict instead of mutating `os.environ`.
