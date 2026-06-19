# `settings/` — Library Configuration Utilities

## Purpose

Shared path helpers, output directory resolution, and environment variable parsing primitives used by both the library and the server package.

Production API settings (`ProductionSettings`) live in `server/settings.py`.

## Files

```
settings/
├── __init__.py      # get_output_dir, PROJECT_ROOT, re-exports
├── environment.py   # Environment variable parsing primitives
└── streaming.py     # TradingView streaming auth for realtime quotes
```

---

## `__init__.py` — Path Helpers

| Symbol | Description |
|---|---|
| `PROJECT_ROOT` | Repository root (four `.parent` hops from this package) |
| `get_output_dir(*subpaths)` | Resolves `BIST_QUANT_OUTPUT_DIR` or `./outputs` |

---

## `streaming.py` — TradingView Auth

Library-side config for `realtime/quotes.py`. Reads `BIST_TRADINGVIEW_AUTH_TOKEN`, `TRADINGVIEW_AUTH_TOKEN`, and connect timeout env vars.

```python
from bist_quant.settings import load_streaming_auth_config

config = load_streaming_auth_config()
```

---

## `environment.py` — Parsing Primitives

| Function | Description |
|---|---|
| `parse_env_str(name, default, environ?)` | Reads string with fallback |
| `parse_env_bool(name, default, environ?)` | Accepts `"1"`, `"true"`, `"yes"`, `"on"`, `"y"` (case-insensitive) |
| `parse_env_int(name, default, minimum?, maximum?, environ?)` | Parses int and clamps to `[minimum, maximum]` |
| `parse_env_list(name, environ?)` | Comma-split into cleaned list |

All functions accept an optional `environ` mapping argument for dependency injection in tests.

---

## Server Settings

For JWT, CORS, rate limits, and API middleware configuration, see `server/settings.py`:

```python
from server.settings import load_production_settings

settings = load_production_settings()
```
