# `security/` — Authentication, Rate Limiting & Sanitization

## Purpose

Provides authentication, rate limiting, and input sanitization for the HTTP API layer. All security primitives follow defensive coding practices: timing-attack-safe comparisons, safe key comparisons, and recursive input sanitization.

## Files

```
security/
├── auth.py           # API key, JWT, and "either" auth modes
├── rate_limiter.py   # Sliding-window in-memory rate limiter
└── sanitization.py   # Recursive payload sanitizer
```

---

## `auth.py` — Request Authentication

Supports four auth modes (configured via `ProductionSettings.auth_mode`):

| Mode | Behavior |
|---|---|
| `"none"` | All requests pass through, `AuthContext.principal = "anonymous"` |
| `"api_key"` | Validates `X-License-Key` header via HMAC-safe comparison |
| `"jwt"` | Validates `Authorization: Bearer <token>` via pyjwt |
| `"either"` | Accepts API key OR JWT (tries both) |

```python
from bist_quant.security.auth import authenticate_request

ok, error_msg, ctx = authenticate_request(request, settings)
if not ok:
    return 401, error_msg
# ctx.principal, ctx.mode, ctx.claims available
```

**Key functions:**

| Function | Description |
|---|---|
| `authenticate_request(request, settings)` | Main auth dispatcher — returns `(ok, error, AuthContext \| None)` |
| `validate_license_key(request, expected_keys)` | HMAC-safe API key check via `X-License-Key` header |
| `decode_jwt(token, settings)` | Validates JWT; returns claims dict or `None` if invalid |
| `resolve_client_ip(request)` | `X-Forwarded-For` → `X-Real-IP` → `request.client.host` |

**Critical:** All secret comparisons use `hmac.compare_digest` — never use `==` for key comparison.

---

## `rate_limiter.py` — Sliding-Window Rate Limiter

In-memory, thread-safe, keyed by arbitrary string (typically `f"{client_ip}:{route}"`).

```python
from bist_quant.security.rate_limiter import InMemoryRateLimiter

limiter = InMemoryRateLimiter(max_requests=100, window_seconds=60)
decision = limiter.check(f"{client_ip}:{path}")
if not decision.allowed:
    return 429, f"Rate limit exceeded. Retry after {decision.retry_after_seconds}s"
```

`RateLimitDecision` fields: `allowed`, `retry_after_seconds`, `remaining`.

**Implementation:** `collections.deque` per key, `threading.Lock` for safety, O(1) eviction of expired timestamps.

---

## `sanitization.py` — Input Payload Sanitizer

Recursively sanitizes any nested dict/list/string payload:

```python
from bist_quant.security.sanitization import sanitize_payload

safe = sanitize_payload(user_input, max_string_length=1000)
```

**What it removes / truncates:**
- C0/C1 control characters (regex `[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]`) — preserves `\t`, `\n`, `\r`.
- Strings longer than `max_string_length` (default 1000).
- Dict keys longer than 128 characters.
- Unknown types are converted to `str`.

---

## Local Rules for Contributors

1. **Always use `hmac.compare_digest` for secret comparisons.** Never use `==`, `in`, or any other operator for comparing API keys, tokens, or hashes.
2. **Sanitize all user-provided inputs** before persistence or display. Call `sanitize_payload()` at the API boundary — do not rely on individual handlers to sanitize.
3. **Rate limit keys must be deterministic.** Use `f"{client_ip}:{path}"` — do not include timestamps or random values in the key.
4. **`pyjwt` is optional.** If it is not installed, `decode_jwt` returns `None`. The auth system must handle this gracefully (fall back to API key mode or reject).
5. **`resolve_client_ip` must be used** for all rate-limiting and auth logging — never use `request.client.host` directly (it will be wrong behind a proxy).
