"""Structured API error helpers for jobs-family endpoints."""

from __future__ import annotations

import re
from typing import Any, Iterable

from fastapi import HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


_JOBS_PATH_PATTERN = re.compile(r"^/api/jobs(?:/|$)")


class ApiErrorDetail(BaseModel):
    """Machine-readable error detail payload."""

    code: str = Field(..., min_length=1)
    detail: str = Field(..., min_length=1)
    hint: str | None = None
    errors: list[dict[str, Any]] | None = None


class ApiErrorEnvelope(BaseModel):
    """Top-level FastAPI error envelope."""

    detail: ApiErrorDetail


def _normalize_errors(errors: Iterable[Any] | None) -> list[dict[str, Any]] | None:
    if errors is None:
        return None

    normalized: list[dict[str, Any]] = []
    for item in errors:
        if isinstance(item, dict):
            normalized.append(dict(item))
        else:
            normalized.append({"error": str(item)})
    return normalized or None


def api_error_detail(
    *,
    code: str,
    detail: str,
    hint: str | None = None,
    errors: Iterable[Any] | None = None,
) -> dict[str, Any]:
    payload = ApiErrorDetail(
        code=code,
        detail=detail,
        hint=hint,
        errors=_normalize_errors(errors),
    )
    return payload.model_dump(exclude_none=True)


def api_http_exception(
    *,
    status_code: int,
    code: str,
    detail: str,
    hint: str | None = None,
    errors: Iterable[Any] | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=api_error_detail(code=code, detail=detail, hint=hint, errors=errors),
    )


def job_validation_http_exception(
    *,
    kind: str,
    detail: str,
    errors: Iterable[Any] | None = None,
    hint: str | None = None,
) -> HTTPException:
    fallback_hint = (
        f"Check the request schema for job kind '{kind}' and send fields in backend contract format."
    )
    return api_http_exception(
        status_code=422,
        code="job_validation_error",
        detail=detail,
        hint=hint or fallback_hint,
        errors=errors,
    )


def unsupported_job_kind_http_exception(kind: str, valid_kinds: set[str]) -> HTTPException:
    return api_http_exception(
        status_code=400,
        code="unsupported_job_kind",
        detail=f"Unsupported job kind '{kind}'.",
        hint=f"Use one of: {sorted(valid_kinds)}",
    )


def job_not_found_http_exception(job_id: str) -> HTTPException:
    return api_http_exception(
        status_code=404,
        code="job_not_found",
        detail=f"Job '{job_id}' not found.",
        hint="List jobs with GET /api/jobs to retrieve a valid job id.",
    )


def retry_unsupported_job_kind_http_exception(kind: str) -> HTTPException:
    return api_http_exception(
        status_code=400,
        code="retry_unsupported_job_kind",
        detail=f"Cannot retry unsupported job kind '{kind}'.",
        hint="Retry is supported only for known kinds accepted by POST /api/jobs.",
    )


def job_request_missing_http_exception(job_id: str) -> HTTPException:
    return api_http_exception(
        status_code=400,
        code="job_request_missing",
        detail=f"Job '{job_id}' request payload missing.",
        hint="Retry requires a persisted non-empty request payload.",
    )


def is_jobs_route_path(path: str) -> bool:
    return bool(_JOBS_PATH_PATTERN.match(path))


async def jobs_request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Return structured 422 payloads for jobs-family request-validation errors."""

    if not is_jobs_route_path(request.url.path):
        return await request_validation_exception_handler(request, exc)

    return JSONResponse(
        status_code=422,
        content={
            "detail": api_error_detail(
                code="request_validation_error",
                detail="Invalid request payload for jobs endpoint.",
                hint="Check required fields, data types, and path/query parameters.",
                errors=exc.errors(),
            )
        },
    )
