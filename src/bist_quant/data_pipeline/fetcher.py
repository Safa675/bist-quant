from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from bist_quant.data_pipeline.errors import CircuitBreakerOpenError, FetchError
from bist_quant.data_pipeline.logging_utils import append_jsonl, log_event
from bist_quant.data_pipeline.types import PipelineConfig, PipelinePaths, RawDataBundle

try:
    import httpx
except Exception as exc:  # pragma: no cover - environment dependency path
    raise ImportError("httpx is required for fundamentals fetch pipeline") from exc


ISY_BASE = "https://www.isyatirim.com.tr/_Layouts/15/IsYatirim.Website/Common"
ISY_MALI_TABLO = f"{ISY_BASE}/Data.aspx/MaliTablo"
ISY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.isyatirim.com.tr/",
}

# Known institutions typically using UFRS group on the endpoint.
BANK_TICKERS = {
    "AKBNK",
    "ALBRK",
    "DENIZ",
    "GARAN",
    "HALKB",
    "ICBCT",
    "ISCTR",
    "KLNMA",
    "QNBFB",
    "QNBFK",
    "QNBTR",
    "SKBNK",
    "TSKB",
    "VAKBN",
    "YKBNK",
}
FINANCE_TICKERS = {
    "AGESA",
    "AKGRT",
    "ANHYT",
    "ANSGR",
    "AVHOL",
    "AVIVA",
    "GUSGR",
    "HDFGS",
    "ISFIN",
    "ISGSY",
    "ISYAT",
    "RAYSG",
    "SEKFK",
    "TURSG",
    "VAKFN",
    "VKFYO",
}
UFRS_TICKERS = BANK_TICKERS | FINANCE_TICKERS


def classify_fetch_error(exc: Exception, status_code: int | None = None) -> str:
    """Map remote fetch exceptions to stable, typed classifications."""
    message = str(exc).lower()
    if "ssl" in message or "certificate" in message:
        return "ssl_cert_issue"
    if status_code == 403:
        return "auth_or_blocked"
    if status_code == 404:
        return "endpoint_missing"
    if status_code == 429:
        return "rate_limited"
    if status_code and status_code >= 500:
        return "server_error"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "connection" in message:
        return "connection_error"
    if "json" in message or "decode" in message:
        return "parse_error"
    return "unknown"


@dataclass
class CircuitBreaker:
    """Simple circuit breaker for noisy upstream data sources."""

    failure_threshold: int
    timeout_seconds: int
    failure_count: int = 0
    opened_at: datetime | None = None

    def ensure_available(self) -> None:
        if self.opened_at is None:
            return
        if datetime.now(timezone.utc) - self.opened_at >= timedelta(seconds=self.timeout_seconds):
            self.reset()
            return
        raise CircuitBreakerOpenError(
            "Fetch circuit breaker is open; too many consecutive upstream failures"
        )

    def record_success(self) -> None:
        self.reset()

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = datetime.now(timezone.utc)

    def reset(self) -> None:
        self.failure_count = 0
        self.opened_at = None


class FundamentalsFetcher:
    """Typed fetch layer with retry/backoff, circuit breaker, and raw cache."""

    def __init__(
        self,
        *,
        config: PipelineConfig,
        paths: PipelinePaths,
        logger,
    ) -> None:
        self.config = config
        self.paths = paths
        self.logger = logger
        self.breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failures,
            timeout_seconds=config.circuit_breaker_timeout_seconds,
        )

    def get_ticker_universe(self) -> list[str]:
        """Get tradable tickers from borsapy, with deterministic fallbacks."""
        try:
            import borsapy as bp

            companies = bp.companies()
            tickers = sorted(companies["ticker"].dropna().astype(str).unique().tolist())
            if tickers:
                log_event(self.logger, "ticker_universe_loaded", source="borsapy", count=len(tickers))
                return tickers
        except Exception as exc:
            log_event(self.logger, "ticker_universe_borsapy_failed", error=str(exc))

        if self.paths.consolidated_parquet.exists():
            panel = pd.read_parquet(self.paths.consolidated_parquet)
            tickers = sorted(panel.index.get_level_values("ticker").astype(str).unique().tolist())
            if tickers:
                log_event(
                    self.logger,
                    "ticker_universe_loaded",
                    source="consolidated_parquet",
                    count=len(tickers),
                )
                return tickers

        xlsx_dir = self.paths.data_dir / "fundamental_data"
        if xlsx_dir.exists():
            tickers = sorted(file.stem.split(".")[0].upper() for file in xlsx_dir.glob("*.xlsx"))
            if tickers:
                log_event(
                    self.logger,
                    "ticker_universe_loaded",
                    source="fundamental_data_xlsx",
                    count=len(tickers),
                )
                return tickers

        raise FetchError("Cannot determine ticker universe from borsapy, parquet, or xlsx fallback")

    def fetch_tickers(self, *, tickers: list[str], force: bool = False) -> RawDataBundle:
        """Fetch all requested tickers and persist raw json cache by ticker."""
        normalized_tickers = [ticker.strip().upper() for ticker in tickers if str(ticker).strip()]
        normalized_tickers = list(dict.fromkeys(normalized_tickers))
        if not normalized_tickers:
            return RawDataBundle(
                raw_by_ticker={},
                errors=[],
                source_name="isyatirim_malitablosu",
                fetched_at=datetime.now(timezone.utc),
            )

        raw_by_ticker: dict[str, dict[str, Any]] = {}
        errors: list[dict[str, Any]] = []
        log_event(
            self.logger,
            "fetch_start",
            ticker_count=len(normalized_tickers),
            force=force,
            periods=list(self.config.periods),
        )
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            for idx, ticker in enumerate(normalized_tickers):
                try:
                    payload, error_record = self._fetch_single_ticker(
                        client=client,
                        ticker=ticker,
                        force=force,
                    )
                except CircuitBreakerOpenError as exc:
                    # Deterministic failure path for remaining symbols.
                    for remaining in normalized_tickers[idx:]:
                        error_record = {
                            "ticker": remaining,
                            "endpoint": ISY_MALI_TABLO,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "classification": "circuit_breaker_open",
                            "retry_count": 0,
                        }
                        errors.append(error_record)
                        append_jsonl(self.paths.alerts_log_jsonl, {"event": "fetch_failure", **error_record})
                    break

                if payload is not None:
                    raw_by_ticker[ticker] = payload
                if error_record is not None:
                    errors.append(error_record)
                    append_jsonl(self.paths.alerts_log_jsonl, {"event": "fetch_failure", **error_record})

                if idx < len(normalized_tickers) - 1:
                    time.sleep(self.config.request_delay_seconds)

        log_event(
            self.logger,
            "fetch_complete",
            success_count=len(raw_by_ticker),
            error_count=len(errors),
            requested_count=len(normalized_tickers),
        )
        return RawDataBundle(
            raw_by_ticker=raw_by_ticker,
            errors=errors,
            source_name="isyatirim_malitablosu",
            fetched_at=datetime.now(timezone.utc),
        )

    def load_cached_raw(self, *, tickers: list[str] | None = None) -> RawDataBundle:
        """Load raw json payloads from disk cache without remote requests."""
        if tickers:
            selected = [ticker.strip().upper() for ticker in tickers if str(ticker).strip()]
            files = [self.paths.raw_dir / f"{ticker}.json" for ticker in selected]
        else:
            files = sorted(self.paths.raw_dir.glob("*.json"))

        raw_by_ticker: dict[str, dict[str, Any]] = {}
        errors: list[dict[str, Any]] = []
        for path in files:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                ticker = str(payload.get("symbol") or payload.get("ticker") or path.stem).upper()
                raw_by_ticker[ticker] = payload
            except Exception as exc:
                errors.append(
                    {
                        "ticker": path.stem.upper(),
                        "endpoint": "raw_cache",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "classification": "cache_read_error",
                        "retry_count": 0,
                    }
                )

        log_event(
            self.logger,
            "raw_cache_loaded",
            ticker_count=len(raw_by_ticker),
            error_count=len(errors),
        )
        return RawDataBundle(
            raw_by_ticker=raw_by_ticker,
            errors=errors,
            source_name="raw_json_cache",
            fetched_at=datetime.now(timezone.utc),
        )

    def _build_params(self, ticker: str) -> dict[str, Any]:
        periods = self.config.periods[:5]
        group = "UFRS" if ticker in UFRS_TICKERS else "XI_29"
        params: dict[str, Any] = {
            "companyCode": ticker,
            "exchange": "TRY",
            "financialGroup": group,
        }
        for i, (year, period) in enumerate(periods, start=1):
            params[f"year{i}"] = year
            params[f"period{i}"] = period
        return params

    def _fetch_single_ticker(
        self,
        *,
        client: httpx.Client,
        ticker: str,
        force: bool,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        raw_file = self.paths.raw_dir / f"{ticker}.json"
        if raw_file.exists() and not force:
            try:
                cached = json.loads(raw_file.read_text(encoding="utf-8"))
                return cached, None
            except Exception as exc:
                # Treat corrupt cache as a recoverable fetch attempt.
                error_record = {
                    "ticker": ticker,
                    "endpoint": "raw_cache",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "classification": "cache_decode_error",
                    "retry_count": 0,
                }
                append_jsonl(self.paths.alerts_log_jsonl, {"event": "cache_decode_error", **error_record})

        params = self._build_params(ticker)
        response: httpx.Response | None = None
        exception: Exception | None = None
        for attempt in range(self.config.max_retries):
            self.breaker.ensure_available()
            try:
                response = client.get(ISY_MALI_TABLO, params=params, headers=ISY_HEADERS)
                response.raise_for_status()
                decoded = response.json()
                if not isinstance(decoded, dict):
                    raise FetchError(f"Unexpected response payload type: {type(decoded).__name__}")
                items = decoded.get("value", [])
                if not isinstance(items, list):
                    items = []
                if not items:
                    # Fallback for alternate payload shapes.
                    list_values = [value for value in decoded.values() if isinstance(value, list)]
                    if list_values:
                        items = list_values[0]

                payload = {
                    "symbol": ticker,
                    "financial_group": params["financialGroup"],
                    "periods_requested": [list(p) for p in self.config.periods[:5]],
                    "url": str(response.url),
                    "status_code": response.status_code,
                    "items": items,
                    "raw_keys": list(decoded.keys()),
                    "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                }
                raw_file.parent.mkdir(parents=True, exist_ok=True)
                raw_file.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                self.breaker.record_success()
                return payload, None
            except Exception as exc:
                self.breaker.record_failure()
                exception = exc
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                error_class = classify_fetch_error(exc, status_code)
                if attempt < self.config.max_retries - 1:
                    wait_seconds = self.config.retry_base_seconds ** attempt
                    log_event(
                        self.logger,
                        "fetch_retry",
                        ticker=ticker,
                        attempt=attempt + 1,
                        classification=error_class,
                        wait_seconds=wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue
                error_record = {
                    "ticker": ticker,
                    "endpoint": str(response.url) if response is not None else ISY_MALI_TABLO,
                    "http_status_code": status_code,
                    "response_snippet": (
                        getattr(getattr(exc, "response", None), "text", "")[:5000]
                        if getattr(exc, "response", None) is not None
                        else ""
                    ),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "classification": error_class,
                    "retry_count": attempt + 1,
                }
                return None, error_record

        if exception is None:
            return None, None
        raise FetchError(f"Unexpected fetch failure path for {ticker}: {exception}") from exception
