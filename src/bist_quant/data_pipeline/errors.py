from __future__ import annotations


class FundamentalsPipelineError(Exception):
    """Base error for the fundamentals reliability pipeline."""


class FetchError(FundamentalsPipelineError):
    """Raised when a remote fetch operation fails."""


class CircuitBreakerOpenError(FetchError):
    """Raised when fetches are blocked by the circuit breaker."""


class SchemaValidationError(FundamentalsPipelineError):
    """Raised when a dataset does not satisfy schema constraints."""


class MergeError(FundamentalsPipelineError):
    """Raised when merge/integration fails."""


class FreshnessGateError(FundamentalsPipelineError):
    """Raised when freshness thresholds are violated."""


class ProvenanceError(FundamentalsPipelineError):
    """Raised when provenance metadata cannot be generated or persisted."""
