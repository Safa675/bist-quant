from __future__ import annotations


class ScreeningError(RuntimeError):
    """Base class for stock screener errors."""

    code = "SCREENING_ERROR"

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or message


class ScreeningValidationError(ScreeningError):
    code = "VALIDATION_ERROR"


class ScreeningDataError(ScreeningError):
    code = "DATA_ERROR"


class ScreeningExecutionError(ScreeningError):
    code = "EXECUTION_ERROR"
