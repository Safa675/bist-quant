from __future__ import annotations


class QuantEngineError(RuntimeError):
    """Base class for library-level quant engine errors."""

    code = "ENGINE_ERROR"

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or message


class QuantEngineValidationError(QuantEngineError):
    code = "VALIDATION_ERROR"


class QuantEngineDataError(QuantEngineError):
    code = "DATA_ERROR"


class QuantEngineExecutionError(QuantEngineError):
    code = "EXECUTION_ERROR"
