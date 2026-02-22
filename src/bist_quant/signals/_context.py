import pandas as pd


def get_runtime_context(config: dict | None) -> dict:
    if not isinstance(config, dict):
        return {}

    runtime_context = config.get("_runtime_context")
    if isinstance(runtime_context, dict):
        return runtime_context

    return {}


def require_context(signal_name: str, context: dict, key: str):
    value = context.get(key)
    if value is None:
        raise ValueError(f"Signal '{signal_name}' requires runtime context '{key}'")
    return value


def parse_int_param(signal_name: str, params: dict, key: str, default: int) -> int:
    value = params.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Signal '{signal_name}' expects integer param '{key}', got {value!r}"
        ) from exc


def parse_float_param(signal_name: str, params: dict, key: str, default: float) -> float:
    value = params.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Signal '{signal_name}' expects float param '{key}', got {value!r}"
        ) from exc


def parse_optional_str_list(signal_name: str, params: dict, key: str) -> list[str] | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"Signal '{signal_name}' expects list[str] param '{key}', got {type(value).__name__}"
        )
    return value


def parse_optional_dict(signal_name: str, params: dict, key: str) -> dict | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(
            f"Signal '{signal_name}' expects dict param '{key}', got {type(value).__name__}"
        )
    return value


def build_high_low_panels(
    signal_name: str,
    context: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_df = context.get("high_df")
    low_df = context.get("low_df")
    if isinstance(high_df, pd.DataFrame) and isinstance(low_df, pd.DataFrame):
        return high_df, low_df

    prices = require_context(signal_name, context, "prices")
    if "Date" not in prices.columns or "Ticker" not in prices.columns:
        raise ValueError(
            f"Signal '{signal_name}' requires prices with Date/Ticker columns to build high/low panels"
        )

    high_df = prices.pivot_table(index="Date", columns="Ticker", values="High").sort_index()
    high_df.columns = [column.split(".")[0].upper() for column in high_df.columns]

    low_df = prices.pivot_table(index="Date", columns="Ticker", values="Low").sort_index()
    low_df.columns = [column.split(".")[0].upper() for column in low_df.columns]

    context["high_df"] = high_df
    context["low_df"] = low_df
    return high_df, low_df
