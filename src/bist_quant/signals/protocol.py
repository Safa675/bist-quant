from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class SignalBuilder(Protocol):
    def __call__(
        self,
        dates: pd.DatetimeIndex,
        loader: Any,
        config: dict[str, Any],
        signal_params: dict[str, Any],
    ) -> pd.DataFrame: ...
