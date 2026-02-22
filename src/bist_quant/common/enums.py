from __future__ import annotations

from enum import Enum
from typing import Any


class RegimeLabel(str, Enum):
    BULL = "Bull"
    BEAR = "Bear"
    RECOVERY = "Recovery"
    STRESS = "Stress"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: Any) -> "RegimeLabel | None":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            text = value.strip()
            for member in cls:
                if text == member.value:
                    return member
            lowered = text.lower()
            for member in cls:
                if lowered == member.value.lower():
                    return member
        return None
